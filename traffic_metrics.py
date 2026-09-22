"""
TraffiSense - metrics/traffic_metrics.py
-----------------------------------------
The Traffic Metrics Module's front door.

One call per frame, one metric block per approach:

    engine = TrafficMetricsEngine(intersection)
    snapshot = engine.update(tracked_vehicles, frame=i, timestamp=t)

    snapshot["north"].vehicle_count
    snapshot["north"].density.density_percent
    snapshot["north"].queue.length_m
    snapshot["north"].waiting.max_wait_s
    snapshot["north"].pressure.pressure

Everything this module does is delegated to the four independent metric
modules; its own job is only to:

  * normalise whatever the tracker handed over (schema.as_frame),
  * label each vehicle with the approach it is standing on,
  * step the shared MotionTracker exactly ONCE per frame, so queue.py and
    waiting_time.py agree about which vehicles are stopped,
  * collect the four results per approach and hand them to pressure.py.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from .density import DensityEstimator, DensityResult
from .motion import MotionTracker
from .pressure import PressureCalculator, PressureConfig, PressureResult
from .queue import QueueEstimator, QueueResult
from .roi import IntersectionROI
from .schema import Detection, FrameData, as_frame
from .waiting_time import WaitingTimeResult, WaitingTimeTracker

__all__ = ["ApproachMetrics", "TrafficMetricsEngine"]


@dataclass
class ApproachMetrics:
    """All five numbers for one approach at one instant."""

    approach: str
    vehicle_count: int
    density: DensityResult
    queue: QueueResult
    waiting: WaitingTimeResult
    pressure: PressureResult
    frame: int = 0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Flat, JSON-friendly view - the shape the signal controller reads."""
        return {
            "approach": self.approach,
            "frame": self.frame,
            "timestamp": round(self.timestamp, 3),
            "vehicle_count": self.vehicle_count,
            "density_percent": round(self.density.density_percent, 2),
            "queue_length_px": round(self.queue.length_px, 1),
            "queue_length_m": (None if self.queue.length_m is None
                               else round(self.queue.length_m, 2)),
            "queued_vehicles": self.queue.queued_count,
            "waiting_time_s": round(self.waiting.max_wait_s, 2),
            "mean_waiting_time_s": round(self.waiting.mean_wait_s, 2),
            "waiting_vehicles": self.waiting.waiting_count,
            "pressure": round(self.pressure.pressure, 2),
            "pressure_level": self.pressure.level,
        }

    def detail(self) -> dict:
        """Full nested view including every module's own breakdown."""
        return {
            "approach": self.approach,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "vehicle_count": self.vehicle_count,
            "density": self.density.to_dict(),
            "queue": self.queue.to_dict(),
            "waiting_time": self.waiting.to_dict(),
            "pressure": self.pressure.to_dict(),
        }


class TrafficMetricsEngine:
    """Runs all four metric modules over a tracked-vehicle stream.

    Parameters
    ----------
    intersection:
        The approach geometry (see roi.py).
    pressure_config:
        Provisional fusion settings; see pressure.py.
    motion:
        Optional shared MotionTracker.  Supply one tuned to your camera:
        its `stop_speed_px_s` should be roughly 0.5 m/s in pixels.
    auto_lane:
        Assign a lane to detections that arrive without one, by testing which
        ROI the vehicle's ground point falls inside.  Leave True unless the
        detection side already labels lanes reliably.
    density_options / queue_options / waiting_options:
        Passed through to the respective modules.
    """

    def __init__(self,
                 intersection: IntersectionROI,
                 pressure_config: Optional[PressureConfig] = None,
                 motion: Optional[MotionTracker] = None,
                 auto_lane: bool = True,
                 density_options: Optional[Mapping[str, Any]] = None,
                 queue_options: Optional[Mapping[str, Any]] = None,
                 waiting_options: Optional[Mapping[str, Any]] = None) -> None:
        self.intersection = intersection
        self.auto_lane = auto_lane
        self.motion = motion if motion is not None else MotionTracker()

        d_opts = dict(density_options or {})
        q_opts = dict(queue_options or {})
        w_opts = dict(waiting_options or {})

        self.density = {roi.name: DensityEstimator(roi, **d_opts)
                        for roi in intersection}
        self.queues = {roi.name: QueueEstimator(roi, motion=self.motion, **q_opts)
                       for roi in intersection}
        self.waiting = WaitingTimeTracker(intersection=intersection,
                                          motion=self.motion, **w_opts)
        self.pressure = PressureCalculator(pressure_config)

        self.last_snapshot: dict[str, ApproachMetrics] = {}
        self.frames_processed = 0

    # ------------------------------------------------------------------
    def _label_lanes(self, detections: list[Detection]) -> list[Detection]:
        if not self.auto_lane:
            return detections
        labelled: list[Detection] = []
        for det in detections:
            if det.lane in self.intersection.approaches:
                labelled.append(det)
                continue
            lane = None
            for roi in self.intersection:
                if roi.contains(det.ground_point):
                    lane = roi.name
                    break
            labelled.append(dataclasses.replace(det, lane=lane)
                            if lane != det.lane else det)
        return labelled

    # ------------------------------------------------------------------
    def update(self,
               detections: Iterable | FrameData,
               frame: int = 0,
               timestamp: Optional[float] = None,
               fps: float = 25.0,
               camera_confidence: float | Mapping[str, float] | None = None,
               optical_flow: float | Mapping[str, float] | None = None
               ) -> dict[str, ApproachMetrics]:
        """Process one frame and return the per-approach metric block."""
        if isinstance(detections, FrameData):
            frame_data = detections
        else:
            frame_data = as_frame(detections, frame=frame,
                                  timestamp=timestamp, fps=fps)

        frame_data.detections = self._label_lanes(frame_data.detections)
        ts = frame_data.timestamp

        # One motion step per frame, shared by queue and waiting time.
        states = self.motion.update(frame_data.detections, ts)

        self.waiting.update(frame_data, timestamp=ts,
                            motion_states=states, update_motion=False)

        snapshot: dict[str, ApproachMetrics] = {}
        for name in self.intersection.names:
            mine = frame_data.by_lane(name)

            density = self.density[name].compute(mine)
            queue = self.queues[name].compute(mine, motion_states=states,
                                              update_motion=False)
            waiting = self.waiting.result(name)

            pressure = self.pressure.compute(
                approach=name,
                vehicle_count=len(mine),
                density=density,
                queue=queue,
                waiting=waiting,
                optical_flow=_per_approach(optical_flow, name),
                camera_confidence=_per_approach(camera_confidence, name),
            )

            snapshot[name] = ApproachMetrics(
                approach=name, vehicle_count=len(mine),
                density=density, queue=queue, waiting=waiting,
                pressure=pressure, frame=frame_data.frame, timestamp=ts,
            )

        self.last_snapshot = snapshot
        self.frames_processed += 1
        return snapshot

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Last snapshot as plain JSON-ready data."""
        return {name: m.to_dict() for name, m in self.last_snapshot.items()}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def pressure_ranking(self) -> list[tuple[str, float]]:
        """Approaches ordered by pressure, highest first.

        Handed to the FUTURE adaptive signal controller; no control decision
        is made inside this package.
        """
        return PressureCalculator.rank(
            {name: m.pressure for name, m in self.last_snapshot.items()})

    def reset(self) -> None:
        self.motion.reset()
        self.waiting.reset()
        self.last_snapshot = {}
        self.frames_processed = 0


def _per_approach(value: float | Mapping[str, float] | None,
                  name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        got = value.get(name)
        return None if got is None else float(got)
    return float(value)
