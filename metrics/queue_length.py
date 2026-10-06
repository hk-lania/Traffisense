"""
TraffiSense - metrics/queue.py
-------------------------------
Queue length = how far the waiting traffic extends back from the stop line.

            direction of travel
                    |
                    v
    ----------------------------------  <- back of queue (the answer)
            [car] [car] [car]
            [car] [car]
            [car]
    ================================== STOP LINE

Input : tracked vehicle boxes for one approach + their motion state
Output: queue length in pixels (and metres when the ROI is calibrated),
        plus how many vehicles are in that queue.

Why it is not simply "distance of the farthest vehicle"
=======================================================
Two vehicles can sit inside the same ROI without being in the same queue - one
stopped at the light, another still rolling in 60 m behind it.  Measuring to
the farthest one would report a queue that does not exist.  So the queue is
built as a CHAIN:

  1. keep only vehicles that are stopped (or barely moving),
  2. start from the one nearest the stop line - it must actually be near it,
     otherwise nobody is queued,
  3. walk outward, adding a vehicle while the gap to the previous one is small
     enough to count as "same queue",
  4. stop at the first big gap; the queue ends at the last vehicle added.

The result is an independent value.  It is NOT converted into the Traffic
Pressure Index here - pressure.py does that later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from .motion import MotionTracker, TrackMotion
from .roi import ApproachROI, IntersectionROI
from .schema import Detection, FrameData, as_detection

__all__ = ["QueueResult", "QueueEstimator", "QueuedVehicle"]

# Typical vehicle footprint used to convert a queue length into a vehicle
# count when no per-vehicle data is available (metres, incl. gap).
DEFAULT_VEHICLE_SLOT_M = 6.0


@dataclass
class QueuedVehicle:
    """One vehicle that was accepted into the queue chain."""

    track_id: int
    vehicle_class: str
    front_px: float          # distance from stop line to its near edge
    rear_px: float           # distance from stop line to its far edge
    speed_px_s: float
    stationary: bool


@dataclass
class QueueResult:
    """Queue state of one approach at one instant."""

    approach: str
    length_px: float
    length_m: Optional[float]
    queued_count: int
    stopped_count: int            # stopped vehicles in the ROI (chain or not)
    vehicles_in_roi: int
    max_queue_px: float
    vehicles: list[QueuedVehicle] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Queue extent as 0 - 1 of the ROI's full length (for later fusion)."""
        if self.max_queue_px <= 0:
            return 0.0
        return max(0.0, min(1.0, self.length_px / self.max_queue_px))

    @property
    def track_ids(self) -> list[int]:
        return [v.track_id for v in self.vehicles]

    def vehicle_equivalents(self, slot_m: float = DEFAULT_VEHICLE_SLOT_M) -> float:
        """Queue length expressed as an approximate number of vehicle slots."""
        if self.length_m is None or slot_m <= 0:
            return float(self.queued_count)
        return self.length_m / slot_m

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "length_px": round(self.length_px, 1),
            "length_m": None if self.length_m is None else round(self.length_m, 2),
            "queued_count": self.queued_count,
            "stopped_count": self.stopped_count,
            "vehicles_in_roi": self.vehicles_in_roi,
            "score": round(self.score, 3),
            "track_ids": self.track_ids,
        }


class QueueEstimator:
    """Estimates queue length for ONE approach ROI.

    Parameters
    ----------
    roi:
        Approach geometry - supplies the stop line and the queue direction.
    motion:
        Shared MotionTracker, so "stopped" means the same thing here as it
        does in waiting_time.py.  If omitted, one is created internally.
    max_gap_m / max_gap_px:
        Largest empty space between two vehicles that still counts as one
        queue.  Metres are used when the ROI is calibrated; otherwise pixels.
    first_gap_m / first_gap_px:
        How close the head of the queue must be to the stop line for a queue
        to exist at all.  Prevents reporting a queue when the road is simply
        occupied further upstream.
    require_stationary:
        True  - only stopped vehicles are queued (correct for a red phase).
        False - every vehicle in the ROI is a candidate (useful for slow
                crawling traffic where nothing is fully at rest).
    creep_speed_px_s:
        A vehicle slower than this counts as queued even if the motion
        tracker has not yet latched it as stopped - handles stop-and-go creep.
    """

    def __init__(self,
                 roi: ApproachROI,
                 motion: Optional[MotionTracker] = None,
                 max_gap_m: float = 9.0,
                 max_gap_px: Optional[float] = None,
                 first_gap_m: float = 12.0,
                 first_gap_px: Optional[float] = None,
                 require_stationary: bool = True,
                 creep_speed_px_s: Optional[float] = None,
                 require_inside: bool = True) -> None:
        self.roi = roi
        self.motion = motion if motion is not None else MotionTracker()
        self.require_stationary = require_stationary
        self.require_inside = require_inside
        self.creep_speed_px_s = (creep_speed_px_s
                                 if creep_speed_px_s is not None
                                 else self.motion.stop_speed_px_s)

        ppm = roi.pixels_per_meter
        self.max_gap_px = float(max_gap_px if max_gap_px is not None
                                else (max_gap_m * ppm if ppm else 120.0))
        self.first_gap_px = float(first_gap_px if first_gap_px is not None
                                  else (first_gap_m * ppm if ppm else 160.0))

    # ------------------------------------------------------------------
    def _relevant(self, detections: Iterable) -> list[Detection]:
        out: list[Detection] = []
        for raw in detections or []:
            det = as_detection(raw)
            if det.lane is not None and det.lane != self.roi.name:
                continue
            if self.require_inside and not self.roi.contains_vehicle(det):
                continue
            out.append(det)
        return out

    def _edges(self, det: Detection) -> tuple[float, float]:
        """(near edge, far edge) distance from the stop line, in pixels."""
        dists = [self.roi.distance_from_stop_line(c) for c in det.corners]
        return min(dists), max(dists)

    # ------------------------------------------------------------------
    def compute(self,
                detections: Iterable,
                motion_states: Optional[Mapping[int, TrackMotion]] = None,
                update_motion: bool = True,
                timestamp: Optional[float] = None) -> QueueResult:
        """Queue length for this approach from one frame's detections.

        Pass `motion_states` (and update_motion=False) when a shared
        MotionTracker has already been updated for this frame - that is what
        the aggregator does, so the tracker is stepped exactly once per frame.
        """
        vehicles = self._relevant(detections)

        if update_motion:
            motion_states = self.motion.update(vehicles, timestamp)
        states: Mapping[int, TrackMotion] = motion_states or {}

        candidates: list[QueuedVehicle] = []
        stopped_count = 0
        for det in vehicles:
            state = states.get(det.track_id) or self.motion.get(det.track_id)
            speed = state.speed_px_s if state else 0.0
            stationary = bool(state.stationary) if state else False
            if stationary:
                stopped_count += 1

            front, rear = self._edges(det)
            if rear <= 0:
                continue        # fully across the stop line, inside the junction

            queued = (not self.require_stationary
                      or stationary
                      or speed <= self.creep_speed_px_s)
            if queued:
                candidates.append(QueuedVehicle(
                    track_id=det.track_id,
                    vehicle_class=det.vehicle_class,
                    front_px=max(front, 0.0),
                    rear_px=rear,
                    speed_px_s=speed,
                    stationary=stationary,
                ))

        chain = self._build_chain(candidates)
        length_px = max((v.rear_px for v in chain), default=0.0)

        return QueueResult(
            approach=self.roi.name,
            length_px=length_px,
            length_m=self.roi.to_meters(length_px),
            queued_count=len(chain),
            stopped_count=stopped_count,
            vehicles_in_roi=len(vehicles),
            max_queue_px=float(self.roi.max_queue_pixels or 0.0),
            vehicles=chain,
        )

    def _build_chain(self, candidates: Sequence[QueuedVehicle]) -> list[QueuedVehicle]:
        """Walk outward from the stop line, breaking at the first large gap."""
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda v: v.front_px)
        head = ordered[0]
        if head.front_px > self.first_gap_px:
            return []           # nothing is actually waiting at the line

        chain = [head]
        rear = head.rear_px
        for vehicle in ordered[1:]:
            gap = vehicle.front_px - rear
            if gap > self.max_gap_px:
                break           # empty road: the queue ended at the last vehicle
            chain.append(vehicle)
            rear = max(rear, vehicle.rear_px)
        return chain


# ---------------------------------------------------------------------------
# Convenience: every approach at once
# ---------------------------------------------------------------------------

def compute_all(intersection: IntersectionROI,
                frame: FrameData | Iterable,
                motion: Optional[MotionTracker] = None,
                **kwargs) -> dict[str, QueueResult]:
    """Queue length for all approaches, sharing one MotionTracker."""
    detections = frame.detections if isinstance(frame, FrameData) else list(frame)
    timestamp = frame.timestamp if isinstance(frame, FrameData) else None
    motion = motion if motion is not None else MotionTracker()
    states = motion.update([as_detection(d) for d in detections], timestamp)
    return {
        roi.name: QueueEstimator(roi, motion=motion, **kwargs).compute(
            detections, motion_states=states, update_motion=False)
        for roi in intersection
    }
