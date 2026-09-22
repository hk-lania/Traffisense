"""
TraffiSense - metrics/waiting_time.py
--------------------------------------
Waiting time = how long a vehicle has been standing still at the approach.

    Vehicle #17
        |
      speed ~ 0
        |
      5 s -> 10 s -> 18 s waiting

Input : tracked vehicle boxes across consecutive frames (track_id is essential)
Output: per-approach waiting statistics - longest wait, mean wait, total
        vehicle-seconds of delay, and how many vehicles are waiting.

Why per-track state is needed
=============================
Density and queue length are instantaneous - they can be computed from a single
frame.  Waiting time cannot: it is an accumulation, so this module keeps a
small record per track_id and adds elapsed time on every frame in which that
vehicle is judged stationary.

Robustness details that matter in practice
==========================================
* Stationary/moving comes from the shared MotionTracker, which smooths speed
  and uses two thresholds (hysteresis).  Detector jitter therefore does not
  reset a 20-second wait back to zero.
* Time is accumulated from TIMESTAMPS, not frame counts, so dropped frames or
  a variable frame rate do not distort the result.
* A brief tracking dropout (the same car re-appearing a few frames later under
  the same id) does not lose the accumulated wait; a genuinely long absence
  retires the record.
* `total_wait_s` keeps accumulating across a vehicle's separate stop episodes,
  which is the delay that actually matters at a junction with stop-and-go flow.

The values here stay independent.  pressure.py is what later normalises and
fuses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .motion import MotionTracker, TrackMotion
from .roi import ApproachROI, IntersectionROI
from .schema import Detection, FrameData, as_detection, normalise_direction

__all__ = ["WaitRecord", "WaitingTimeResult", "WaitingTimeTracker"]


@dataclass
class WaitRecord:
    """Accumulated waiting state of a single tracked vehicle."""

    track_id: int
    vehicle_class: str = "vehicle"
    lane: Optional[str] = None
    current_wait_s: float = 0.0     # length of the stop episode in progress
    total_wait_s: float = 0.0       # sum over all stop episodes
    stop_episodes: int = 0
    waiting: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_update: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class": self.vehicle_class,
            "lane": self.lane,
            "current_wait_s": round(self.current_wait_s, 2),
            "total_wait_s": round(self.total_wait_s, 2),
            "stop_episodes": self.stop_episodes,
            "waiting": self.waiting,
        }


@dataclass
class WaitingTimeResult:
    """Waiting statistics of one approach at one instant."""

    approach: str
    max_wait_s: float = 0.0
    mean_wait_s: float = 0.0
    total_wait_s: float = 0.0       # vehicle-seconds of delay on this approach
    waiting_count: int = 0
    vehicles_in_roi: int = 0
    longest_track_id: Optional[int] = None
    normalisation_cap_s: float = 60.0

    @property
    def score(self) -> float:
        """Longest wait as 0 - 1 against the cap (for later fusion)."""
        if self.normalisation_cap_s <= 0:
            return 0.0
        return max(0.0, min(1.0, self.max_wait_s / self.normalisation_cap_s))

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "max_wait_s": round(self.max_wait_s, 2),
            "mean_wait_s": round(self.mean_wait_s, 2),
            "total_wait_s": round(self.total_wait_s, 2),
            "waiting_count": self.waiting_count,
            "vehicles_in_roi": self.vehicles_in_roi,
            "longest_track_id": self.longest_track_id,
            "score": round(self.score, 3),
        }


class WaitingTimeTracker:
    """Accumulates per-vehicle waiting time and reports it per approach.

    Parameters
    ----------
    intersection:
        Used to decide which approach a vehicle belongs to when the detection
        does not already carry a lane label.  Optional: with lane-labelled
        detections the tracker works without any geometry.
    motion:
        Shared MotionTracker (the same instance queue.py uses, ideally).
    normalisation_cap_s:
        Wait considered "as bad as it gets" - only used for the 0-1 score,
        never inside the raw seconds.
    forget_after_s:
        A track absent for longer than this is retired.  Shorter absences are
        treated as a tracking dropout and the accumulated wait survives.
    count_only_in_roi:
        Ignore vehicles that are inside the frame but outside every ROI.
    """

    def __init__(self,
                 intersection: Optional[IntersectionROI] = None,
                 motion: Optional[MotionTracker] = None,
                 normalisation_cap_s: float = 60.0,
                 forget_after_s: float = 5.0,
                 count_only_in_roi: bool = True) -> None:
        self.intersection = intersection
        self.motion = motion if motion is not None else MotionTracker()
        self.normalisation_cap_s = float(normalisation_cap_s)
        self.forget_after_s = float(forget_after_s)
        self.count_only_in_roi = count_only_in_roi
        self.records: dict[int, WaitRecord] = {}
        self._last_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    def _approach_of(self, det: Detection) -> Optional[str]:
        if det.lane:
            return det.lane
        if self.intersection is None:
            return None
        for roi in self.intersection:
            if roi.contains_vehicle(det):
                return roi.name
        return None

    # ------------------------------------------------------------------
    def update(self,
               frame: FrameData | Iterable,
               timestamp: Optional[float] = None,
               motion_states: Optional[Mapping[int, TrackMotion]] = None,
               update_motion: bool = True) -> dict[int, WaitRecord]:
        """Advance every track's waiting time by one frame."""
        if isinstance(frame, FrameData):
            detections = frame.detections
            timestamp = frame.timestamp if timestamp is None else timestamp
        else:
            detections = [as_detection(d) for d in (frame or [])]
            if timestamp is None:
                timestamp = max((d.timestamp for d in detections), default=0.0)

        detections = [as_detection(d) for d in detections]

        if update_motion:
            motion_states = self.motion.update(detections, timestamp)
        states: Mapping[int, TrackMotion] = motion_states or {}

        for det in detections:
            approach = self._approach_of(det)
            if self.count_only_in_roi and approach is None:
                continue

            record = self.records.get(det.track_id)
            if record is None:
                record = WaitRecord(
                    track_id=det.track_id,
                    vehicle_class=det.vehicle_class,
                    lane=approach,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    last_update=timestamp,
                )
                self.records[det.track_id] = record

            record.lane = approach or record.lane
            record.vehicle_class = det.vehicle_class or record.vehicle_class

            dt = max(0.0, timestamp - record.last_update)
            state = states.get(det.track_id) or self.motion.get(det.track_id)
            stationary = bool(state.stationary) if state else False

            if stationary:
                if not record.waiting:
                    record.waiting = True
                    record.stop_episodes += 1
                    record.current_wait_s = 0.0
                record.current_wait_s += dt
                record.total_wait_s += dt
            else:
                record.waiting = False
                record.current_wait_s = 0.0

            record.last_seen = timestamp
            record.last_update = timestamp

        self._retire(timestamp)
        self._last_timestamp = timestamp
        return self.records

    def _retire(self, timestamp: float) -> None:
        gone = [tid for tid, r in self.records.items()
                if timestamp - r.last_seen > self.forget_after_s]
        for tid in gone:
            del self.records[tid]

    # ------------------------------------------------------------------
    def result(self, approach: str | ApproachROI) -> WaitingTimeResult:
        """Waiting statistics for one approach, as of the last update()."""
        name = approach.name if isinstance(approach, ApproachROI) else \
            (normalise_direction(approach) or str(approach))

        mine = [r for r in self.records.values() if r.lane == name]
        waiting = [r for r in mine if r.waiting]

        if not waiting:
            return WaitingTimeResult(
                approach=name, vehicles_in_roi=len(mine),
                normalisation_cap_s=self.normalisation_cap_s,
            )

        waits = [r.current_wait_s for r in waiting]
        longest = max(waiting, key=lambda r: r.current_wait_s)
        return WaitingTimeResult(
            approach=name,
            max_wait_s=max(waits),
            mean_wait_s=sum(waits) / len(waits),
            total_wait_s=sum(r.total_wait_s for r in mine),
            waiting_count=len(waiting),
            vehicles_in_roi=len(mine),
            longest_track_id=longest.track_id,
            normalisation_cap_s=self.normalisation_cap_s,
        )

    def results(self, intersection: Optional[IntersectionROI] = None
                ) -> dict[str, WaitingTimeResult]:
        """Waiting statistics for every approach."""
        target = intersection or self.intersection
        if target is not None:
            return {roi.name: self.result(roi.name) for roi in target}
        lanes = sorted({r.lane for r in self.records.values() if r.lane})
        return {lane: self.result(lane) for lane in lanes}

    # ------------------------------------------------------------------
    def wait_for(self, track_id: int) -> float:
        """Current waiting time of one vehicle, in seconds."""
        record = self.records.get(track_id)
        return record.current_wait_s if record else 0.0

    def reset(self) -> None:
        self.records.clear()
        self._last_timestamp = None
