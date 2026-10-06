"""
TraffiSense - metrics/motion.py
--------------------------------
Per-track speed estimation shared by queue.py and waiting_time.py.

Both "is this vehicle queued?" and "how long has it been waiting?" rest on the
same question: is this track_id currently moving?  Answering it once here keeps
the two modules consistent and stops the same smoothing code being written
twice.

Method
======
Frame-to-frame speed is unusable on its own.  A parked car's box still jitters
by a pixel or so every frame, and at 25 fps a 0.6-pixel wobble reads as ~20
px/s - fast enough to look like moving traffic.  So speed is measured over a
short TIME WINDOW instead of between consecutive frames:

    speed_px_s = |p_now - p_(now - window)| / window

Averaging over ~0.5 s divides the jitter by the window length while keeping
the estimate responsive enough to catch a vehicle pulling away.  A light
exponential moving average smooths what is left, and the moving/stopped
decision uses HYSTERESIS - two thresholds instead of one:

    stopped  when smoothed speed falls below `stop_speed_px_s`
    moving   when it rises above `go_speed_px_s`  (go > stop)

A single threshold would make a vehicle flicker between states right at the
boundary and shred the waiting-time counter.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional

from .schema import Detection

__all__ = ["TrackMotion", "MotionTracker"]


@dataclass
class TrackMotion:
    """Current motion state of one tracked vehicle."""

    track_id: int
    position: tuple[float, float]
    timestamp: float
    speed_px_s: float = 0.0
    stationary: bool = False
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_seen: int = 1
    displacement_px: float = 0.0       # over the current speed window
    history: Deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=120), repr=False)
    _stopped_since: Optional[float] = None
    _stop_candidate_frames: int = 0

    def speed_m_s(self, pixels_per_meter: Optional[float]) -> Optional[float]:
        if not pixels_per_meter:
            return None
        return self.speed_px_s / pixels_per_meter


class MotionTracker:
    """Keeps speed / stationary state for every track_id across frames.

    Parameters
    ----------
    stop_speed_px_s:
        Below this smoothed speed a vehicle is a candidate for "stopped".
        Rule of thumb: pick ~ (0.5 m/s x pixels_per_meter).
    go_speed_px_s:
        Above this smoothed speed a stopped vehicle is released back to
        "moving".  Must be >= stop_speed_px_s (hysteresis band).
    min_stop_frames:
        How many consecutive slow frames are needed before we believe it.
    smoothing:
        EMA factor for speed, 0 < a <= 1.  Lower = smoother/slower to react.
    forget_after_s:
        Tracks not seen for this long are dropped, so memory does not grow
        without bound on a long video.
    """

    def __init__(self,
                 stop_speed_px_s: float = 4.0,
                 go_speed_px_s: Optional[float] = None,
                 min_stop_frames: int = 3,
                 smoothing: float = 0.5,
                 window_s: float = 0.5,
                 forget_after_s: float = 5.0) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")
        self.window_s = float(window_s)
        self.stop_speed_px_s = float(stop_speed_px_s)
        self.go_speed_px_s = float(go_speed_px_s if go_speed_px_s is not None
                                   else stop_speed_px_s * 2.0)
        if self.go_speed_px_s < self.stop_speed_px_s:
            raise ValueError("go_speed_px_s must be >= stop_speed_px_s")
        self.min_stop_frames = int(min_stop_frames)
        self.smoothing = float(smoothing)
        self.forget_after_s = float(forget_after_s)
        self.tracks: dict[int, TrackMotion] = {}

    # ------------------------------------------------------------------
    def update(self, detections: Iterable[Detection],
               timestamp: Optional[float] = None) -> dict[int, TrackMotion]:
        """Feed one frame's detections; returns the motion state of each."""
        detections = list(detections)
        if timestamp is None:
            timestamp = max((d.timestamp for d in detections), default=0.0)

        seen: dict[int, TrackMotion] = {}
        for det in detections:
            seen[det.track_id] = self._update_one(det, timestamp)

        self._forget_stale(timestamp)
        return seen

    def _update_one(self, det: Detection, timestamp: float) -> TrackMotion:
        point = det.ground_point
        state = self.tracks.get(det.track_id)

        if state is None:
            state = TrackMotion(
                track_id=det.track_id, position=point, timestamp=timestamp,
                speed_px_s=0.0, stationary=False,
                first_seen=timestamp, last_seen=timestamp,
            )
            state.history.append((timestamp, point[0], point[1]))
            # A brand-new track has no history, so it starts as "unknown/moving"
            # and needs min_stop_frames of evidence before it counts as queued.
            self.tracks[det.track_id] = state
            return state

        state.history.append((timestamp, point[0], point[1]))
        raw_speed, displacement = self._windowed_speed(state, point, timestamp)
        if raw_speed is not None:
            a = self.smoothing
            state.speed_px_s = a * raw_speed + (1 - a) * state.speed_px_s
            state.displacement_px = displacement

        state.position = point
        state.timestamp = timestamp
        state.last_seen = timestamp
        state.frames_seen += 1
        self._apply_hysteresis(state, timestamp)
        return state

    def _windowed_speed(self, state: TrackMotion,
                        point: tuple[float, float],
                        timestamp: float) -> tuple[Optional[float], float]:
        """Speed over the last `window_s` seconds of this track's history.

        Falls back to the longest span available while the window is still
        filling, and returns None when no usable time span exists yet.
        """
        # Drop samples older than the window, but always keep one so a short
        # history still yields a measurement.
        cutoff = timestamp - self.window_s
        while len(state.history) > 2 and state.history[1][0] < cutoff:
            state.history.popleft()

        t0, x0, y0 = state.history[0]
        dt = timestamp - t0
        if dt <= 0:
            return None, 0.0
        displacement = math.hypot(point[0] - x0, point[1] - y0)
        return displacement / dt, displacement

    def _apply_hysteresis(self, state: TrackMotion, timestamp: float) -> None:
        if state.stationary:
            # Only a clear burst of movement releases a stopped vehicle.
            if state.speed_px_s > self.go_speed_px_s:
                state.stationary = False
                state._stopped_since = None
                state._stop_candidate_frames = 0
                state.displacement_px = 0.0
        else:
            if state.speed_px_s < self.stop_speed_px_s:
                state._stop_candidate_frames += 1
                if state._stop_candidate_frames >= self.min_stop_frames:
                    state.stationary = True
                    # Back-date the stop to when the vehicle actually slowed.
                    state._stopped_since = timestamp
            else:
                state._stop_candidate_frames = 0

    def _forget_stale(self, timestamp: float) -> None:
        stale = [tid for tid, s in self.tracks.items()
                 if timestamp - s.last_seen > self.forget_after_s]
        for tid in stale:
            del self.tracks[tid]

    # ------------------------------------------------------------------
    def get(self, track_id: int) -> Optional[TrackMotion]:
        return self.tracks.get(track_id)

    def is_stationary(self, track_id: int, default: bool = False) -> bool:
        state = self.tracks.get(track_id)
        return default if state is None else state.stationary

    def speed(self, track_id: int, default: float = 0.0) -> float:
        state = self.tracks.get(track_id)
        return default if state is None else state.speed_px_s

    def reset(self) -> None:
        self.tracks.clear()
