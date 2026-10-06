"""
TraffiSense - metrics/simulate.py
--------------------------
A synthetic tracked-vehicle stream, so the metrics modules can be developed,
tested and demonstrated before the detection side is wired up.

It produces EXACTLY the records the real tracker is expected to produce -
plain dictionaries:

    {"track_id": 17, "class": "motorcycle", "bbox": [x1, y1, x2, y2],
     "frame": 240, "timestamp": 9.6, "lane": "north"}

Two traffic styles
==================
`style="india"` (the default) models an Indian urban junction, because that is
what this project's footage will look like:

  * two-wheelers dominate - roughly half of all arrivals, plus auto-rickshaws,
  * NO LANE DISCIPLINE - vehicles take any lateral position that fits, rather
    than sitting in numbered lanes,
  * filtering - a motorcycle finds a gap between two cars and advances to the
    front of the queue, because car-following here is decided by which vehicle
    is actually in front of you, not by which lane you are in.

`style="lane"` models lane-disciplined traffic for comparison.

This matters for the metrics, not just the picture.  Under lane-less packing a
queue holds far more vehicles per metre, so vehicle count and queue length stop
agreeing with each other - which is exactly the case density is there to cover.
The queue chain in queue.py measures distance from the stop line and never
assumes lanes, so it handles both styles unchanged; this simulator is what
proves that.

Also modelled: car-following with realistic deceleration, a fixed-time signal
cycling N -> E -> S -> W, Poisson arrivals at four different rates, sub-pixel
detector jitter and occasional dropped detections.

This file is test scaffolding, not part of the deliverable pipeline.  Delete
it once real tracker output is available.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from metrics.roi import ApproachROI, IntersectionROI, rectangular_intersection

# class -> (length_m, width_m, free_speed_m_s)
VEHICLE_TYPES = {
    "car":        (4.0, 1.7, 12.0),
    "motorcycle": (1.9, 0.75, 13.0),
    "auto":       (2.8, 1.4, 10.0),
    "bus":        (10.0, 2.5, 9.0),
    "truck":      (7.0, 2.4, 9.0),
    "bicycle":    (1.7, 0.6, 5.0),
}

# Arrival mix by class.
MIX_INDIAN_URBAN = {
    "motorcycle": 0.46,
    "car": 0.26,
    "auto": 0.16,
    "bus": 0.05,
    "truck": 0.04,
    "bicycle": 0.03,
}

MIX_LANE_DISCIPLINED = {
    "car": 0.72,
    "motorcycle": 0.10,
    "bus": 0.08,
    "truck": 0.10,
}


@dataclass
class _Vehicle:
    track_id: int
    vehicle_class: str
    lane: str
    lateral_m: float           # offset from the centre of the road, metres
    length_m: float
    width_m: float
    free_speed: float
    distance_m: float          # front edge -> stop line, along the approach
    speed: float = 0.0

    @property
    def rear_m(self) -> float:
        return self.distance_m + self.length_m


class IntersectionSimulator:
    """Generates frames of tracked vehicles for a four-arm junction."""

    def __init__(self,
                 intersection: Optional[IntersectionROI] = None,
                 fps: float = 25.0,
                 arrival_rates: Optional[dict[str, float]] = None,
                 style: str = "india",
                 green_seconds: float = 20.0,
                 amber_seconds: float = 3.0,
                 min_gap_m: float = 1.2,
                 lateral_clearance_m: float = 0.25,
                 filter_lateral_speed: float = 1.0,
                 accel: float = 1.8,
                 decel: float = 4.0,
                 jitter_px: float = 0.6,
                 dropout_probability: float = 0.02,
                 seed: int = 7) -> None:
        if style not in ("india", "lane"):
            raise ValueError("style must be 'india' or 'lane'")
        self.style = style
        self.lane_discipline = (style == "lane")
        self.mix = MIX_LANE_DISCIPLINED if self.lane_discipline else MIX_INDIAN_URBAN

        self.intersection = intersection or rectangular_intersection()
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        # vehicles per second per approach - deliberately unequal
        self.arrival_rates = arrival_rates or {
            "north": 0.75, "east": 0.40, "south": 0.28, "west": 0.16}
        self.green_seconds = green_seconds
        self.amber_seconds = amber_seconds
        self.min_gap_m = min_gap_m
        self.lateral_clearance_m = lateral_clearance_m
        self.filter_lateral_speed = filter_lateral_speed
        self.accel = accel
        self.decel = decel
        self.jitter_px = jitter_px
        self.dropout_probability = dropout_probability

        self.rng = random.Random(seed)
        self.time = 0.0
        self.frame = 0
        self._next_id = 1
        self.vehicles: dict[str, list[_Vehicle]] = {
            name: [] for name in self.intersection.names}

        self._order = self.intersection.names
        self._geometry = {roi.name: self._approach_geometry(roi)
                          for roi in self.intersection}

    # ------------------------------------------------------------------
    def _approach_geometry(self, roi: ApproachROI) -> dict:
        ppm = roi.pixels_per_meter or 8.0
        laterals = [roi.lateral_offset(p) for p in roi.polygon]
        half_width_px = max(abs(min(laterals)), abs(max(laterals)))
        return {
            "roi": roi,
            "ppm": ppm,
            "half_width_px": half_width_px,
            "half_width_m": half_width_px / ppm,
            "spawn_m": (roi.max_queue_pixels or 0.0) / ppm,
            "lane_offsets_m": self._lane_offsets(half_width_px / ppm,
                                                 roi.lane_count),
        }

    @staticmethod
    def _lane_offsets(half_width_m: float, lanes: int) -> list[float]:
        lanes = max(1, int(lanes))
        step = (2 * half_width_m) / (lanes + 1)
        return [-half_width_m + step * (i + 1) for i in range(lanes)]

    # ------------------------------------------------------------------
    @property
    def green_approach(self) -> str:
        cycle = (self.green_seconds + self.amber_seconds) * len(self._order)
        phase = self.time % cycle
        index = int(phase // (self.green_seconds + self.amber_seconds))
        within = phase - index * (self.green_seconds + self.amber_seconds)
        if within > self.green_seconds:
            return "__all_red__"
        return self._order[index]

    # ------------------------------------------------------------------
    def _overlaps_laterally(self, a: _Vehicle, b: _Vehicle) -> bool:
        limit = (a.width_m + b.width_m) / 2.0 + self.lateral_clearance_m
        return abs(a.lateral_m - b.lateral_m) < limit

    def _free_lateral_slot(self, approach: str, candidate: _Vehicle,
                           near_m: float) -> Optional[float]:
        """Find a lateral position where `candidate` fits at `near_m`.

        Lane-disciplined traffic picks one of the fixed lane centres.
        Lane-less traffic tries random positions across the road - which is
        what lets a narrow motorcycle slot in beside a car.
        """
        geometry = self._geometry[approach]
        half = geometry["half_width_m"] - candidate.width_m / 2.0
        if half <= 0:
            return None

        if self.lane_discipline:
            options = [o for o in geometry["lane_offsets_m"] if abs(o) <= half]
            self.rng.shuffle(options)
        else:
            options = [self.rng.uniform(-half, half) for _ in range(12)]

        for option in options:
            candidate.lateral_m = option
            clash = False
            for other in self.vehicles[approach]:
                if not self._overlaps_laterally(candidate, other):
                    continue
                # Longitudinal overlap at the entry position?
                if (near_m - candidate.length_m - self.min_gap_m < other.rear_m
                        and other.distance_m < near_m + self.min_gap_m):
                    clash = True
                    break
            if not clash:
                return option
        return None

    def _spawn(self) -> None:
        for name, rate in self.arrival_rates.items():
            if name not in self.vehicles:
                continue
            if self.rng.random() > rate * self.dt:
                continue

            geometry = self._geometry[name]
            classes = list(self.mix)
            weights = [self.mix[c] for c in classes]
            vehicle_class = self.rng.choices(classes, weights=weights)[0]
            length, width, free_speed = VEHICLE_TYPES[vehicle_class]

            start = geometry["spawn_m"] - length - 0.5
            candidate = _Vehicle(
                track_id=self._next_id, vehicle_class=vehicle_class, lane=name,
                lateral_m=0.0, length_m=length, width_m=width,
                free_speed=free_speed * self.rng.uniform(0.85, 1.1),
                distance_m=start, speed=free_speed * 0.8,
            )

            slot = self._free_lateral_slot(name, candidate, start)
            if slot is None:
                continue                 # no room to enter yet

            candidate.lateral_m = slot
            self.vehicles[name].append(candidate)
            self._next_id += 1

    def _leader_of(self, vehicle: _Vehicle,
                   ahead: list[_Vehicle]) -> Optional[_Vehicle]:
        """Nearest vehicle in front whose body actually blocks this one.

        With no lane discipline this is what produces filtering: a motorcycle
        is only held up by vehicles it cannot get past laterally.
        """
        best = None
        for other in ahead:
            if other is vehicle or other.distance_m >= vehicle.distance_m:
                continue
            if not self._overlaps_laterally(vehicle, other):
                continue
            if best is None or other.distance_m > best.distance_m:
                best = other
        return best

    def _clear_distance(self, vehicle: _Vehicle, others: list[_Vehicle],
                        lateral: float) -> float:
        """How much clear road `vehicle` would have at lateral position
        `lateral`, or -1 if another vehicle already occupies that space."""
        probe = _Vehicle(**{**vehicle.__dict__, "lateral_m": lateral})
        clear = 1e9
        for other in others:
            if other is vehicle or not self._overlaps_laterally(probe, other):
                continue
            if other.distance_m >= vehicle.distance_m:
                # Alongside or behind: only a problem if bodies would overlap.
                if other.distance_m < vehicle.rear_m + self.min_gap_m:
                    return -1.0
                continue
            if other.rear_m > vehicle.distance_m:
                return -1.0
            clear = min(clear, vehicle.distance_m - other.rear_m)
        return clear

    def _try_filtering(self, vehicle: _Vehicle, others: list[_Vehicle],
                       geometry: dict) -> None:
        """Two-wheelers edge sideways into gaps when traffic is slow.

        This is what fills an Indian queue across its whole width instead of
        in neat rows, and it is the behaviour that decouples vehicle count
        from queue length.
        """
        if self.lane_discipline or vehicle.width_m > 1.0:
            return
        if vehicle.speed > 2.0:
            return                            # only in slow or stopped traffic

        half = geometry["half_width_m"] - vehicle.width_m / 2.0
        if half <= 0:
            return

        step = self.filter_lateral_speed * self.dt
        current = self._clear_distance(vehicle, others, vehicle.lateral_m)
        best_lateral = None
        best_clear = current
        for direction in (-1.0, 1.0):
            candidate = max(-half, min(half, vehicle.lateral_m + direction * step))
            clear = self._clear_distance(vehicle, others, candidate)
            if clear > best_clear + 0.05:
                best_clear = clear
                best_lateral = candidate
        if best_lateral is not None:
            vehicle.lateral_m = best_lateral

    def _step_physics(self) -> None:
        green = self.green_approach
        for name, queue in self.vehicles.items():
            stop_for_light = (name != green)
            geometry = self._geometry[name]
            queue.sort(key=lambda v: v.distance_m)   # nearest the line first

            for vehicle in queue:
                self._try_filtering(vehicle, queue, geometry)

            for index, vehicle in enumerate(queue):
                target = -1e9
                if stop_for_light and vehicle.distance_m > -0.5:
                    target = 0.0

                leader = self._leader_of(vehicle, queue[:index])
                if leader is not None:
                    target = max(target, leader.rear_m + self.min_gap_m)

                gap = vehicle.distance_m - target
                if gap <= 0:
                    desired = 0.0
                else:
                    # Speed that can still be shed before reaching the target.
                    desired = min(vehicle.free_speed,
                                  math.sqrt(max(0.0, 2 * self.decel * gap)))

                if desired > vehicle.speed:
                    vehicle.speed = min(desired,
                                        vehicle.speed + self.accel * self.dt)
                else:
                    vehicle.speed = max(desired,
                                        vehicle.speed - self.decel * self.dt)

                vehicle.distance_m -= vehicle.speed * self.dt

            # Retire vehicles that have cleared the junction.
            self.vehicles[name] = [v for v in queue if v.distance_m > -14.0]

    # ------------------------------------------------------------------
    def _bbox(self, vehicle: _Vehicle) -> list[float]:
        geometry = self._geometry[vehicle.lane]
        roi: ApproachROI = geometry["roi"]
        ppm = geometry["ppm"]
        u = np.array(roi.queue_direction, float)
        perp = np.array([-u[1], u[0]])
        origin = np.array(roi.stop_line_midpoint, float)

        lateral = vehicle.lateral_m * ppm
        front_px = vehicle.distance_m * ppm
        rear_px = vehicle.rear_m * ppm
        half_w_px = vehicle.width_m * ppm / 2.0

        corners = []
        for along in (front_px, rear_px):
            for side in (-half_w_px, half_w_px):
                corners.append(origin + u * along + perp * (lateral + side))
        corners = np.asarray(corners)

        jitter = (self.rng.gauss(0, self.jitter_px),
                  self.rng.gauss(0, self.jitter_px))
        x1, y1 = corners.min(axis=0) + jitter
        x2, y2 = corners.max(axis=0) + jitter
        return [float(x1), float(y1), float(x2), float(y2)]

    # ------------------------------------------------------------------
    def step(self) -> list[dict]:
        """Advance the simulation one frame and return its detections."""
        self._spawn()
        self._step_physics()

        detections: list[dict] = []
        for name, queue in self.vehicles.items():
            for vehicle in queue:
                if self.rng.random() < self.dropout_probability:
                    continue        # detector missed it this frame
                detections.append({
                    "track_id": vehicle.track_id,
                    "class": vehicle.vehicle_class,
                    "bbox": self._bbox(vehicle),
                    "frame": self.frame,
                    "timestamp": round(self.time, 4),
                    "lane": name,
                })

        self.frame += 1
        self.time += self.dt
        return detections

    def run(self, seconds: float = 60.0) -> Iterator[tuple[int, float, list[dict]]]:
        """Yield (frame_index, timestamp, detections) for `seconds` of video."""
        total = int(seconds * self.fps)
        for _ in range(total):
            frame, timestamp = self.frame, self.time
            yield frame, timestamp, self.step()

    # ------------------------------------------------------------------
    def true_queue_count(self, approach: str, speed_threshold: float = 0.4) -> int:
        """Ground truth: vehicles actually stopped on this approach.

        Used by the tests to check the estimator against reality.
        """
        return sum(1 for v in self.vehicles.get(approach, [])
                   if v.speed < speed_threshold and v.distance_m > -0.5)

    def true_queue_length_m(self, approach: str,
                            speed_threshold: float = 0.4) -> float:
        stopped = [v for v in self.vehicles.get(approach, [])
                   if v.speed < speed_threshold and v.distance_m > -0.5]
        return max((v.rear_m for v in stopped), default=0.0)
