"""
TraffiSense - metrics/roi.py
-----------------------------
Road geometry for one intersection: the road region (ROI) of each approach,
its stop line, and the pixel -> metre scale.

Every metric in this package is geometry-relative:

    density      = how much of the ROI polygon is covered by vehicles
    queue length = how far back from the STOP LINE the stopped vehicles reach
    waiting time = how long a vehicle inside the ROI stays still

So this file is the single source of truth for "where is the road".

Coordinate convention
=====================
Image pixels: x to the right, y downwards (standard OpenCV/YOLO convention).

Stop line: a segment ((x1, y1), (x2, y2)) drawn across the road.
Queue direction: the unit vector pointing from the stop line BACK into the
approach (i.e. upstream, where the waiting vehicles are). It is derived
automatically from the ROI centroid, so you only have to draw two things per
approach: the polygon and the stop line.

Only numpy is required.  OpenCV is optional and used nowhere here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .schema import DIRECTIONS, normalise_direction

__all__ = ["ApproachROI", "IntersectionROI", "polygon_area", "point_in_polygon"]

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Plain geometry helpers (numpy only, no OpenCV needed)
# ---------------------------------------------------------------------------

def polygon_area(polygon: Sequence[Point]) -> float:
    """Area of a simple polygon via the shoelace formula (always positive)."""
    pts = np.asarray(polygon, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Even-odd ray casting test for a single point."""
    x, y = point
    pts = np.asarray(polygon, dtype=float)
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


def points_in_polygon(xs: np.ndarray, ys: np.ndarray,
                      polygon: Sequence[Point]) -> np.ndarray:
    """Vectorised even-odd test.  `xs`/`ys` are same-shaped arrays."""
    pts = np.asarray(polygon, dtype=float)
    inside = np.zeros(np.shape(xs), dtype=bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        straddles = (y1 > ys) != (y2 > ys)
        x_cross = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        inside ^= straddles & (xs < x_cross)
    return inside


# ---------------------------------------------------------------------------
# One approach (one arm of the intersection)
# ---------------------------------------------------------------------------

@dataclass
class ApproachROI:
    """Road region of a single approach, plus its stop line and scale.

    Parameters
    ----------
    name:
        'north' | 'east' | 'south' | 'west' (or a custom lane name).
    polygon:
        Points of the drivable region for this approach, in image pixels.
    stop_line:
        ((x1, y1), (x2, y2)) - the segment vehicles must not cross on red.
    pixels_per_meter:
        Optional scale so queue length can also be reported in metres.
        Calibrate once: measure a known road feature (lane width ~3.5 m,
        a lane-marking dash ~3 m) in pixels and divide.
    lane_count:
        Number of traffic lanes on this approach.  Used to convert queue
        length into an approximate vehicle count, and for reporting only.
    max_queue_pixels:
        Length of the ROI along the queue direction; filled in automatically.
        Used as the normalisation cap for the queue score.
    """

    name: str
    polygon: list[Point]
    stop_line: tuple[Point, Point]
    pixels_per_meter: Optional[float] = None
    lane_count: int = 1
    max_queue_pixels: Optional[float] = None

    # Derived, filled in __post_init__
    _origin: np.ndarray = field(init=False, repr=False)
    _u: np.ndarray = field(init=False, repr=False)       # unit queue direction
    _area_px: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = normalise_direction(self.name) or str(self.name)
        self.polygon = [(float(x), float(y)) for x, y in self.polygon]
        if len(self.polygon) < 3:
            raise ValueError(f"[{self.name}] ROI polygon needs >= 3 points")

        (sx1, sy1), (sx2, sy2) = self.stop_line
        self.stop_line = ((float(sx1), float(sy1)), (float(sx2), float(sy2)))

        self._origin = np.array([(sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0], float)

        # Normal of the stop line...
        line = np.array([sx2 - sx1, sy2 - sy1], float)
        norm = np.linalg.norm(line)
        if norm == 0:
            raise ValueError(f"[{self.name}] stop line is a single point")
        normal = np.array([-line[1], line[0]]) / norm

        # ...oriented so that it points INTO the approach (towards the queue).
        centroid = np.asarray(self.polygon, float).mean(axis=0)
        if np.dot(centroid - self._origin, normal) < 0:
            normal = -normal
        self._u = normal

        self._area_px = polygon_area(self.polygon)
        if self._area_px <= 0:
            raise ValueError(f"[{self.name}] ROI polygon has zero area")

        if self.max_queue_pixels is None:
            dists = [self.distance_from_stop_line(p) for p in self.polygon]
            self.max_queue_pixels = float(max(dists)) if dists else 0.0

    # --- basic properties -------------------------------------------------
    @property
    def area_px(self) -> float:
        """Total road area of this approach, in square pixels."""
        return self._area_px

    @property
    def queue_direction(self) -> tuple[float, float]:
        """Unit vector from the stop line back up the approach."""
        return (float(self._u[0]), float(self._u[1]))

    @property
    def stop_line_midpoint(self) -> Point:
        return (float(self._origin[0]), float(self._origin[1]))

    # --- geometry queries -------------------------------------------------
    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def contains_vehicle(self, detection) -> bool:
        """True when a detected vehicle belongs to this approach.

        Tests the ground point (bottom-centre of the box) first, then the
        centroid.  The second test matters: the head of a queue always noses
        onto the stop line, and a ground point one pixel past it would
        otherwise drop the most important vehicle on the approach - the one
        the queue is measured from.
        """
        if self.contains(detection.ground_point):
            return True
        return self.contains(detection.centroid)

    def distance_from_stop_line(self, point: Point) -> float:
        """Signed distance in pixels along the queue direction.

        > 0 : upstream of the stop line (waiting side)
        < 0 : already past the stop line (inside the junction)
        """
        vec = np.asarray(point, float) - self._origin
        return float(np.dot(vec, self._u))

    def lateral_offset(self, point: Point) -> float:
        """Signed distance across the road (perpendicular to queue direction)."""
        vec = np.asarray(point, float) - self._origin
        perp = np.array([-self._u[1], self._u[0]])
        return float(np.dot(vec, perp))

    def to_meters(self, pixels: float) -> Optional[float]:
        if not self.pixels_per_meter:
            return None
        return pixels / self.pixels_per_meter

    def mask(self, resolution: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Rasterise the ROI polygon onto a bounded grid.

        Returns (mask, xs, ys, cell_area) where `mask` is a boolean grid of
        points inside the polygon and `cell_area` is the pixel-area each grid
        cell represents.  Cached after the first call.
        """
        cached = getattr(self, "_mask_cache", None)
        if cached is not None and cached[0] == resolution:
            return cached[1]

        pts = np.asarray(self.polygon, float)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)

        nx = max(int(resolution * width / max(width, height)), 8)
        ny = max(int(resolution * height / max(width, height)), 8)

        # Sample at cell centres.
        xs_1d = x_min + (np.arange(nx) + 0.5) * width / nx
        ys_1d = y_min + (np.arange(ny) + 0.5) * height / ny
        xs, ys = np.meshgrid(xs_1d, ys_1d)

        mask = points_in_polygon(xs, ys, self.polygon)
        cell_area = (width / nx) * (height / ny)

        result = (mask, xs, ys, float(cell_area))
        self._mask_cache = (resolution, result)
        return result

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "polygon": [list(p) for p in self.polygon],
            "stop_line": [list(self.stop_line[0]), list(self.stop_line[1])],
            "pixels_per_meter": self.pixels_per_meter,
            "lane_count": self.lane_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ApproachROI":
        return cls(
            name=data["name"],
            polygon=[tuple(p) for p in data["polygon"]],
            stop_line=(tuple(data["stop_line"][0]), tuple(data["stop_line"][1])),
            pixels_per_meter=data.get("pixels_per_meter"),
            lane_count=int(data.get("lane_count", 1)),
        )


# ---------------------------------------------------------------------------
# The whole intersection
# ---------------------------------------------------------------------------

@dataclass
class IntersectionROI:
    """The four approaches of one junction."""

    approaches: dict[str, ApproachROI] = field(default_factory=dict)

    @classmethod
    def from_list(cls, items: Iterable[ApproachROI]) -> "IntersectionROI":
        return cls({a.name: a for a in items})

    def add(self, approach: ApproachROI) -> None:
        self.approaches[approach.name] = approach

    def __getitem__(self, name: str) -> ApproachROI:
        return self.approaches[normalise_direction(name) or name]

    def __contains__(self, name: str) -> bool:
        return (normalise_direction(name) or name) in self.approaches

    def __iter__(self):
        return iter(self.approaches.values())

    @property
    def names(self) -> list[str]:
        """Approach names, compass-ordered when they are compass directions."""
        known = [d for d in DIRECTIONS if d in self.approaches]
        extra = sorted(n for n in self.approaches if n not in DIRECTIONS)
        return known + extra

    # --- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return {"approaches": [a.to_dict() for a in self]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "IntersectionROI":
        return cls.from_list(ApproachROI.from_dict(a) for a in data["approaches"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "IntersectionROI":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Convenience: a rectangular four-arm junction, handy for tests and demos
# ---------------------------------------------------------------------------

def rectangular_intersection(frame_width: int = 1280,
                             frame_height: int = 720,
                             road_half_width: int = 30,
                             pixels_per_meter: float = 8.0,
                             lane_count: int = 2) -> IntersectionROI:
    """A synthetic '+' shaped junction centred in the frame.

    Defaults describe a 7.5 m wide two-lane road (60 px at 8 px/m) seen by a
    camera covering roughly 40 m of the north/south arms and 75 m of the
    east/west arms.

    Useful for unit tests and for the demo; replace with real ROIs drawn on
    your own camera view before deployment.
    """
    cx, cy = frame_width / 2.0, frame_height / 2.0
    hw = road_half_width
    box = hw  # half-size of the central junction box

    def approach(name, polygon, stop_line):
        return ApproachROI(name=name, polygon=polygon, stop_line=stop_line,
                           pixels_per_meter=pixels_per_meter,
                           lane_count=lane_count)

    north = approach(
        "north",
        [(cx - hw, 0), (cx + hw, 0), (cx + hw, cy - box), (cx - hw, cy - box)],
        ((cx - hw, cy - box), (cx + hw, cy - box)),
    )
    south = approach(
        "south",
        [(cx - hw, cy + box), (cx + hw, cy + box),
         (cx + hw, frame_height), (cx - hw, frame_height)],
        ((cx - hw, cy + box), (cx + hw, cy + box)),
    )
    west = approach(
        "west",
        [(0, cy - hw), (cx - box, cy - hw), (cx - box, cy + hw), (0, cy + hw)],
        ((cx - box, cy - hw), (cx - box, cy + hw)),
    )
    east = approach(
        "east",
        [(cx + box, cy - hw), (frame_width, cy - hw),
         (frame_width, cy + hw), (cx + box, cy + hw)],
        ((cx + box, cy - hw), (cx + box, cy + hw)),
    )
    return IntersectionROI.from_list([north, east, south, west])
