"""
TraffiSense - metrics/schema.py
--------------------------------
Input contract for the Traffic Metrics Module.

This module is the ONLY place that knows what the tracker's output looks like.
Everything downstream (density, queue, waiting_time, pressure) works on the
normalised `Detection` / `FrameData` objects defined here.

Why this file exists
====================
Earlier integration attempts failed with:

    AttributeError: 'dict' object has no attribute 'track_id'

...because the code assumed the tracker handed over objects, while it actually
handed over dictionaries.  `as_detection()` accepts BOTH, plus the common key
spellings, so the metrics layer never breaks again when the detection side
changes its output format.

Expected per-vehicle record (any of these shapes works)
=======================================================
    {"track_id": 17, "class": "car", "bbox": [x1, y1, x2, y2],
     "frame": 240, "timestamp": 8.0, "lane": "north"}

    {"id": 17, "label": "car", "x1": .., "y1": .., "x2": .., "y2": ..,
     "frame_id": 240, "ts": 8.0, "direction": "N"}

    SomeTrackObject(track_id=17, cls="car", xyxy=(..), frame=240, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "Detection",
    "FrameData",
    "as_detection",
    "as_frame",
    "normalise_direction",
    "DIRECTIONS",
]

DIRECTIONS = ("north", "east", "south", "west")

# Accepted spellings for a compass approach.
_DIRECTION_ALIASES = {
    "n": "north", "north": "north", "nb": "north", "northbound": "north",
    "e": "east", "east": "east", "eb": "east", "eastbound": "east",
    "s": "south", "south": "south", "sb": "south", "southbound": "south",
    "w": "west", "west": "west", "wb": "west", "westbound": "west",
}

# Key aliases, in priority order.
_TRACK_ID_KEYS = ("track_id", "tracking_id", "trackId", "tid", "id")
_CLASS_KEYS = ("class", "cls", "class_name", "label", "category", "name", "type")
_BBOX_KEYS = ("bbox", "box", "xyxy", "bbox_xyxy", "tlbr", "coords")
_FRAME_KEYS = ("frame", "frame_id", "frame_idx", "frame_index", "frame_no", "fid")
_TIME_KEYS = ("timestamp", "time", "ts", "t", "time_s", "seconds")
_LANE_KEYS = ("lane", "side", "direction", "approach", "arm", "leg", "lane_id")
_CONF_KEYS = ("confidence", "conf", "score", "det_conf")


def _get(source: Any, keys: Sequence[str]) -> Any:
    """Fetch the first present key/attribute from a dict OR an object."""
    if isinstance(source, Mapping):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return None
    for key in keys:
        value = getattr(source, key, None)
        if value is not None:
            return value
    return None


def normalise_direction(value: Any) -> Optional[str]:
    """'N' / 'north' / 'Northbound' -> 'north'.  Unknown values pass through
    lower-cased so a project using custom lane names still works."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return _DIRECTION_ALIASES.get(text, text)


def _extract_bbox(source: Any) -> tuple[float, float, float, float]:
    raw = _get(source, _BBOX_KEYS)
    if raw is None:
        # Fall back to separate corner fields.
        x1 = _get(source, ("x1", "left", "xmin"))
        y1 = _get(source, ("y1", "top", "ymin"))
        x2 = _get(source, ("x2", "right", "xmax"))
        y2 = _get(source, ("y2", "bottom", "ymax"))
        if None in (x1, y1, x2, y2):
            # Last resort: centre + size (xywh).
            cx = _get(source, ("cx", "x_center", "xc"))
            cy = _get(source, ("cy", "y_center", "yc"))
            w = _get(source, ("w", "width"))
            h = _get(source, ("h", "height"))
            if None in (cx, cy, w, h):
                raise ValueError(f"Cannot read a bounding box from: {source!r}")
            x1, y1 = float(cx) - float(w) / 2.0, float(cy) - float(h) / 2.0
            x2, y2 = float(cx) + float(w) / 2.0, float(cy) + float(h) / 2.0
        raw = (x1, y1, x2, y2)

    values = [float(v) for v in list(raw)[:4]]
    if len(values) != 4:
        raise ValueError(f"Bounding box needs 4 numbers, got: {raw!r}")
    x1, y1, x2, y2 = values
    # Tolerate boxes given as (x, y, w, h) when the last two look like sizes.
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


@dataclass(frozen=True)
class Detection:
    """One tracked vehicle in one frame, in a form the metrics modules trust."""

    track_id: int
    bbox: tuple[float, float, float, float]          # x1, y1, x2, y2 (pixels)
    vehicle_class: str = "vehicle"
    frame: int = 0
    timestamp: float = 0.0                            # seconds
    lane: Optional[str] = None                        # 'north' | 'east' | ...
    confidence: float = 1.0

    # --- derived geometry -------------------------------------------------
    @property
    def x1(self) -> float: return self.bbox[0]

    @property
    def y1(self) -> float: return self.bbox[1]

    @property
    def x2(self) -> float: return self.bbox[2]

    @property
    def y2(self) -> float: return self.bbox[3]

    @property
    def width(self) -> float: return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float: return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float: return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0,
                (self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: the best cheap proxy for where the
        vehicle touches the road, which is what queue geometry cares about."""
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        x1, y1, x2, y2 = self.bbox
        return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))

    def scaled(self, factor: float) -> "Detection":
        """Box scaled about its centre so its AREA becomes `factor` x original.
        Used by density.py to correct for bounding boxes over-covering the road."""
        if factor <= 0:
            raise ValueError("scale factor must be positive")
        k = factor ** 0.5
        cx, cy = self.centroid
        hw, hh = self.width * k / 2.0, self.height * k / 2.0
        return Detection(
            track_id=self.track_id,
            bbox=(cx - hw, cy - hh, cx + hw, cy + hh),
            vehicle_class=self.vehicle_class,
            frame=self.frame,
            timestamp=self.timestamp,
            lane=self.lane,
            confidence=self.confidence,
        )


def as_detection(source: Any,
                 frame: Optional[int] = None,
                 timestamp: Optional[float] = None) -> Detection:
    """Normalise ANY tracker record (dict / object / Detection) into a Detection.

    `frame` and `timestamp` act as defaults when the record itself carries none.
    """
    if isinstance(source, Detection):
        if frame is None and timestamp is None:
            return source
        return Detection(
            track_id=source.track_id, bbox=source.bbox,
            vehicle_class=source.vehicle_class,
            frame=source.frame if source.frame else (frame or 0),
            timestamp=source.timestamp if source.timestamp else (timestamp or 0.0),
            lane=source.lane, confidence=source.confidence,
        )

    raw_id = _get(source, _TRACK_ID_KEYS)
    if raw_id is None:
        raise ValueError(
            f"Detection has no track id (looked for {_TRACK_ID_KEYS}): {source!r}"
        )

    raw_frame = _get(source, _FRAME_KEYS)
    raw_time = _get(source, _TIME_KEYS)
    raw_conf = _get(source, _CONF_KEYS)
    raw_class = _get(source, _CLASS_KEYS)

    return Detection(
        track_id=int(raw_id),
        bbox=_extract_bbox(source),
        vehicle_class=str(raw_class).lower() if raw_class is not None else "vehicle",
        frame=int(raw_frame) if raw_frame is not None else int(frame or 0),
        timestamp=float(raw_time) if raw_time is not None else float(timestamp or 0.0),
        lane=normalise_direction(_get(source, _LANE_KEYS)),
        confidence=float(raw_conf) if raw_conf is not None else 1.0,
    )


@dataclass
class FrameData:
    """All tracked vehicles observed in a single video frame."""

    frame: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)

    def by_lane(self, lane: str) -> list[Detection]:
        lane = normalise_direction(lane)
        return [d for d in self.detections if d.lane == lane]

    def __len__(self) -> int:
        return len(self.detections)


def as_frame(detections: Iterable[Any],
             frame: int = 0,
             timestamp: Optional[float] = None,
             fps: float = 25.0) -> FrameData:
    """Build a FrameData from a raw list of tracker records.

    If no timestamp is supplied anywhere, one is derived from `frame / fps`
    so waiting-time still works on frame-only pipelines.
    """
    if timestamp is None:
        timestamp = frame / fps if fps else float(frame)

    items = [as_detection(d, frame=frame, timestamp=timestamp)
             for d in (detections or [])]

    # Adopt a timestamp carried by the detections themselves, if present.
    stamps = [d.timestamp for d in items if d.timestamp]
    if stamps:
        timestamp = min(stamps)

    return FrameData(frame=frame, timestamp=float(timestamp), detections=items)
