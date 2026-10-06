"""
TraffiSense - tracking/byte_tracker.py
---------------------------------------
STAND-IN tracker so the full pipeline runs end to end. Hitarth's ByteTrack
module replaces this file - keep the same `update()` signature and the
rest of the pipeline does not change.

    from tracking import ByteTracker
    tracker = ByteTracker(fps=25)
    tracked = tracker.update(detections, frame_index, timestamp)

Input  - Riddhi's detect(frame) output, one dict per box:
         {"class_name": "Car", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
Output - the metrics input contract (see metrics/README.md), one dict per
         confirmed track:
         {"track_id": 17, "class": "car", "bbox": [...], "frame": 240,
          "timestamp": 9.6, "confidence": 0.91}

How it works (the ByteTrack idea, simplified, numpy only):
  1. predict every track forward with its recent velocity,
  2. match HIGH-confidence boxes to tracks by IoU,
  3. match the leftover tracks to LOW-confidence boxes (this is ByteTrack's
     trick - a partly hidden vehicle keeps its id instead of getting a new one),
  4. start new tracks from unmatched high-confidence boxes,
  5. keep unmatched tracks alive for `track_buffer` frames before dropping them.
Stable ids matter: waiting time is accumulated per track_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np

__all__ = ["ByteTracker", "normalise_class", "CLASS_MAP"]


# Riddhi's model classes (and COCO classes from the pretrained fallback model)
# mapped onto the vocabulary the metrics module understands.
CLASS_MAP: dict[str, Optional[str]] = {
    # Riddhi's Roboflow dataset
    "car": "car",
    "bus": "bus",
    "bicyclist": "bicycle",
    "commercial-vehicle": "truck",
    "emergency-vehicle": "van",
    "pickup-truck": "van",
    "semi-truck": "truck",
    "trailer": "truck",
    "pedestrian": None,          # not a vehicle - dropped
    # COCO (pretrained yolov8n.pt)
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "truck": "truck",
    "person": None,
}
_NON_VEHICLE_DEFAULT = {"traffic light", "stop sign", "fire hydrant", "bench", "dog", "cow"}


def normalise_class(name: str) -> Optional[str]:
    """Return the metrics class name, or None if the box is not a vehicle."""
    key = str(name).strip().lower()
    if key in CLASS_MAP:
        return CLASS_MAP[key]
    if key in _NON_VEHICLE_DEFAULT:
        return None
    return key


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Pair rows and columns, best IoU first, ignoring pairs under threshold."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used_r, used_c = set(), set()
    for r, c in order:
        if iou[r, c] < threshold:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return pairs


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    vehicle_class: str
    confidence: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    misses: int = 0

    def predicted(self) -> np.ndarray:
        return self.bbox + self.velocity * (self.misses + 1)

    def update(self, bbox: np.ndarray, vehicle_class: str, confidence: float) -> None:
        step = (bbox - self.bbox) / (self.misses + 1)
        self.velocity = 0.6 * self.velocity + 0.4 * step
        self.bbox = bbox
        self.vehicle_class = vehicle_class
        self.confidence = confidence
        self.hits += 1
        self.misses = 0


class ByteTracker:
    """Simplified ByteTrack. Same interface Hitarth's tracker should expose."""

    def __init__(self,
                 fps: float = 25.0,
                 high_thresh: float = 0.5,
                 low_thresh: float = 0.1,
                 match_iou: float = 0.2,
                 low_match_iou: float = 0.4,
                 track_buffer_s: float = 1.2,
                 min_hits: int = 2):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.track_buffer = max(1, int(round(track_buffer_s * fps)))
        self.min_hits = min_hits
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self,
               detections: Iterable[Mapping],
               frame_index: int,
               timestamp: float) -> list[dict]:
        boxes, scores, classes = [], [], []
        for det in detections:
            vehicle_class = normalise_class(det.get("class_name", det.get("class", "vehicle")))
            if vehicle_class is None:
                continue
            score = float(det.get("confidence", 1.0))
            if score < self.low_thresh:
                continue
            boxes.append([float(v) for v in det["bbox"]])
            scores.append(score)
            classes.append(vehicle_class)

        boxes_np = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores_np = np.asarray(scores, dtype=float)
        high = np.where(scores_np >= self.high_thresh)[0]
        low = np.where(scores_np < self.high_thresh)[0]

        predicted = np.asarray([t.predicted() for t in self._tracks]).reshape(-1, 4)
        unmatched_tracks = list(range(len(self._tracks)))
        matched_this_frame: set[int] = set()

        # Stage 1: high-confidence boxes vs. every track
        pairs = _greedy_match(_iou_matrix(predicted, boxes_np[high]), self.match_iou)
        for t_idx, d_local in pairs:
            d_idx = high[d_local]
            self._tracks[t_idx].update(boxes_np[d_idx], classes[d_idx], scores_np[d_idx])
            matched_this_frame.add(t_idx)
        unmatched_tracks = [i for i in unmatched_tracks if i not in matched_this_frame]
        unmatched_high = [high[j] for j in range(len(high)) if j not in {p[1] for p in pairs}]

        # Stage 2: leftover tracks vs. low-confidence boxes
        if unmatched_tracks and len(low):
            sub = predicted[unmatched_tracks]
            pairs2 = _greedy_match(_iou_matrix(sub, boxes_np[low]), self.low_match_iou)
            for r, d_local in pairs2:
                t_idx = unmatched_tracks[r]
                d_idx = low[d_local]
                self._tracks[t_idx].update(boxes_np[d_idx], classes[d_idx], scores_np[d_idx])
                matched_this_frame.add(t_idx)
            unmatched_tracks = [i for i in unmatched_tracks if i not in matched_this_frame]

        # Age the tracks that found nothing, drop the ones gone too long
        for t_idx in unmatched_tracks:
            self._tracks[t_idx].misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.track_buffer]

        # Stage 3: new tracks from unmatched high-confidence boxes
        for d_idx in unmatched_high:
            self._tracks.append(_Track(self._next_id, boxes_np[d_idx], classes[d_idx], scores_np[d_idx]))
            self._next_id += 1

        return [
            {
                "track_id": t.track_id,
                "class": t.vehicle_class,
                "bbox": [round(float(v), 1) for v in t.bbox],
                "frame": int(frame_index),
                "timestamp": float(timestamp),
                "confidence": round(float(t.confidence), 3),
            }
            for t in self._tracks
            if t.misses == 0 and t.hits >= self.min_hits
        ]
