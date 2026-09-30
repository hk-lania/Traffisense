"""
TraffiSense - metrics/density.py
---------------------------------
Traffic density = how much of the road is covered by vehicles.

    density (%) = occupied road area / total road ROI area x 100

Input : tracked vehicle bounding boxes for one approach
Output: an independent density value per approach (0 - 100 %)

Two things make a naive implementation wrong, and both are handled here.

1. OVERLAP.  Adding up bounding-box areas double-counts every place two boxes
   overlap - common in dense traffic and with perspective - and can push
   "density" past 100 %.  The default method rasterises the ROI onto a grid and
   marks covered cells, so an area covered twice is still counted once.  This
   is a UNION of boxes, not a sum.

2. BOXES ARE BIGGER THAN VEHICLES.  An axis-aligned box around a car at an
   angle contains a lot of empty asphalt, and a motorcycle's box is mostly air.
   `use_class_footprint` shrinks each box about its centre by a per-class factor
   so the covered area approximates the vehicle's actual road footprint.
   Set it to False if you want the raw geometric box coverage.

The output is deliberately left as a standalone number.  Normalisation and
fusion into the Traffic Pressure Index happen later, in pressure.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .roi import ApproachROI, IntersectionROI
from .schema import Detection, FrameData, as_detection

__all__ = ["DensityResult", "DensityEstimator", "CLASS_FOOTPRINT"]


# Fraction of the bounding box actually occupied by the vehicle on the road.
# Tune these against your own camera view; they are deliberately conservative.
CLASS_FOOTPRINT: dict[str, float] = {
    "car": 0.80,
    "van": 0.82,
    "suv": 0.82,
    "truck": 0.88,
    "bus": 0.90,
    "lorry": 0.88,
    "motorcycle": 0.50,
    "motorbike": 0.50,
    "bike": 0.50,
    "scooter": 0.50,
    "bicycle": 0.40,
    "cycle": 0.40,
    "auto": 0.65,
    "rickshaw": 0.65,
    "autorickshaw": 0.65,
    "tempo": 0.80,
    "person": 0.25,
    "vehicle": 0.75,      # generic fallback
}
_DEFAULT_FOOTPRINT = 0.75


@dataclass
class DensityResult:
    """Density of one approach at one instant."""

    approach: str
    density_percent: float          # 0 - 100
    occupied_area_px: float
    roi_area_px: float
    vehicle_count: int
    method: str = "mask"

    @property
    def score(self) -> float:
        """Density as a 0 - 1 value, ready for later normalisation/fusion."""
        return max(0.0, min(1.0, self.density_percent / 100.0))

    @property
    def level(self) -> str:
        """Human-readable band - for display and viva, not for control logic."""
        d = self.density_percent
        if d < 10:
            return "free"
        if d < 25:
            return "light"
        if d < 45:
            return "moderate"
        if d < 65:
            return "heavy"
        return "congested"

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "density_percent": round(self.density_percent, 2),
            "occupied_area_px": round(self.occupied_area_px, 1),
            "roi_area_px": round(self.roi_area_px, 1),
            "vehicle_count": self.vehicle_count,
            "level": self.level,
            "method": self.method,
        }


class DensityEstimator:
    """Computes road occupancy for ONE approach ROI.

    Parameters
    ----------
    roi:
        The approach whose road area is being measured.
    use_class_footprint:
        Shrink each box to its per-class road footprint (see CLASS_FOOTPRINT).
    method:
        'mask'  - union of boxes on a raster grid; overlap-safe (default).
        'sum'   - sum of box areas clipped to the ROI bounding box; faster,
                  but double-counts overlaps.  Kept for comparison/benchmarks.
    resolution:
        Grid resolution along the ROI's longer side for the 'mask' method.
        220 gives sub-1 % area error on typical ROIs and costs ~1 ms.
    require_inside:
        Only count a detection that sits on this approach (see
        ApproachROI.contains_vehicle).  Leave True when detections are not
        already sorted by lane.
    """

    def __init__(self,
                 roi: ApproachROI,
                 use_class_footprint: bool = True,
                 method: str = "mask",
                 resolution: int = 220,
                 require_inside: bool = True,
                 class_footprint: Optional[Mapping[str, float]] = None) -> None:
        if method not in ("mask", "sum"):
            raise ValueError("method must be 'mask' or 'sum'")
        self.roi = roi
        self.use_class_footprint = use_class_footprint
        self.method = method
        self.resolution = int(resolution)
        self.require_inside = require_inside
        self.class_footprint = dict(class_footprint or CLASS_FOOTPRINT)

    # ------------------------------------------------------------------
    def footprint_factor(self, vehicle_class: str) -> float:
        if not self.use_class_footprint:
            return 1.0
        return self.class_footprint.get((vehicle_class or "").lower(),
                                        _DEFAULT_FOOTPRINT)

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

    # ------------------------------------------------------------------
    def compute(self, detections: Iterable) -> DensityResult:
        """Density for this approach from one frame's detections."""
        vehicles = self._relevant(detections)
        roi_area = self.roi.area_px

        if not vehicles:
            return DensityResult(self.roi.name, 0.0, 0.0, roi_area, 0, self.method)

        boxes = [v.scaled(self.footprint_factor(v.vehicle_class))
                 if self.use_class_footprint else v
                 for v in vehicles]

        if self.method == "mask":
            occupied = self._occupied_area_mask(boxes)
        else:
            occupied = self._occupied_area_sum(boxes)

        density = 100.0 * occupied / roi_area if roi_area > 0 else 0.0
        density = max(0.0, min(100.0, density))
        return DensityResult(
            approach=self.roi.name,
            density_percent=density,
            occupied_area_px=occupied,
            roi_area_px=roi_area,
            vehicle_count=len(vehicles),
            method=self.method,
        )

    # --- the two area methods -----------------------------------------
    def _occupied_area_mask(self, boxes: Sequence[Detection]) -> float:
        roi_mask, xs, ys, cell_area = self.roi.mask(self.resolution)
        covered = np.zeros_like(roi_mask, dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = box.bbox
            covered |= (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)
        inside = covered & roi_mask

        # Scale by the ratio between the true polygon area and the rasterised
        # one, so grid quantisation does not bias the percentage.
        raster_roi_area = roi_mask.sum() * cell_area
        if raster_roi_area <= 0:
            return 0.0
        correction = self.roi.area_px / raster_roi_area
        return float(inside.sum() * cell_area * correction)

    def _occupied_area_sum(self, boxes: Sequence[Detection]) -> float:
        pts = np.asarray(self.roi.polygon, float)
        rx1, ry1 = pts.min(axis=0)
        rx2, ry2 = pts.max(axis=0)
        total = 0.0
        for box in boxes:
            x1, y1, x2, y2 = box.bbox
            w = max(0.0, min(x2, rx2) - max(x1, rx1))
            h = max(0.0, min(y2, ry2) - max(y1, ry1))
            total += w * h
        return min(total, self.roi.area_px)


# ---------------------------------------------------------------------------
# Convenience: every approach at once
# ---------------------------------------------------------------------------

def compute_all(intersection: IntersectionROI,
                frame: FrameData | Iterable,
                **kwargs) -> dict[str, DensityResult]:
    """Density for all approaches of an intersection in one call."""
    detections = frame.detections if isinstance(frame, FrameData) else list(frame)
    return {
        roi.name: DensityEstimator(roi, **kwargs).compute(detections)
        for roi in intersection
    }
