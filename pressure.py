"""
TraffiSense - metrics/pressure.py
----------------------------------
Combines the independent traffic metrics of one approach into a single
comparable number: the traffic PRESSURE of that approach.

    pressure = 100 x  SUM( weight_i x normalised_metric_i ) / SUM( weight_i )

                      optionally scaled by camera confidence

    +-------------------------------------------------------------------+
    |  P R O V I S I O N A L                                            |
    |                                                                   |
    |  This is NOT the final Traffic Pressure Index.  The weights and   |
    |  caps below are starting values so the pipeline runs end to end   |
    |  and can be demonstrated.  They live in PressureConfig - in a     |
    |  JSON file, not in the code - precisely so they can be retuned    |
    |  (or the whole formula replaced) once the individual modules are  |
    |  validated against real footage.  Nothing upstream depends on     |
    |  these numbers: density, queue and waiting time each stay         |
    |  independent and meaningful on their own.                         |
    +-------------------------------------------------------------------+

Design notes
============
* Each raw metric is first NORMALISED to 0 - 1 against a cap, because the raw
  units are incomparable: vehicles, metres, seconds and percent cannot be
  added together.  A value at or above its cap normalises to 1.0.
* Every result carries a full breakdown (raw value, normalised value, weight,
  contribution), so any pressure number can be explained rather than trusted -
  which is exactly what a reviewer or examiner will ask for.
* Camera confidence, when supplied, scales the result: unreliable footage
  should not drive a confident signal decision.  It is applied as a separate,
  clearly-labelled multiplier rather than being mixed into the weighted sum.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

__all__ = ["PressureConfig", "MetricContribution", "PressureResult",
           "PressureCalculator", "DEFAULT_WEIGHTS", "DEFAULT_CAPS"]


# --- PROVISIONAL starting values -------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "queue": 0.30,        # how far back the jam reaches
    "waiting": 0.30,      # how long people have already been stuck
    "count": 0.25,        # how many vehicles are demanding the green
    "density": 0.15,      # how tightly the road is packed
    "flow": 0.00,         # optical flow, off by default (see notes below)
}

DEFAULT_CAPS: dict[str, float] = {
    "count": 25.0,        # vehicles on one approach = "as busy as it gets"
    "queue_m": 80.0,      # metres of queue = "as long as it gets"
    "waiting_s": 60.0,    # seconds of wait = "as delayed as it gets"
    "density_pct": 100.0,
    "flow": 10.0,         # mean optical-flow magnitude at free-flow speed
}


@dataclass
class PressureConfig:
    """Tunable, serialisable settings for the provisional pressure formula."""

    weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS))
    caps: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CAPS))

    # 'linear'     : min(x / cap, 1)          - simple, predictable
    # 'saturating' : 1 - exp(-x / cap * 3)    - early growth, soft ceiling
    normalisation: str = "linear"

    # Optical flow is an INVERSE indicator: high flow means traffic is moving,
    # so pressure should fall.  When enabled, its normalised value is inverted.
    invert_flow: bool = True

    # Camera confidence handling.
    apply_camera_confidence: bool = True
    confidence_floor: float = 0.3     # never scale below this, to avoid a
                                      # bad frame silently zeroing an approach

    scale: float = 100.0              # output range: 0 - scale
    version: str = "provisional-0.1"

    def __post_init__(self) -> None:
        if self.normalisation not in ("linear", "saturating"):
            raise ValueError("normalisation must be 'linear' or 'saturating'")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must not be negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("at least one weight must be positive")

    # --- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PressureConfig":
        base = cls()
        weights = dict(base.weights)
        weights.update(data.get("weights", {}))
        caps = dict(base.caps)
        caps.update(data.get("caps", {}))
        return cls(
            weights=weights,
            caps=caps,
            normalisation=data.get("normalisation", base.normalisation),
            invert_flow=data.get("invert_flow", base.invert_flow),
            apply_camera_confidence=data.get("apply_camera_confidence",
                                             base.apply_camera_confidence),
            confidence_floor=data.get("confidence_floor", base.confidence_floor),
            scale=data.get("scale", base.scale),
            version=data.get("version", base.version),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PressureConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class MetricContribution:
    """One metric's share of the final pressure - the explainability record."""

    name: str
    raw: Optional[float]
    normalised: float
    weight: float
    contribution: float       # weight x normalised, before rescaling

    def to_dict(self) -> dict:
        return {
            "raw": None if self.raw is None else round(self.raw, 3),
            "normalised": round(self.normalised, 3),
            "weight": self.weight,
            "contribution": round(self.contribution, 4),
        }


@dataclass
class PressureResult:
    """Pressure of one approach, with the full breakdown behind it."""

    approach: str
    pressure: float                    # 0 - config.scale (100 by default)
    raw_pressure: float                # before camera-confidence scaling
    camera_confidence: Optional[float]
    components: dict[str, MetricContribution] = field(default_factory=dict)
    config_version: str = "provisional-0.1"

    @property
    def score(self) -> float:
        return self.pressure / 100.0

    @property
    def level(self) -> str:
        p = self.pressure
        if p < 20:
            return "low"
        if p < 40:
            return "moderate"
        if p < 60:
            return "high"
        if p < 80:
            return "very high"
        return "critical"

    def explain(self) -> str:
        """One-line-per-metric breakdown - useful in the demo and the viva."""
        lines = [f"{self.approach}: pressure {self.pressure:.1f} "
                 f"({self.level}), config {self.config_version}"]
        for name, c in sorted(self.components.items(),
                              key=lambda kv: -kv[1].contribution):
            raw = "n/a" if c.raw is None else f"{c.raw:.2f}"
            lines.append(f"   {name:<8} raw={raw:>8}  norm={c.normalised:.2f}"
                         f"  w={c.weight:.2f}  -> {c.contribution:.3f}")
        if self.camera_confidence is not None:
            lines.append(f"   camera confidence x{self.camera_confidence:.2f} "
                         f"({self.raw_pressure:.1f} -> {self.pressure:.1f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "approach": self.approach,
            "pressure": round(self.pressure, 2),
            "raw_pressure": round(self.raw_pressure, 2),
            "level": self.level,
            "camera_confidence": self.camera_confidence,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "config_version": self.config_version,
        }


class PressureCalculator:
    """Applies the provisional pressure formula to one approach's metrics."""

    def __init__(self, config: Optional[PressureConfig] = None) -> None:
        self.config = config or PressureConfig()

    # ------------------------------------------------------------------
    def normalise(self, value: Optional[float], cap: float) -> float:
        if value is None or cap <= 0:
            return 0.0
        x = max(0.0, float(value))
        if self.config.normalisation == "saturating":
            return float(1.0 - math.exp(-3.0 * x / cap))
        return min(1.0, x / cap)

    # ------------------------------------------------------------------
    def compute(self,
                approach: str,
                vehicle_count: Optional[int] = None,
                density: Any = None,
                queue: Any = None,
                waiting: Any = None,
                optical_flow: Optional[float] = None,
                camera_confidence: Optional[float] = None) -> PressureResult:
        """Fuse one approach's metrics into a pressure value.

        `density`, `queue` and `waiting` accept either the result objects from
        the other modules (DensityResult / QueueResult / WaitingTimeResult) or
        plain numbers (percent, metres, seconds).
        """
        cfg = self.config
        caps = cfg.caps

        density_pct = _value(density, ("density_percent",))
        queue_m = _value(queue, ("length_m",))
        queue_px = _value(queue, ("length_px",))
        queue_max_px = _value(queue, ("max_queue_px",))
        wait_s = _value(waiting, ("max_wait_s",))

        if vehicle_count is None:
            vehicle_count = _value(queue, ("vehicles_in_roi",)) \
                or _value(density, ("vehicle_count",))

        # Queue normalises against metres when calibrated, otherwise against
        # the ROI's own length in pixels - so an uncalibrated camera still
        # produces a meaningful 0 - 1 value.
        if queue_m is not None:
            queue_norm = self.normalise(queue_m, caps["queue_m"])
            queue_raw: Optional[float] = queue_m
        elif queue_px is not None and queue_max_px:
            queue_norm = max(0.0, min(1.0, queue_px / queue_max_px))
            queue_raw = queue_px
        else:
            queue_norm = 0.0
            queue_raw = queue_px

        normalised = {
            "count": self.normalise(vehicle_count, caps["count"]),
            "queue": queue_norm,
            "waiting": self.normalise(wait_s, caps["waiting_s"]),
            "density": self.normalise(density_pct, caps["density_pct"]),
        }
        raws: dict[str, Optional[float]] = {
            "count": None if vehicle_count is None else float(vehicle_count),
            "queue": queue_raw,
            "waiting": wait_s,
            "density": density_pct,
        }

        if cfg.weights.get("flow", 0.0) > 0:
            flow_norm = self.normalise(optical_flow, caps.get("flow", 10.0))
            normalised["flow"] = (1.0 - flow_norm) if cfg.invert_flow else flow_norm
            raws["flow"] = optical_flow

        components: dict[str, MetricContribution] = {}
        weight_sum = 0.0
        weighted = 0.0
        for name, norm in normalised.items():
            weight = float(cfg.weights.get(name, 0.0))
            if weight <= 0:
                continue
            contribution = weight * norm
            weighted += contribution
            weight_sum += weight
            components[name] = MetricContribution(
                name=name, raw=raws.get(name), normalised=norm,
                weight=weight, contribution=contribution,
            )

        raw_pressure = cfg.scale * (weighted / weight_sum) if weight_sum else 0.0

        confidence_used: Optional[float] = None
        pressure = raw_pressure
        if cfg.apply_camera_confidence and camera_confidence is not None:
            confidence_used = max(cfg.confidence_floor,
                                  min(1.0, float(camera_confidence)))
            pressure = raw_pressure * confidence_used

        return PressureResult(
            approach=approach,
            pressure=pressure,
            raw_pressure=raw_pressure,
            camera_confidence=confidence_used,
            components=components,
            config_version=cfg.version,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def rank(results: Mapping[str, PressureResult]) -> list[tuple[str, float]]:
        """Approaches ordered by pressure, highest first.

        Provided for the FUTURE signal controller to consume; deciding what to
        do with the ranking is explicitly out of scope for this module.
        """
        return sorted(((name, r.pressure) for name, r in results.items()),
                      key=lambda kv: -kv[1])


def _value(source: Any, attributes: Sequence[str]) -> Optional[float]:
    """Read a metric out of a result object, a dict, or a bare number."""
    if source is None:
        return None
    if isinstance(source, (int, float)):
        return float(source)
    for attr in attributes:
        if isinstance(source, Mapping):
            if source.get(attr) is not None:
                return float(source[attr])
        else:
            value = getattr(source, attr, None)
            if value is not None:
                return float(value)
    return None
