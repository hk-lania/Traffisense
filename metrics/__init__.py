"""
TraffiSense - Traffic Metrics Module (Tanish)

Per-approach density, queue length, waiting time and (provisional) pressure,
computed from tracked-vehicle output. See metrics/README.md.

    from metrics import TrafficMetricsEngine, IntersectionROI
"""

from .schema import Detection, FrameData, as_detection, as_frame, normalise_direction
from .roi import ApproachROI, IntersectionROI, rectangular_intersection
from .motion import MotionTracker, TrackMotion
from .density import CLASS_FOOTPRINT, DensityEstimator, DensityResult
from .queue_length import QueueEstimator, QueueResult, QueuedVehicle
from .waiting_time import WaitingTimeResult, WaitingTimeTracker
from .pressure import PressureCalculator, PressureConfig, PressureResult
from .traffic_metrics import ApproachMetrics, TrafficMetricsEngine

__all__ = [
    "Detection", "FrameData", "as_detection", "as_frame", "normalise_direction",
    "ApproachROI", "IntersectionROI", "rectangular_intersection",
    "MotionTracker", "TrackMotion",
    "CLASS_FOOTPRINT", "DensityEstimator", "DensityResult",
    "QueueEstimator", "QueueResult", "QueuedVehicle",
    "WaitingTimeResult", "WaitingTimeTracker",
    "PressureCalculator", "PressureConfig", "PressureResult",
    "ApproachMetrics", "TrafficMetricsEngine",
]
