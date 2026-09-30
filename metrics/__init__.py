"""
TraffiSense — metrics package
==============================
Traffic density, queue, waiting-time, and pressure metrics for
multi-approach intersection analysis.
"""

from .schema import Detection, FrameData, as_detection, as_frame, normalise_direction
from .roi import ApproachROI, IntersectionROI, rectangular_intersection
from .motion import MotionTracker, TrackMotion
from .density import DensityEstimator, DensityResult, CLASS_FOOTPRINT
from .queue import QueueEstimator, QueueResult
from .waiting_time import WaitingTimeTracker, WaitingTimeResult
from .pressure import PressureCalculator, PressureConfig, PressureResult
from .traffic_metrics import TrafficMetricsEngine, ApproachMetrics

__all__ = [
    # schema
    "Detection", "FrameData", "as_detection", "as_frame", "normalise_direction",
    # roi
    "ApproachROI", "IntersectionROI", "rectangular_intersection",
    # motion
    "MotionTracker", "TrackMotion",
    # density
    "DensityEstimator", "DensityResult", "CLASS_FOOTPRINT",
    # queue
    "QueueEstimator", "QueueResult",
    # waiting_time
    "WaitingTimeTracker", "WaitingTimeResult",
    # pressure
    "PressureCalculator", "PressureConfig", "PressureResult",
    # engine
    "TrafficMetricsEngine", "ApproachMetrics",
]
