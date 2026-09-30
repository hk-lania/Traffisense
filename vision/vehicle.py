"""
vehicle.py
TraffiSense Vision Module

Standard vehicle representation used throughout the project.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Vehicle:
    """
    Represents one tracked vehicle.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float

    bbox: Tuple[int, int, int, int]

    center: Tuple[int, int]

    frame_number: int

    timestamp: float