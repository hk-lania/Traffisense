"""
line.py
TraffiSense Vision Module

Represents a virtual counting line.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Line:
    """
    Represents a virtual line used for
    vehicle counting.
    """

    start: Tuple[int, int]
    end: Tuple[int, int]