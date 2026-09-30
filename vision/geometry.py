"""
geometry.py
TraffiSense Vision Module

Reusable geometry utilities.
"""

from typing import Tuple


Point = Tuple[int, int]


def side_of_line(point: Point, line_start: Point, line_end: Point) -> float:
    """
    Returns the signed position of a point relative to a line.

    > 0 : Point lies on one side
    < 0 : Point lies on the opposite side
    = 0 : Point lies exactly on the line
    """

    return (
        (line_end[0] - line_start[0]) * (point[1] - line_start[1])
        - (line_end[1] - line_start[1]) * (point[0] - line_start[0])
    )