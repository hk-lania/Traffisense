"""
line_crossing.py
TraffiSense Vision Module

Detects when tracked vehicles cross a virtual counting line.
"""

from typing import Dict, List

from .geometry import side_of_line
from .line import Line
from .vehicle import Vehicle


class LineCrossingDetector:

    def __init__(self, line: Line):

        self.line = line

        self.previous_side: Dict[int, float] = {}

        self.crossed_ids = set()

        self.total_crossings = 0

    def update(
        self,
        vehicles: List[Vehicle],
    ) -> List[Vehicle]:

        crossed = []

        for vehicle in vehicles:

            current_side = side_of_line(
                vehicle.center,
                self.line.start,
                self.line.end,
            )

            if vehicle.track_id not in self.previous_side:

                self.previous_side[vehicle.track_id] = current_side
                continue

            previous_side = self.previous_side[
                vehicle.track_id
            ]

            if (

                previous_side * current_side < 0

                and vehicle.track_id
                not in self.crossed_ids

            ):

                self.crossed_ids.add(
                    vehicle.track_id
                )

                self.total_crossings += 1

                crossed.append(vehicle)

            self.previous_side[
                vehicle.track_id
            ] = current_side

        return crossed

    def get_total_crossings(self):

        return self.total_crossings