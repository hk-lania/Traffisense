"""
roi.py
TraffiSense Vision Module

Polygonal Region of Interest (ROI)
"""

from typing import List, Tuple

import cv2
import numpy as np

from .vehicle import Vehicle


class ROI:

    def __init__(
        self,
        points: List[Tuple[int, int]],
        approach_id: int,
        name: str,
    ):

        self.points = np.array(points, dtype=np.int32)

        self.approach_id = approach_id

        self.name = name

        self._mask = None

        self._mask_shape = None

    def get_mask(self, frame_shape):

        h, w = frame_shape[:2]

        if self._mask is None or self._mask_shape != (h, w):

            self._mask = np.zeros((h, w), dtype=np.uint8)

            cv2.fillPoly(
                self._mask,
                [self.points],
                255,
            )

            self._mask_shape = (h, w)

        return self._mask

    def contains_point(self, point):

        return (

            cv2.pointPolygonTest(

                self.points,

                (float(point[0]), float(point[1])),

                False,

            )

            >= 0

        )

    def filter_vehicles(self, vehicles: List[Vehicle]):

        return [

            vehicle

            for vehicle in vehicles

            if self.contains_point(vehicle.center)

        ]

    def draw(self, frame):

        overlay = frame.copy()

        cv2.fillPoly(

            overlay,

            [self.points],

            (0, 255, 0),

        )

        cv2.addWeighted(

            overlay,

            0.20,

            frame,

            0.80,

            0,

            frame,

        )

        cv2.polylines(

            frame,

            [self.points],

            True,

            (0, 255, 0),

            3,

        )

        centroid = self.points.mean(axis=0).astype(int)

        cv2.putText(

            frame,

            self.name,

            tuple(centroid),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2,

        )

        return frame