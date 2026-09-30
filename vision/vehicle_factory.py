"""
vehicle_factory.py
TraffiSense Vision Module

Converts YOLO + ByteTrack detections into Vehicle objects.
"""

from typing import List

from .vehicle import Vehicle


class VehicleFactory:

    @staticmethod
    def create(results, frame_number: int, timestamp: float) -> List[Vehicle]:

        vehicles = []

        if len(results) == 0:
            return vehicles

        result = results[0]

        if result.boxes is None:
            return vehicles

        boxes = result.boxes

        for box in boxes:

            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2,
            )

            vehicle = Vehicle(
                track_id=int(box.id.item()),
                class_id=int(box.cls.item()),
                class_name=result.names[int(box.cls.item())],
                confidence=float(box.conf.item()),
                bbox=(x1, y1, x2, y2),
                center=center,
                frame_number=frame_number,
                timestamp=timestamp,
            )

            vehicles.append(vehicle)

        return vehicles