"""
advanced_queue_estimator.py

Standalone Queue Length Estimation
"""

import math


class QueueEstimator:

    def __init__(self):
        self.previous_positions = {}
        self.speed_threshold = 5.0

    def compute(self, vehicles):

        queued = []
        moving = []
        total_speed = 0

        for vehicle in vehicles:

            # Read dictionary values
            track_id = vehicle["track_id"]
            center = vehicle["center"]

            # First appearance
            if track_id not in self.previous_positions:

                self.previous_positions[track_id] = center
                moving.append(vehicle)
                continue

            previous = self.previous_positions[track_id]

            dx = center[0] - previous[0]
            dy = center[1] - previous[1]

            speed = math.sqrt(dx * dx + dy * dy)

            total_speed += speed

            if speed < self.speed_threshold:
                queued.append(vehicle)
            else:
                moving.append(vehicle)

            self.previous_positions[track_id] = center

        if len(vehicles) > 0:
            average_speed = total_speed / len(vehicles)
        else:
            average_speed = 0

        return {
            "queue_length": len(queued),
            "average_speed": average_speed,
            "queued_vehicles": queued,
            "moving_vehicles": moving
        }