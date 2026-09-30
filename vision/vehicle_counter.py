"""
vehicle_counter.py
TraffiSense Vision Module

Counts unique vehicles using ByteTrack IDs.
"""

from collections import defaultdict


class VehicleCounter:
    """
    Counts each tracked vehicle only once.
    """

    def __init__(self):

        self.seen_ids = set()

        self.class_counts = defaultdict(int)

        self.total_count = 0

    def update(self, vehicles):

        for vehicle in vehicles:

            if vehicle.track_id in self.seen_ids:
                continue

            self.seen_ids.add(vehicle.track_id)

            self.class_counts[vehicle.class_name] += 1

            self.total_count += 1

    def get_total_count(self):

        return self.total_count

    def get_class_counts(self):

        return dict(self.class_counts)