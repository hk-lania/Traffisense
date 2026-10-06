"""Tests for the stand-in ByteTracker (tracking/byte_tracker.py)."""

import unittest

from metrics import TrafficMetricsEngine, rectangular_intersection
from metrics.simulate import IntersectionSimulator
from tracking import ByteTracker, normalise_class


def box(x, y, w=40, h=20):
    return [x, y, x + w, y + h]


class TrackerTests(unittest.TestCase):

    def test_moving_vehicle_keeps_its_id(self):
        tracker = ByteTracker(fps=25)
        ids = set()
        for f in range(50):
            out = tracker.update([{"class_name": "Car", "confidence": 0.9,
                                   "bbox": box(100 + 3 * f, 200)}], f, f / 25)
            ids.update(o["track_id"] for o in out)
        self.assertEqual(len(ids), 1)

    def test_two_separate_vehicles_get_two_ids(self):
        tracker = ByteTracker(fps=25)
        for f in range(5):
            out = tracker.update([
                {"class_name": "Car", "confidence": 0.9, "bbox": box(100, 100)},
                {"class_name": "Bus", "confidence": 0.9, "bbox": box(400, 100, 80, 30)},
            ], f, f / 25)
        self.assertEqual(len({o["track_id"] for o in out}), 2)
        self.assertEqual(sorted(o["class"] for o in out), ["bus", "car"])

    def test_low_confidence_box_keeps_the_id(self):
        """ByteTrack's second stage: a partly hidden vehicle keeps its track."""
        tracker = ByteTracker(fps=25)
        for f in range(5):
            out = tracker.update([{"class_name": "Car", "confidence": 0.9,
                                   "bbox": box(100, 100)}], f, f / 25)
        first = out[0]["track_id"]
        out = tracker.update([{"class_name": "Car", "confidence": 0.2,
                               "bbox": box(101, 100)}], 5, 0.2)
        self.assertEqual(out[0]["track_id"], first)

    def test_short_dropout_keeps_the_id(self):
        tracker = ByteTracker(fps=25)
        for f in range(5):
            out = tracker.update([{"class_name": "Car", "confidence": 0.9,
                                   "bbox": box(100, 100)}], f, f / 25)
        first = out[0]["track_id"]
        for f in range(5, 10):                       # 5 missed frames
            tracker.update([], f, f / 25)
        out = tracker.update([{"class_name": "Car", "confidence": 0.9,
                               "bbox": box(100, 100)}], 10, 0.4)
        self.assertEqual(out[0]["track_id"], first)

    def test_pedestrians_are_dropped_and_classes_mapped(self):
        self.assertIsNone(normalise_class("Pedestrian"))
        self.assertIsNone(normalise_class("person"))
        self.assertEqual(normalise_class("Semi-Truck"), "truck")
        self.assertEqual(normalise_class("Bicyclist"), "bicycle")
        self.assertEqual(normalise_class("motorcycle"), "motorcycle")

    def test_metrics_with_tracker_ids_match_true_ids(self):
        """Feed simulated boxes WITHOUT ids through the tracker; the metrics
        must come out close to the ones computed with the true ids."""
        true_engine = TrafficMetricsEngine(rectangular_intersection())
        tracked_engine = TrafficMetricsEngine(rectangular_intersection())
        sim = IntersectionSimulator(rectangular_intersection(), fps=25)
        tracker = ByteTracker(fps=25)
        errors = []
        for frame, t, dets in sim.run(60):
            raw = [{"class_name": d["class"], "confidence": 0.85, "bbox": d["bbox"]}
                   for d in dets]
            truth = true_engine.update(dets, frame=frame, timestamp=t)
            tracked = tracked_engine.update(tracker.update(raw, frame, t),
                                            frame=frame, timestamp=t)
            if frame % 25 == 0:
                errors += [abs(truth[n].pressure.pressure - tracked[n].pressure.pressure)
                           for n in truth]
        self.assertLess(sum(errors) / len(errors), 3.0)   # pressure is 0-100


if __name__ == "__main__":
    unittest.main()
