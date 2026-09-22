"""Input-contract tests: the metrics layer must accept whatever the tracker
hands over - dicts, objects, different key spellings - without breaking.

These exist because of two real failures during integration:
    ModuleNotFoundError: No module named 'tracker'
    AttributeError: 'dict' object has no attribute 'track_id'
"""

import unittest

from metrics.schema import (Detection, as_detection, as_frame,
                            normalise_direction)


class _FakeTrack:
    """Stand-in for a tracker object (what ByteTrack-style code returns)."""

    def __init__(self):
        self.track_id = 7
        self.cls = "CAR"
        self.xyxy = (10.0, 20.0, 30.0, 60.0)
        self.frame = 5
        self.timestamp = 0.2
        self.side = "N"


class TestDetectionParsing(unittest.TestCase):

    def test_plain_dict(self):
        det = as_detection({"track_id": 3, "class": "car",
                            "bbox": [0, 0, 10, 20], "frame": 1,
                            "timestamp": 0.04, "lane": "north"})
        self.assertEqual(det.track_id, 3)
        self.assertEqual(det.vehicle_class, "car")
        self.assertEqual(det.bbox, (0.0, 0.0, 10.0, 20.0))
        self.assertEqual(det.lane, "north")

    def test_alternative_key_names(self):
        det = as_detection({"id": 9, "label": "Bus", "x1": 5, "y1": 5,
                            "x2": 25, "y2": 45, "frame_id": 2, "ts": 0.08,
                            "direction": "EB"})
        self.assertEqual(det.track_id, 9)
        self.assertEqual(det.vehicle_class, "bus")
        self.assertEqual(det.lane, "east")
        self.assertEqual(det.frame, 2)

    def test_tracker_object(self):
        det = as_detection(_FakeTrack())
        self.assertEqual(det.track_id, 7)
        self.assertEqual(det.vehicle_class, "car")
        self.assertEqual(det.lane, "north")
        self.assertEqual(det.bbox, (10.0, 20.0, 30.0, 60.0))

    def test_centre_width_height_form(self):
        det = as_detection({"track_id": 1, "cx": 50, "cy": 50, "w": 20, "h": 10})
        self.assertEqual(det.bbox, (40.0, 45.0, 60.0, 55.0))

    def test_reversed_corners_are_repaired(self):
        det = as_detection({"track_id": 1, "bbox": [30, 60, 10, 20]})
        self.assertEqual(det.bbox, (10.0, 20.0, 30.0, 60.0))

    def test_missing_track_id_is_an_error(self):
        with self.assertRaises(ValueError):
            as_detection({"class": "car", "bbox": [0, 0, 1, 1]})

    def test_missing_box_is_an_error(self):
        with self.assertRaises(ValueError):
            as_detection({"track_id": 1, "class": "car"})


class TestDetectionGeometry(unittest.TestCase):

    def setUp(self):
        self.det = Detection(track_id=1, bbox=(10.0, 20.0, 30.0, 60.0))

    def test_dimensions(self):
        self.assertEqual(self.det.width, 20.0)
        self.assertEqual(self.det.height, 40.0)
        self.assertEqual(self.det.area, 800.0)

    def test_points(self):
        self.assertEqual(self.det.centroid, (20.0, 40.0))
        self.assertEqual(self.det.ground_point, (20.0, 60.0))

    def test_scaling_preserves_centre_and_scales_area(self):
        scaled = self.det.scaled(0.5)
        self.assertAlmostEqual(scaled.area, 400.0, places=6)
        self.assertAlmostEqual(scaled.centroid[0], 20.0, places=6)
        self.assertAlmostEqual(scaled.centroid[1], 40.0, places=6)


class TestDirections(unittest.TestCase):

    def test_aliases(self):
        for value in ("N", "n", "north", "NORTH", "northbound", "NB"):
            self.assertEqual(normalise_direction(value), "north")
        self.assertEqual(normalise_direction("W"), "west")
        self.assertIsNone(normalise_direction(None))

    def test_unknown_lane_names_pass_through(self):
        self.assertEqual(normalise_direction("Ring Road A"), "ring road a")


class TestFrames(unittest.TestCase):

    def test_timestamp_derived_from_fps_when_absent(self):
        frame = as_frame([{"track_id": 1, "bbox": [0, 0, 1, 1]}],
                         frame=50, fps=25.0)
        self.assertAlmostEqual(frame.timestamp, 2.0, places=6)

    def test_by_lane_filters(self):
        frame = as_frame([
            {"track_id": 1, "bbox": [0, 0, 1, 1], "lane": "north"},
            {"track_id": 2, "bbox": [0, 0, 1, 1], "lane": "south"},
        ], frame=0, timestamp=0.0)
        self.assertEqual(len(frame.by_lane("N")), 1)
        self.assertEqual(frame.by_lane("N")[0].track_id, 1)

    def test_empty_frame(self):
        self.assertEqual(len(as_frame([], frame=0)), 0)


if __name__ == "__main__":
    unittest.main()
