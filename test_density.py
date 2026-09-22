"""Density tests - every case has a hand-computable answer.

ROI used throughout: a 100 x 100 square (area = 10 000 px^2) whose stop line
is its bottom edge.
"""

import unittest

from metrics.density import CLASS_FOOTPRINT, DensityEstimator
from metrics.roi import ApproachROI


def square_roi(size=100.0):
    return ApproachROI(
        name="north",
        polygon=[(0, 0), (size, 0), (size, size), (0, size)],
        stop_line=((0, size), (size, size)),
        pixels_per_meter=10.0,
    )


def box(track_id, x1, y1, x2, y2, cls="car"):
    return {"track_id": track_id, "class": cls, "bbox": [x1, y1, x2, y2],
            "lane": "north"}


class TestDensityBasics(unittest.TestCase):

    def setUp(self):
        self.roi = square_roi()
        # Raw geometric coverage, so the arithmetic is exact and checkable.
        self.est = DensityEstimator(self.roi, use_class_footprint=False,
                                    resolution=400)

    def test_roi_area(self):
        self.assertAlmostEqual(self.roi.area_px, 10000.0, places=6)

    def test_empty_road_is_zero(self):
        result = self.est.compute([])
        self.assertEqual(result.density_percent, 0.0)
        self.assertEqual(result.vehicle_count, 0)
        self.assertEqual(result.level, "free")

    def test_single_box_quarter_of_roi(self):
        # 50 x 50 = 2500 of 10 000 -> 25 %
        result = self.est.compute([box(1, 10, 10, 60, 60)])
        self.assertAlmostEqual(result.density_percent, 25.0, delta=0.7)
        self.assertEqual(result.vehicle_count, 1)

    def test_overlapping_boxes_are_not_double_counted(self):
        # 2500 + 2500 - 625 overlap = 4375 -> 43.75 %  (NOT 50 %)
        boxes = [box(1, 0, 0, 50, 50), box(2, 25, 25, 75, 75)]
        result = self.est.compute(boxes)
        self.assertAlmostEqual(result.density_percent, 43.75, delta=0.9)

    def test_sum_method_double_counts_by_design(self):
        boxes = [box(1, 0, 0, 50, 50), box(2, 25, 25, 75, 75)]
        summed = DensityEstimator(self.roi, use_class_footprint=False,
                                  method="sum").compute(boxes)
        self.assertAlmostEqual(summed.density_percent, 50.0, delta=0.1)
        self.assertGreater(summed.density_percent,
                           self.est.compute(boxes).density_percent)

    def test_box_partly_outside_roi_counts_only_the_inside_part(self):
        # Half the box hangs off the left edge -> 25 x 50 = 1250 -> 12.5 %
        result = self.est.compute([box(1, -25, 0, 25, 50)])
        self.assertAlmostEqual(result.density_percent, 12.5, delta=0.7)

    def test_density_never_exceeds_100(self):
        boxes = [box(i, -50, -50, 150, 150) for i in range(3)]
        result = self.est.compute(boxes)
        self.assertLessEqual(result.density_percent, 100.0)
        self.assertGreater(result.density_percent, 95.0)

    def test_other_lanes_are_ignored(self):
        other = box(1, 10, 10, 60, 60)
        other["lane"] = "south"
        self.assertEqual(self.est.compute([other]).density_percent, 0.0)


class TestClassFootprint(unittest.TestCase):

    def test_footprint_shrinks_coverage_by_the_class_factor(self):
        roi = square_roi()
        raw = DensityEstimator(roi, use_class_footprint=False,
                               resolution=400).compute([box(1, 10, 10, 60, 60)])
        adjusted = DensityEstimator(roi, use_class_footprint=True,
                                    resolution=400).compute([box(1, 10, 10, 60, 60)])
        expected = raw.density_percent * CLASS_FOOTPRINT["car"]
        self.assertAlmostEqual(adjusted.density_percent, expected, delta=0.8)

    def test_motorcycle_covers_less_than_a_car_of_the_same_box(self):
        roi = square_roi()
        est = DensityEstimator(roi, resolution=400)
        car = est.compute([box(1, 10, 10, 60, 60, "car")])
        bike = est.compute([box(1, 10, 10, 60, 60, "motorcycle")])
        self.assertLess(bike.density_percent, car.density_percent)

    def test_unknown_class_uses_the_fallback_and_does_not_crash(self):
        est = DensityEstimator(square_roi(), resolution=400)
        result = est.compute([box(1, 10, 10, 60, 60, "spaceship")])
        self.assertGreater(result.density_percent, 0.0)


class TestDensityScoring(unittest.TestCase):

    def test_score_is_density_over_100(self):
        est = DensityEstimator(square_roi(), use_class_footprint=False,
                               resolution=400)
        result = est.compute([box(1, 0, 0, 50, 50)])
        self.assertAlmostEqual(result.score, result.density_percent / 100.0,
                               places=9)

    def test_levels(self):
        est = DensityEstimator(square_roi(), use_class_footprint=False,
                               resolution=400)
        self.assertEqual(est.compute([]).level, "free")
        self.assertEqual(est.compute([box(1, 0, 0, 100, 70)]).level, "congested")


if __name__ == "__main__":
    unittest.main()
