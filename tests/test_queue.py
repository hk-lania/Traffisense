"""Queue-length tests.

Geometry used throughout: a 60 px wide, 400 px long approach whose stop line
is the bottom edge (y = 400).  The queue direction therefore points upwards,
and a vehicle's distance from the stop line is simply (400 - y).

The scale is 10 px/m, so 40 px = 4 m and the default 9 m gap rule becomes
90 px - which keeps every expected answer easy to check by hand.
"""

import unittest

from metrics.motion import MotionTracker, TrackMotion
from metrics.queue_length import QueueEstimator
from metrics.roi import ApproachROI

PPM = 10.0
STOP_Y = 400.0


def approach():
    return ApproachROI(
        name="north",
        polygon=[(0, 0), (60, 0), (60, STOP_Y), (0, STOP_Y)],
        stop_line=((0, STOP_Y), (60, STOP_Y)),
        pixels_per_meter=PPM,
        lane_count=1,
    )


def vehicle(track_id, front_px, length_px=40.0, cls="car"):
    """A box whose near edge sits `front_px` back from the stop line."""
    y2 = STOP_Y - front_px
    y1 = y2 - length_px
    return {"track_id": track_id, "class": cls, "bbox": [10, y1, 50, y2],
            "lane": "north"}


def states(*specs):
    """Fake motion states: (track_id, stationary, speed)."""
    out = {}
    for track_id, stationary, speed in specs:
        out[track_id] = TrackMotion(track_id=track_id, position=(0.0, 0.0),
                                    timestamp=0.0, speed_px_s=speed,
                                    stationary=stationary)
    return out


class TestQueueGeometry(unittest.TestCase):

    def test_queue_direction_points_away_from_the_stop_line(self):
        roi = approach()
        self.assertAlmostEqual(roi.queue_direction[0], 0.0, places=6)
        self.assertAlmostEqual(roi.queue_direction[1], -1.0, places=6)

    def test_distance_from_stop_line(self):
        roi = approach()
        self.assertAlmostEqual(roi.distance_from_stop_line((30, 400)), 0.0)
        self.assertAlmostEqual(roi.distance_from_stop_line((30, 300)), 100.0)
        # Past the line, inside the junction -> negative.
        self.assertAlmostEqual(roi.distance_from_stop_line((30, 420)), -20.0)


class TestQueueChain(unittest.TestCase):

    def setUp(self):
        self.est = QueueEstimator(approach(), motion=MotionTracker())

    def run_queue(self, vehicles, motion):
        return self.est.compute(vehicles, motion_states=motion,
                                update_motion=False)

    def test_empty_approach(self):
        result = self.run_queue([], {})
        self.assertEqual(result.length_px, 0.0)
        self.assertEqual(result.queued_count, 0)

    def test_three_stopped_vehicles_bumper_to_bumper(self):
        # Fronts at 0, 50, 100; each 40 px long -> rear of the last = 140 px.
        vehicles = [vehicle(1, 0), vehicle(2, 50), vehicle(3, 100)]
        motion = states((1, True, 0.0), (2, True, 0.0), (3, True, 0.0))
        result = self.run_queue(vehicles, motion)
        self.assertEqual(result.queued_count, 3)
        self.assertAlmostEqual(result.length_px, 140.0, places=3)
        self.assertAlmostEqual(result.length_m, 14.0, places=3)

    def test_large_gap_breaks_the_queue(self):
        # Vehicle 3 starts 160 px behind the pair, beyond the 90 px gap rule.
        vehicles = [vehicle(1, 0), vehicle(2, 50), vehicle(3, 250)]
        motion = states((1, True, 0.0), (2, True, 0.0), (3, True, 0.0))
        result = self.run_queue(vehicles, motion)
        self.assertEqual(result.queued_count, 2)
        self.assertAlmostEqual(result.length_px, 90.0, places=3)

    def test_no_queue_when_nothing_waits_near_the_stop_line(self):
        # Head of the group is 200 px back - beyond the 120 px first-gap rule.
        vehicles = [vehicle(1, 200), vehicle(2, 250)]
        motion = states((1, True, 0.0), (2, True, 0.0))
        result = self.run_queue(vehicles, motion)
        self.assertEqual(result.queued_count, 0)
        self.assertEqual(result.length_px, 0.0)

    def test_moving_vehicles_are_not_queued(self):
        vehicles = [vehicle(1, 0), vehicle(2, 50)]
        motion = states((1, False, 200.0), (2, False, 200.0))
        result = self.run_queue(vehicles, motion)
        self.assertEqual(result.queued_count, 0)
        self.assertEqual(result.stopped_count, 0)
        self.assertEqual(result.vehicles_in_roi, 2)

    def test_stopped_head_with_moving_traffic_behind(self):
        vehicles = [vehicle(1, 0), vehicle(2, 50), vehicle(3, 100)]
        motion = states((1, True, 0.0), (2, True, 0.0), (3, False, 250.0))
        result = self.run_queue(vehicles, motion)
        self.assertEqual(result.queued_count, 2)
        self.assertAlmostEqual(result.length_px, 90.0, places=3)

    def test_vehicle_past_the_stop_line_is_excluded(self):
        # Entirely inside the junction: front -60, rear -20.
        crossed = {"track_id": 9, "class": "car",
                   "bbox": [10, STOP_Y + 20, 50, STOP_Y + 60], "lane": "north"}
        result = self.run_queue([crossed], states((9, True, 0.0)))
        self.assertEqual(result.queued_count, 0)

    def test_vehicle_nosing_over_the_line_still_counts(self):
        # Front 10 px past the line, rear 30 px behind it - the queue head.
        nosed = {"track_id": 4, "class": "car",
                 "bbox": [10, STOP_Y - 30, 50, STOP_Y + 10], "lane": "north"}
        result = self.run_queue([nosed], states((4, True, 0.0)))
        self.assertEqual(result.queued_count, 1)
        self.assertAlmostEqual(result.length_px, 30.0, places=3)

    def test_creeping_traffic_counts_as_queued(self):
        # Not latched as stationary, but slower than the creep threshold.
        vehicles = [vehicle(1, 0), vehicle(2, 50)]
        motion = states((1, False, 1.0), (2, False, 1.0))
        result = self.run_queue(vehicles, motion)
        self.assertEqual(result.queued_count, 2)

    def test_score_is_relative_to_the_roi_length(self):
        vehicles = [vehicle(1, 0), vehicle(2, 50), vehicle(3, 100)]
        motion = states((1, True, 0.0), (2, True, 0.0), (3, True, 0.0))
        result = self.run_queue(vehicles, motion)
        self.assertAlmostEqual(result.score, 140.0 / STOP_Y, places=6)

    def test_vehicle_equivalents(self):
        vehicles = [vehicle(1, 0), vehicle(2, 50), vehicle(3, 100)]
        motion = states((1, True, 0.0), (2, True, 0.0), (3, True, 0.0))
        result = self.run_queue(vehicles, motion)
        # 14 m of queue at a 6 m slot -> about 2.3 vehicle slots.
        self.assertAlmostEqual(result.vehicle_equivalents(6.0), 14.0 / 6.0,
                               places=6)


class TestQueueWithRealMotion(unittest.TestCase):
    """The estimator driven by the real MotionTracker, not fake states."""

    def test_stationary_vehicles_are_detected_over_time(self):
        est = QueueEstimator(approach(),
                             motion=MotionTracker(stop_speed_px_s=4.0,
                                                  min_stop_frames=3))
        vehicles = [vehicle(1, 0), vehicle(2, 50)]
        result = None
        for step in range(30):           # 30 frames at 25 fps = 1.2 s
            result = est.compute(vehicles, timestamp=step / 25.0)
        self.assertEqual(result.queued_count, 2)
        self.assertAlmostEqual(result.length_px, 90.0, places=3)

    def test_departing_vehicles_leave_the_queue(self):
        est = QueueEstimator(approach(),
                             motion=MotionTracker(stop_speed_px_s=4.0,
                                                  go_speed_px_s=8.0,
                                                  min_stop_frames=3))
        for step in range(30):           # stopped at the line
            est.compute([vehicle(1, 0), vehicle(2, 50)], timestamp=step / 25.0)

        result = None
        for step in range(30, 70):       # then pulling away at ~150 px/s
            shift = (step - 30) * 6.0
            moving = [vehicle(1, -shift), vehicle(2, 50 - shift)]
            result = est.compute(moving, timestamp=step / 25.0)
        self.assertEqual(result.queued_count, 0)


if __name__ == "__main__":
    unittest.main()
