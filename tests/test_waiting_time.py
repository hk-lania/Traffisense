"""Waiting-time tests.

Waiting time is the only metric that accumulates across frames, so these tests
feed sequences of frames rather than single snapshots.
"""

import unittest

from metrics.motion import MotionTracker
from metrics.roi import ApproachROI, IntersectionROI
from metrics.waiting_time import WaitingTimeTracker

FPS = 25.0
STOP_Y = 400.0


def approach():
    return ApproachROI(
        name="north",
        polygon=[(0, 0), (60, 0), (60, STOP_Y), (0, STOP_Y)],
        stop_line=((0, STOP_Y), (60, STOP_Y)),
        pixels_per_meter=10.0,
    )


def intersection():
    return IntersectionROI.from_list([approach()])


def box_at(track_id, y_bottom, jitter=0.0):
    return {"track_id": track_id, "class": "car",
            "bbox": [10 + jitter, y_bottom - 40 + jitter,
                     50 + jitter, y_bottom + jitter],
            "lane": "north"}


def new_tracker(**kwargs):
    motion = MotionTracker(stop_speed_px_s=4.0, go_speed_px_s=8.0,
                           min_stop_frames=3)
    return WaitingTimeTracker(intersection(), motion=motion, **kwargs)


class TestWaitingAccumulation(unittest.TestCase):

    def test_a_parked_vehicle_accumulates_time(self):
        tracker = new_tracker()
        for step in range(int(10 * FPS)):        # 10 seconds of video
            tracker.update([box_at(1, 380)], timestamp=step / FPS)

        result = tracker.result("north")
        self.assertEqual(result.waiting_count, 1)
        self.assertEqual(result.longest_track_id, 1)
        # A short latch delay is expected (speed window + min_stop_frames);
        # it must be under a second, not seconds.
        self.assertGreater(result.max_wait_s, 9.0)
        self.assertLessEqual(result.max_wait_s, 10.0)

    def test_detector_jitter_does_not_reset_the_wait(self):
        tracker = new_tracker()
        jitters = [0.0, 0.8, -0.7, 0.5, -0.9, 0.3, -0.4, 0.6]
        for step in range(int(10 * FPS)):
            tracker.update([box_at(1, 380, jitters[step % len(jitters)])],
                           timestamp=step / FPS)
        self.assertGreater(tracker.result("north").max_wait_s, 9.0)

    def test_a_moving_vehicle_never_waits(self):
        tracker = new_tracker()
        for step in range(int(5 * FPS)):
            # 200 px/s upstream -> clearly moving
            tracker.update([box_at(1, 100 + step * 8.0)], timestamp=step / FPS)
        result = tracker.result("north")
        self.assertEqual(result.waiting_count, 0)
        self.assertEqual(result.max_wait_s, 0.0)

    def test_wait_resets_when_the_vehicle_drives_off(self):
        tracker = new_tracker()
        for step in range(int(8 * FPS)):         # stopped for 8 s
            tracker.update([box_at(1, 380)], timestamp=step / FPS)
        stopped = tracker.result("north").max_wait_s
        self.assertGreater(stopped, 7.0)

        start = int(8 * FPS)
        for step in range(start, start + int(3 * FPS)):   # then accelerates
            tracker.update([box_at(1, 380 - (step - start) * 9.0)],
                           timestamp=step / FPS)

        result = tracker.result("north")
        self.assertEqual(result.waiting_count, 0)
        self.assertEqual(tracker.wait_for(1), 0.0)
        # ...but the cumulative delay this vehicle suffered is retained.
        self.assertGreater(tracker.records[1].total_wait_s, 7.0)

    def test_total_wait_sums_separate_stop_episodes(self):
        tracker = new_tracker()
        step = 0

        def hold(seconds, y):
            nonlocal step
            for _ in range(int(seconds * FPS)):
                tracker.update([box_at(1, y)], timestamp=step / FPS)
                step += 1

        def move(seconds, y_start):
            nonlocal step
            for i in range(int(seconds * FPS)):
                tracker.update([box_at(1, y_start - i * 9.0)],
                               timestamp=step / FPS)
                step += 1

        hold(5, 380)
        move(2, 380)
        hold(5, 200)

        record = tracker.records[1]
        self.assertEqual(record.stop_episodes, 2)
        self.assertGreater(record.total_wait_s, 8.5)
        self.assertLess(record.total_wait_s, 10.5)

    def test_several_vehicles_report_max_and_mean(self):
        tracker = new_tracker()
        for step in range(int(10 * FPS)):
            frame = [box_at(1, 380)]
            if step >= int(5 * FPS):             # a second car joins later
                frame.append(box_at(2, 330))
            tracker.update(frame, timestamp=step / FPS)

        result = tracker.result("north")
        self.assertEqual(result.waiting_count, 2)
        self.assertGreater(result.max_wait_s, 9.0)
        self.assertLess(result.mean_wait_s, result.max_wait_s)
        self.assertGreater(result.mean_wait_s, 6.0)

    def test_short_tracking_dropout_preserves_the_wait(self):
        tracker = new_tracker()
        for step in range(int(12 * FPS)):
            # The detector misses this vehicle for half a second at t = 5 s.
            if int(5 * FPS) <= step < int(5.5 * FPS):
                tracker.update([], timestamp=step / FPS)
            else:
                tracker.update([box_at(1, 380)], timestamp=step / FPS)
        self.assertGreater(tracker.result("north").max_wait_s, 10.5)

    def test_long_absence_retires_the_record(self):
        tracker = new_tracker(forget_after_s=2.0)
        for step in range(int(5 * FPS)):
            tracker.update([box_at(1, 380)], timestamp=step / FPS)
        for step in range(int(5 * FPS), int(12 * FPS)):
            tracker.update([], timestamp=step / FPS)
        self.assertNotIn(1, tracker.records)
        self.assertEqual(tracker.result("north").waiting_count, 0)

    def test_score_is_capped_at_one(self):
        tracker = new_tracker(normalisation_cap_s=5.0)
        for step in range(int(20 * FPS)):
            tracker.update([box_at(1, 380)], timestamp=step / FPS)
        self.assertEqual(tracker.result("north").score, 1.0)

    def test_vehicles_outside_every_roi_are_ignored(self):
        tracker = new_tracker()
        outside = {"track_id": 5, "class": "car", "bbox": [900, 900, 940, 940]}
        for step in range(int(5 * FPS)):
            tracker.update([outside], timestamp=step / FPS)
        self.assertEqual(len(tracker.records), 0)


if __name__ == "__main__":
    unittest.main()
