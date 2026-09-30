"""End-to-end tests: the whole module against a simulated tracker stream.

The unit tests prove each metric is arithmetically right.  These prove the
assembled engine behaves like traffic:

  * the busiest approach ends up with the highest pressure,
  * the estimated queue length matches the simulator's ground truth,
  * vehicles held on red accumulate real waiting time,
  * every reported value stays inside its valid range,
  * nothing breaks when the tracker sends plain dicts with no lane label.
"""

import statistics
from collections import Counter
import unittest

from metrics import (PressureConfig, TrafficMetricsEngine,
                     rectangular_intersection)
from simulate import IntersectionSimulator

SECONDS = 120.0
FPS = 25.0


class TestEndToEnd(unittest.TestCase):
    """One 120-second run, inspected from several angles."""

    @classmethod
    def setUpClass(cls):
        intersection = rectangular_intersection()
        engine = TrafficMetricsEngine(intersection)
        sim = IntersectionSimulator(intersection, fps=FPS, seed=7)

        cls.names = intersection.names
        cls.pressures = {name: [] for name in cls.names}
        cls.densities = {name: [] for name in cls.names}
        cls.queue_errors = []
        cls.queue_relative_errors = []
        cls.max_wait = 0.0
        cls.max_queue_score = 0.0
        cls.rows = []

        for frame, timestamp, detections in sim.run(SECONDS):
            snapshot = engine.update(detections, frame=frame, timestamp=timestamp)
            for name, metrics in snapshot.items():
                cls.pressures[name].append(metrics.pressure.pressure)
                cls.densities[name].append(metrics.density.density_percent)
                cls.max_wait = max(cls.max_wait, metrics.waiting.max_wait_s)
                cls.max_queue_score = max(cls.max_queue_score, metrics.queue.score)
                cls.rows.append(metrics.to_dict())

                truth = sim.true_queue_length_m(name)
                if truth > 8.0:      # only judge accuracy on real queues
                    error = abs(metrics.queue.length_m - truth)
                    cls.queue_errors.append(error)
                    cls.queue_relative_errors.append(error / truth)

        cls.engine = engine
        cls.sim = sim

    # --- behaviour ----------------------------------------------------
    def test_the_busiest_approach_has_the_highest_mean_pressure(self):
        means = {n: statistics.mean(v) for n, v in self.pressures.items()}
        busiest = max(means, key=means.get)
        self.assertEqual(busiest, "north",
                         f"north has the highest arrival rate but means were {means}")
        self.assertGreater(means["north"], means["west"] * 1.5)

    def test_queue_length_matches_the_simulated_truth(self):
        self.assertGreater(len(self.queue_errors), 500,
                           "not enough real queues formed to judge accuracy")
        # The median is what the estimate looks like normally.  The mean is
        # pulled up by the moment a green starts, when the simulator calls a
        # vehicle "moving" a fraction of a second before the estimator does.
        self.assertLess(statistics.median(self.queue_errors), 1.5)
        self.assertLess(statistics.mean(self.queue_errors), 6.5)
        self.assertLess(statistics.mean(self.queue_relative_errors), 0.25)

    def test_vehicles_held_on_red_accumulate_waiting_time(self):
        self.assertGreater(self.max_wait, 20.0)

    def test_queues_actually_form(self):
        self.assertGreater(self.max_queue_score, 0.25)

    def test_density_rises_with_the_queue(self):
        # On the busiest approach, the densest frames must not be the emptiest.
        busy = self.densities["north"]
        self.assertGreater(max(busy), 10.0)
        self.assertLess(min(busy), 5.0)

    # --- output hygiene ------------------------------------------------
    def test_every_value_stays_in_range(self):
        for row in self.rows:
            self.assertGreaterEqual(row["vehicle_count"], 0)
            self.assertGreaterEqual(row["density_percent"], 0.0)
            self.assertLessEqual(row["density_percent"], 100.0)
            self.assertGreaterEqual(row["queue_length_px"], 0.0)
            self.assertGreaterEqual(row["waiting_time_s"], 0.0)
            self.assertGreaterEqual(row["pressure"], 0.0)
            self.assertLessEqual(row["pressure"], 100.0)

    def test_snapshot_has_one_block_per_approach(self):
        snapshot = self.engine.to_dict()
        self.assertEqual(sorted(snapshot), sorted(self.names))
        for block in snapshot.values():
            for key in ("vehicle_count", "density_percent", "queue_length_m",
                        "waiting_time_s", "pressure"):
                self.assertIn(key, block)

    def test_ranking_is_ordered(self):
        ranking = self.engine.pressure_ranking()
        values = [pressure for _, pressure in ranking]
        self.assertEqual(values, sorted(values, reverse=True))


class TestLanelessTraffic(unittest.TestCase):
    """Indian-style traffic: two-wheelers, no lane discipline, filtering.

    The queue chain measures distance from the stop line and never assumes
    lanes: vehicles take any lateral position that fits, and two-wheelers edge
    sideways into gaps. The metrics must hold up under that, because it is
    what the project's own footage will look like.
    """

    @staticmethod
    def _run(style: str) -> tuple[list[float], Counter]:
        """Queue errors and the class mix of queued vehicles, for one style."""
        intersection = rectangular_intersection()
        engine = TrafficMetricsEngine(intersection)
        sim = IntersectionSimulator(intersection, fps=FPS, seed=7, style=style)

        errors = []
        classes = Counter()
        for frame, timestamp, detections in sim.run(90.0):
            snapshot = engine.update(detections, frame=frame, timestamp=timestamp)
            for name, metrics in snapshot.items():
                for vehicle in metrics.queue.vehicles:
                    classes[vehicle.vehicle_class] += 1
                truth = sim.true_queue_length_m(name)
                if truth > 8.0:
                    errors.append(abs(metrics.queue.length_m - truth))
        return errors, classes

    def test_queue_length_is_accurate_without_lane_discipline(self):
        errors, _ = self._run("india")
        self.assertGreater(len(errors), 500)
        self.assertLess(statistics.median(errors), 2.0)
        self.assertLess(statistics.mean(errors), 6.5)

    def test_queue_length_is_accurate_with_lane_discipline(self):
        errors, _ = self._run("lane")
        self.assertGreater(len(errors), 500)
        self.assertLess(statistics.median(errors), 2.0)

    def test_two_wheelers_dominate_the_indian_queue(self):
        # Guards the mixed-class path: if two-wheelers stopped reaching the
        # queue, the class footprints and narrow boxes would go untested.
        _, classes = self._run("india")
        total = sum(classes.values())
        two_wheelers = classes["motorcycle"] + classes["bicycle"]
        self.assertGreater(two_wheelers / total, 0.30)
        self.assertGreater(classes["auto"] / total, 0.10)


class TestEngineRobustness(unittest.TestCase):

    def test_unlabelled_detections_are_assigned_to_an_approach(self):
        intersection = rectangular_intersection()
        engine = TrafficMetricsEngine(intersection)
        sim = IntersectionSimulator(intersection, fps=FPS, seed=11)

        assigned = 0
        total = 0
        for frame, timestamp, detections in sim.run(20.0):
            expected = {d["track_id"]: d["lane"] for d in detections}
            stripped = [{k: v for k, v in d.items() if k != "lane"}
                        for d in detections]
            snapshot = engine.update(stripped, frame=frame, timestamp=timestamp)
            for name, metrics in snapshot.items():
                for track_id in metrics.queue.track_ids:
                    total += 1
                    if expected.get(track_id) == name:
                        assigned += 1
        self.assertGreater(total, 20)
        self.assertEqual(assigned, total,
                         "auto lane assignment disagreed with the simulator")

    def test_empty_frames_do_not_break_the_engine(self):
        engine = TrafficMetricsEngine(rectangular_intersection())
        for frame in range(50):
            snapshot = engine.update([], frame=frame, timestamp=frame / FPS)
        self.assertTrue(all(m.pressure.pressure == 0.0
                            for m in snapshot.values()))

    def test_camera_confidence_scales_every_approach(self):
        intersection = rectangular_intersection()
        clear = TrafficMetricsEngine(intersection)
        foggy = TrafficMetricsEngine(intersection)
        sim_a = IntersectionSimulator(intersection, fps=FPS, seed=3)
        sim_b = IntersectionSimulator(intersection, fps=FPS, seed=3)

        clear_snapshot = foggy_snapshot = None
        for (f, t, dets_a), (_, _, dets_b) in zip(sim_a.run(30.0), sim_b.run(30.0)):
            clear_snapshot = clear.update(dets_a, frame=f, timestamp=t)
            foggy_snapshot = foggy.update(dets_b, frame=f, timestamp=t,
                                          camera_confidence=0.5)

        for name in intersection.names:
            expected = clear_snapshot[name].pressure.pressure * 0.5
            self.assertAlmostEqual(foggy_snapshot[name].pressure.pressure,
                                   expected, places=4)

    def test_custom_pressure_config_changes_the_outcome(self):
        intersection = rectangular_intersection()
        queue_only = PressureConfig(weights={"queue": 1.0, "waiting": 0.0,
                                             "count": 0.0, "density": 0.0,
                                             "flow": 0.0})
        engine = TrafficMetricsEngine(intersection, pressure_config=queue_only)
        sim = IntersectionSimulator(intersection, fps=FPS, seed=5)
        snapshot = None
        for f, t, dets in sim.run(45.0):
            snapshot = engine.update(dets, frame=f, timestamp=t)
        for metrics in snapshot.values():
            self.assertEqual(set(metrics.pressure.components), {"queue"})

    def test_reset_clears_accumulated_state(self):
        intersection = rectangular_intersection()
        engine = TrafficMetricsEngine(intersection)
        sim = IntersectionSimulator(intersection, fps=FPS, seed=2)
        for f, t, dets in sim.run(30.0):
            engine.update(dets, frame=f, timestamp=t)
        engine.reset()
        self.assertEqual(engine.frames_processed, 0)
        self.assertEqual(len(engine.motion.tracks), 0)
        self.assertEqual(len(engine.waiting.records), 0)


if __name__ == "__main__":
    unittest.main()
