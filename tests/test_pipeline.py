"""Tests for the metrics -> controller glue in run_pipeline.py."""

import types
import unittest

from run_pipeline import SignalPlanner


def metrics(pressure, count):
    return types.SimpleNamespace(pressure=types.SimpleNamespace(pressure=pressure),
                                 vehicle_count=count)


class PlannerTests(unittest.TestCase):

    def setUp(self):
        self.planner = SignalPlanner(["north", "east", "south", "west"])
        self.planner.observe({"north": metrics(10, 3), "east": metrics(50, 12),
                              "south": metrics(30, 6), "west": metrics(5, 1)})

    def test_highest_pressure_goes_first(self):
        direction, green, info = self.planner.next_phase()
        self.assertEqual(direction, "east")
        self.assertEqual(green, 10 + 2 * 12)          # controller's own rule
        self.assertEqual(info["vehicles"], 12)

    def test_every_approach_served_once_per_cycle(self):
        order = [self.planner.next_phase()[0] for _ in range(4)]
        self.assertEqual(order, ["east", "south", "north", "west"])
        # next cycle starts again from the highest pressure
        self.assertEqual(self.planner.next_phase()[0], "east")

    def test_green_time_is_clamped(self):
        self.planner.observe({"north": metrics(90, 100)})
        direction, green, _ = self.planner.next_phase()
        self.assertEqual((direction, green), ("north", 60))

    def test_no_data_yet_still_gives_a_safe_phase(self):
        planner = SignalPlanner(["north", "east"])
        direction, green, _ = planner.next_phase()
        self.assertIn(direction, ("north", "east"))
        self.assertEqual(green, 10)


if __name__ == "__main__":
    unittest.main()
