"""Pressure-fusion tests.

The formula is provisional, so these tests pin down its PROPERTIES rather than
specific magic numbers:

  * empty approach -> 0, every metric at its cap -> 100
  * every metric is monotonic (more queue can never mean less pressure)
  * the weights actually control the outcome
  * the contribution breakdown adds up to the reported pressure
  * camera confidence scales the result and respects its floor
  * a config survives a save/load round trip

If the weights are retuned later, these tests still hold - which is the point.
"""

import json
import tempfile
import unittest
from pathlib import Path

from metrics.pressure import PressureCalculator, PressureConfig


class TestPressureRange(unittest.TestCase):

    def setUp(self):
        self.calc = PressureCalculator()
        self.caps = self.calc.config.caps

    def test_empty_approach_is_zero(self):
        result = self.calc.compute("north", vehicle_count=0, density=0.0,
                                   queue=0.0, waiting=0.0)
        self.assertEqual(result.pressure, 0.0)
        self.assertEqual(result.level, "low")

    def test_everything_at_cap_is_one_hundred(self):
        result = self.calc.compute(
            "north",
            vehicle_count=self.caps["count"],
            density={"density_percent": self.caps["density_pct"]},
            queue={"length_m": self.caps["queue_m"]},
            waiting={"max_wait_s": self.caps["waiting_s"]},
        )
        self.assertAlmostEqual(result.pressure, 100.0, places=6)

    def test_beyond_cap_is_clamped(self):
        result = self.calc.compute(
            "north",
            vehicle_count=self.caps["count"] * 10,
            density={"density_percent": 400.0},
            queue={"length_m": self.caps["queue_m"] * 5},
            waiting={"max_wait_s": self.caps["waiting_s"] * 5},
        )
        self.assertAlmostEqual(result.pressure, 100.0, places=6)

    def test_half_of_every_cap_is_half_pressure(self):
        result = self.calc.compute(
            "north",
            vehicle_count=self.caps["count"] / 2,
            density={"density_percent": self.caps["density_pct"] / 2},
            queue={"length_m": self.caps["queue_m"] / 2},
            waiting={"max_wait_s": self.caps["waiting_s"] / 2},
        )
        self.assertAlmostEqual(result.pressure, 50.0, places=6)


class TestMonotonicity(unittest.TestCase):

    def setUp(self):
        self.calc = PressureCalculator()

    def _pressure(self, **kwargs):
        base = {"vehicle_count": 5, "density": 20.0,
                "queue": {"length_m": 20.0}, "waiting": {"max_wait_s": 10.0}}
        base.update(kwargs)
        return self.calc.compute("north", **base).pressure

    def test_more_queue_means_more_pressure(self):
        self.assertGreater(self._pressure(queue={"length_m": 40.0}),
                           self._pressure())

    def test_longer_wait_means_more_pressure(self):
        self.assertGreater(self._pressure(waiting={"max_wait_s": 30.0}),
                           self._pressure())

    def test_more_vehicles_mean_more_pressure(self):
        self.assertGreater(self._pressure(vehicle_count=15), self._pressure())

    def test_higher_density_means_more_pressure(self):
        self.assertGreater(self._pressure(density=60.0), self._pressure())


class TestWeightsAndBreakdown(unittest.TestCase):

    def test_a_single_weight_isolates_one_metric(self):
        config = PressureConfig(weights={"queue": 1.0, "waiting": 0.0,
                                         "count": 0.0, "density": 0.0,
                                         "flow": 0.0})
        calc = PressureCalculator(config)
        result = calc.compute("north", vehicle_count=50, density=100.0,
                              queue={"length_m": config.caps["queue_m"] / 4},
                              waiting={"max_wait_s": 999.0})
        self.assertAlmostEqual(result.pressure, 25.0, places=6)
        self.assertEqual(set(result.components), {"queue"})

    def test_contributions_reconstruct_the_pressure(self):
        calc = PressureCalculator()
        result = calc.compute("north", vehicle_count=8, density=30.0,
                              queue={"length_m": 25.0},
                              waiting={"max_wait_s": 18.0})
        total_weight = sum(c.weight for c in result.components.values())
        total = sum(c.contribution for c in result.components.values())
        self.assertAlmostEqual(result.pressure, 100.0 * total / total_weight,
                               places=6)

    def test_explain_mentions_every_component(self):
        calc = PressureCalculator()
        result = calc.compute("north", vehicle_count=8, density=30.0,
                              queue={"length_m": 25.0},
                              waiting={"max_wait_s": 18.0})
        text = result.explain()
        for name in ("queue", "waiting", "count", "density"):
            self.assertIn(name, text)

    def test_optical_flow_is_inverted_when_enabled(self):
        config = PressureConfig(weights={"queue": 0.0, "waiting": 0.0,
                                         "count": 0.0, "density": 0.0,
                                         "flow": 1.0})
        calc = PressureCalculator(config)
        still = calc.compute("north", optical_flow=0.0).pressure
        flowing = calc.compute("north", optical_flow=config.caps["flow"]).pressure
        self.assertAlmostEqual(still, 100.0, places=6)
        self.assertAlmostEqual(flowing, 0.0, places=6)

    def test_saturating_normalisation_rises_faster_early(self):
        linear = PressureCalculator(PressureConfig(normalisation="linear"))
        saturating = PressureCalculator(PressureConfig(normalisation="saturating"))
        args = {"vehicle_count": 3, "density": 10.0,
                "queue": 8.0, "waiting": {"max_wait_s": 6.0}}
        self.assertGreater(saturating.compute("north", **args).pressure,
                           linear.compute("north", **args).pressure)


class TestCameraConfidence(unittest.TestCase):

    def test_confidence_scales_the_result(self):
        calc = PressureCalculator()
        full = calc.compute("north", vehicle_count=10, density=30.0,
                            queue={"length_m": 30.0},
                            waiting={"max_wait_s": 20.0})
        halved = calc.compute("north", vehicle_count=10, density=30.0,
                              queue={"length_m": 30.0},
                              waiting={"max_wait_s": 20.0},
                              camera_confidence=0.5)
        self.assertAlmostEqual(halved.pressure, full.pressure * 0.5, places=6)
        self.assertAlmostEqual(halved.raw_pressure, full.pressure, places=6)

    def test_confidence_floor_prevents_a_blackout_zeroing_an_approach(self):
        calc = PressureCalculator()
        result = calc.compute("north", vehicle_count=10, density=30.0,
                              queue={"length_m": 30.0},
                              waiting={"max_wait_s": 20.0},
                              camera_confidence=0.0)
        self.assertEqual(result.camera_confidence,
                         calc.config.confidence_floor)
        self.assertGreater(result.pressure, 0.0)

    def test_confidence_can_be_switched_off(self):
        config = PressureConfig(apply_camera_confidence=False)
        calc = PressureCalculator(config)
        result = calc.compute("north", vehicle_count=10, density=30.0,
                              queue={"length_m": 30.0},
                              waiting={"max_wait_s": 20.0},
                              camera_confidence=0.2)
        self.assertEqual(result.pressure, result.raw_pressure)
        self.assertIsNone(result.camera_confidence)


class TestInputFlexibility(unittest.TestCase):

    def test_accepts_bare_numbers_dicts_and_result_objects(self):
        from metrics.density import DensityResult
        from metrics.queue_length import QueueResult
        from metrics.waiting_time import WaitingTimeResult

        calc = PressureCalculator()
        numbers = calc.compute("north", vehicle_count=10, density=40.0,
                               queue=20.0, waiting=15.0)
        dicts = calc.compute("north", vehicle_count=10,
                             density={"density_percent": 40.0},
                             queue={"length_m": 20.0},
                             waiting={"max_wait_s": 15.0})
        objects = calc.compute(
            "north", vehicle_count=10,
            density=DensityResult("north", 40.0, 0.0, 1.0, 10),
            queue=QueueResult("north", 200.0, 20.0, 3, 3, 10, 400.0),
            waiting=WaitingTimeResult("north", max_wait_s=15.0),
        )
        self.assertAlmostEqual(numbers.pressure, dicts.pressure, places=6)
        self.assertAlmostEqual(dicts.pressure, objects.pressure, places=6)

    def test_uncalibrated_camera_falls_back_to_pixel_queue_ratio(self):
        from metrics.queue_length import QueueResult
        calc = PressureCalculator()
        result = calc.compute(
            "north", vehicle_count=0,
            queue=QueueResult("north", length_px=200.0, length_m=None,
                              queued_count=4, stopped_count=4,
                              vehicles_in_roi=4, max_queue_px=400.0),
        )
        self.assertAlmostEqual(result.components["queue"].normalised, 0.5,
                               places=6)


class TestConfigPersistence(unittest.TestCase):

    def test_round_trip(self):
        config = PressureConfig(weights={"queue": 0.5, "waiting": 0.2,
                                         "count": 0.2, "density": 0.1,
                                         "flow": 0.0},
                                version="tuned-2026-09")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pressure.json"
            config.save(path)
            loaded = PressureConfig.load(path)
        self.assertEqual(loaded.weights, config.weights)
        self.assertEqual(loaded.version, "tuned-2026-09")

    def test_partial_config_keeps_defaults(self):
        config = PressureConfig.from_dict({"weights": {"queue": 0.9}})
        self.assertEqual(config.weights["queue"], 0.9)
        self.assertIn("waiting", config.weights)
        self.assertEqual(config.caps["waiting_s"], 60.0)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            PressureConfig(weights={"queue": -1.0})
        with self.assertRaises(ValueError):
            PressureConfig(normalisation="magic")


class TestRanking(unittest.TestCase):

    def test_highest_pressure_first(self):
        calc = PressureCalculator()
        results = {
            "north": calc.compute("north", vehicle_count=20, density=50.0,
                                  queue=60.0, waiting=40.0),
            "east": calc.compute("east", vehicle_count=2, density=5.0,
                                 queue=2.0, waiting=1.0),
            "south": calc.compute("south", vehicle_count=10, density=25.0,
                                  queue=30.0, waiting=20.0),
        }
        ranking = PressureCalculator.rank(results)
        self.assertEqual([name for name, _ in ranking],
                         ["north", "south", "east"])


if __name__ == "__main__":
    unittest.main()
