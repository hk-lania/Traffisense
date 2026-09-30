"""Unit tests for the TraffiSense signal controller."""

import sys
import os
import unittest

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.signal_controller import SignalController
from controller.signal_state import SignalState
from controller.direction import Direction


class TestController(unittest.TestCase):

    def test_initial_state(self):
        """Verify that all signals are initially RED."""
        controller = SignalController()
        signals = controller.phase_manager.get_signals()
        for direction, signal in signals.items():
            self.assertEqual(signal.get_state(), SignalState.RED,
                             f"{direction} should be RED initially")

    def test_emergency_stop(self):
        """Verify that emergency stop forces all signals to RED."""
        controller = SignalController()
        # Force a random state first
        controller.phase_manager.set_phase(
            SignalState.GREEN, SignalState.GREEN,
            SignalState.RED, SignalState.RED,
        )
        # Trigger emergency stop
        controller.emergency_stop()
        # Verify all are red
        signals = controller.phase_manager.get_signals()
        for direction, signal in signals.items():
            self.assertEqual(signal.get_state(), SignalState.RED,
                             f"{direction} should be RED after emergency stop")

    def test_reset_behavior(self):
        """Verify the PhaseManager can reset signals individually."""
        controller = SignalController()
        pm = controller.phase_manager
        pm.set_phase(
            SignalState.YELLOW, SignalState.YELLOW,
            SignalState.YELLOW, SignalState.YELLOW,
        )
        pm.reset_all()
        for direction, signal in pm.get_signals().items():
            self.assertEqual(signal.get_state(), SignalState.RED,
                             f"{direction} should be RED after reset")


if __name__ == "__main__":
    unittest.main()
