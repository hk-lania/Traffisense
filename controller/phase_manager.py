from typing import Dict
from .direction import Direction
from .signal_state import SignalState
from .traffic_signal import TrafficSignal

class PhaseManager:
    """
    Manages all four traffic signals at the intersection.
    Handles updating their states simultaneously.
    """
    def __init__(self):
        # Initialize signals for all four directions
        self.signals: Dict[Direction, TrafficSignal] = {
            Direction.NORTH: TrafficSignal(Direction.NORTH),
            Direction.SOUTH: TrafficSignal(Direction.SOUTH),
            Direction.EAST: TrafficSignal(Direction.EAST),
            Direction.WEST: TrafficSignal(Direction.WEST)
        }

    def set_phase(self, north: SignalState, south: SignalState, east: SignalState, west: SignalState):
        """
        Sets the state for all four signals simultaneously.
        """
        self.signals[Direction.NORTH].set_state(north)
        self.signals[Direction.SOUTH].set_state(south)
        self.signals[Direction.EAST].set_state(east)
        self.signals[Direction.WEST].set_state(west)

    def get_signals(self) -> Dict[Direction, TrafficSignal]:
        """
        Returns the collection of traffic signals.
        """
        return self.signals

    def reset_all(self):
        """
        Forces all signals back to RED (used in emergency stops).
        """
        for signal in self.signals.values():
            signal.reset()
