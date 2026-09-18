from .signal_state import SignalState
from .direction import Direction

class TrafficSignal:
    """
    Represents a single traffic signal for a specific direction.
    Maintains its current state (RED, YELLOW, or GREEN).
    """
    def __init__(self, direction: Direction):
        self.direction = direction
        self.current_state = SignalState.RED

    def set_state(self, state: SignalState):
        """Sets the traffic signal to a new state."""
        if not isinstance(state, SignalState):
            raise ValueError("State must be a valid SignalState Enum.")
        self.current_state = state

    def get_state(self) -> SignalState:
        """Returns the current state of the signal."""
        return self.current_state

    def reset(self):
        """Resets the signal to its default safe state (RED)."""
        self.current_state = SignalState.RED

    def __str__(self) -> str:
        """Pretty-prints the signal state."""
        state_icon = {
            SignalState.RED: "🔴 RED",
            SignalState.YELLOW: "🟡 YELLOW",
            SignalState.GREEN: "🟢 GREEN"
        }
        
        # Adjust spacing for alignment in dashboard (NORTH/SOUTH/EAST/WEST)
        dir_name = self.direction.name
        padding = " " * (6 - len(dir_name))
        
        return f"{dir_name}{padding} → {state_icon.get(self.current_state)}"
