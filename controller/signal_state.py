from enum import Enum

class SignalState(Enum):
    """
    Represents the three possible states of a traffic signal.
    Using an Enum prevents errors from typos like "Red" vs "red".
    """
    RED = 1
    YELLOW = 2
    GREEN = 3

    def __str__(self):
        """Returns the string representation of the state (e.g., 'RED')."""
        return self.name
