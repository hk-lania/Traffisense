from enum import Enum

class Direction(Enum):
    """
    Represents the four approaches of the intersection.
    """
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4

    def __str__(self):
        """Returns the string representation of the direction (e.g., 'NORTH')."""
        return self.name
