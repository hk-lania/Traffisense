from abc import ABC, abstractmethod

class OutputInterface(ABC):
    """
    Abstract Base Class for output mechanisms.
    This abstraction allows the core controller to send output
    without knowing if it's going to a console, a file, or hardware pins.
    """
    
    @abstractmethod
    def display(self, message: str):
        """
        Abstract method to display or transmit a message.
        Must be implemented by subclasses.
        """
        pass
