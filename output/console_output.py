from .output_interface import OutputInterface

class ConsoleOutput(OutputInterface):
    """
    Concrete implementation of OutputInterface that prints to the terminal.
    Used for the software simulation phase of the project.
    """
    
    def display(self, message: str):
        """
        Prints the message to the console.
        """
        print(message)
