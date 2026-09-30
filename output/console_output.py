import sys

from .output_interface import OutputInterface


class ConsoleOutput(OutputInterface):
    """
    Concrete implementation of OutputInterface that prints to the terminal.
    Used for the software simulation phase of the project.
    """

    def display(self, message: str):
        """
        Prints the message to the console.
        Handles Unicode gracefully on Windows terminals (cp1252).
        """
        try:
            print(message)
        except UnicodeEncodeError:
            safe = message.encode(sys.stdout.encoding or "utf-8",
                                  errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace")
            print(safe)
