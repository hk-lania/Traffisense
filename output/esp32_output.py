import sys
from .output_interface import OutputInterface
from controller.direction import Direction
from controller.signal_state import SignalState

class ESP32Output(OutputInterface):
    """
    Hardware implementation of the OutputInterface.
    This class is responsible for sending the actual electrical commands
    (via PySerial/USB) to the ESP32 to change the physical LED lights.
    """
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600):
        # We wrap this in a try-except because the hardware teammate 
        # might not have the ESP32 plugged in while testing.
        try:
            import serial
            self.connection = serial.Serial(port, baudrate, timeout=1)
            self.hardware_connected = True
            print(f"✅ ESP32 Connected on {port}")
        except ImportError:
            self.hardware_connected = False
            print("⚠️ 'pyserial' library not found. Run: pip install pyserial")
        except Exception as e:
            self.hardware_connected = False
            print(f"⚠️ Could not connect to ESP32 on {port}. Hardware disabled. Error: {e}")

    def display(self, message: str):
        """
        The abstract interface method. 
        For hardware, we might not need to print the text dashboard,
        but we can print it to the console anyway for debugging.
        """
        print(message)
        
    def transmit_hardware_state(self, phase_manager):
        """
        Extracts the RED/YELLOW/GREEN states from HK's PhaseManager 
        and converts them into a simple string that the ESP32 C++ code can parse.
        Example format sent over serial: "N:G,S:R,E:R,W:R\n"
        """
        if not self.hardware_connected:
            return

        signals = phase_manager.get_signals()
        
        # Convert states to a tiny 1-letter format to save bandwidth to the ESP32
        state_map = {
            SignalState.RED: "R",
            SignalState.YELLOW: "Y",
            SignalState.GREEN: "G"
        }
        
        # Build the payload: "N:G,S:R,E:R,W:R"
        payload = (
            f"N:{state_map[signals[Direction.NORTH].get_state()]},"
            f"S:{state_map[signals[Direction.SOUTH].get_state()]},"
            f"E:{state_map[signals[Direction.EAST].get_state()]},"
            f"W:{state_map[signals[Direction.WEST].get_state()]}\n"
        )
        
        # Send the string to the ESP32 over the USB serial cable
        try:
            self.connection.write(payload.encode('utf-8'))
        except Exception as e:
            print(f"Failed to send data to ESP32: {e}")
