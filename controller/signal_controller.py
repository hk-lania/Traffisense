import time
from .signal_state import SignalState
from .direction import Direction
from .phase_manager import PhaseManager
from output.console_output import ConsoleOutput

class SignalController:
    """
    The main controller for the TraffiSense system.
    Manages timing, cycles, and emergency modes without depending directly on hardware.
    """
    def __init__(self):
        self.phase_manager = PhaseManager()
        self.output = ConsoleOutput()
        self.running = False

    def display_signals(self):
        """
        Displays all four signals in a formatted dashboard via the output interface.
        """
        dashboard = []
        dashboard.append("==========================================")
        dashboard.append("      TRAFFISENSE TRAFFIC CONTROLLER      ")
        dashboard.append("==========================================")
        dashboard.append("Direction        Signal")
        dashboard.append("-----------------------")
        
        for direction, signal in self.phase_manager.get_signals().items():
            dashboard.append(str(signal))
            
        dashboard.append("==========================================\n")
        
        self.output.display("\n".join(dashboard))

    def run_cycle(self, green_time: int = 10, yellow_time: int = 3):
        """
        Executes a single standard 4-phase traffic cycle.
        """
        self.running = True
        
        # Helper to set phase, display, and wait
        def run_phase(n: SignalState, s: SignalState, e: SignalState, w: SignalState, duration: int, phase_name: str):
            if not self.running: return # Check if stopped (e.g. by emergency)
            self.output.display(f"--- Starting {phase_name} ({duration}s) ---")
            self.phase_manager.set_phase(n, s, e, w)
            self.display_signals()
            time.sleep(duration)

        try:
            # Phase 1: North Only
            run_phase(SignalState.GREEN, SignalState.RED, SignalState.RED, SignalState.RED, green_time, "Phase 1: NORTH Go")
            run_phase(SignalState.YELLOW, SignalState.RED, SignalState.RED, SignalState.RED, yellow_time, "NORTH Stopping")
            
            # Phase 2: East Only
            run_phase(SignalState.RED, SignalState.RED, SignalState.GREEN, SignalState.RED, green_time, "Phase 2: EAST Go")
            run_phase(SignalState.RED, SignalState.RED, SignalState.YELLOW, SignalState.RED, yellow_time, "EAST Stopping")
            
            # Phase 3: South Only
            run_phase(SignalState.RED, SignalState.GREEN, SignalState.RED, SignalState.RED, green_time, "Phase 3: SOUTH Go")
            run_phase(SignalState.RED, SignalState.YELLOW, SignalState.RED, SignalState.RED, yellow_time, "SOUTH Stopping")
            
            # Phase 4: West Only
            run_phase(SignalState.RED, SignalState.RED, SignalState.RED, SignalState.GREEN, green_time, "Phase 4: WEST Go")
            run_phase(SignalState.RED, SignalState.RED, SignalState.RED, SignalState.YELLOW, yellow_time, "WEST Stopping")
            
            self.output.display("--- Cycle Complete ---\n")
        except KeyboardInterrupt:
            self.output.display("\nCycle interrupted by user.")
        finally:
            self.running = False

    def calculate_green_time(self, car_count: int) -> int:
        """
        Calculates the appropriate green time based on traffic density (car count).
        Minimum green time is 10 seconds, maximum is 60 seconds.
        Each car adds 2 seconds to the base time.
        """
        base_time = 10
        calculated_time = base_time + (car_count * 2)
        return min(max(calculated_time, 10), 60) # Clamp between 10 and 60 seconds

    def run_adaptive_cycle(self, densities: dict):
        """
        Runs a cycle where timings are dynamically adjusted based on the input densities.
        `densities` should be a dict mapping Direction to an integer car count.
        """
        self.output.display("\n🧠 ADAPTIVE ALGORITHM TRIGGERED")
        self.output.display(f"Input Data Received: North={densities.get(Direction.NORTH, 0)} cars, "
                            f"East={densities.get(Direction.EAST, 0)} cars, "
                            f"South={densities.get(Direction.SOUTH, 0)} cars, "
                            f"West={densities.get(Direction.WEST, 0)} cars")

        # Calculate timings
        t_north = self.calculate_green_time(densities.get(Direction.NORTH, 0))
        t_east = self.calculate_green_time(densities.get(Direction.EAST, 0))
        t_south = self.calculate_green_time(densities.get(Direction.SOUTH, 0))
        t_west = self.calculate_green_time(densities.get(Direction.WEST, 0))
        
        self.output.display(f"Calculated Green Times -> N:{t_north}s, E:{t_east}s, S:{t_south}s, W:{t_west}s\n")
        
        self.running = True
        yellow_time = 3

        def run_phase(n: SignalState, s: SignalState, e: SignalState, w: SignalState, duration: int, phase_name: str):
            if not self.running: return
            self.output.display(f"--- Starting {phase_name} ({duration}s) ---")
            self.phase_manager.set_phase(n, s, e, w)
            self.display_signals()
            time.sleep(duration)

        try:
            # Phase 1: North Only
            run_phase(SignalState.GREEN, SignalState.RED, SignalState.RED, SignalState.RED, t_north, "Phase 1: NORTH Go")
            run_phase(SignalState.YELLOW, SignalState.RED, SignalState.RED, SignalState.RED, yellow_time, "NORTH Stopping")
            
            # Phase 2: East Only
            run_phase(SignalState.RED, SignalState.RED, SignalState.GREEN, SignalState.RED, t_east, "Phase 2: EAST Go")
            run_phase(SignalState.RED, SignalState.RED, SignalState.YELLOW, SignalState.RED, yellow_time, "EAST Stopping")
            
            # Phase 3: South Only
            run_phase(SignalState.RED, SignalState.GREEN, SignalState.RED, SignalState.RED, t_south, "Phase 3: SOUTH Go")
            run_phase(SignalState.RED, SignalState.YELLOW, SignalState.RED, SignalState.RED, yellow_time, "SOUTH Stopping")
            
            # Phase 4: West Only
            run_phase(SignalState.RED, SignalState.RED, SignalState.RED, SignalState.GREEN, t_west, "Phase 4: WEST Go")
            run_phase(SignalState.RED, SignalState.RED, SignalState.RED, SignalState.YELLOW, yellow_time, "WEST Stopping")
            
            self.output.display("--- Adaptive Cycle Complete ---\n")
        except KeyboardInterrupt:
            self.output.display("\nCycle interrupted by user.")
        finally:
            self.running = False

    def emergency_stop(self):
        """
        Immediately sets all signals to RED and displays an emergency alert.
        """
        self.running = False # Stop any running cycle if this was threaded (it's synchronous here, but good practice)
        self.phase_manager.reset_all()
        
        alert = "\n🚨 EMERGENCY STOP ACTIVATED 🚨\nAll signals set to RED for safety."
        self.output.display(alert)
        self.display_signals()

    def fixed_time_mode(self):
        """
        Fallback mode using predefined conservative timings.
        """
        self.output.display("\n⚙️ FIXED-TIME MODE ACTIVATED")
        self.output.display("Using fallback timings: 30s Green, 3s Yellow")
        self.output.display("(Software fallback if adaptive timing is unavailable)")
        
        self.run_cycle(green_time=30, yellow_time=3)
