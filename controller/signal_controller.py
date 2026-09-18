import time
from .signal_state import SignalState
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
