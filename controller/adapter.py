import time
from typing import Dict, List, Tuple
from controller.signal_controller import SignalController
from controller.signal_state import SignalState
from controller.direction import Direction

class SignalAdapter:
    """
    Non-blocking adapter that bridges Tanish's Metrics Engine and HK's Signal Controller.
    
    Tanish's engine provides high-frequency (per-frame) pressure updates.
    HK's controller originally used blocking time.sleep() calls.
    
    This adapter maintains a non-blocking state machine that can be stepped every frame,
    allocating green times dynamically based on the latest traffic pressure.
    """
    
    STR_TO_DIR = {
        "north": Direction.NORTH,
        "east": Direction.EAST,
        "south": Direction.SOUTH,
        "west": Direction.WEST
    }
    
    def __init__(self, min_green: float = 10.0, max_green: float = 60.0, yellow_time: float = 3.0, all_red_time: float = 2.0):
        self.controller = SignalController()
        
        # Timing configurations
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        
        # State machine
        self.state = "ALL_RED"
        self.current_approach = "north"
        self.next_transition_time = 0.0
        
        # Cycle order (Round Robin to prevent starvation)
        self.cycle_order = ["north", "east", "south", "west"]
        self.cycle_index = -1
        
        # Force initial state
        self._apply_hardware_state()
        
    def step(self, current_time: float, pressure_snapshot: dict) -> str:
        """
        Steps the signal state machine. Called every video frame.
        
        Args:
            current_time: The video timestamp or system time.
            pressure_snapshot: Dict mapping approach name to its metrics block.
            
        Returns:
            The name of the currently green approach, or 'all_red'/'yellow'.
        """
        if current_time < self.next_transition_time:
            return self._get_status_string()
            
        # Time to transition!
        if self.state == "GREEN":
            # Switch to yellow
            self.state = "YELLOW"
            self.next_transition_time = current_time + self.yellow_time
            self._apply_hardware_state()
            print(f"[Adapter] {self.current_approach.upper()} transitioning to YELLOW for {self.yellow_time}s")
            
        elif self.state == "YELLOW":
            # Switch to all-red clearance
            self.state = "ALL_RED"
            self.next_transition_time = current_time + self.all_red_time
            self._apply_hardware_state()
            print(f"[Adapter] Clearance ALL_RED for {self.all_red_time}s")
            
        elif self.state == "ALL_RED":
            # Pick next approach for green
            self._advance_cycle(pressure_snapshot)
            
            # Calculate dynamic green time based on pressure
            pressure_val = pressure_snapshot[self.current_approach].pressure.pressure
            green_duration = self.min_green + (pressure_val / 100.0) * (self.max_green - self.min_green)
            green_duration = round(green_duration, 1)
            
            self.state = "GREEN"
            self.next_transition_time = current_time + green_duration
            self._apply_hardware_state()
            print(f"[Adapter] {self.current_approach.upper()} gets GREEN for {green_duration}s (Pressure: {pressure_val:.1f})")
            
        return self._get_status_string()

    def _advance_cycle(self, snapshot: dict):
        """Advances the round-robin index, skipping approaches that are completely empty."""
        for _ in range(4):
            self.cycle_index = (self.cycle_index + 1) % 4
            candidate = self.cycle_order[self.cycle_index]
            
            # If there's at least one vehicle or some pressure, don't skip
            if snapshot[candidate].vehicle_count > 0 or snapshot[candidate].pressure.pressure > 5.0:
                self.current_approach = candidate
                return
                
        # If the entire intersection is empty, just default to the next one
        self.current_approach = self.cycle_order[self.cycle_index]
        
    def _apply_hardware_state(self):
        """Translates the adapter's state machine into HK's 4-way signal states."""
        states = {
            Direction.NORTH: SignalState.RED,
            Direction.EAST: SignalState.RED,
            Direction.SOUTH: SignalState.RED,
            Direction.WEST: SignalState.RED,
        }
        
        if self.state != "ALL_RED":
            active_dir = self.STR_TO_DIR[self.current_approach]
            if self.state == "GREEN":
                states[active_dir] = SignalState.GREEN
            elif self.state == "YELLOW":
                states[active_dir] = SignalState.YELLOW
                
        self.controller.phase_manager.set_phase(
            states[Direction.NORTH],
            states[Direction.SOUTH],
            states[Direction.EAST],
            states[Direction.WEST]
        )
        
        # Optional: Print HK's dashboard if you want full terminal UI
        # self.controller.display_signals()
        
    def _get_status_string(self) -> str:
        if self.state == "GREEN":
            return self.current_approach
        elif self.state == "YELLOW":
            return f"{self.current_approach} (yellow)"
        return "all_red"
