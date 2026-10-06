# TraffiSense — Traffic Signal Controller

This is the **Traffic Signal Controller** module for the TraffiSense B.Tech IPD Project. 

It provides a modular, software-based traffic signal simulation designed to be independent of hardware, allowing easy integration with physical signals (ESP32/RS-485) and an adaptive algorithm in the future.

## 🚀 Features (Implemented)
- **Four-Way Intersection Model:** Controls North, South, East, and West approaches.
- **Signal States:** Accurate RED, YELLOW, and GREEN state management.
- **Phase Management:** Runs a standard 4-phase traffic cycle.
- **Emergency Stop:** Immediately sets all signals to RED for safety.
- **Fixed-Time Fallback:** Uses predefined conservative timings (30s Green, 3s Yellow).
- **Console Simulator:** Professional terminal-based dashboard output.
- **Hardware-Ready Architecture:** Abstract `OutputInterface` allows swapping the terminal output for physical hardware output without rewriting the core logic.

## 🚧 Future Scope (Not Implemented Here)
- AI/ML Vehicle Detection (YOLO)
- Camera & Computer Vision
- Adaptive Timing Algorithm (This controller will receive timings from it)
- ESP32 Microcontroller Integration
- RS-485 Communication Layer
- Relay Drivers & Physical Traffic Lights

## 📂 Project Structure
```
TraffiSense/
├── main.py                  # Entry point / Interactive Menu
├── controller/              # Core Logic
│   ├── direction.py         # Enum for N/S/E/W
│   ├── phase_manager.py     # Manages all 4 signals
│   ├── signal_controller.py # Main brain (timing, cycles, emergency)
│   ├── signal_state.py      # Enum for RED/YELLOW/GREEN
│   └── traffic_signal.py    # Represents a single traffic pole
├── output/                  # Output Mechanisms
│   ├── console_output.py    # Terminal simulator
│   └── output_interface.py  # Abstract base class for future hardware
└── tests/                   # Verification
    └── test_controller.py   # Basic unit tests
```

## 🛠️ How to Run

1. Open a terminal.
2. Go to the project root (the `Traffisense` folder).
3. Run the main application:
   ```bash
   python3 main.py
   ```
4. Follow the interactive on-screen menu!

## 🧪 Testing
Run the simple test suite to verify core mechanics:
```bash
python3 tests/test_controller.py   # or: python3 run_tests.py (whole project)
```

To drive this controller from live traffic data, see `run_pipeline.py --controller live`
in the project root README.
