import sys
from controller.signal_controller import SignalController

def display_menu():
    print("\n" + "="*26)
    print("  TRAFFISENSE CONTROLLER  ")
    print("="*26)
    print("1. Run Normal Cycle")
    print("2. Emergency Stop")
    print("3. Fixed-Time Mode")
    print("4. Exit")
    print("="*26)

def main():
    controller = SignalController()
    
    # Initial state display
    print("\n--- Initializing TraffiSense Controller ---")
    controller.display_signals()

    while True:
        display_menu()
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            print("\nStarting Normal Cycle (10s Green, 3s Yellow)...")
            controller.run_cycle(green_time=10, yellow_time=3)
        elif choice == '2':
            controller.emergency_stop()
        elif choice == '3':
            controller.fixed_time_mode()
        elif choice == '4':
            print("\nShutting down TraffiSense Controller...")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nForced shutdown. Exiting...")
        sys.exit(0)
