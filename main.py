import sys
from controller.signal_controller import SignalController
from controller.direction import Direction

def display_menu():
    print("\n" + "="*30)
    print("  TRAFFISENSE CONTROLLER  ")
    print("="*30)
    print("1. Run Normal Cycle")
    print("2. Run Adaptive Cycle (Input Density)")
    print("3. Emergency Stop")
    print("4. Fixed-Time Mode")
    print("5. Exit")
    print("="*30)

def main():
    controller = SignalController()
    
    # Initial state display
    print("\n--- Initializing TraffiSense Controller ---")
    controller.display_signals()

    while True:
        display_menu()
        choice = input("Select an option (1-5): ").strip()
        
        if choice == '1':
            print("\nStarting Normal Cycle (10s Green, 3s Yellow)...")
            controller.run_cycle(green_time=10, yellow_time=3)
        elif choice == '2':
            print("\n--- Simulate AI Input (Car Density) ---")
            try:
                n_cars = int(input("Enter number of cars in NORTH: "))
                e_cars = int(input("Enter number of cars in EAST: "))
                s_cars = int(input("Enter number of cars in SOUTH: "))
                w_cars = int(input("Enter number of cars in WEST: "))
                
                fake_ai_input = {
                    Direction.NORTH: n_cars,
                    Direction.EAST: e_cars,
                    Direction.SOUTH: s_cars,
                    Direction.WEST: w_cars
                }
                
                controller.run_adaptive_cycle(fake_ai_input)
            except ValueError:
                print("Invalid input. Please enter numbers only.")
        elif choice == '3':
            controller.emergency_stop()
        elif choice == '4':
            controller.fixed_time_mode()
        elif choice == '5':
            print("\nShutting down TraffiSense Controller...")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nForced shutdown. Exiting...")
        sys.exit(0)
