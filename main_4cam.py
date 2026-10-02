"""
TraffiSense - main_4cam.py
---------------------------
Runs the full TraffiSense pipeline on 4 separate camera feeds, one for each approach.

Usage:
  python main_4cam.py \
    --north datasets/cam_north.mp4 \
    --east datasets/cam_east.mp4 \
    --south datasets/cam_south.mp4 \
    --west datasets/cam_west.mp4
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from metrics import PressureConfig, TrafficMetricsEngine, rectangular_intersection
from controller.adapter import SignalAdapter

from vision.video_loader import VideoLoader
from vision.tracker import Tracker
from vision.vehicle_factory import VehicleFactory
from vision.density import DensityMap


def draw_dashboard(frame, snapshot: dict, fps: float, current_green: str):
    """Draws Hitarth's vision telemetry HUD on the combined 2x2 grid."""
    # Semi-transparent overlay for the HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (380, 310), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.rectangle(frame, (20, 20), (380, 310), (0, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "TraffiSense 4-Cam Dashboard", (30, 45), font, 0.65, (0, 255, 255), 2)

    total_vehicles = sum(m.vehicle_count for m in snapshot.values())
    total_queued = sum(m.queue.queued_count for m in snapshot.values())
    max_pressure_name = max(snapshot.items(), key=lambda x: x[1].pressure.pressure)[0]
    max_p_val = snapshot[max_pressure_name].pressure.pressure

    items = [
        ("Total Vehicles", total_vehicles),
        ("Total Queued", total_queued),
        ("Max Pressure", f"{max_p_val:.1f} ({max_pressure_name})"),
        ("Signal Status", current_green.upper()),
        ("Overall FPS", f"{fps:.1f}")
    ]

    y = 80
    spacing = 30
    for label, value in items:
        val_str = str(value).lower()
        color = (0, 255, 0) if "Signal Status" in label and "yellow" not in val_str else (255, 255, 255)
        if "yellow" in val_str: color = (0, 255, 255)
        if "red" in val_str: color = (0, 0, 255)
        cv2.putText(frame, f"{label} : {value}", (30, y), font, 0.60, color, 2)
        y += spacing

    cv2.putText(frame, "Approaches:", (30, y), font, 0.55, (0, 255, 255), 1)
    y += 20
    for name, m in snapshot.items():
        text = f"{name[0].upper()}: Cnt={m.vehicle_count} Q={m.queue.queued_count} P={m.pressure.pressure:.0f}"
        cv2.putText(frame, text, (30, y), font, 0.50, (200, 200, 200), 1)
        y += 20


def build_grid(frames: dict, labels: dict, width=640, height=360):
    """Resizes 4 frames, adds labels, and arranges them in a 2x2 grid."""
    grid_frames = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for d in ["north", "east", "south", "west"]:
        if d in frames and frames[d] is not None:
            resized = cv2.resize(frames[d], (width, height))
        else:
            resized = np.zeros((height, width, 3), dtype=np.uint8)
            
        # Draw camera label
        cv2.rectangle(resized, (0, 0), (120, 30), (0, 0, 0), -1)
        cv2.putText(resized, d.upper(), (10, 22), font, 0.7, (0, 255, 255), 2)
        
        # Add a green/red border based on signal status
        color = (0, 255, 0) if labels.get(d) == "green" else (0, 0, 255)
        if labels.get(d) == "yellow": color = (0, 255, 255)
        cv2.rectangle(resized, (0, 0), (width, height), color, 4)
            
        grid_frames[d] = resized
            
    top_row = cv2.hconcat([grid_frames["north"], grid_frames["east"]])
    bottom_row = cv2.hconcat([grid_frames["west"], grid_frames["south"]])
    return cv2.vconcat([top_row, bottom_row])


def main():
    parser = argparse.ArgumentParser(description="TraffiSense 4-Camera Pipeline")
    parser.add_argument("--north", type=str, help="Video for North approach")
    parser.add_argument("--east", type=str, help="Video for East approach")
    parser.add_argument("--south", type=str, help="Video for South approach")
    parser.add_argument("--west", type=str, help="Video for West approach")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Save output video to path (e.g. out.avi)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")
    args = parser.parse_args()

    sources = {
        "north": args.north,
        "east": args.east,
        "south": args.south,
        "west": args.west
    }
    
    # Initialize components
    loaders = {}
    density_estimators = {}
    for d, path in sources.items():
        if path and Path(path).exists():
            loaders[d] = VideoLoader(path)
            density_estimators[d] = DensityMap()
            print(f"[Init] Loaded {d.upper()} camera from {path}")
        else:
            print(f"[Init] Skipping {d.upper()} camera (no valid path provided)")
            
    if not loaders:
        print("Error: No valid video sources provided! Please pass at least one video (e.g., --north video.mp4)")
        return

    tracker = Tracker()
    intersection = rectangular_intersection()
    config = PressureConfig.load(args.config) if args.config else PressureConfig()
    
    # auto_lane=False because we are explicitly setting `vehicle.lane = direction` for each camera!
    engine = TrafficMetricsEngine(intersection, pressure_config=config, auto_lane=False)
    adapter = SignalAdapter(min_green=5.0, max_green=30.0)

    frame_number = 0
    previous_time = time.time()
    video_start_time = time.time()

    out_writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Grid is 2x2 of 640x360 frames = 1280x720
        out_writer = cv2.VideoWriter(args.save, fourcc, 25.0, (1280, 720))

    print("\nStarting 4-Camera Multi-Tracking stream. Press 'q' to quit.")
    
    while True:
        frame_number += 1
        current_time = time.time()
        simulated_time = current_time - video_start_time
        
        all_vehicles = []
        annotated_frames = {}
        
        # Process each camera sequentially
        for direction, loader in loaders.items():
            frame = loader.get_frame()
            if frame is None:
                continue # In a real system we'd loop it, here we just skip if a video ends
                
            # 1. Track
            annotated, results = tracker.get_annotated_frame(frame)
            vehicles = VehicleFactory.create(results, frame_number, simulated_time)
            
            # 2. Tag with explicit lane direction
            for v in vehicles:
                v.lane = direction
                
            all_vehicles.extend(vehicles)
            
            # 3. Apply Heatmap Visualizations
            if len(vehicles) > 0:
                density_out = density_estimators[direction].compute(vehicles, frame.shape)
                annotated = density_estimators[direction].visualize(annotated, density_out["density_map"])
                
            annotated_frames[direction] = annotated
            
        if not annotated_frames:
            print("All video streams ended.")
            break
            
        # Update Tanish's Engine with combined detections from ALL cameras
        snapshot = engine.update(all_vehicles, frame=frame_number, timestamp=simulated_time)
        
        # Step the Signal Adapter
        current_green = adapter.step(simulated_time, snapshot)
        
        # Figure out signal states for border colors
        signal_labels = {d: "red" for d in ["north", "east", "south", "west"]}
        if current_green != "all_red":
            if "(yellow)" in current_green:
                signal_labels[current_green.replace(" (yellow)", "")] = "yellow"
            else:
                signal_labels[current_green] = "green"
                
        # Build 2x2 Grid Display
        grid = build_grid(annotated_frames, signal_labels, width=640, height=360)
        
        # Draw Central Dashboard
        fps = 1 / max((current_time - previous_time), 0.001)
        previous_time = current_time
        draw_dashboard(grid, snapshot, fps, current_green)
        
        if out_writer is not None:
            out_writer.write(grid)
            
        try:
            cv2.imshow("TraffiSense 4-Cam AI Pipeline", grid)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            pass # Ignore if running headlessly without GUI support
            
        if args.max_frames > 0 and frame_number >= args.max_frames:
            print(f"Reached max frames ({args.max_frames}). Stopping.")
            break

    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
