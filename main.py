"""
TraffiSense - main.py
----------------------
Runs the complete TraffiSense pipeline end-to-end.

Modes:
  1. Real Video Pipeline (Hitarth's Vision + Tanish's Engine):
     python main.py --source datasets/AICity22_Track1_MTMC_Tracking/validation/S05/c018/vdo.avi

  2. Synthetic Simulator (Tanish's Engine standalone):
     python main.py --seconds 60

Integration Notes:
  - This script bridges Hitarth's `Vehicle` dataclasses from YOLO+ByteTrack 
    seamlessly into Tanish's `TrafficMetricsEngine`.
  - The `metrics/schema.py` natively parses the `Vehicle` dataclass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from metrics import PressureConfig, TrafficMetricsEngine, rectangular_intersection
from simulate import IntersectionSimulator


from controller.adapter import SignalAdapter

HEADER = (f"{'approach':<8}{'count':>7}{'density%':>10}{'queue m':>9}"
          f"{'queued':>8}{'wait s':>8}{'pressure':>10}  level")

def print_snapshot(timestamp: float, snapshot: dict, green: str) -> None:
    print(f"\n t = {timestamp:6.1f} s     green: {green}")
    print(" " + HEADER)
    print(" " + "-" * len(HEADER))
    for name, metrics in snapshot.items():
        row = metrics.to_dict()
        queue_m = row["queue_length_m"]
        queue_text = "n/a" if queue_m is None else f"{queue_m:.1f}"
        print(f" {name:<8}{row['vehicle_count']:>7}"
              f"{row['density_percent']:>10.1f}{queue_text:>9}"
              f"{row['queued_vehicles']:>8}{row['waiting_time_s']:>8.1f}"
              f"{row['pressure']:>10.1f}  {row['pressure_level']}")

def draw_dashboard(frame, snapshot: dict, fps: float, current_green: str):
    """Draws Hitarth's vision telemetry HUD on the OpenCV frame."""
    cv2.rectangle(frame, (10, 10), (360, 290), (30, 30, 30), -1)
    cv2.rectangle(frame, (10, 10), (360, 290), (0, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "TraffiSense AI Dashboard", (20, 35), font, 0.75, (0, 255, 255), 2)

    total_vehicles = sum(m.vehicle_count for m in snapshot.values())
    total_queued = sum(m.queue.queued_vehicles for m in snapshot.values())
    max_pressure_name = max(snapshot.items(), key=lambda x: x[1].pressure.pressure)[0]
    max_p_val = snapshot[max_pressure_name].pressure.pressure

    items = [
        ("Total Vehicles", total_vehicles),
        ("Total Queued", total_queued),
        ("Max Pressure", f"{max_p_val:.1f} ({max_pressure_name})"),
        ("Signal Status", current_green.upper()),
        ("FPS", f"{fps:.1f}")
    ]

    y = 70
    spacing = 30
    for label, value in items:
        color = (0, 255, 0) if "Signal Status" in label and "yellow" not in value.lower() else (255, 255, 255)
        if "yellow" in value.lower(): color = (0, 255, 255)
        if "red" in value.lower(): color = (0, 0, 255)
        cv2.putText(frame, f"{label} : {value}", (20, y), font, 0.60, color, 2)
        y += spacing

    cv2.putText(frame, "Approaches:", (20, y), font, 0.55, (0, 255, 255), 1)
    y += 20
    for name, m in snapshot.items():
        text = f"{name[0].upper()}: Cnt={m.vehicle_count} Q={m.queue.queued_vehicles} P={m.pressure.pressure:.0f}"
        cv2.putText(frame, text, (20, y), font, 0.50, (200, 200, 200), 1)
        y += 20

def run_real_video(source: str, engine: TrafficMetricsEngine, args: argparse.Namespace) -> list[dict]:
    from vision.video_loader import VideoLoader
    from vision.tracker import Tracker
    from vision.vehicle_factory import VehicleFactory
    from vision.density import DensityMap
    from vision.config import DISPLAY_WIDTH, DISPLAY_HEIGHT

    print(f"  source     : REAL VIDEO ({source})")
    loader = VideoLoader(source)
    info = loader.get_info()
    if info is None:
        print(f"Error: Could not load video from {source}")
        return []

    tracker = Tracker()
    density_estimator = DensityMap()
    adapter = SignalAdapter(min_green=5.0, max_green=30.0)

    log: list[dict] = []
    frame_number = 0
    previous_time = time.time()
    video_start_time = time.time()

    print("\nStarting video stream. Press 'q' to quit OpenCV window.")
    while True:
        frame = loader.get_frame()
        if frame is None:
            break
            
        frame_number += 1
        current_time = time.time()
        simulated_video_time = current_time - video_start_time
        
        annotated, results = tracker.get_annotated_frame(frame)
        vehicles = VehicleFactory.create(results, frame_number, simulated_video_time)
        snapshot = engine.update(vehicles, frame=frame_number, timestamp=simulated_video_time)
        
        # Step the signal adapter with our video timestamp
        current_green = adapter.step(simulated_video_time, snapshot)

        if len(vehicles) > 0:
            density_out = density_estimator.compute(vehicles, frame.shape)
            annotated = density_estimator.visualize(annotated, density_out["density_map"])
            
        fps = 1 / max((current_time - previous_time), 0.001)
        previous_time = current_time
        
        draw_dashboard(annotated, snapshot, fps, current_green)
        
        display = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("TraffiSense AI", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        report_interval = max(1, int(args.report_every * info["fps"]))
        if frame_number % report_interval == 0:
            print_snapshot(simulated_video_time, snapshot, green=current_green)
            
        log.append({
            "frame": frame_number,
            "timestamp": round(simulated_video_time, 2),
            "green": current_green,
            "approaches": {n: m.to_dict() for n, m in snapshot.items()},
        })

    cv2.destroyAllWindows()
    return log

def main() -> None:
    parser = argparse.ArgumentParser(description="TraffiSense end-to-end pipeline")
    parser.add_argument("--source", type=str, default=None,
                        help="path to video file (if provided, runs YOLO pipeline)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="seconds of simulated traffic (used if --source is None)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--report-every", type=float, default=5.0)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    intersection = rectangular_intersection()
    config = PressureConfig.load(args.config) if args.config else PressureConfig()
    engine = TrafficMetricsEngine(intersection, pressure_config=config)
    adapter = SignalAdapter(min_green=5.0, max_green=30.0)

    print("========== TraffiSense ==========")
    print("  approaches :", ", ".join(intersection.names))
    print("  pressure   :", config.version)

    if args.source:
        log = run_real_video(args.source, engine, args)
    else:
        print(f"  source     : SIMULATOR ({args.seconds:.0f} s @ {args.fps:.0f} fps)")
        sim = IntersectionSimulator(intersection, fps=args.fps)
        log = []
        report_interval = max(1, int(args.report_every * args.fps))
        for frame_index, timestamp, detections in sim.run(args.seconds):
            snapshot = engine.update(detections, frame=frame_index, timestamp=timestamp)
            current_green = adapter.step(timestamp, snapshot)
            
            if frame_index % report_interval == 0:
                print_snapshot(timestamp, snapshot, current_green)
            log.append({
                "frame": frame_index,
                "timestamp": round(timestamp, 2),
                "green": current_green,
                "approaches": {n: m.to_dict() for n, m in snapshot.items()},
            })

    ranking = engine.pressure_ranking()
    print("\nFinal pressure ranking (input for the future signal controller):")
    for position, (name, pressure) in enumerate(ranking, start=1):
        print(f"  {position}. {name:<6} {pressure:6.1f}")

    if args.json:
        Path(args.json).write_text(json.dumps(log, indent=2))
        print(f"\nWrote {len(log)} snapshots to {args.json}")


if __name__ == "__main__":
    main()
