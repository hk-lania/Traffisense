"""
TraffiSense - main.py
----------------------
Runs the complete TraffiSense pipeline end-to-end.

Modes:
  1. Real Video Pipeline (Hitarth's Vision + Tanish's Engine):
     python draw_roi.py --video clip.mp4 --out config/intersection.json   # once per camera
     python main.py --source clip.mp4                                      # uses config/intersection.json
     python main.py --source clip.mp4 --camera-confidence --optical-flow  # Tanish's vision modules

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

from metrics import (IntersectionROI, PressureConfig, TrafficMetricsEngine,
                     rectangular_intersection)
from simulate import IntersectionSimulator


from controller.adapter import SignalAdapter

DEFAULT_ROIS = Path(__file__).resolve().parent / "config" / "intersection.json"

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
    total_queued = sum(m.queue.queued_count for m in snapshot.values())
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
        val_str = str(value).lower()
        color = (0, 255, 0) if "Signal Status" in label and "yellow" not in val_str else (255, 255, 255)
        if "yellow" in val_str: color = (0, 255, 255)
        if "red" in val_str: color = (0, 0, 255)
        cv2.putText(frame, f"{label} : {value}", (20, y), font, 0.60, color, 2)
        y += spacing

    cv2.putText(frame, "Approaches:", (20, y), font, 0.55, (0, 255, 255), 1)
    y += 20
    for name, m in snapshot.items():
        text = f"{name[0].upper()}: Cnt={m.vehicle_count} Q={m.queue.queued_count} P={m.pressure.pressure:.0f}"
        cv2.putText(frame, text, (20, y), font, 0.50, (200, 200, 200), 1)
        y += 20

def load_intersection(rois_path: str | None, frame_size: tuple[int, int]):
    """Road areas + stop lines for THIS camera, drawn with draw_roi.py.

    Without them the metrics fall back to the simulator's made-up junction,
    which does not match real footage - density, queue and waiting time are
    then meaningless, so say so loudly."""
    path = Path(rois_path) if rois_path else DEFAULT_ROIS
    if path.exists():
        print(f"  rois       : {path}")
        return IntersectionROI.load(path)
    print("\n  !! No ROI file found at", path)
    print("  !! Using the SIMULATOR's road layout - metrics will NOT match this video.")
    print("  !! Draw the real one once per camera:")
    print(f"  !!   python draw_roi.py --video <your video> --out {path}\n")
    width, height = frame_size
    return rectangular_intersection(frame_width=width, frame_height=height)


def run_real_video(source: str, engine_factory, adapter: SignalAdapter,
                   args: argparse.Namespace) -> tuple[list[dict], TrafficMetricsEngine]:
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
        raise SystemExit(1)

    engine = engine_factory(load_intersection(args.rois, info["resolution"]))
    video_fps = info["fps"] or 25.0

    tracker = Tracker()
    density_estimator = DensityMap()

    confidence_module = flow_module = None
    if args.camera_confidence:
        from vision.camera_confidence import CameraConfidenceEstimator
        confidence_module = CameraConfidenceEstimator()
    if args.optical_flow:
        from vision.optical_flow import OpticalFlowEstimator
        flow_module = OpticalFlowEstimator()

    log: list[dict] = []
    frame_number = 0
    previous_time = time.time()

    print("\nStarting video stream. Press 'q' to quit OpenCV window.")
    while True:
        frame = loader.get_frame()
        if frame is None:
            break
            
        frame_number += 1
        current_time = time.time()
        # Video time, not wall-clock time: waiting time and speeds must follow
        # the footage even when YOLO runs slower (or faster) than real time.
        simulated_video_time = (frame_number - 1) / video_fps

        annotated, results = tracker.get_annotated_frame(frame)
        vehicles = VehicleFactory.create(results, frame_number, simulated_video_time)

        confidence = confidence_module.compute(frame)["confidence"] if confidence_module else None
        flow = flow_module.compute(frame)["average_motion"] if flow_module else None
        snapshot = engine.update(vehicles, frame=frame_number, timestamp=simulated_video_time,
                                 camera_confidence=confidence, optical_flow=flow)
        
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
    return log, engine

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
    parser.add_argument("--port", type=str, default=None, help="COM port for ESP32 hardware (e.g., COM3)")
    parser.add_argument("--rois", type=str, default=None,
                        help="ROI file for this camera (default: config/intersection.json)")
    parser.add_argument("--camera-confidence", action="store_true",
                        help="scale pressure by vision/camera_confidence.py")
    parser.add_argument("--optical-flow", action="store_true",
                        help="record vision/optical_flow.py (weight 0 for now; slow on CPU)")
    args = parser.parse_args()

    config = PressureConfig.load(args.config) if args.config else PressureConfig()
    # One adapter only: opening the ESP32 serial port twice fails.
    adapter = SignalAdapter(min_green=5.0, max_green=30.0, com_port=args.port)

    print("========== TraffiSense ==========")
    print("  pressure   :", config.version)

    if args.source:
        log, engine = run_real_video(
            args.source,
            lambda intersection: TrafficMetricsEngine(intersection, pressure_config=config),
            adapter, args)
    else:
        intersection = rectangular_intersection()
        engine = TrafficMetricsEngine(intersection, pressure_config=config)
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
