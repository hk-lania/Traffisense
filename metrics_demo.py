"""
TraffiSense - metrics_demo.py
------------------------------
Runs the Traffic Metrics Module end to end against the synthetic tracker
stream and prints the per-approach metric block.

    python metrics_demo.py              # 60 s of simulated traffic
    python metrics_demo.py --seconds 120       # longer run
    python metrics_demo.py --json out.json     # also write a per-second metrics log
    python metrics_demo.py --explain           # show the pressure breakdown

Swapping in the real pipeline means replacing ONE loop:

    for frame_index, timestamp, tracked_vehicles in source:
        snapshot = engine.update(tracked_vehicles, frame=frame_index,
                                 timestamp=timestamp)

where `tracked_vehicles` is whatever YOLO + ByteTrack hands over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import PressureConfig, TrafficMetricsEngine, rectangular_intersection
from metrics.simulate import IntersectionSimulator

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


def main() -> None:
    parser = argparse.ArgumentParser(description="TraffiSense traffic metrics demo")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--report-every", type=float, default=5.0,
                        help="seconds between printed snapshots")
    parser.add_argument("--config", type=str, default=None,
                        help="pressure config JSON (see config/pressure.json)")
    parser.add_argument("--json", type=str, default=None,
                        help="write a per-report metrics log here")
    parser.add_argument("--explain", action="store_true",
                        help="print the pressure breakdown for each approach")
    parser.add_argument("--camera-confidence", type=float, default=None,
                        help="0-1 confidence from your camera_confidence module")
    args = parser.parse_args()

    intersection = rectangular_intersection()
    config = PressureConfig.load(args.config) if args.config else PressureConfig()
    engine = TrafficMetricsEngine(intersection, pressure_config=config)
    sim = IntersectionSimulator(intersection, fps=args.fps)

    print("TraffiSense - Traffic Metrics Module")
    print(f"  approaches : {', '.join(intersection.names)}")
    print(f"  pressure   : {config.version}  weights={config.weights}")
    print(f"  source     : simulated tracker output "
          f"({args.seconds:.0f} s @ {args.fps:.0f} fps)")

    log: list[dict] = []
    report_interval = max(1, int(args.report_every * args.fps))

    for frame_index, timestamp, detections in sim.run(args.seconds):
        snapshot = engine.update(
            detections, frame=frame_index, timestamp=timestamp,
            camera_confidence=args.camera_confidence,
        )

        if frame_index % report_interval == 0:
            print_snapshot(timestamp, snapshot, sim.green_approach)
            if args.explain:
                print()
                for metrics in snapshot.values():
                    print(metrics.pressure.explain())
            log.append({
                "frame": frame_index,
                "timestamp": round(timestamp, 2),
                "green": sim.green_approach,
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
