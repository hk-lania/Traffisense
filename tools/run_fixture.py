"""
TraffiSense - tools/run_fixture.py
-----------------------------------
Replay a tracked-vehicle fixture through the metrics engine.

    python tools/run_fixture.py --fixture fixtures/traffic.json \
        --rois config/intersection.json

    python tools/run_fixture.py --fixture fixtures/traffic.json \
        --rois config/intersection.json --video videos/traffic.mp4 \
        --annotate out/annotated.mp4

What it gives you
=================
* the per-approach metric block printed as the clip plays,
* a JSON log of every reported snapshot (--json),
* a summary of the whole clip - peak queue, longest wait, mean pressure,
* and, with --annotate, a video showing the ROIs, the stop lines, which
  vehicles were judged queued, where the back of each queue was placed, and
  the live numbers.

That last one is the real test.  Numbers in a terminal cannot tell you the
stop line is in the wrong place or the ROI covers the pavement; watching the
queue marker sit at the back of the actual queue can.

Tuning loop
===========
Building the fixture is slow (it runs YOLO); replaying it is fast.  So tune
against the same fixture until the overlay looks right:

    ROI or stop line wrong ....... redraw with tools/draw_roi.py
    queue jitters or never forms . --stop-speed
    queue merges separate groups .. --max-gap
    waiting time stuck at zero ... --stop-speed too low, or ids are churning
    pressure ordering looks wrong  edit config/pressure.json

Nothing here needs the fixture to be rebuilt.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from metrics import (IntersectionROI, MotionTracker, PressureConfig,
                     TrafficMetricsEngine)

# BGR colours
COLOUR_ROI = (90, 90, 90)
COLOUR_STOP = (60, 60, 235)
COLOUR_QUEUED = (40, 165, 245)
COLOUR_MOVING = (90, 210, 90)
COLOUR_OTHER = (150, 150, 150)
COLOUR_QUEUE_END = (40, 165, 245)
COLOUR_PANEL = (24, 24, 24)
COLOUR_TEXT = (240, 240, 240)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(path: str) -> dict:
    data = json.loads(Path(path).read_text())
    if "frames" not in data:
        raise SystemExit(f"{path} does not look like a fixture "
                         "(no 'frames' key)")
    total = sum(len(f["detections"]) for f in data["frames"])
    print(f"fixture : {path}")
    print(f"          {len(data['frames'])} frames, {total} detections, "
          f"{data.get('fps', 25.0):.1f} fps")
    print(f"          source: {data.get('source', 'unknown')}")
    return data


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def road_half_width(roi) -> float:
    laterals = [roi.lateral_offset(p) for p in roi.polygon]
    return max(abs(min(laterals)), abs(max(laterals)))


def draw_overlay(image, intersection, snapshot, detections, fixture_fps):
    import cv2

    overlay = image.copy()
    for roi in intersection:
        points = np.array(roi.polygon, np.int32)
        cv2.fillPoly(overlay, [points], (55, 55, 55))
    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

    for roi in intersection:
        points = np.array(roi.polygon, np.int32)
        cv2.polylines(image, [points], True, COLOUR_ROI, 1, cv2.LINE_AA)
        (sx1, sy1), (sx2, sy2) = roi.stop_line
        cv2.line(image, (int(sx1), int(sy1)), (int(sx2), int(sy2)),
                 COLOUR_STOP, 2, cv2.LINE_AA)

    queued_ids = set()
    for metrics in snapshot.values():
        queued_ids.update(metrics.queue.track_ids)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        track_id = det["track_id"]
        colour = COLOUR_QUEUED if track_id in queued_ids else COLOUR_MOVING
        if det.get("lane") is None and track_id not in queued_ids:
            colour = colour
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(image, str(track_id), (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    # Back-of-queue marker: a bar across the road at the measured extent.
    for name, metrics in snapshot.items():
        if metrics.queue.length_px <= 0:
            continue
        roi = intersection[name]
        u = np.array(roi.queue_direction, float)
        perp = np.array([-u[1], u[0]])
        origin = np.array(roi.stop_line_midpoint, float)
        half = road_half_width(roi)
        end = origin + u * metrics.queue.length_px
        a = end - perp * half
        b = end + perp * half
        cv2.line(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 COLOUR_QUEUE_END, 2, cv2.LINE_AA)
        label = (f"{metrics.queue.length_m:.0f} m"
                 if metrics.queue.length_m is not None
                 else f"{metrics.queue.length_px:.0f} px")
        cv2.putText(image, label, (int(end[0]) + 6, int(end[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_QUEUE_END, 1,
                    cv2.LINE_AA)

    draw_panel(image, snapshot)
    return image


def draw_panel(image, snapshot):
    import cv2

    rows = len(snapshot) + 2
    height = 20 * rows + 12
    width = 430
    cv2.rectangle(image, (10, 10), (10 + width, 10 + height), COLOUR_PANEL, -1)
    cv2.rectangle(image, (10, 10), (10 + width, 10 + height), (70, 70, 70), 1)

    header = f"{'approach':<9}{'veh':>5}{'dens%':>8}{'queue':>9}{'wait s':>9}{'press':>8}"
    cv2.putText(image, header, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (170, 170, 170), 1, cv2.LINE_AA)

    y = 54
    for name, metrics in snapshot.items():
        queue = (f"{metrics.queue.length_m:.0f}m"
                 if metrics.queue.length_m is not None
                 else f"{metrics.queue.length_px:.0f}px")
        row = (f"{name:<9}{metrics.vehicle_count:>5}"
               f"{metrics.density.density_percent:>8.1f}{queue:>9}"
               f"{metrics.waiting.max_wait_s:>9.1f}"
               f"{metrics.pressure.pressure:>8.1f}")
        cv2.putText(image, row, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    COLOUR_TEXT, 1, cv2.LINE_AA)
        y += 20

    ranking = sorted(snapshot.items(), key=lambda kv: -kv[1].pressure.pressure)
    top = ranking[0][0] if ranking and ranking[0][1].pressure.pressure > 0 else "-"
    cv2.putText(image, f"highest pressure: {top}", (20, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_QUEUED, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------

def run(args) -> None:
    fixture = load_fixture(args.fixture)
    intersection = IntersectionROI.load(args.rois)
    config = PressureConfig.load(args.config) if args.config else PressureConfig()

    motion_options = {}
    if args.stop_speed is not None:
        motion_options["stop_speed_px_s"] = args.stop_speed
        motion_options["go_speed_px_s"] = args.stop_speed * 2.0
    motion = MotionTracker(**motion_options) if motion_options else None

    queue_options = {}
    if args.max_gap is not None:
        queue_options["max_gap_m"] = args.max_gap

    engine = TrafficMetricsEngine(intersection, pressure_config=config,
                                  motion=motion, queue_options=queue_options)

    fps = float(fixture.get("fps", 25.0))
    report_every = max(1, int(args.report_every * fps))

    writer = None
    capture = None
    if args.annotate:
        import cv2
        if not args.video:
            raise SystemExit("--annotate needs --video (the original footage)")
        capture = cv2.VideoCapture(args.video)
        if not capture.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        size = tuple(fixture.get("frame_size") or
                     (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        out_path = Path(args.annotate)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        print(f"annotate: {out_path}  {size[0]}x{size[1]} @ {fps:.1f} fps")

    print(f"rois    : {args.rois}  ({', '.join(intersection.names)})")
    print(f"pressure: {config.version}  {config.weights}\n")

    log: list[dict] = []
    history = {name: [] for name in intersection.names}
    peak_queue = {name: 0.0 for name in intersection.names}
    peak_wait = {name: 0.0 for name in intersection.names}
    peak_density = {name: 0.0 for name in intersection.names}

    frames_by_index = {f["frame"]: f for f in fixture["frames"]}
    video_frame = 0

    for entry in fixture["frames"]:
        snapshot = engine.update(entry["detections"],
                                 frame=entry["frame"],
                                 timestamp=entry["timestamp"])

        for name, metrics in snapshot.items():
            history[name].append(metrics.pressure.pressure)
            peak_queue[name] = max(peak_queue[name],
                                   metrics.queue.length_m
                                   if metrics.queue.length_m is not None
                                   else metrics.queue.length_px)
            peak_wait[name] = max(peak_wait[name], metrics.waiting.max_wait_s)
            peak_density[name] = max(peak_density[name],
                                     metrics.density.density_percent)

        if entry["frame"] % report_every == 0:
            print_snapshot(entry["timestamp"], snapshot)
            log.append({
                "frame": entry["frame"],
                "timestamp": round(entry["timestamp"], 2),
                "approaches": {n: m.to_dict() for n, m in snapshot.items()},
            })

        if writer is not None:
            import cv2
            # Walk the video forward to this fixture frame (fixtures may be
            # strided, so frame indices are not necessarily consecutive).
            while video_frame <= entry["frame"]:
                ok, image = capture.read()
                if not ok:
                    image = None
                    break
                video_frame += 1
            if image is not None:
                writer.write(draw_overlay(image, intersection, snapshot,
                                          entry["detections"], fps))

    if writer is not None:
        writer.release()
    if capture is not None:
        capture.release()

    print_summary(intersection.names, history, peak_queue, peak_wait,
                  peak_density, fixture)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(log, indent=2))
        print(f"\nWrote {len(log)} snapshots to {args.json}")
    if args.annotate:
        print(f"Wrote annotated video to {args.annotate}")


def print_snapshot(timestamp: float, snapshot: dict) -> None:
    header = (f"{'approach':<8}{'count':>7}{'density%':>10}{'queue':>10}"
              f"{'queued':>8}{'wait s':>8}{'pressure':>10}")
    print(f"\n t = {timestamp:7.1f} s")
    print(" " + header)
    print(" " + "-" * len(header))
    for name, metrics in snapshot.items():
        queue = (f"{metrics.queue.length_m:.1f}m"
                 if metrics.queue.length_m is not None
                 else f"{metrics.queue.length_px:.0f}px")
        print(f" {name:<8}{metrics.vehicle_count:>7}"
              f"{metrics.density.density_percent:>10.1f}{queue:>10}"
              f"{metrics.queue.queued_count:>8}"
              f"{metrics.waiting.max_wait_s:>8.1f}"
              f"{metrics.pressure.pressure:>10.1f}")


def print_summary(names, history, peak_queue, peak_wait, peak_density,
                  fixture) -> None:
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    unit = "m" if fixture.get("calibrated", True) else "px"
    header = (f"{'approach':<10}{'mean press':>12}{'max press':>11}"
              f"{'peak queue':>12}{'peak wait':>11}{'peak dens':>11}")
    print(header)
    print("-" * len(header))
    for name in names:
        values = history[name] or [0.0]
        print(f"{name:<10}{statistics.mean(values):>12.1f}{max(values):>11.1f}"
              f"{peak_queue[name]:>11.1f}{unit:<1}{peak_wait[name]:>10.1f}s"
              f"{peak_density[name]:>10.1f}%")

    means = {n: statistics.mean(history[n] or [0.0]) for n in names}
    busiest = max(means, key=means.get)
    print(f"\nBusiest approach over the clip: {busiest} "
          f"(mean pressure {means[busiest]:.1f})")
    print("Sanity checks: does the busiest approach match what you see in the "
          "video?\nDoes the queue marker sit at the back of the real queue?")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a tracked-vehicle fixture through the metrics engine")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--rois", required=True)
    parser.add_argument("--config", default=None,
                        help="pressure config JSON")
    parser.add_argument("--video", default=None,
                        help="original footage, needed for --annotate")
    parser.add_argument("--annotate", default=None,
                        help="write an annotated video here")
    parser.add_argument("--json", default=None,
                        help="write the metrics log here")
    parser.add_argument("--report-every", type=float, default=5.0)
    parser.add_argument("--stop-speed", type=float, default=None,
                        help="stopped-speed threshold in px/s "
                             "(about 0.5 x pixels-per-metre)")
    parser.add_argument("--max-gap", type=float, default=None,
                        help="largest gap inside one queue, in metres")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
