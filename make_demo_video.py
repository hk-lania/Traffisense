"""
TraffiSense - tools/make_demo_video.py
---------------------------------------
Render the traffic simulator to an .mp4 AND write the matching fixture, so the
whole chain can be demonstrated before any real footage or detection pipeline
exists.

    python tools/make_demo_video.py --seconds 90 --out-dir demo

    -> demo/demo.mp4            top-down synthetic junction
    -> demo/demo_fixture.json   its tracked vehicles, in the module's format
    -> demo/intersection.json   the ROIs for that junction

Then run the real thing over it:

    python tools/run_fixture.py --fixture demo/demo_fixture.json \
        --rois demo/intersection.json --video demo/demo.mp4 \
        --annotate demo/annotated.mp4

The boxes in the fixture are the simulator's own, so the demo shows the metric
modules working WITHOUT a detector in the way.  That separation is the point:
if something looks wrong here, it is a metrics bug; if it only looks wrong on
real footage, it is a detection or calibration issue.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from metrics import rectangular_intersection
from simulate import IntersectionSimulator

ROAD = (58, 58, 58)
GRASS = (38, 52, 38)
MARKING = (190, 190, 190)
STOP = (235, 235, 235)

VEHICLE_COLOURS = {
    "car": (200, 170, 90),
    "motorcycle": (110, 200, 235),
    "auto": (110, 235, 170),
    "bus": (90, 120, 230),
    "truck": (150, 120, 210),
}


def draw_scene(width: int, height: int, intersection) -> np.ndarray:
    """The static road background: asphalt, lane markings, stop lines."""
    scene = np.full((height, width, 3), GRASS, np.uint8)

    for roi in intersection:
        cv2.fillPoly(scene, [np.array(roi.polygon, np.int32)], ROAD)

    # The junction box itself, so the arms join up.
    xs = [p[0] for roi in intersection for p in roi.polygon]
    ys = [p[1] for roi in intersection for p in roi.polygon]
    cx, cy = width / 2.0, height / 2.0
    half = min(cx - min(xs), cy - min(ys), max(xs) - cx, max(ys) - cy)
    box_half = max(20.0, min(60.0, half))
    for roi in intersection:
        (sx1, sy1), (sx2, sy2) = roi.stop_line
        cv2.fillPoly(scene, [np.array(
            [(int(cx - box_half), int(cy - box_half)),
             (int(cx + box_half), int(cy - box_half)),
             (int(cx + box_half), int(cy + box_half)),
             (int(cx - box_half), int(cy + box_half))], np.int32)], ROAD)

    # Lane markings down the middle of each arm.
    for roi in intersection:
        u = np.array(roi.queue_direction, float)
        origin = np.array(roi.stop_line_midpoint, float)
        length = roi.max_queue_pixels or 0.0
        step = 26.0
        along = 12.0
        while along < length:
            a = origin + u * along
            b = origin + u * min(along + 13.0, length)
            cv2.line(scene, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     MARKING, 1, cv2.LINE_AA)
            along += step

        (sx1, sy1), (sx2, sy2) = roi.stop_line
        cv2.line(scene, (int(sx1), int(sy1)), (int(sx2), int(sy2)),
                 STOP, 2, cv2.LINE_AA)

    return scene


def draw_vehicles(frame, detections, signal_text: str, timestamp: float):
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        colour = VEHICLE_COLOURS.get(det["class"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), 1)

    # Banner along the bottom, so it cannot collide with the metrics panel
    # that run_fixture.py draws in the top-left corner.
    height = frame.shape[0]
    cv2.rectangle(frame, (0, height - 26), (frame.shape[1], height),
                  (18, 18, 18), -1)
    cv2.putText(frame, f"TraffiSense simulated junction   t = {timestamp:6.1f} s"
                       f"   green: {signal_text}",
                (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (225, 225, 225), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the simulator to video")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--out-dir", default="demo")
    parser.add_argument("--style", choices=("india", "lane"), default="india",
                        help="india: two-wheelers, no lane discipline, "
                             "filtering (default).  lane: lane-disciplined.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intersection = rectangular_intersection(args.width, args.height)
    sim = IntersectionSimulator(intersection, fps=args.fps, seed=args.seed,
                                style=args.style)
    scene = draw_scene(args.width, args.height, intersection)

    video_path = out_dir / "demo.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit("Could not open the video writer (missing codecs?)")

    frames = []
    for frame_index, timestamp, detections in sim.run(args.seconds):
        image = draw_vehicles(scene.copy(), detections,
                              sim.green_approach, timestamp)
        writer.write(image)
        frames.append({
            "frame": frame_index,
            "timestamp": round(timestamp, 4),
            "detections": detections,
        })
        if frame_index % int(args.fps * 10) == 0:
            print(f"  rendered {timestamp:5.1f} s "
                  f"({len(detections)} vehicles)")

    writer.release()

    fixture = {
        "video": str(video_path),
        "fps": args.fps,
        "source_fps": args.fps,
        "frame_size": [args.width, args.height],
        "stride": 1,
        "source": "TraffiSense simulator (synthetic ground truth, no detector)",
        "classes": sorted({d["class"] for f in frames for d in f["detections"]}),
        "frames": frames,
    }
    fixture_path = out_dir / "demo_fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    intersection.save(out_dir / "intersection.json")

    total = sum(len(f["detections"]) for f in frames)
    print(f"\nWrote {video_path}  ({len(frames)} frames)")
    print(f"Wrote {fixture_path}  ({total} detections, "
          f"{fixture_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {out_dir / 'intersection.json'}")
    print("\nNext:")
    print(f"  python tools/run_fixture.py --fixture {fixture_path} "
          f"--rois {out_dir / 'intersection.json'} --video {video_path} "
          f"--annotate {out_dir / 'annotated.mp4'}")


if __name__ == "__main__":
    main()
