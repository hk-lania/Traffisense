"""
TraffiSense - tools/draw_roi.py
--------------------------------
Draw the road ROI and stop line for each approach on a frame of your own
footage, and save them as the JSON that metrics/roi.py loads.

    python tools/draw_roi.py --video videos/traffic.mp4 --out config/intersection.json
    python tools/draw_roi.py --image frame.jpg --approaches north south

Controls
--------
    left click    add a point to the current shape
    ENTER / n     finish the current shape and move on
    u             undo the last point
    r             restart the current approach
    q / ESC       quit without saving the rest

For each approach you draw, in order:
    1. the ROAD POLYGON  - the drivable area of that approach
    2. the STOP LINE     - exactly two points, across the road

Calibration
-----------
After the shapes, you are asked for a pixel-per-metre scale.  Measure
something of known length in the frame - a lane is about 3.5 m wide, a broken
lane-marking dash about 3 m long - and divide its pixel length by its metres.
Leave it blank to skip; queue length is then reported in pixels only.

This tool is the only place OpenCV is needed.  The metrics package itself
runs on numpy alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
except ImportError:                                      # pragma: no cover
    print("This tool needs OpenCV:  pip install opencv-python")
    sys.exit(1)

import numpy as np

WINDOW = "TraffiSense - draw ROI"
COLOURS = {"polygon": (60, 220, 60), "stop": (60, 60, 240)}


def load_frame(video: str | None, image: str | None, frame_index: int):
    if image:
        frame = cv2.imread(image)
        if frame is None:
            raise SystemExit(f"Could not read image: {image}")
        return frame
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("Could not read a frame from the video")
    return frame


def collect_points(frame, title: str, kind: str, limit: int | None = None):
    """Let the user click points; returns them once ENTER is pressed."""
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if limit is None or len(points) < limit:
                points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        canvas = frame.copy()
        colour = COLOURS[kind]
        for point in points:
            cv2.circle(canvas, point, 4, colour, -1)
        if len(points) > 1:
            closed = kind == "polygon" and limit is None
            cv2.polylines(canvas, [np.array(points, np.int32)], closed, colour, 2)

        hint = f"{title}   [{len(points)} points]   ENTER=done  u=undo  r=restart  q=quit"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(canvas, hint, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("n")):
            if kind == "polygon" and len(points) < 3:
                print("A polygon needs at least 3 points.")
                continue
            if kind == "stop" and len(points) != 2:
                print("A stop line needs exactly 2 points.")
                continue
            return points
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):
            raise KeyboardInterrupt


def ask_float(prompt: str):
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print("  not a number - skipping")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw TraffiSense ROIs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--image")
    parser.add_argument("--frame", type=int, default=0,
                        help="which video frame to draw on")
    parser.add_argument("--approaches", nargs="+",
                        default=["north", "east", "south", "west"])
    parser.add_argument("--out", default="config/intersection.json")
    args = parser.parse_args()

    frame = load_frame(args.video, args.image, args.frame)
    print(f"Frame size: {frame.shape[1]} x {frame.shape[0]}")

    approaches = []
    try:
        for name in args.approaches:
            polygon = collect_points(frame, f"{name}: ROAD POLYGON", "polygon")
            stop_line = collect_points(frame, f"{name}: STOP LINE (2 points)",
                                       "stop", limit=2)
            ppm = ask_float(f"  {name}: pixels per metre (blank to skip): ")
            lanes = ask_float(f"  {name}: number of lanes [1]: ") or 1
            approaches.append({
                "name": name,
                "polygon": [list(p) for p in polygon],
                "stop_line": [list(stop_line[0]), list(stop_line[1])],
                "pixels_per_meter": ppm,
                "lane_count": int(lanes),
            })
            print(f"  {name}: {len(polygon)} polygon points saved")
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        cv2.destroyAllWindows()

    if not approaches:
        print("Nothing to save.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"approaches": approaches}, indent=2))
    print(f"\nWrote {len(approaches)} approaches to {out}")
    print("Load it with:  IntersectionROI.load('%s')" % out)


if __name__ == "__main__":
    main()
