"""
TraffiSense - tools/make_fixture.py
------------------------------------
Turn a traffic video into a FIXTURE: a JSON file of tracked vehicles in the
exact format the metrics module consumes.

    python tools/make_fixture.py --video videos/traffic.mp4 --out fixtures/traffic.json

Why this exists
===============
The metrics module consumes tracked vehicles, not video.  Until the project's
own detection pipeline is ready, this script produces the same input using a
PRETRAINED YOLO model with ByteTrack, so the four metric modules can be
exercised on real footage today.

This is a TEST FIXTURE GENERATOR, not part of the traffic-analysis work:
nothing is trained, nothing is designed, and the file is throwaway.  When
Hitarth's pipeline is ready, point `run_fixture.py` at its output instead -
or have it dump the same JSON - and this script is no longer needed.

Requirements
------------
    pip install ultralytics

The first run downloads the model weights (~6 MB for yolov8n), so it needs an
internet connection once.  Use --model yolov8s.pt for better accuracy on small
or distant vehicles, at roughly 3x the runtime.

Speed
-----
Tracking is the slow part, not the metrics.  On a laptop CPU expect a few
frames per second, so start with a 30-60 second clip:

    python tools/make_fixture.py --video clip.mp4 --max-seconds 60

The point of a fixture is that this cost is paid ONCE.  After that,
`run_fixture.py` replays it in seconds, as many times as you like, while you
tune ROIs, thresholds and weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# COCO class ids that count as road traffic, with the names this project uses.
COCO_VEHICLES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def build_fixture(video: str,
                  model_name: str = "yolov8n.pt",
                  confidence: float = 0.30,
                  imgsz: int = 960,
                  max_seconds: float | None = None,
                  stride: int = 1,
                  device: str | None = None,
                  tracker: str = "bytetrack.yaml",
                  progress_every: int = 25) -> dict:
    """Run detection + tracking over `video` and collect the frames."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "This script needs ultralytics:\n"
            "    pip install ultralytics\n"
            "It is only used to create test input - the metrics package "
            "itself does not depend on it."
        )
    import cv2

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    limit = int(max_seconds * fps) if max_seconds else None
    print(f"video   : {video}  {width}x{height} @ {fps:.1f} fps"
          f"{f', {total} frames' if total else ''}")
    print(f"model   : {model_name}   tracker: {tracker}   conf: {confidence}")
    if limit:
        print(f"limit   : first {max_seconds:.0f} s ({limit} frames)")
    if stride > 1:
        print(f"stride  : every {stride} frames "
              f"(effective {fps / stride:.1f} fps)")

    model = YOLO(model_name)
    frames: list[dict] = []
    kept_classes = set(COCO_VEHICLES)

    frame_index = 0
    processed = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if limit is not None and frame_index >= limit:
            break
        if frame_index % stride:
            frame_index += 1
            continue

        timestamp = frame_index / fps

        # persist=True keeps track ids stable across calls - that is the whole
        # point of a fixture, since waiting time is measured per track id.
        results = model.track(image, persist=True, verbose=False,
                              conf=confidence, imgsz=imgsz, tracker=tracker,
                              device=device)

        detections = []
        result = results[0] if results else None
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            classes = boxes.cls.int().tolist()
            confidences = boxes.conf.tolist()
            coordinates = boxes.xyxy.tolist()
            for track_id, class_id, conf, (x1, y1, x2, y2) in zip(
                    ids, classes, confidences, coordinates):
                if class_id not in kept_classes:
                    continue
                detections.append({
                    "track_id": int(track_id),
                    "class": COCO_VEHICLES[class_id],
                    "bbox": [round(float(x1), 1), round(float(y1), 1),
                             round(float(x2), 1), round(float(y2), 1)],
                    "frame": frame_index,
                    "timestamp": round(timestamp, 4),
                    "confidence": round(float(conf), 3),
                })

        frames.append({
            "frame": frame_index,
            "timestamp": round(timestamp, 4),
            "detections": detections,
        })

        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"  {timestamp:6.1f} s   {len(detections):3d} vehicles "
                  f"(frame {frame_index})")

        frame_index += 1

    capture.release()

    counts = [len(f["detections"]) for f in frames]
    track_ids = {d["track_id"] for f in frames for d in f["detections"]}
    print(f"\nframes processed : {len(frames)}")
    print(f"unique track ids : {len(track_ids)}")
    if counts:
        print(f"vehicles / frame : min {min(counts)}  "
              f"mean {sum(counts) / len(counts):.1f}  max {max(counts)}")
    if not track_ids:
        print("\nWARNING: nothing was tracked.  Try a lower --conf, a larger "
              "--imgsz, or a bigger --model.")

    return {
        "video": str(video),
        "fps": fps / stride,
        "source_fps": fps,
        "frame_size": [width, height],
        "stride": stride,
        "source": f"{model_name} + {tracker} (pretrained, test fixture only)",
        "classes": sorted({d["class"] for f in frames
                           for d in f["detections"]}),
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tracked-vehicle fixture from a traffic video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="fixtures/traffic.json")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="yolov8n.pt (fast) ... yolov8x.pt (accurate)")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1,
                        help="process every Nth frame to go faster")
    parser.add_argument("--device", default=None,
                        help="'cpu', 'mps' on Apple silicon, or a CUDA index")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    args = parser.parse_args()

    fixture = build_fixture(
        video=args.video, model_name=args.model, confidence=args.conf,
        imgsz=args.imgsz, max_seconds=args.max_seconds, stride=args.stride,
        device=args.device, tracker=args.tracker,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture))
    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {out}  ({size_mb:.1f} MB)")
    print("\nNext:")
    print(f"  python tools/draw_roi.py --video {args.video} "
          f"--out config/intersection.json")
    print(f"  python tools/run_fixture.py --fixture {out} "
          f"--rois config/intersection.json --video {args.video} "
          f"--annotate out/annotated.mp4")


if __name__ == "__main__":
    main()
