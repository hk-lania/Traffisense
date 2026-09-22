"""
TraffiSense - integration_example.py
-------------------------------------
Template for wiring the Traffic Metrics Module to the REAL pipeline.

Only one function needs to change: `tracked_vehicles_for_frame()`.  Make it
return the detection/tracking output for one frame, in the documented shape,
and everything downstream works unchanged.

    python integration_example.py --video videos/traffic.mp4 \
        --rois config/intersection.json --json metrics_log.json

The optional hooks at the bottom show where the two modules you already have -
optical flow and camera confidence - plug in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import PressureConfig, IntersectionROI, TrafficMetricsEngine


# ---------------------------------------------------------------------------
# 1. REPLACE THIS with the real detection + tracking call
# ---------------------------------------------------------------------------
def tracked_vehicles_for_frame(frame, frame_index: int, timestamp: float) -> list[dict]:
    """Return one record per tracked vehicle visible in `frame`.

    Required fields (see metrics/schema.py for the accepted spellings):

        track_id   int     stable id from the tracker - NOT the detection index
        class      str     'car' | 'motorcycle' | 'bus' | ...
        bbox       [x1, y1, x2, y2]  pixels, top-left / bottom-right
        frame      int     frame index
        timestamp  float   seconds since the start of the video
        lane       str     'north' | 'east' | 'south' | 'west'  (optional -
                           the engine infers it from the ROIs when missing)

    Typical shape of the real call:

        detections = detector.detect(frame)          # YOLO
        tracks = tracker.update(detections, frame)   # ByteTrack
        return [
            {"track_id": t.track_id,
             "class": t.class_name,
             "bbox": list(t.tlbr),
             "frame": frame_index,
             "timestamp": timestamp}
            for t in tracks
        ]
    """
    raise NotImplementedError(
        "Connect the detection/tracking pipeline here - see the docstring."
    )


# ---------------------------------------------------------------------------
# 2. The loop below does not need to change
# ---------------------------------------------------------------------------
def run(video_path: str,
        rois_path: str,
        config_path: str | None = None,
        json_out: str | None = None,
        show: bool = False) -> None:
    import cv2                                    # only needed for video I/O

    intersection = IntersectionROI.load(rois_path)
    config = PressureConfig.load(config_path) if config_path else PressureConfig()
    engine = TrafficMetricsEngine(intersection, pressure_config=config)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    log: list[dict] = []
    frame_index = 0
    previous_gray = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps

        vehicles = tracked_vehicles_for_frame(frame, frame_index, timestamp)

        # --- optional hooks --------------------------------------------
        confidence = camera_confidence(frame)
        flow = optical_flow_magnitude(frame, previous_gray)
        previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        snapshot = engine.update(
            vehicles, frame=frame_index, timestamp=timestamp,
            camera_confidence=confidence, optical_flow=flow,
        )

        if frame_index % int(fps) == 0:           # log once per second
            log.append({
                "frame": frame_index,
                "timestamp": round(timestamp, 2),
                "approaches": {n: m.to_dict() for n, m in snapshot.items()},
            })
            print(json.dumps(log[-1]["approaches"], indent=None))

        frame_index += 1

    capture.release()

    if json_out:
        Path(json_out).write_text(json.dumps(log, indent=2))
        print(f"Wrote {len(log)} snapshots to {json_out}")


# ---------------------------------------------------------------------------
# 3. Hooks for the two modules that already exist
# ---------------------------------------------------------------------------
def camera_confidence(frame) -> float | None:
    """Plug in the existing camera-confidence module (brightness + Laplacian).

        from camera_confidence import score_frame
        return score_frame(frame)          # expected range 0.0 - 1.0

    Returning None simply leaves pressure unscaled.
    """
    return None


def optical_flow_magnitude(frame, previous_gray) -> float | None:
    """Plug in the existing Farneback optical-flow module.

        from optical_flow import mean_magnitude
        return mean_magnitude(previous_gray, current_gray)

    Optical flow carries weight 0.0 in the provisional pressure config, so it
    is recorded but does not affect the output until that weight is raised.
    """
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraffiSense metrics on real video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--rois", default="config/intersection.json")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    run(args.video, args.rois, args.config, args.json)
