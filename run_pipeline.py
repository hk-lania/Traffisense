"""
TraffiSense - run_pipeline.py
------------------------------
The whole system, end to end:

    video frame
      -> riddhi_detection.detect()        (Riddhi   - YOLOv8 vehicle detection)
      -> tracking.ByteTracker.update()    (Hitarth  - ByteTrack; stand-in for now)
      -> metrics.TrafficMetricsEngine     (Tanish   - density, queue, waiting, pressure)
      -> SignalPlanner                    (glue     - picks the next green from pressure)
      -> controller.SignalController      (Harikrishna - signal phases, ESP32 output)

Usage (from the project root):

    # 1. once per camera: draw the road area + stop line of each approach
    python tools/draw_roi.py --video clip.mp4 --out config/intersection.json

    # 2. run it - prints metrics and the signal decisions, no hardware needed
    python run_pipeline.py --video clip.mp4

    # watch it: boxes, track ids, per-approach pressure, current green
    python run_pipeline.py --video clip.mp4 --show

    # drive Harikrishna's real controller (blocking phases run in a thread),
    # optionally out to the ESP32 board
    python run_pipeline.py --video clip.mp4 --controller live
    python run_pipeline.py --video clip.mp4 --controller live --esp32 /dev/tty.usbserial-0001

How the next green is chosen
    Every approach gets one green per cycle (no approach can be starved).
    Within a cycle, the approach with the highest current pressure goes next.
    Green time comes from the controller's own rule,
    SignalController.calculate_green_time(vehicle_count) = 10 s + 2 s/vehicle,
    clamped to 10-60 s. Pressure is still PROVISIONAL (config/pressure.json),
    so it only decides the ORDER here, not the duration.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from controller.signal_controller import SignalController
from metrics import (IntersectionROI, MotionTracker, PressureConfig,
                     TrafficMetricsEngine)
from tracking import ByteTracker

ROOT = Path(__file__).resolve().parent
DEFAULT_ROIS = ROOT / "config" / "intersection.json"
DEFAULT_PRESSURE = ROOT / "config" / "pressure.json"
YELLOW_S = 3  # matches SignalController
WARMUP_S = 2.0  # seconds of metrics before the first signal decision


# ---------------------------------------------------------------------------
# Metrics -> controller glue
# ---------------------------------------------------------------------------
class SignalPlanner:
    """Turns the latest metrics snapshot into the next (direction, green_time)."""

    def __init__(self, approaches: list[str]):
        self.approaches = list(approaches)
        self._served: set[str] = set()
        self._snapshot: dict = {}
        self._lock = threading.Lock()
        self._rules = SignalController()   # only used for calculate_green_time()

    def observe(self, snapshot: dict) -> None:
        with self._lock:
            self._snapshot = snapshot

    def next_phase(self) -> tuple[str, int, dict]:
        """Pick the highest-pressure approach not yet served this cycle."""
        with self._lock:
            snapshot = dict(self._snapshot)
        if len(self._served) >= len(self.approaches):
            self._served.clear()
        waiting = [a for a in self.approaches if a not in self._served]

        def pressure(name: str) -> float:
            m = snapshot.get(name)
            return m.pressure.pressure if m is not None else 0.0

        chosen = max(waiting, key=pressure)
        self._served.add(chosen)
        metrics = snapshot.get(chosen)
        count = metrics.vehicle_count if metrics is not None else 0
        green = self._rules.calculate_green_time(count)
        info = {"pressure": round(pressure(chosen), 1), "vehicles": count,
                "ranking": sorted(((a, round(pressure(a), 1)) for a in self.approaches),
                                  key=lambda x: -x[1])}
        return chosen, green, info


class LoggedSignal:
    """--controller log: follows the plan in VIDEO time and prints decisions.
    Nothing sleeps, so a recorded clip is processed as fast as possible."""

    def __init__(self, planner: SignalPlanner):
        self.planner = planner
        self.green: Optional[str] = None
        self.phase_ends_at = WARMUP_S   # let the metrics see some traffic first
        self.decisions: list[dict] = []

    def tick(self, timestamp: float) -> None:
        if timestamp < self.phase_ends_at:
            return
        direction, green, info = self.planner.next_phase()
        self.green = direction
        self.phase_ends_at = timestamp + green + YELLOW_S
        self.decisions.append({"t": round(timestamp, 1), "green": direction,
                               "green_s": green, **info})
        ranking = "  ".join(f"{a}={p}" for a, p in info["ranking"])
        print(f"[t={timestamp:7.1f}s] SIGNAL -> {direction.upper():5} green {green:2d}s "
              f"(pressure {info['pressure']}, {info['vehicles']} vehicles)   [{ranking}]")


class NoSignal:
    """--controller off: metrics only."""
    green = None
    decisions: list = []

    def tick(self, timestamp: float) -> None:
        pass


class LiveSignal:
    """--controller live: runs Harikrishna's SignalController.set_signal() in a
    background thread (it sleeps through each phase), fed by the planner."""

    def __init__(self, planner: SignalPlanner, esp32_port: Optional[str] = None):
        self.planner = planner
        self.controller = SignalController()
        if esp32_port:
            from output.esp32_output import ESP32Output
            self.controller.output = ESP32Output(port=esp32_port)
        self.green: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def tick(self, timestamp: float) -> None:
        pass  # the thread runs on wall-clock time

    def _loop(self) -> None:
        while not self._stop.is_set():
            direction, green, info = self.planner.next_phase()
            self.green = direction
            print(f"[controller] {direction.upper()} green {green}s "
                  f"(pressure {info['pressure']}, {info['vehicles']} vehicles)")
            self.controller.set_signal(direction.upper(), green)

    def stop(self) -> None:
        self._stop.set()
        self.controller.emergency_stop()  # all RED on shutdown


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(video: str,
        rois: str | Path = DEFAULT_ROIS,
        pressure_config: str | Path | None = DEFAULT_PRESSURE,
        model: Optional[str] = None,
        controller_mode: str = "log",
        esp32: Optional[str] = None,
        stop_speed: Optional[float] = None,
        use_camera_confidence: bool = False,
        use_optical_flow: bool = False,
        stride: int = 1,
        max_seconds: Optional[float] = None,
        report_every: float = 5.0,
        json_out: Optional[str] = None,
        show: bool = False,
        detector: Optional[Callable] = None) -> dict:
    import cv2

    rois = Path(rois)
    if not rois.exists():
        raise SystemExit(
            f"ROI file not found: {rois}\n"
            f"Draw one for this camera first:\n"
            f"  python tools/draw_roi.py --video {video} --out {rois}")
    intersection = IntersectionROI.load(rois)
    config = PressureConfig.load(pressure_config) if pressure_config else PressureConfig()
    motion = (MotionTracker(stop_speed_px_s=stop_speed, go_speed_px_s=2.0 * stop_speed)
              if stop_speed else None)
    engine = TrafficMetricsEngine(intersection, pressure_config=config, motion=motion)

    if detector is None:
        from riddhi_detection.inference import detect, load_model
        load_model(model)
        detector = detect

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = ByteTracker(fps=fps / max(1, stride))

    confidence_module = flow_module = None
    if use_camera_confidence:
        from vision.camera_confidence import CameraConfidenceEstimator
        confidence_module = CameraConfidenceEstimator()
    if use_optical_flow:
        from vision.optical_flow import OpticalFlowEstimator
        flow_module = OpticalFlowEstimator()

    planner = SignalPlanner(intersection.names)
    if controller_mode == "live":
        signal = LiveSignal(planner, esp32)
        # let the metrics see a couple of seconds of traffic before the first decision
        starter = threading.Timer(WARMUP_S, signal.start)
        starter.daemon = True
        starter.start()
    elif controller_mode == "off":
        signal = NoSignal()
    else:
        signal = LoggedSignal(planner)

    print(f"TraffiSense pipeline | video={video} fps={fps:.1f} stride={stride} "
          f"| approaches={', '.join(intersection.names)} | pressure {config.version} "
          f"| controller={controller_mode}")

    log: list[dict] = []
    frame_index = 0
    next_report = 0.0
    started = time.time()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if max_seconds is not None and timestamp > max_seconds:
                break
            if frame_index % stride:
                frame_index += 1
                continue

            detections = detector(frame)
            tracked = tracker.update(detections, frame_index, timestamp)

            confidence = confidence_module.compute(frame)["confidence"] if confidence_module else None
            flow = flow_module.compute(frame)["average_motion"] if flow_module else None

            snapshot = engine.update(tracked, frame=frame_index, timestamp=timestamp,
                                     camera_confidence=confidence, optical_flow=flow)
            planner.observe(snapshot)
            signal.tick(timestamp)

            if timestamp >= next_report:
                next_report += report_every
                rows = {n: m.to_dict() for n, m in snapshot.items()}
                log.append({"frame": frame_index, "timestamp": round(timestamp, 2),
                            "green": signal.green, "approaches": rows})
                print(f"           metrics  " + "  ".join(
                    f"{n}: {r['vehicle_count']} veh, q {r['queued_vehicles']}, "
                    f"wait {r['waiting_time_s']:.0f}s, P {r['pressure']:.0f}"
                    for n, r in rows.items()))

            if show:
                _draw(cv2, frame, intersection, tracked, snapshot, signal.green)
                cv2.imshow("TraffiSense", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()
        if isinstance(signal, LiveSignal):
            starter.cancel()
            signal.stop()

    elapsed = time.time() - started
    print(f"\nProcessed {frame_index} frames in {elapsed:.1f}s "
          f"({frame_index / max(elapsed, 1e-9):.1f} fps)")
    print("Final pressure ranking:", engine.pressure_ranking())

    result = {"snapshots": log,
              "decisions": getattr(signal, "decisions", []),
              "ranking": engine.pressure_ranking()}
    if json_out:
        Path(json_out).write_text(json.dumps(result, indent=2))
        print(f"Wrote {json_out}")
    return result


def _draw(cv2, frame, intersection, tracked, snapshot, green) -> None:
    import numpy as np
    for roi in intersection:
        colour = (0, 200, 0) if roi.name == green else (0, 0, 220)
        pts = np.array(roi.polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, colour, 2)
        (x1, y1), (x2, y2) = roi.stop_line
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
    for v in tracked:
        x1, y1, x2, y2 = (int(c) for c in v["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)
        cv2.putText(frame, f"{v['track_id']} {v['class']}", (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    cv2.rectangle(frame, (0, 0), (430, 14 + 22 * len(snapshot)), (0, 0, 0), -1)
    y = 22
    for name, m in snapshot.items():
        text = (f"{name:<5} veh {m.vehicle_count:2d}  wait {m.waiting.max_wait_s:4.0f}s  "
                f"P {m.pressure.pressure:5.1f}" + ("  GREEN" if name == green else ""))
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22


def main() -> None:
    p = argparse.ArgumentParser(description="TraffiSense: video -> detection -> tracking -> metrics -> signal")
    p.add_argument("--video", required=True, help="traffic video file (or camera index, e.g. 0)")
    p.add_argument("--rois", default=str(DEFAULT_ROIS), help="ROI file from tools/draw_roi.py")
    p.add_argument("--pressure-config", default=str(DEFAULT_PRESSURE))
    p.add_argument("--model", default=None,
                   help="YOLO weights (default: riddhi_detection/models/best.pt, else yolov8n.pt)")
    p.add_argument("--controller", choices=["log", "live", "off"], default="log",
                   help="log: print decisions (default); live: drive SignalController; off: metrics only")
    p.add_argument("--esp32", default=None, help="serial port of the ESP32 (with --controller live)")
    p.add_argument("--stop-speed", type=float, default=None,
                   help="px/s below which a vehicle counts as stopped (~0.5 x pixels-per-metre)")
    p.add_argument("--camera-confidence", action="store_true",
                   help="scale pressure by vision/camera_confidence.py")
    p.add_argument("--optical-flow", action="store_true",
                   help="record vision/optical_flow.py (weight 0 in the provisional config; slow on CPU)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame (faster)")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--report-every", type=float, default=5.0)
    p.add_argument("--json", default=None, help="write metrics + decisions here")
    p.add_argument("--show", action="store_true", help="show the annotated video")
    a = p.parse_args()

    video = int(a.video) if a.video.isdigit() else a.video
    run(video, rois=a.rois, pressure_config=a.pressure_config, model=a.model,
        controller_mode=a.controller, esp32=a.esp32, stop_speed=a.stop_speed,
        use_camera_confidence=a.camera_confidence, use_optical_flow=a.optical_flow,
        stride=a.stride, max_seconds=a.max_seconds, report_every=a.report_every,
        json_out=a.json, show=a.show)


if __name__ == "__main__":
    main()
