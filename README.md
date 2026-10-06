# TraffiSense

Adaptive traffic signal controller (B.Tech IPD). A camera watches a four-way
junction; the system measures how much pressure is building on each approach
and gives the next green to the approach that needs it most.

```
video ─► detection ─► tracking ─► traffic metrics ─► signal planner ─► signal controller ─► ESP32 / LEDs
         Riddhi       Hitarth     Tanish             (run_pipeline)    Harikrishna
```

| Folder | Owner | What it does | Docs |
|---|---|---|---|
| `riddhi_detection/` | Riddhi | YOLOv8 vehicle detection — `detect(frame)` | [README](riddhi_detection/README.md) |
| `tracking/` | Hitarth | ByteTrack ids across frames — **stand-in for now**, see below | `tracking/byte_tracker.py` |
| `metrics/` | Tanish | density, queue length, waiting time, provisional pressure per approach | [README](metrics/README.md) |
| `vision/` | Tanish | camera confidence (brightness + blur), optical flow | |
| `controller/`, `output/` | Harikrishna | signal phases, emergency stop, fixed-time fallback, console + ESP32 output | [README](controller/README.md) |
| `run_pipeline.py` | all | connects everything above | this file |

## Setup

```bash
pip install -r requirements.txt
python run_tests.py          # whole project, no pytest needed
```

## Run the whole system

```bash
# 1. once per camera: click the road area and stop line of each approach
python tools/draw_roi.py --video clip.mp4 --out config/intersection.json

# 2. run - prints metrics and each signal decision (no hardware needed)
python run_pipeline.py --video clip.mp4

#    ...watch it: boxes, track ids, pressure per approach, current green
python run_pipeline.py --video clip.mp4 --show

#    ...drive the real controller (and optionally the ESP32)
python run_pipeline.py --video clip.mp4 --controller live
python run_pipeline.py --video clip.mp4 --controller live --esp32 /dev/tty.usbserial-0001
```

Useful options: `--stride 2` (process every 2nd frame, faster on a laptop),
`--max-seconds 60`, `--json out.json` (metrics + decisions log),
`--stop-speed` (calibration, see metrics/README), `--camera-confidence`,
`--optical-flow`, `--controller off` (metrics only).

**How the next green is chosen.** Every approach gets one green per cycle, so
none can be starved. Within the cycle, the approach with the highest current
pressure goes next. The green time comes from the controller's own rule
(`calculate_green_time`: 10 s + 2 s per vehicle, 10–60 s). The pressure
formula is still provisional (`config/pressure.json`), so for now it only
decides the *order*, not the duration.

## Run each part on its own

```bash
python main.py                                    # Harikrishna: interactive controller menu
python metrics_demo.py                            # Tanish: metrics on simulated traffic
python riddhi_detection/demo.py --source clip.mp4 # Riddhi: detection on a video
python vision/check_camera_confidence.py clip.mp4
python vision/check_optical_flow.py clip.mp4
```

## Interfaces between the parts

These are the hand-off formats. If you change one, update the next part.

**Detection → tracking** — `riddhi_detection.inference.detect(frame)` returns
```python
[{"class_id": 2, "class_name": "Car", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}, ...]
```

**Tracking → metrics** — `tracker.update(detections, frame_index, timestamp)` returns
```python
[{"track_id": 17, "class": "car", "bbox": [x1, y1, x2, y2], "frame": 240, "timestamp": 9.6}, ...]
```
`track_id` must stay the same for the same vehicle across frames — waiting time depends on it.

**Metrics → controller** — `engine.update(...)` returns a snapshot per approach
(`vehicle_count`, `density`, `queue`, `waiting`, `pressure`); the planner turns
it into `SignalController.set_signal("EAST", green_seconds)`.

## Open items

- **Hitarth — tracking.** `tracking/byte_tracker.py` is a simplified, numpy-only
  ByteTrack so the pipeline runs today. Replace its internals with your
  ByteTrack and keep the same `update()` input/output; nothing else changes.
  `tests/test_tracker.py` checks ids stay stable.
- **Riddhi — trained weights.** `models/best.pt` is git-ignored (size), so the
  detector falls back to the pretrained `yolov8n.pt` until it is shared.
  Put it in `riddhi_detection/models/best.pt`.
- **Detector classes.** The trained model has no motorcycle / auto-rickshaw
  class; on Indian footage those are a large share of traffic. Worth checking
  before final testing.
- **Pressure formula.** Still provisional — finalise only after each metric is
  validated on the project's own footage (see metrics/README).

## Layout

```
Traffisense/
├── run_pipeline.py        whole system: video → detection → tracking → metrics → signals
├── main.py                controller menu (Harikrishna)
├── metrics_demo.py        metrics demo on simulated traffic (Tanish)
├── run_tests.py           runs every test in tests/
├── requirements.txt
├── config/                pressure.json, intersection_example.json (+ your intersection.json)
├── controller/            signal controller (Harikrishna)
├── output/                console + ESP32 output (Harikrishna)
├── riddhi_detection/      YOLOv8 detector, training + evaluation (Riddhi)
├── tracking/              ByteTrack stand-in (Hitarth's slot)
├── metrics/               traffic metrics module (Tanish)
├── vision/                camera confidence, optical flow (Tanish)
├── tools/                 draw_roi, make_fixture, run_fixture, make_demo_video
├── tests/                 controller, metrics, tracker, pipeline tests
└── archive/               early prototypes, kept for reference
```
