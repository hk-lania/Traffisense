# TraffiSense

Adaptive traffic signal controller (B.Tech IPD). A camera watches a four-way
junction; the system measures how much pressure is building on each approach
and gives green time to the approaches that need it.

```
video ─► YOLOv8 + ByteTrack ─► traffic metrics ─► SignalAdapter ─► signal controller ─► ESP32 / LEDs
         Riddhi's model,        Tanish             Hitarth           Harikrishna
         Hitarth's tracking
```

| Part | Owner | Where | Docs |
|---|---|---|---|
| Vehicle detection model (YOLOv8n, 83.2 % mAP) | Riddhi | `riddhi_detection/`, weights in `weights/` | [README](riddhi_detection/README.md) |
| Tracking, video pipeline, signal adapter, 4-camera mode | Hitarth | `vision/`, `controller/adapter.py`, `main.py`, `main_4cam.py` | |
| Density, queue length, waiting time, pressure; camera confidence, optical flow | Tanish | `metrics/`, `vision/camera_confidence.py`, `vision/optical_flow.py` | [README](metrics/README.md) |
| Signal controller, phases, emergency stop, ESP32 output | Harikrishna | `controller/`, `output/`, `controller_menu.py` | |

## Setup

```bash
pip install -r requirements.txt
python run_tests.py
```

## Run

```bash
# simulated traffic - no video or model needed
python main.py --seconds 60

# real video: draw the road area + stop line of each approach once per camera
python draw_roi.py --video clip.mp4 --out config/intersection.json
python main.py --source clip.mp4

# with Tanish's camera-confidence and optical-flow modules
python main.py --source clip.mp4 --camera-confidence --optical-flow

# with the ESP32 connected
python main.py --source clip.mp4 --port /dev/tty.usbserial-0001    # Mac
python main.py --source clip.mp4 --port COM3                        # Windows

# four cameras, one per approach
python main_4cam.py --north n.mp4 --east e.mp4 --south s.mp4 --west w.mp4

# Harikrishna's interactive controller menu
python controller_menu.py
```

Without `config/intersection.json`, `main.py` falls back to the simulator's road
layout and prints a warning — the metrics only mean something once the real
road areas are drawn for the camera.

## Hand-off formats

- **Tracking → metrics:** one record per tracked vehicle per frame with
  `track_id`, `class`, `bbox` `[x1, y1, x2, y2]`, `frame`, `timestamp`
  (`vision/vehicle.py` objects are accepted directly — see `metrics/schema.py`).
  `track_id` must stay stable across frames; waiting time depends on it.
- **Metrics → adapter:** `engine.update()` returns a snapshot per approach
  (`vehicle_count`, `density`, `queue`, `waiting`, `pressure`).
- **Adapter → ESP32:** one byte per change over serial at 115200 baud —
  `N/E/S/W` green, `n/e/s/w` yellow, `A` all red.

## Still open

- The pressure formula and weights in `config/pressure.json` are provisional —
  finalise them only after each metric is validated on the project's footage.
- `main_4cam.py` still uses the simulator's road layout; each camera needs its
  own drawn approach area before its metrics are meaningful.
- The detection model has no motorcycle / auto-rickshaw class, which matters
  on Indian footage.
