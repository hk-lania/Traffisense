# TraffiSense — Traffic Metrics Module

Per-approach traffic metrics computed from tracked vehicle output.

```
Camera → YOLO detection → ByteTrack tracking → [ THIS MODULE ] → Traffic Pressure Index → Signal control
                          (Hitarth)              (Tanish)          (later, together)      (later)
```

For each of the four approaches, every frame:

```
North:  vehicle_count   density   queue_length   waiting_time   pressure
East:   vehicle_count   density   queue_length   waiting_time   pressure
South:  vehicle_count   density   queue_length   waiting_time   pressure
West:   vehicle_count   density   queue_length   waiting_time   pressure
```

**Scope.** This module does not detect, track, or train anything. No YOLO, no
ByteTrack, no model weights. It starts where the tracker's output begins, and
stops before any signal-timing decision is made.

---

## Quick start

```bash
cd traffisense_metrics
python main.py                    # 60 s of simulated traffic, metrics printed live
python main.py --explain          # also show how each pressure value was built
python run_tests.py               # 88 tests, no pytest needed
```

Requires Python 3.9+ and **numpy**. OpenCV is needed only by `tools/draw_roi.py`
and the video loop in `integration_example.py` — the metrics themselves run on
numpy alone, so nothing here can break with a `cv2` import error.

Sample output:

```
 approach  count  density%  queue m  queued  wait s  pressure  level
 -------------------------------------------------------------------
 north        14      18.5     32.0      12    21.4      39.5  moderate
 east          0       0.0      0.0       0     0.0       0.0  low
 south         6      10.2     20.6       5    39.0      34.8  moderate
 west          2       1.5      4.2       1    26.6      17.1  low

north: pressure 39.5 (moderate), config provisional-0.1
   count    raw=   14.00  norm=0.56  w=0.25  -> 0.140
   queue    raw=   32.01  norm=0.40  w=0.30  -> 0.120
   waiting  raw=   21.36  norm=0.36  w=0.30  -> 0.107
   density  raw=   18.50  norm=0.18  w=0.15  -> 0.028
```

---

## Testing it on video

The module consumes tracked vehicles, not video — so a clip on its own proves
nothing until something produces detections from it. Two paths, and the first
needs no footage and no detector at all.

**A. The synthetic junction — works right now, nothing to install**

```bash
python tools/make_demo_video.py --seconds 90 --out-dir demo
python tools/run_fixture.py --fixture demo/demo_fixture.json \
    --rois demo/intersection.json --video demo/demo.mp4 \
    --annotate demo/annotated.mp4
```

The first command renders a junction to `demo.mp4` and writes the matching
tracked vehicles; the second runs all four metrics over them and produces an
annotated video showing the ROIs, the stop lines, which vehicles were judged
queued, where the back of each queue was placed, and the live numbers.

The simulated traffic is **Indian by default** (`--style india`): about half
two-wheelers, plus auto-rickshaws, buses and trucks, with **no lane
discipline** — vehicles take any lateral position that fits and two-wheelers
edge sideways into gaps. `--style lane` gives lane-disciplined traffic for
comparison. This matters for correctness, not just appearance: the queue chain
measures distance from the stop line and never assumes lanes, and the tests
prove it holds up under both.

Because the boxes come from the simulator rather than a detector, this path
isolates the metrics. If something looks wrong here it is a metrics bug; if it
only looks wrong on real footage it is a detection or calibration issue.

**B. Real footage**

```bash
pip install ultralytics                                      # once
python tools/make_fixture.py --video clip.mp4 --max-seconds 60 \
    --out fixtures/clip.json                                 # slow, run once
python tools/draw_roi.py --video clip.mp4 --out config/intersection.json
python tools/run_fixture.py --fixture fixtures/clip.json \
    --rois config/intersection.json --video clip.mp4 --annotate out/clip.mp4
```

`make_fixture.py` runs a **pretrained** YOLO with ByteTrack purely to generate
test input. Nothing is trained or designed there — it is scaffolding that
stands in for Hitarth's pipeline, and it is deleted once his output is
available. Building the fixture is the slow part; replaying it is seconds, so
tune ROIs, thresholds and weights against the same fixture:

| symptom | fix |
|---|---|
| ROI or stop line in the wrong place | redraw with `draw_roi.py` |
| queue never forms, or flickers | `--stop-speed` (≈ 0.5 × pixels-per-metre) |
| two separate groups merged into one queue | `--max-gap` |
| waiting time stuck at zero | stop speed too low, or track ids are churning |
| pressure ordering looks wrong | edit `config/pressure.json` |

Watch the annotated video, not just the terminal. Numbers cannot tell you the
stop line is in the wrong place; a queue marker sitting in the middle of a
moving stream can.

---

## The input contract

This is the one thing the detection side needs to agree to. One record per
tracked vehicle per frame:

```python
{
    "track_id":  17,                      # stable id from the tracker, not a detection index
    "class":     "car",                   # car / motorcycle / auto / bus / truck
    "bbox":      [x1, y1, x2, y2],        # pixels, top-left and bottom-right
    "frame":     240,                     # frame index
    "timestamp": 9.6,                     # seconds since video start
    "lane":      "north"                  # optional — inferred from the ROIs if absent
}
```

Plain dictionaries, tracker objects, and several key spellings all work
(`id`/`tid`, `label`/`cls`, `xyxy`/`x1,y1,x2,y2`, `side`/`direction`, `N`/`north`).
That flexibility is deliberate — it is what fixes the two failures hit during
the earlier integration attempt:

```
ModuleNotFoundError: No module named 'tracker'
AttributeError: 'dict' object has no attribute 'track_id'
```

`metrics/schema.py` is the only file that knows the tracker's format. If
Hitarth's output ever changes shape, that is the single file to touch.

**`track_id` is the one field that cannot be faked.** Waiting time is an
accumulation over frames, so a vehicle must keep the same id between them. If
ids churn, waiting time collapses toward zero — and that is a tracker issue,
not a metrics issue.

---

## Using it

```python
from metrics import TrafficMetricsEngine, IntersectionROI

intersection = IntersectionROI.load("config/intersection.json")
engine = TrafficMetricsEngine(intersection)

for frame_index, timestamp, tracked_vehicles in stream:
    snapshot = engine.update(tracked_vehicles, frame=frame_index, timestamp=timestamp)

    north = snapshot["north"]
    north.vehicle_count              # 14
    north.density.density_percent    # 18.5
    north.queue.length_m             # 32.0   (length_px when uncalibrated)
    north.waiting.max_wait_s         # 21.4
    north.pressure.pressure          # 39.5

    engine.to_dict()                 # JSON-ready block for all four approaches
    engine.pressure_ranking()        # [('north', 39.5), ('south', 34.8), ...]
```

`integration_example.py` is a ready-made version of that loop for real video —
one function to fill in.

---

## The four modules

### 1. `density.py` — how much road is covered

```
density (%) = occupied road area / total road ROI area × 100
```

Two things a naive version gets wrong, both handled here:

- **Overlap.** Adding up box areas double-counts where boxes overlap and can
  push "density" past 100 %. The default method rasterises the ROI onto a grid
  and marks covered cells, so an area covered twice counts once — a *union*,
  not a sum. (`method="sum"` is kept for comparison, and the tests show it
  reporting 50 % where the truth is 43.75 %.)
- **Boxes are bigger than vehicles.** A box around an angled car holds a lot of
  empty asphalt; a motorcycle's box is mostly air. Each box is shrunk about its
  centre by a per-class factor (`CLASS_FOOTPRINT`) before the area is measured.

Expect 0–5 % on an empty approach and 25–40 % on a fully queued one. It rarely
approaches 100 %, because the ROI covers the whole road width including the
gaps between vehicles — that is the honest number, not a bug.

### 2. `queue.py` — how far back the jam reaches

Measured from the **stop line**, along the approach:

```
 back of queue  ← the answer
 ┌─────────────────────────────┐
 │  [car] [car] [car]          │
 │  [car] [car]                │
 └─────────────────────────────┘
 ═══════════════ STOP LINE ════
```

The queue is built as a **chain**, not as "distance to the farthest vehicle":

1. keep only vehicles that are stopped (or barely creeping),
2. start from the one nearest the stop line — it must actually *be* near it,
3. walk outward while each gap stays under the gap rule (9 m by default),
4. stop at the first big gap.

Without the chain rule, a car still rolling in 60 m behind a stopped queue
would be reported as part of it. Vehicles fully past the stop line are excluded;
a vehicle whose nose has crept over it is still counted, because that vehicle is
the head of the queue.

Reported in pixels always, and in metres when the ROI is calibrated.
Against the simulator's ground truth over 120 s of lane-less Indian traffic
(6 588 samples with a real queue present): **median error 0.07 m, mean 4.9 m,
mean relative error 15 %**. The median is what the estimate looks like
normally; the mean is pulled up by the moment a green starts, when the
simulator calls a vehicle "moving" a fraction of a second before the estimator
does. Lane-disciplined traffic scores slightly better (mean 4.0 m) — the logic
is the same either way.

**On Indian traffic specifically**: nothing here assumes lanes. A vehicle's
position is reduced to one number — distance from the stop line — so a
motorcycle filtering up the side of a queue is just another vehicle at its own
distance. What does change is that a queue holds far more vehicles per metre,
which is precisely why vehicle count alone is a poor signal there and why
density is carried alongside it.

### 3. `waiting_time.py` — how long vehicles have been stuck

```
Vehicle #17 → speed ≈ 0 → 5 s → 10 s → 18 s waiting
```

The only metric that accumulates across frames, so it keeps a small record per
`track_id`. Three details that decide whether it works on real footage:

- **Time comes from timestamps, not frame counts**, so dropped frames or a
  variable frame rate do not distort it.
- **Speed is measured over a ~0.5 s window, not between consecutive frames.**
  At 25 fps a 0.6-pixel detector wobble reads as ~20 px/s — fast enough to look
  like moving traffic. Averaging over a window divides that jitter down.
- **Hysteresis**: a vehicle is judged stopped below one threshold and moving
  above a higher one, so it cannot flicker between states and shred a 20-second
  count.

Reports `max_wait_s` (the worst-off vehicle), `mean_wait_s`, and `total_wait_s`
— the vehicle-seconds of delay on that approach, which is the quantity an
adaptive controller ultimately exists to reduce.

### 4. `pressure.py` — one comparable number per approach

```
pressure = 100 × Σ(weight_i × normalised_metric_i) / Σ(weight_i)      [× camera confidence]
```

> **PROVISIONAL — this is not the final Traffic Pressure Index.**
> The weights and caps are starting values so the pipeline runs end to end and
> can be demonstrated. They live in `config/pressure.json`, not in the code,
> precisely so they can be retuned — or the whole formula replaced — once the
> individual modules are validated on real footage. Nothing upstream depends on
> them: density, queue and waiting time each stay meaningful on their own.

Raw metrics are normalised to 0–1 against caps first, because vehicles, metres,
seconds and percent cannot be added together. Every result carries a full
breakdown (`result.explain()`), so any pressure value can be *explained* rather
than trusted — which is exactly what a reviewer will ask for.

Camera confidence, when supplied, scales the result: unreliable footage should
not drive a confident signal decision. It has a floor (0.3), so one dark frame
cannot silently zero an approach that is actually jammed.

Current defaults: queue 0.30, waiting 0.30, count 0.25, density 0.15, optical
flow 0.00 (recorded, not yet weighted).

---

## Calibrating it to real footage

Three steps, in order. Nothing else needs changing.

**1. Draw the ROIs and stop lines.**

```bash
python tools/draw_roi.py --video videos/traffic.mp4 --out config/intersection.json
```

Click the drivable area of each approach, then two points for its stop line.
Draw only the road, not the pavement — every density value is a fraction of
whatever polygon is drawn here.

**2. Set the pixel-per-metre scale.** Measure something of known length in the
frame: a lane is about 3.5 m wide, a broken lane-marking dash about 3 m long.
Divide pixels by metres. Without it, queue length is reported in pixels only
and everything else still works.

**3. Set the stopped-speed threshold** to roughly 0.5 m/s in pixels:

```python
from metrics import MotionTracker, TrafficMetricsEngine

motion = MotionTracker(stop_speed_px_s=0.5 * pixels_per_meter,
                       go_speed_px_s=1.2 * pixels_per_meter)
engine = TrafficMetricsEngine(intersection, motion=motion)
```

Sanity check afterwards: park the camera on a red phase. Queue length should be
stable, waiting times should climb steadily, and density should sit well above
its empty-road value. If waiting time keeps resetting to zero, the stop
threshold is too low or the tracker is swapping ids.

---

## Tests

```bash
python run_tests.py -v
```

91 tests, all passing, no external test framework required.

| File | Covers |
|---|---|
| `test_schema.py` | dicts, tracker objects, alternative key names, malformed boxes |
| `test_density.py` | hand-computed areas, overlap handling, partial-ROI boxes, class footprints |
| `test_queue.py` | chain building, gap breaking, nose-over-the-line, moving traffic excluded |
| `test_waiting_time.py` | accumulation, jitter immunity, stop/go episodes, tracking dropouts |
| `test_pressure.py` | range, monotonicity, weights, breakdown arithmetic, config round trip |
| `test_integration.py` | full engine vs. simulated traffic, both traffic styles, queue length vs. ground truth |

Density and queue tests use numbers checkable by hand — a 50×50 box in a
100×100 ROI is 25 %, two boxes overlapping by 625 px² give 43.75 % and not
50 %, three bumper-to-bumper cars give a 140 px queue.

`simulate.py` generates the test traffic: a signal cycling N→E→S→W, Poisson
arrivals at four different rates, car-following physics, mixed vehicle classes,
lane-less lateral packing with two-wheeler filtering, sub-pixel detector jitter
and occasional dropped detections. It is scaffolding — delete it once the real
tracker output is available.

---

## Files

```
traffisense_metrics/
├── metrics/
│   ├── schema.py            input contract — dicts OR objects (the AttributeError fix)
│   ├── roi.py               road polygons, stop lines, pixel↔metre scale
│   ├── motion.py            shared per-track speed / stopped state
│   ├── density.py           ◄ assigned module 1
│   ├── queue.py             ◄ assigned module 2
│   ├── waiting_time.py      ◄ assigned module 3
│   ├── pressure.py          ◄ assigned module 4 (provisional)
│   └── traffic_metrics.py   the engine that runs all four per frame
├── tests/                   91 unit + integration tests
├── tools/
│   ├── draw_roi.py          click ROIs and stop lines on your own footage
│   ├── make_fixture.py      pretrained YOLO + ByteTrack → tracked-vehicle JSON
│   ├── run_fixture.py       replay a fixture; prints metrics, annotates video
│   └── make_demo_video.py   render the simulator to mp4 + matching fixture
├── config/
│   ├── pressure.json        the provisional weights, editable without code
│   └── intersection_example.json
├── simulate.py              synthetic tracker output for testing
├── main.py                  runnable demo
├── integration_example.py   template for the real video pipeline
└── run_tests.py
```

`motion.py` exists because "is this vehicle stopped?" is asked by both queue
length and waiting time. Answering it once, in one place, keeps the two metrics
consistent — a vehicle can never be queued in one and moving in the other.

---

## Viva answers

**Density** — "The proportion of the road region occupied by vehicles, measured
as the union of the vehicle footprints over the ROI area, so overlapping boxes
are not double-counted."

**Queue length** — "How far the stopped traffic extends back from the stop line.
Vehicles are chained outward from the line and the queue ends at the first gap
larger than about 9 metres, so traffic still approaching is not counted as
queued."

**Waiting time** — "How long each tracked vehicle has been stationary, measured
from timestamps and accumulated per track id. Speed is estimated over a short
time window with two thresholds, so detector jitter does not reset the count."

**Pressure** — "A provisional weighted combination of the four metrics, each
normalised to 0–1 first. The weights sit in a config file because the final
Traffic Pressure Index has not been fixed yet; the module reports the full
breakdown behind every value."

**Why classical CV and not another model?** — "The detection side already runs a
deep model. These metrics are geometry over its output — area ratios, distances
from a stop line, elapsed time per track — so they are cheap, deterministic,
and explainable. Another network here would add training cost and remove the
ability to say exactly why a number came out the way it did."

---

## What comes next

- [ ] Watch the synthetic demo once, so you know what "working" looks like
- [ ] Draw real ROIs on the project's own footage and calibrate pixels-per-metre
- [ ] Build a fixture from one 60 s clip and tune against it
- [ ] Run the module against Hitarth's tracker output and confirm `track_id` is stable
- [ ] Log a full signal cycle and check queue length against a manual count on video
- [ ] Wire the existing camera-confidence module into `camera_confidence()`
- [ ] Wire the existing optical-flow module in and decide whether it earns a weight
- [ ] **Only then**: tune the weights and finalise the Traffic Pressure Index
- [ ] Hand `pressure_ranking()` to the signal controller

The order matters. Every module already produces a standalone, validated number;
fusing them is the last step, not the first.
