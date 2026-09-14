# Vision pipeline — real shopper data from video

Replaces the simulated logs (`simulation.py`) with data extracted from real store
video. **Stage 1 (done): detect and track people, and what they carry.**

## Model choice

`YOLO11` via [ultralytics](https://docs.ultralytics.com), restricted to the COCO
`person` class, with **BoT-SORT** for IDs across frames and a **stitching pass**
that re-joins broken IDs by clothing colour (see "Keeping one ID per shopper").

* `yolo11n-pose.pt` — default: boxes **plus 17 body keypoints**, same speed as
  the plain model (~2.2x realtime on this CPU at stride 3 / 640px). The
  keypoints are what lets the floor point be estimated when the feet are hidden
  (next section).
* `yolo11s-pose.pt` / `yolo11m-pose.pt` — more accurate, proportionally slower
* `yolo11n.pt` etc. — boxes only; the floor point is then always the box bottom
* Licence: ultralytics is **AGPL-3.0**. Fine for internal/research use; a closed
  commercial product needs an ultralytics licence or an Apache-2.0 alternative
  (YOLOX, RT-DETR).

Weights are cached in `data/models/`.

### Where are the feet? — the floor point when a shelf hides the legs

In a CCTV view a shelf often hides everything below the waist. The detection box
then stops at the shelf, and "bottom of the box" is the shelf edge, not the
feet — the person lands on the map a metre or two too far *up* the image. In
the sample video this affects **41 %** of detections.

So `foot_x,foot_y` is now chosen per detection from the pose keypoints
(`estimate_foot()` in `detect_people.py`), and `foot_src` records how:

| `foot_src` | when | how |
|---|---|---|
| `box` | an ankle is visible | bottom-centre of the box, as before |
| `knees` | hips + knees visible | continue down the thigh: knee + (knee − hip) × 0.249 / 0.229 |
| `torso` | shoulders + hips visible | head-top + (hip − shoulder) / 0.311 |
| `position` | only the head | head-top + height(y), a per-video linear fit of standing height against image y on the full-body detections |

The ratios are body proportions as fractions of standing height, measured on 271
full-body detections in the sample video (head→shoulder 0.211, shoulder→hip
0.311, hip→knee 0.229, knee→sole 0.249) and matching anthropometric tables.
`foot_x` uses the hips' midpoint when it is visible — a better plumb line than
the head.

Accuracy, checked by hiding the lower body of the full-body detections and
comparing with their real box bottom (people 140–430 px tall):

| estimator | median error | 90th pct |
|---|---|---|
| knees | 8 px | 26 px |
| torso | 10 px | 30 px |
| position (head only) | 33 px | 85 px |

The estimate is never above the visible box bottom and never below the frame.
The viewer draws estimated floor points hollow with a dashed drop-line from the
visible bottom, and the live map draws a dashed ring round them; `Skeleton` in
the player shows the keypoints themselves. `stats.foot_source` in the job result
counts the sources; `stats.height_model` is the fitted `position` model.

Limits: a person bending over or sitting is measured as if standing; children
are shorter than the proportions assume (the ratios still hold, the position
model does not); the `torso` estimate degrades when the camera looks steeply
down, because the torso foreshortens more than the legs.

### Steady floor points — smoothing

Even a perfectly still shopper wobbles a few pixels frame to frame, and
`foot_src` flips between `box` / `knees` / `torso` as an ankle comes and goes,
jumping `foot_y` by 10–30 px; the homography magnifies that into a zig-zag on
the plan (worst far from the camera). `smooth_feet()` runs after stitching, per
visitor, and rewrites `foot_x,foot_y`:

1. a centred rolling **median** over 0.5 s of analysed frames removes the
   single-frame spikes;
2. a **leaky dead-band** ignores motion smaller than 3 % of the box height: a
   standing shopper becomes one fixed point, a walking one follows the median
   continuously (trailing by at most the dead-band, a few pixels).

On the sample video (yolo11m-pose, every frame) the map step during "still"
seconds drops from 7.2 px (90th pct) to 1.2 px and the path-length /
displacement ratio from 8.3 to 2.7. `--no-smooth` keeps the raw points,
`--smooth-window` (seconds) and `--smooth-deadband` (fraction of height) tune
it; `stats.smooth` reports how many frames were held. The live pipeline applies
a causal version (`FootSmoother`: running median + the same dead-band, ~0.25 s
behind).

Lowering the input resolution does **not** help here: at `--imgsz 416` the
still-frame jitter is the same and a quarter of the detections are lost; at
`--imgsz 960` the model sees ~10 % more detections, mostly small far-away
people, at higher confidence.

### Held objects — YOLO-World

A second detector runs on every processed frame and looks for the things people
carry. It is [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)
(`yolov8s-worldv2.pt`), an *open-vocabulary* model: the classes are free-text
prompts, embedded with CLIP once at start-up, so it is not limited to COCO's 80
classes — "cardboard box" and "shopping basket" are not COCO classes at all.
Default prompts:

```
shopping bag, handbag, backpack, cardboard box, shopping basket, shopping cart, bottle, phone
```

Each object is attributed to the person whose box contains at least 40 % of it
(`assign_holder()` in `detect_people.py`; on a tie the person whose centre is
nearer). Objects that are inside nobody — shelf stock, a bag on the floor — are
kept with an empty `person_id`, so the viewer can hide them.

Cost: ~0.8 s per frame on this CPU on top of ~0.35 s for people, i.e. the run is
about 3x slower with objects on. `--no-objects` turns it off; `yolov8m-worldv2.pt`
is more accurate and slower again. Needs the `clip` package (see `requirements.txt`).

## Install

```bash
pip install -r vision/requirements.txt
```

## Use

```bash
# 1. download a source video (yt-dlp; any supported URL)
python vision/download_video.py "https://www.youtube.com/watch?v=KMJS66jBtVQ"

# 2. detect + track people
python vision/detect_people.py data/videos/KMJS66jBtVQ.mp4 \
    --stride 3 --preview data/tracks/KMJS66jBtVQ_preview.mp4
```

`--stride N` processes every Nth frame (the main speed knob), `--seconds` /
`--start` analyse a slice, `--model` picks the weights. Held-object detection is
on by default: `--no-objects` skips it, `--object-classes "shopping bag,box,…"`
changes what to look for, `--object-model` / `--object-conf` tune it. Floor-point
smoothing is on by default (`--no-smooth`, see "Steady floor points").

Output — `data/tracks/<video>_tracks.csv`, one row per person per processed frame:

```
frame,time_s,track_id,conf,x1,y1,x2,y2,cx,cy,foot_x,foot_y,foot_src,w,h,keypoints
```

`foot_x,foot_y` is the floor contact point — the input the next stage needs to
map a person onto the store map — and `foot_src` says whether it was seen or
estimated (see above). `keypoints` is the 17 COCO keypoints as `x:y:conf;…`.

And `data/tracks/<video>_objects.csv`, one row per detected object per processed
frame:

```
frame,time_s,label,conf,x1,y1,x2,y2,cx,cy,w,h,person_id
```

`person_id` is the `track_id` of the person holding it, empty when nobody is.

## Server

The endpoints are a Flask blueprint mounted by `server.py` under `/api/vision`
(silently skipped if the vision extras aren't installed):

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/vision/videos` | locally downloaded videos |
| POST | `/api/vision/download` | `{"url": ...}` → download one |
| POST | `/api/vision/jobs` | `{"video","stride","seconds","start","model","conf","preview","tracker","reid","track_buffer","stitch","stitch_gap","stitch_sim","objects","object_classes","object_model","object_conf"}` → starts a background job (`stride` 0/missing = auto) |
| GET  | `/api/vision/jobs` | all jobs + live progress |
| GET  | `/api/vision/jobs/<id>` | one job (status, progress, stats) |
| GET  | `/api/vision/jobs/<id>/tracks` | raw tracks CSV |
| GET  | `/api/vision/jobs/<id>/tracks.json` | per-track trails + `items` (what each track held), for the viewer |
| GET  | `/api/vision/jobs/<id>/objects` | raw objects CSV |
| GET  | `/api/vision/objects/<video_id>` | objects CSV matching `/tracks/<video_id>` (same run) |
| GET  | `/api/vision/jobs/<id>/preview` | annotated mp4 |

## The app — five tabs

`http://localhost:5000/` is a tab shell (`vision/shell.html`); each tab is its own
page in an iframe, loaded on first open and kept alive afterwards, so a
half-finished calibration or a video position survives tab switching. The tab is
in the URL hash, so `/#livemap` links straight to one.

| Tab | Page | What it is |
|---|---|---|
| Simulation | `/viewer` (`viewer.html`) | the original simulated-log viewer, animation / analytics / paths / chat |
| Video | `/vision` (`vision/player.html`) | the video with YOLO detections drawn over it |
| Calibration | `/calibrate` (`vision/calibrate.html`) | match points between camera and store plan |
| Live map | `/livemap` (`vision/livemap.html`) | video and plan side by side, people placed on the plan |
| Zones | `/zones` (`vision/zones.html`) | draw polygons on the plan, see them projected into the camera |

## Live map tab

The payoff view: the camera on the left, the store plan on the right, and every
detected person drawn on the plan at their mapped floor position — play, scrub,
or step frame by frame and watch them move through the store. Trails show the
last 6 s. The side panel lists who is in frame with their plan coordinates.

It needs both a tracks CSV and a saved calibration for the video; when one is
missing it says which and offers a button through to the Calibration tab.

## Zones tab — polygons on the plan, projected into the camera

`http://localhost:5000/zones`

Draw an area on the store plan and the same area is drawn over the video,
warped through the inverse of the saved homography (map → camera). If the
outline in the camera does not hug the same patch of floor — a shelf base, the
entrance mat, the queue lane — the calibration is off *there*, which the RMS
number alone never tells you. It is also where the entrance / sections / queue /
checkout polygons for the next stage get drawn.

* `+ New zone` (or <kbd>n</kbd>), then click the corners on the plan; click the
  first corner again, double-click, right-click or <kbd>Enter</kbd> closes it,
  <kbd>Backspace</kbd> drops the last corner, <kbd>Esc</kbd> cancels. The
  half-finished polygon is previewed on the camera too, dashed.
* drag a corner handle to move it — both panes update live; click a zone to
  select it, <kbd>Del</kbd> deletes it; rename, recolour or hide it in the list.
* the detected people are drawn on both panes and each zone shows how many are
  inside it right now (point-in-polygon on the mapped foot point), so a zone can
  be checked against actual traffic while the video plays.
* `Save zones` (<kbd>Ctrl+S</kbd>) → `data/zones/<map>.json`. Zones are in
  **map pixels**, so they belong to the map image, not the video: every camera
  calibrated against the same plan shares them.

A map edge that crosses the camera's horizon would flip to the other side of the
image; edges are subdivided and the points behind the camera dropped, so such a
zone degrades to a clipped outline rather than a bow-tie.

Endpoints: `GET/POST /api/vision/zones/<map>` — body
`{"zones": [{"id","name","color","pts": [[x,y],…],"visible"}], "map_size": [w,h]}`.

## Player app

`http://localhost:5000/vision` — the source video with the detections drawn over it
by a canvas overlay synced to `video.currentTime` (not a pre-rendered mp4, so
everything is toggleable and seekable):

* boxes, track id + confidence, movement trails, floor points (hollow + dashed
  drop-line when the feet were hidden and the point is estimated), skeleton —
  each toggleable
* held objects: a dashed box in the holder's colour with the label, hidden when
  the holder is filtered out; unheld ones (shelf stock) are off by default
* per-track "holding: shopping bag ×12" in the list, and a whole-video tally of
  objects by label
* min-confidence and trail-length sliders
* frame-stepping (buttons or arrow keys), space to play/pause
* live counts: people in frame, average confidence, per-track "tracked Ns"
* deep links to a moment: `/vision#t=52`

Extra endpoints it uses: `GET /api/vision/video/<video_id>` (the mp4, Range
requests supported) and `GET /api/vision/tracks/<video_id>` (the most complete
tracks CSV for that video; `?job=<id>` picks a specific run).

The pre-rendered `--preview` mp4 is still there for a quick look without a server.

## Calibration app — camera pixels → store map

`http://localhost:5000/calibrate`

The shop floor is a plane, so a **homography** (3x3 matrix) maps a point on the
camera image to a point on the store map. The page fits one from point pairs you
click:

1. pick the store map (`Upload map…` puts a PNG/JPG/SVG in `data/map/`)
2. scrub to a frame where floor landmarks are visible
3. click a landmark on the camera frame, then the same spot on the map — repeat
   **4+ times**, spread out over the floor, never all on one line
4. from the 4th pair on, the detected people show up live as dots on the map;
   scrub the video and watch them move
5. `Save calibration` → `data/calibration/<video_id>.json`

The fit is `cv2.findHomography` (plain least squares — every clicked pair counts).
The side panel shows the per-pair reprojection error in map pixels and the RMS;
pairs over 25 px are flagged red, so a mis-click is easy to spot and delete.

### Fixing a bad fit

* **Edit** — drag any marker (either pane) to move it; the fit updates on drop.
  Click a table row to select a pair, `Delete` removes it, `×` removes it from
  the row, `Undo` drops the last one.
* **Test a point** — switch modes and click anywhere: on the camera frame it
  draws where that point lands on the map, on the map it draws where it lands in
  the camera image. This is the fastest way to see *how* a fit is wrong, not just
  that the number is high.
* **Map grid warped onto the camera** — projects the map border and an 8x8 grid
  into the camera view. If the border does not sit on the walls and the grid does
  not lie flat on the floor, the fit is wrong regardless of the RMS.
* **Camera warped onto the map** (opacity slider) — the reverse: the camera frame
  itself, perspective-warped onto the plan (a CSS `matrix3d` built from H). When
  the calibration is right the floor, the fixtures and the plan line up; when it
  is wrong you see exactly *how*. This is the single most useful check.

Switching the map clears the pairs — the map-side clicks belong to the image they
were made on.

### Choosing landmarks on a schematic plan

A hand-drawn plan is not a survey. Fixtures are usually placed about right, but
their proportions are not, so pick landmarks whose *position* is meaningful:

* good — the base corners of a big fixture (the promo island, the counter), where
  a wall shelf meets the floor, a fixture's floor centre;
* bad — anything drawn as a symbol rather than to scale. In `newmap.png` the
  entrance mat is a thin strip, while in the camera it is a large square mat:
  its corners are not the same points, and using them wrecks the fit.

A homography only holds for points **on one plane**. The usual causes of a bad
fit are, in order:

1. clicking things that are not on the floor (a shelf top, a sign, someone's
   head) — always click where an object *meets the floor*;
2. a map that is not a true, to-scale top-down plan of the same store;
3. wide-angle CCTV lens distortion, which no homography can absorb — spread the
   points over the whole floor and expect a few px of residual;
4. all the points sitting on one line or crowded in one corner.
Escape cancels a half-finished pair, arrow keys step the video.

Endpoints: `GET/POST /api/vision/maps`, `GET /api/vision/maps/<name>`,
`GET/POST /api/vision/calibration/<video_id>` (POST without `"save": true` just
returns the fit, which is how the live preview works).

**Note:** `data/map/store_map_placeholder.png` is a redrawn copy of the map from
chat — replace it with the real file via `Upload map…` so the proportions match.

### Applying it to tracks

```bash
python vision/homography.py KMJS66jBtVQ
```

Reads the saved calibration and writes `<video>_tracks_map.csv` — the tracks CSV
plus `map_x,map_y` columns, i.e. every person's position in store-map
coordinates. `vision/homography.py` also exposes `load_calibration()`,
`to_map()` and `solve_homography()` for the next stage.

## Keeping one ID per shopper

Three things used to make an ID jump to someone else or restart mid-walk, and
each has a fix in `detect_people.py` / `stitch.py`:

1. **Too few frames.** The sample video is 13 fps and stride 5 gave the tracker
   2.6 frames/s; a walking shopper moved most of a body width between samples, so
   the box no longer overlapped its own previous box. Stride is now automatic,
   ≈6 analysed frames/s (`TARGET_TRACK_FPS`), whatever the video's rate.
2. **Weak detections thrown away before the tracker.** ByteTrack/BoT-SORT match
   low-confidence boxes (a shopper half behind a shelf, conf 0.15) to *existing*
   tracks in a second pass — but the detector was cut at 0.35 first, so that pass
   never ran. The detector now runs down to 0.1; `--conf` (0.35) is the bar for
   the first-stage match and for starting a *new* track, and only boxes that
   belong to an established track reach the CSV.
3. **No notion of appearance.** Two people crossing swapped IDs, and anyone
   re-emerging after an occlusion got a new one. The tracker is now BoT-SORT with
   its ReID gate on (`with_reid`, the detector's own features, no extra model, no
   measurable cost) and no camera-motion compensation (fixed camera). Its lost
   buffer is given in seconds (`--track-buffer`, 4 s) and converted to analysed
   frames. The generated YAML sits in `data/models/tracker_*.yaml`; pass your own
   with `--tracker path.yaml`.

Then, after the whole video has been seen, **stitching** (`stitch.py`) joins
fragments that are plausibly one person: B starts within `--stitch-gap` (8 s) of
A ending, about where A was heading (A's last floor point extrapolated with its
velocity, plus walking-speed slack that grows with the gap), and B's *clothes*
match A's. The clothing descriptor is an HSV histogram of the torso (between
shoulders and hips from the keypoints; upper-middle of the box without them) and
of the thighs when visible. Pixels are split into *colourful* (hue × saturation
bins) and *grey/black/white* (brightness bins only) — without that split a black
coat and a white hoodie look alike, because both scatter their meaningless hues
over every bin. Links are chosen greedily by a cost mixing distance, gap and
clothing dissimilarity; each fragment gets at most one predecessor and one
successor. `stats.stitch.log` lists every link with its numbers.

`--stitch-sim` (0.6) was set by eye on the sample video: at 0.6 all 9 links in
the first minute were the same person; at 0.55 two of 14 were wrong (a pink top
and a maroon top at the crowded till). A wrong join corrupts two visitors, a
missed one just leaves an extra ID, so the default is conservative. Re-run the
pass alone with other values via `python vision/stitch.py … --sim 0.5` — it
re-reads the video frames for the descriptor but skips detection (seconds, not
minutes).

Measured on the first 60 s of the sample video (7 people in frame on average):

| | old (ByteTrack, stride 5, conf 0.35) | new (BoT-SORT + ReID, stride 2, stitch 0.6) |
|---|---|---|
| analysed frames / s | 2.6 | 6.5 |
| IDs from the tracker | 33 | 40 |
| IDs after stitching | 33 | 31 |
| median ID lifetime | 18 samples ≈ 7 s | 90+ samples ≈ 15 s |
| ID ends with another starting nearby ≤ 3 s later | 11 | 0–3 |

Tracker IDs go *up* with more frames (more chances to break); stitching brings
them back down, and the lifetime is what matters for the map trails.

## Live stream

`live.py` runs the same detector, tracker and clothing descriptor on a camera,
an RTSP/HTTP url or a video file, and publishes events instead of writing a CSV
at the end. It is the path for the live demo; the offline analyser stays for
recorded footage.

```bash
python vision/live.py rtsp://user:pass@camera/stream --out data/live/cam1.jsonl
python vision/live.py 0                                        # webcam
python vision/live.py data/videos/KMJS66jBtVQ.mp4 --seconds 60 # a file, paced like a camera
python vision/live.py data/videos/KMJS66jBtVQ.mp4 --fast       # a file, every frame, no pacing (tests)
```

Or from the UI: the **Live** tab (`/live`) starts it through `POST
/api/vision/live/start`, shows the annotated frame, the visitors in view and
the event feed (`GET /api/vision/live/stream`, server-sent events; `…/events?since=N`
to poll; `…/status`; `…/frame.jpg`; `POST …/stop`).

What is different from the offline path, and why:

* **Stitching is online.** A new tracker id is not judged at birth — one frame
  of clothing is not enough. It is *pending* for `--decide-after` seconds
  (2 s), then matched against the *lost pool*: fragments that have not been
  seen for a moment, within `--stitch-gap` (8 s), with the same distance /
  velocity / clothing gate as the offline pass (`link_cost` in `stitch.py`).
  A pool fragment that comes back to life meanwhile is no longer a candidate.
  So **people rows are published `decide-after` seconds late**, already
  carrying their final `track_id`; a consumer that cannot wait can listen to
  the `merge` events and re-key instead. For zone/dwell analytics a 2 s delay
  is invisible.
* **Ingest keeps only the newest frame.** A reader thread drops the backlog,
  so a slow moment becomes dropped frames, never a growing delay; timestamps
  are wall-clock (`ts`), not frame counts; on a read failure the capture is
  reopened with exponential back-off, and `status.ingest` counts drops and
  reconnects. A file is paced at its own fps so the pipeline sees it as a
  camera would (`--fast` disables that for tests).
* **Rate is "about 6 analysed frames per second"** (`--fps`), whatever the
  camera delivers — a frame is processed when `1/fps` has elapsed. The
  tracker's lost buffer is derived from that.
* **Held objects run off the critical path.** YOLO-World is the expensive
  detector and bags do not change every frame, so it runs in its own thread on
  the most recent frame, at most every `--object-interval` seconds (1 s), and its
  results are attributed to the *visitor* ids current at that moment (they wait
  for the same `decide-after` window).

Events (one JSON object per line in the `--out` file, same over SSE):

```
{"type":"people",  "seq":…, "t":12.3, "ts":1757…, "rows":[<offline CSV fields> + "ts"]}
{"type":"objects", "seq":…, "t":12.0, "frame":…, "objects":[{label, conf, box…, person_id}]}
{"type":"merge",   "seq":…, "t":12.3, "from":31, "to":38, "visitor":27, "gap_s":1.5, "dist_px":…, "sim":0.73}
{"type":"status",  "seq":…, "state":"running", "frame_ms":…, "ingest":{…}, "objects":{…}, "stitch":{…}}
```

`t` is seconds since the pipeline started (`--fast` on a file: the file's own
clock, so results line up with the offline CSV). Tested on the sample video in
`--fast` mode: the online stitcher makes the same links as the offline pass
(10 merges in the first minute, 44 raw ids → 34 visitors).

**CPU budget.** People-only (yolo11n-pose, 640 px) measured 70 ms per frame on
this laptop when it was cool and 280 ms an hour later under browser/Docker load
and throttling — the same code, the same frames, a 4x spread. At 70 ms, 6
analysed fps fits; at 280 ms the pipeline degrades to ~3 fps by dropping
frames (it never queues them), which is where IDs start to jump. YOLO-World
costs 600–850 ms per run and competes for the same cores: with it on every
second the tracker fell to ~2 fps. For a CPU-only demo: run objects every
2–3 s (`--object-interval`) or turn them off, close what else is using the
cores, plug the laptop in with a performance power plan — or use a small GPU,
which removes the question. `status.achieved_fps` against `target_fps`,
`frame_ms` against the `1000/fps` budget, and `ingest.frames_dropped` show live
whether the machine keeps up; the Live tab colours them.

**Learned re-identification (optional, off).** `--embed yolo11n-cls.pt` adds an
ImageNet-classifier embedding of the torso to the clothing similarity
(`Embedder` in `stitch.py`; also `--embed` on `stitch.py` for offline re-runs).
Measured on the labelled pairs from the sample video it does *not* separate
people: cosine 0.87–0.99 for same-person pairs, 0.86–0.96 for different people,
median 0.91 between strangers — a classifier backbone is not a ReID model. The
hook is there so a real ReID network (OSNet-style weights, an extra dependency)
can be dropped in when the crowded till needs it; `EMB_COS_LO` / `EMB_WEIGHT`
would be recalibrated for it.

## Deploying a shareable demo

The repo ships a `Dockerfile` that runs everything — the simulation viewer,
the video tabs and the live pipeline — on a CPU host, plus the sample video,
its analysis, the store map and the camera calibration (`data/…`, un-ignored
for exactly those files). Model weights are fetched at build time. Image
≈2 GB (torch CPU). `docker build -t shopview . && docker run -p 7860:7860
shopview` runs it locally.

Where to put it:

* **Railway** (what the demo uses). railway.com → *New Project → Deploy from
  GitHub repo* → pick `alondotan/shopview`. The `Dockerfile` is detected, the
  build takes ~10 min (torch + model weights), then *Settings → Networking →
  Generate Domain* gives the public URL. Optional variable:
  `ANTHROPIC_API_KEY` for the chat tab. Railway does not fetch Git LFS
  objects, so the `Dockerfile` downloads the sample video from GitHub itself
  when it finds the LFS pointer. Memory: the container needs ~1.5 GB.
* **Hugging Face Spaces** — Docker Spaces now require a PRO subscription
  (free CPU or not); `README.md` carries the front matter they need, so with
  PRO it is `git push https://huggingface.co/spaces/<user>/<space> master:main`
  (password = a write token). The video is in LFS, as Spaces require.
* **Render / Fly.io** — same Dockerfile, `$PORT` is honoured. Render's free
  tier (512 MB RAM) is too small for torch + YOLO; pick ≥ 2 GB.

What to expect on a shared CPU host: the pre-computed analysis (Video, Live
map, Zones tabs) is instant. The **Live** tab runs YOLO on the host's CPU: on
2 vCPU expect 2–4 analysed fps with held objects off, so stitching still works
but IDs will jump more than on a fast machine. New analysis jobs on other
videos run at a fraction of realtime. Uploads (maps, calibrations, live JSONL)
land on ephemeral disk and vanish on redeploy.

## Results on the sample video

CCTV of a small retail store, 111 s, 1270x720, 13.09 fps.

| | |
|---|---|
| frames processed (stride 3) | 484 |
| person detections | 3 558 |
| avg people in frame | 7.35 |
| max people in frame | 12 |
| raw track IDs | 85 |
| wall time | 50 s (2.2x realtime, CPU) |

85 raw IDs for ~12–15 actual shoppers — ByteTrack restarts an ID whenever a person
is occluded by a shelf. Merging those into real visitors is the next stage.

With objects on (same video, stride 3): see `--no-objects` timing above; the
detector finds mostly `shopping bag` and `handbag`, plus the odd `cardboard box`
being carried to the till. Two YOLO-World habits to know about: it labels the
same bag "shopping bag" in one frame and "handbag" in the next (the per-track
tally shows both), and it sometimes calls a stack of boxed stock on a shelf
"cardboard box" — those have no holder and are hidden by default.

## Next stages

1. ~~Track stitching / re-ID~~ — done, see "Keeping one ID per shopper" and
   "Live stream". Still open: a real ReID network (OSNet-style) behind the
   `Embedder` hook for crowded tills, where torso crops get contaminated by the
   person in front — the ImageNet classifier tried there does not separate people.
2. ~~Homography~~ — done, see the calibration app above. Still to do: turn
   `map_x,map_y` into the `row,col` cells `store.py` uses.
3. **Zones & events** — the polygons from the Zones tab (`data/zones/<map>.json`)
   → `ENTERED`, `BROWSING`, `JOINED_QUEUE`… in the same schema as `*_video.csv`.
4. **Per-visit summary** — dwell, sections, funnel stage → `week_summary`-shaped
   CSV, which the existing viewer and chat already consume unchanged.
