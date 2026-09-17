# Vision pipeline — real shopper data from video

Replaces the simulated logs (`simulation.py`) with data extracted from real store
video. **Stage 1 (done): detect and track people, and what they carry.**

## Model choice

Three models, all permissively licensed, all run through **onnxruntime** on the
CPU (`detector.py`, `bytetrack.py`):

* **YOLOX** ([Megvii](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0)
  finds the people — and, in the same pass, the COCO objects the held-object
  stage wants (handbag, backpack, bottle, cell phone…).
* **RTMPose** ([OpenMMLab](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose),
  Apache-2.0) puts **17 COCO body keypoints** on each person box, a few ms per
  crop. The keypoints are what lets the floor point be estimated when the feet
  are hidden (next section).
* **ByteTrack** ([Zhang et al.](https://github.com/ifzhang/ByteTrack), MIT,
  ported in `bytetrack.py`) keeps IDs across frames; a **stitching pass** then
  re-joins broken IDs by clothing colour (see "Keeping one ID per shopper").

`--model` is `<detector>+<pose>`:

| spec | per frame, this laptop | notes |
|---|---|---|
| `yolox_s+rtmpose-m` | ~110 ms (people + pose) | **default** — YOLOX-s is about YOLO11-n's accuracy (COCO AP 40.5) |
| `yolox_tiny+rtmpose-s` | ~50 ms | fast; small far-away people are missed more |
| `yolox_m+rtmpose-m` | ~180 ms | the shipped sample analysis |
| `yolox_s` | ~35 ms | boxes only; the floor point is then always the box bottom |

The network input is a property of the weights (416 px for tiny/nano, 640 px
otherwise; the frame is letter-boxed into it), so there is no `--imgsz`.
Weights are fetched into `data/models/` on first use (`python vision/detector.py
--fetch` does it ahead of time — the Dockerfile does).

### Licences

| component | licence | role |
|---|---|---|
| YOLOX code + COCO weights | Apache-2.0 | people + COCO objects |
| RTMPose code + weights (via mmpose / rtmlib) | Apache-2.0 | keypoints |
| ByteTrack (reference implementation, ported) | MIT | tracking |
| OWLv2 weights (optional, `--object-model owlv2`) | Apache-2.0 | free-text objects |
| CLIP weights (optional, `--embed clip`) | MIT | appearance embedding |
| onnxruntime / OpenCV / numpy / torch / transformers | MIT / Apache-2.0 / BSD | runtime |
| lap, yt-dlp, Flask | BSD-2 / Unlicense / BSD-3 | assignment, download, server |

No AGPL anywhere: the earlier ultralytics stack (YOLO11, YOLO-World, its
BoT-SORT) was AGPL-3.0, which would have required releasing the whole service's
source or buying an Ultralytics licence. One caveat that is legal rather than
technical: the RTMPose "body7" checkpoints were trained on seven public pose
datasets (COCO, MPII, AI Challenger, CrowdPose, Halpe, PoseTrack18, sub-JHMDB),
some of which carry research-only terms for the *data*; the weights themselves
are released under Apache-2.0 by OpenMMLab, as is standard practice, but a
COCO-only RTMPose checkpoint exists if a lawyer wants the cleaner lineage. The
YOLOX weights are COCO-only.

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

On the sample video (every frame, measured with the earlier detector) the map step during "still"
seconds drops from 7.2 px (90th pct) to 1.2 px and the path-length /
displacement ratio from 8.3 to 2.7. `--no-smooth` keeps the raw points,
`--smooth-window` (seconds) and `--smooth-deadband` (fraction of height) tune
it; `stats.smooth` reports how many frames were held. The live pipeline applies
a causal version (`FootSmoother`: running median + the same dead-band, ~0.25 s
behind).

Lowering the input resolution does **not** help here: at 416 px (the
`yolox_tiny` input) the still-frame jitter is the same and a quarter of the
detections are lost.

### Held objects — COCO classes for free, OWLv2 for free text

Every processed frame also looks for the things people carry. Two backends
(`--object-model`):

* **`yolox`** (default) — no second model at all: the YOLOX pass that found the
  people is an 80-class COCO detector, so the wanted COCO classes are simply
  read off it. Of the default list that is `handbag`, `backpack`, `bottle` and
  `cell phone` (`COCO_ALIASES` maps "phone", "bag", "purse"…); the rest are
  reported as skipped at start-up. Free, and what the Live tab and the Docker
  image use.
* **`owlv2`** / **`owlv2-large`** — [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)
  (Google, Apache-2.0) through `transformers`, an *open-vocabulary* model: the
  classes are free-text prompts, so "cardboard box" and "shopping basket" work
  even though they are not COCO classes. ~1 s per frame on this laptop
  (`--object-interval` in the live pipeline keeps it off the critical path),
  so it is for offline runs where the labels matter. Needs
  `pip install -r vision/requirements-owl.txt` (torch + transformers, ~1 GB of
  weights on first use).

Default classes (COCO names or prompts, `--object-classes`):

```
shopping bag, handbag, backpack, cardboard box, shopping basket, shopping cart, bottle, phone
```

Each object is attributed to the person whose box contains at least 40 % of it
(`assign_holder()` in `detect_people.py`; on a tie the person whose centre is
nearer). Objects that are inside nobody — shelf stock, a bag on the floor — are
kept with an empty `person_id`, so the viewer can hide them. `--no-objects`
turns the stage off.

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
`--start` analyse a slice, `--model` picks the weights (`yolox_s+rtmpose-m`
default, see "Model choice"). Held-object detection is on by default:
`--no-objects` skips it, `--object-classes "handbag,backpack,…"` changes what to
look for, `--object-model owlv2` switches to free-text prompts, `--object-conf`
tunes the threshold. Floor-point smoothing is on by default (`--no-smooth`, see
"Steady floor points").

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
| POST | `/api/vision/jobs` | `{"video","stride","seconds","start","model","conf","preview","track_buffer","stitch","stitch_gap","stitch_sim","objects","object_classes","object_model","object_conf"}` → starts a background job (`stride` 0/missing = auto) |
| GET  | `/api/vision/jobs` | all jobs + live progress |
| GET  | `/api/vision/jobs/<id>` | one job (status, progress, stats) |
| GET  | `/api/vision/jobs/<id>/tracks` | raw tracks CSV |
| GET  | `/api/vision/jobs/<id>/tracks.json` | per-track trails + `items` (what each track held), for the viewer |
| GET  | `/api/vision/jobs/<id>/objects` | raw objects CSV |
| GET  | `/api/vision/objects/<video_id>` | objects CSV matching `/tracks/<video_id>` (same run) |
| GET  | `/api/vision/jobs/<id>/preview` | annotated mp4 |
| GET/POST/DELETE | `/api/vision/scenes/<name>` | a multi-camera scene (map, cameras, shared landmarks) |
| POST | `/api/vision/scenes/<name>/calibrate` | fit every camera's homography from the shared landmarks; `save` writes the per-camera files too |
| POST | `/api/vision/scenes/<name>/fuse` | join the cameras' tracks into visitors → `data/fusion/<name>.json` + `_tracks.csv` |
| GET  | `/api/vision/scenes/<name>/fusion` | the saved fusion result (`fusion.csv` for the CSV) |

## The app — seven tabs

`http://localhost:5000/` is a tab shell (`vision/shell.html`); each tab is its own
page in an iframe, loaded on first open and kept alive afterwards, so a
half-finished calibration or a video position survives tab switching. The tab is
in the URL hash, so `/#livemap` links straight to one.

| Tab | Page | What it is |
|---|---|---|
| Simulation | `/viewer` (`viewer.html`) | the original simulated-log viewer, animation / analytics / paths / chat |
| Video | `/vision` (`vision/player.html`) | the video with the detections drawn over it |
| Calibration | `/calibrate` (`vision/calibrate.html`) | match points between camera and store plan |
| Live map | `/livemap` (`vision/livemap.html`) | video and plan side by side, people placed on the plan |
| Zones | `/zones` (`vision/zones.html`) | draw polygons on the plan, see them projected into the camera |
| Multi-cam calibration | `/multical` (`vision/multical.html`) | one map, all the cameras of a store, shared landmarks |
| Multi-cam | `/multiview` (`vision/multiview.html`) | every camera and the plan in sync, tracks fused into one path per shopper |

## Live map tab

The payoff view: the camera on the left, the store plan on the right, and every
detected person drawn on the plan at their mapped floor position — play, scrub,
or step frame by frame and watch them move through the store. Trails show the
last 6 s. The side panel lists who is in frame with their plan coordinates.

It needs both a tracks CSV and a saved calibration for the video; when one is
missing it says which and offers a button through to the Calibration tab.

## Several cameras on one store

A real store has more than one camera, and one shopper walks from one view
into the next — or stands where two of them overlap. The two multi-cam tabs
handle that; `1ye32v77GE0` and `-8zyEwAa50Q` are the sample pair: the same
checkout area, recorded at the same moment from two angles (scene
`checkout`, plan `data/map/checkout_plan.png` — a schematic drawn from the
footage; replace it with the real plan via *Upload map…*).

A **scene** (`data/scenes/<name>.json`) is the list of videos that share one
map and one clock, plus the *shared landmarks*. Every camera still gets its
own `data/calibration/<video>.json`, written by the joint tool, so the
single-camera tabs (Live map, Zones) keep working per camera.

### Multi-cam calibration tab — one map, all cameras

`http://localhost:5000/multical`. *New scene…* → name, map, tick the cameras
(an *offset* per camera lines up clocks that do not start together: seconds
to add to that video's time). Then:

1. click a floor landmark **on the map** — a numbered marker appears;
2. click the same spot **in every camera that sees it** (skip the cameras
   that do not — a landmark needs the map plus at least one camera);
3. click the map again for the next landmark. Each camera is fitted from
   the landmarks it has, from the 4th one on; the side panel shows one RMS
   per camera, the table one cell per landmark × camera with its error.

Once a camera has a fit, the selected landmark is drawn dashed in that
camera at the spot the fit *expects* it — a hint of where to click, and a
quick check on the fit. Everything from the single-camera tool is there,
across all cameras: drag any marker (map or camera), *Test a point* puts a
click from any pane on the map **and on every other camera** (two cameras
that disagree about a floor spot are the whole point of the joint view),
the map grid warped into each camera, and every camera warped onto the plan
at once (opacity slider) — when the calibrations agree, the same counter
lands in the same place from each of them. The detected people of every
camera are drawn on the map in the camera's colour (circle = camera 1,
square = camera 2), so scrubbing shows whether the two views put the same
shopper at the same spot. *Save all* writes the scene and one calibration
file per camera. The per-camera slider scrubs one video; the *All cameras*
slider moves all of them together.

### Multi-cam tab — the cameras and the plan together

`http://localhost:5000/multiview`. All cameras of the scene play in sync
(camera 1 is the clock, the others are nudged to it), the plan beside them.
Colour means **visitor**: one person keeps one colour in every camera and
on the plan. On the plan each camera's own floor point is a small
camera-shaped dot, the fused position a big one (double ring = seen by two
or more cameras at that instant), with the fused trail. The side panel lists
who is in view and which cameras see them, the fusion stats, every
cross-camera link with its evidence, and the three parameters with a
*Re-fuse* button (a second, saved to `data/fusion/`). Re-fuse after changing
a calibration.

### Fusion — joining tracks from different cameras (`fusion.py`)

```bash
python vision/fusion.py checkout            # → data/fusion/checkout.json + checkout_tracks.csv
python vision/fusion.py checkout --max-dist 80 --max-gap 10 --min-sim 0.35
```

Each camera's tracks (after its own stitching) are mapped onto the plan
through its homography; then tracks from *different* cameras are linked
when they are plausibly the same person:

* **overlap** — both cameras see the person at the same time for at least
  `min_overlap_s` (1 s) and the two mapped points stay close: median
  distance under `max_dist_px` (60 map px) and most samples within it;
* **hand-off** — B starts within `max_gap_s` (6 s) after A ends, about
  where A was heading (last point + velocity, slack ≈ `max_dist_px` per
  second of gap) — the person walked out of one view into the other;
* and in both cases the **clothes must agree**: the same torso/legs colour
  histogram the per-camera stitching uses (`stitch.appearance`), averaged
  over each track, must score at least `min_sim` (0.4). On the sample pair —
  a steep view against a shallow one — the same person scores 0.46–0.68 and
  different people 0.24–0.38. This is what stops two shoppers side by side
  at the till from being merged when the geometry cannot tell them apart:
  a floor point estimated behind a counter is easily 50–100 map px off, so
  `max_dist_px` has to be generous (100). The histograms need the video
  frames (a few seconds per video, cached in `data/fusion/appearance_*`).

A track whose floor points were mostly *estimated* (feet hidden behind a
counter — `foot_src` ≠ `box`) gets up to 1.5× the distance, because such
points are a few times less accurate than seen feet; that is what joins the
shopper at the belt whom the shallow camera places a metre too far along
the lane. A long co-location at some distance outranks a brief brush at none (two
shoppers pass within a metre of each other all the time; they do not stand
together for a quarter of a minute), so the link cost also falls with the
shared time, up to 15 s.

Links are taken cheapest first with union-find. A merge is refused when it
would put two tracks **of the same camera that overlap in time** into one
visitor — one person is not two boxes in one frame. Same-camera fragments
are never linked directly (the per-camera stitch already did that with
clothing), but they do end up together when the other camera bridges them
(A1 ↔ B ↔ A2): `stats.visitors_bridged` counts those, and that is what the
second camera buys — it covers the first one's occlusions. The fused trail
is sampled every 0.2 s as the confidence-weighted mean of the cameras that
see the person then (estimated floor points count half); holes stay holes.

Output: `data/fusion/<scene>.json` (visitors with their member tracks per
camera, trails, links, stats) and `data/fusion/<scene>_tracks.csv`:

```
time_s,visitor_id,map_x,map_y,n_cams,cams
```

one row per visitor per 0.2 s in map pixels — the input for the zones /
events stage, whichever camera(s) saw the person.

Known limit: in a shallow view the feet of the people nearest the camera
leave the frame, the floor point is clamped to the bottom edge and lands on
the plan 50–100 px from where the other camera puts them — the calibration
is not wrong, the point is. With the clothing gate the right link usually
still wins; when two long tracks of one camera both fit, the nearer one
does. `max_dist_px` is in map pixels because the plan's scale is unknown —
set it to roughly one metre on your plan.

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
  the holder is filtered out; unheld ones (shelf stock) can be toggled off
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
2. **Weak detections thrown away before the tracker.** ByteTrack matches
   low-confidence boxes (a shopper half behind a shelf, conf 0.15) to *existing*
   tracks in a second pass — but the detector was cut at 0.35 first, so that pass
   never ran. The detector now runs down to 0.1; `--conf` (0.35) is the bar for
   the first-stage match and for starting a *new* track, and only boxes that
   belong to an established track reach the CSV.
3. **No notion of appearance.** Two people crossing swapped IDs, and anyone
   re-emerging after an occlusion got a new one. ByteTrack itself is
   IoU-only (a fixed camera needs no motion compensation); the appearance work
   is done by the stitching pass below, which is where it can see a whole
   fragment's clothing rather than one frame's. The tracker's lost buffer is
   given in seconds (`--track-buffer`, 4 s) and converted to analysed frames;
   the settings actually used are in `stats.tracker` (`tracker_settings()` in
   `detect_people.py`).

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

**Quick re-appearance.** The commonest break is a shelf: the legs disappear,
the box shrinks to head and shoulders, the tracker's IoU match fails
(ByteTrack's low-confidence second pass never looks at *lost* tracks) and a new id is
born a few frames later — while the head never left the picture. The clothing
descriptor is at its worst just then (torso half hidden, shelf in the crop), so
it is not asked to prove such a match, only not to contradict it. A link is a
*quick re-appearance* when

* B starts **≤ 1 s** after A ends (`QUICK_GAP_S`);
* B's first **head** lands within **0.3 body heights** of where A's head was
  heading (`QUICK_HEAD_HEIGHTS`). The head is the face keypoints' x at the box
  top (`head_point()`): the box centre would shift by half a body width when a
  full-body box turns into a head-and-shoulders one, the face does not;
* B's first floor point is well inside the walking slack, at most 70 % of it
  (`QUICK_FOOT_FRACTION`) — someone who *replaces* A at the edge of the
  plausible range is not A;

and then the clothes need a similarity of only `QUICK_MIN_SIM` 0.35 instead of
0.6. The three gates are tight on purpose and were set on labelled pairs from
both sample videos: the true re-appearances all had gaps ≤ 0.85 s, head
distances ≤ 0.25 h and floor points ≤ 67 % of the slack, while at the crowded
till of the first sample a *different* person appears at the same spot 1.1–1.3 s
later with clothing similarity 0.52–0.59, or 0.4 s later at 95 % of the slack.
Such links carry `"quick": true` in the log. The torso crop is also clipped to
the detection box now: when a shelf hides the hips the pose model still
guesses them, below the box, and that strip is shelf, not shirt.

Effect: on `-1bRhYjw1qE` the man in the white shirt walking behind the shelf at
0.9–1.5 s keeps his id (1→6), the bald man in grey survives two breaks (8→9→11)
and the man in black behind the rack one (5→7) — 16 visitors → 12, every link
checked by eye. On `KMJS66jBtVQ` it adds three correct links (the man carrying
a box 36→49→64, a toddler 76→86, the man in the red cap 107→115) and no wrong
one — 60 visitors → 54. The live stitcher applies the same rule, with the
fragments it has 2 s after the new id appears.

`--stitch-sim` (0.6) was set by eye on the sample video: at 0.6 all 9 links in
the first minute were the same person; at 0.55 two of 14 were wrong (a pink top
and a maroon top at the crowded till). A wrong join corrupts two visitors, a
missed one just leaves an extra ID, so the default is conservative. Re-run the
pass alone with other values via `python vision/stitch.py … --sim 0.5` — it
re-reads the video frames for the descriptor but skips detection (seconds, not
minutes).

Measured on the first 60 s of the sample video (7 people in frame on average,
with the earlier YOLO11 detector — the mechanism, not the detector, is what
changed between the columns):

| | old (stride 5, conf 0.35, no stitching) | new (stride 2, stitch 0.6) |
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
* **Held objects run off the critical path.** With the default `yolox`
  object model they come out of the same pass as the people and are reported
  every `--object-interval` seconds (1 s). With `owlv2` the detector is the
  expensive part and bags do not change every frame, so it runs in its own
  thread on the most recent frame at that interval. Either way the results are
  attributed to the *visitor* ids current at that moment (they wait for the
  same `decide-after` window).

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

**CPU budget.** People + pose (`yolox_s+rtmpose-m`) measure ~115 ms per
frame on this laptop with ~10 people in view (the pose model's cost grows with
the head count; YOLOX-s alone is ~30 ms). At 115 ms, 6 analysed fps fits with
room to spare; a machine 3x slower degrades to ~3 fps by dropping frames (the
pipeline never queues them), which is where IDs start to jump. COCO objects
are free; OWLv2 costs ~1 s per run and competes for the same cores — run it
every 2–3 s (`--object-interval`) or leave objects on `yolox`. `yolox_tiny+rtmpose-s`
halves the cost again if the machine cannot keep up — or use a small GPU
(`onnxruntime-gpu`, `--device cuda`), which removes the question. `status.achieved_fps` against `target_fps`,
`frame_ms` against the `1000/fps` budget, and `ingest.frames_dropped` show live
whether the machine keeps up; the Live tab colours them.

**Learned re-identification (optional, off).** `--embed clip` adds a CLIP
image embedding of the torso (`openai/clip-vit-base-patch32`, MIT; needs
`vision/requirements-owl.txt`) to the clothing similarity (`Embedder` in
`stitch.py`; also `--embed` on `stitch.py` for offline re-runs). The earlier
ImageNet-classifier embedding, measured on the labelled pairs from the sample
video, did *not* separate people: cosine 0.87–0.99 for same-person pairs,
0.86–0.96 for different people, median 0.91 between strangers — a
classification backbone is not a ReID model, and CLIP is not expected to do
much better. The hook is there so a real ReID network (OSNet-style weights,
an extra dependency) can be dropped in when the crowded till needs it; `EMB_COS_LO` / `EMB_WEIGHT`
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
  build takes a few minutes (onnxruntime + ~90 MB of model weights, no torch),
  then *Settings → Networking → Generate Domain* gives the public URL. Every
  push to `master` redeploys. Optional variable: `ANTHROPIC_API_KEY` for the
  chat tab and the actions stage. Railway does not fetch Git LFS objects, so
  the `Dockerfile` downloads the sample videos from GitHub itself when it
  finds the LFS pointers. Memory: the container needs ~1 GB.
* **Hugging Face Spaces** — Docker Spaces now require a PRO subscription
  (free CPU or not); `README.md` carries the front matter they need, so with
  PRO it is `git push https://huggingface.co/spaces/<user>/<space> master:main`
  (password = a write token). The video is in LFS, as Spaces require.
* **Render / Fly.io** — same Dockerfile, `$PORT` is honoured. The image has
  no torch (onnxruntime + opencv + ffmpeg, ~1.8 GB); 1 GB of RAM is enough.

What to expect on a shared CPU host: the pre-computed analysis (Video, Live
map, Zones tabs) is instant. The **Live** tab runs YOLOX + RTMPose on the
host's CPU: on 2 vCPU expect 2–4 analysed fps, so stitching still works but
IDs will jump more than on a fast machine (`yolox_tiny+rtmpose-s` in the
model field helps). New analysis jobs on other
videos run at a fraction of realtime. Uploads (maps, calibrations, live JSONL)
land on ephemeral disk and vanish on redeploy.

## Results on the sample videos

The shipped analysis (`data/tracks/*.csv`) is `yolox_m+rtmpose-m`, every
frame, on this laptop's CPU; the people/tracking numbers are from the default
`yolox` object run, the object rows from a second run with `--object-model
owlv2 --device mps` (free-text labels; 14 and 12 minutes on the Apple GPU):

| | KMJS66jBtVQ (small shop) | -1bRhYjw1qE (second clip) |
|---|---|---|
| video | 111 s, 1270x720, 13.09 fps | 60 s, 1280x720, 30 fps |
| frames processed | 1 452 | 1 810 |
| person detections | 15 671 | 6 176 |
| avg / max people in frame | 10.8 / 15 | 3.4 / 6 |
| raw tracker IDs → visitors after stitching | 95 → 61 | 19 → 14 |
| feet estimated (hidden by a shelf) | 40 % | 29 % |
| wall time | 251 s (0.44x realtime) | 222 s (0.27x realtime) |
| objects (`yolox`, COCO) | handbag 2 254, backpack 211 (2 224 held) | bottle 3 267 (shelf stock, unheld), handbag 107 |
| objects (`owlv2`, shipped) | shopping basket 9 349, shopping cart 3 567, shopping bag 1 857, handbag 1 700, cardboard box 658 (5 281 held, 29 visitors) | bottle 17 216 (the fridge), shopping bag 724, cardboard box 508 (1 298 held, 10 visitors) |

The earlier YOLO11-m run at 960 px on the same clips gave 83 → 54 and 17 → 12
ids, with 56 % / 36 % of the feet estimated — the same picture, so the
calibration, zones and stitching thresholds carried over unchanged. OWLv2
labels the display racks "shopping basket" and the shelf stock "bottle" /
"cardboard box" — those have no holder and are hidden by default in the
viewer; the held ones are what the per-visitor tally shows.

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
5. ~~Several cameras~~ — done, see "Several cameras on one store". Still open:
   a ReID embedding across cameras (view-invariant where the colour histogram
   is only roughly so), floor points for people cut off by the frame edge
   (extrapolate below the frame instead of clamping), and automatic calibration of a new camera from
   the shoppers themselves (simultaneous floor points in two views are
   correspondences — tried on the sample pair; at a checkout too few people
   move for RANSAC to lock on, an aisle camera would do better).
