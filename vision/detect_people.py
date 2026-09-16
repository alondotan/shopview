"""Stage 1 — detect and track people in a store video.

Runs YOLOX for the boxes and RTMPose for the body keypoints (``detector.py``,
both Apache-2.0), ByteTrack for IDs across frames (``bytetrack.py``, MIT; see
``tracker_settings``) and a post-run stitching pass that re-joins broken IDs
by clothing colour + position + time (``stitch.py``), and writes one CSV row
per person-detection per processed frame:

    frame,time_s,track_id,raw_id,conf,x1,y1,x2,y2,cx,cy,foot_x,foot_y,w,h

``track_id`` is the visitor after stitching, ``raw_id`` the tracker's own id.

`cx,cy` is the box centre; `foot_x,foot_y` is where the person touches the
floor — the point the later homography / zone-mapping stage needs. With a pose
model (the default, ``yolox_s+rtmpose-m``) that is the bottom of the box when the
feet are visible, and otherwise *estimated* from the visible body parts, because
in a CCTV view a shelf often hides everything below the waist and the box then
stops at the shelf, not at the feet. ``foot_src`` says which:

    box       feet visible → bottom-centre of the box
    knees     hips + knees visible → extrapolated down the legs
    torso     shoulders + hips visible → head-top + torso length / 0.311
    position  only the head → this video's own height-vs-image-y model

``keypoints`` holds the 17 COCO keypoints as ``x:y:conf;…`` for the viewer.

Optionally (``objects=True``, the default) the things people carry are
detected too — by default the COCO classes YOLOX already finds in the same pass
(handbag, backpack, bottle, cell phone…), or with ``object_model="owlv2"`` an
open-vocabulary detector that takes free-text prompts (shopping basket,
cardboard box…) — and each object is attributed to the person whose box
contains it. Those go to a sibling CSV, ``<video>_objects.csv``:

    frame,time_s,label,conf,x1,y1,x2,y2,cx,cy,w,h,person_id

``person_id`` is the track_id of the holder, or empty when the object is not
inside anyone's box (on a shelf, on the floor).
"""

from __future__ import annotations
import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bytetrack import ByteTracker
from detector import (
    COCO_CLASSES, DEFAULT_MODEL, KP_VISIBLE, MODEL_DIR, PERSON_CLASS, RtmPose, Yolox,
    draw_objects, draw_people, nms_boxes, parse_model_spec,
)
from stitch import Fragment, appearance, head_point, stitch

ROOT = Path(__file__).resolve().parent.parent
TRACK_DIR = ROOT / "data" / "tracks"

# ── tracker ───────────────────────────────────────────────────────────────
# How many *analysed* frames per second the tracker should see. IoU matching
# needs consecutive boxes of the same person to overlap; below ~4 fps a walking
# shopper moves most of a body width between samples and IDs start jumping.
TARGET_TRACK_FPS = 6.0
# The tracker's second stage matches weak detections (a half-hidden shopper at
# conf 0.15) to existing tracks — but only if the detector lets them through.
TRACK_LOW_THRESH = 0.1
TRACK_BUFFER_S = 4.0        # keep a lost track alive this long before dropping it


def auto_stride(fps: float) -> int:
    return max(1, round(fps / TARGET_TRACK_FPS))


def tracker_settings(fps: float, stride: int, conf: float,
                     buffer_s: float = TRACK_BUFFER_S) -> dict:
    """ByteTrack settings for a fixed store camera (see ``bytetrack.py``):
    the detector runs down at ``TRACK_LOW_THRESH`` so the tracker's second
    stage sees weak detections; ``conf`` is the first-stage / new-track bar;
    ``buffer`` is in analysed frames, so it is derived from seconds."""
    return {
        "track_high": round(conf, 3),
        "track_low": TRACK_LOW_THRESH,
        "new_track": round(max(conf, 0.45), 3),
        "buffer": max(5, round(buffer_s * fps / stride)),
        "match_thresh": 0.8,
        "fuse_score": True,
    }


class PeopleModel:
    """Detector + pose + tracker for one video: ``model(frame, run_conf)`` →
    (tracked people, every detection of the frame). Each tracked person is a
    dict with ``raw_id``, the tracker's box, ``conf`` and ``kp`` — 17 COCO
    keypoints as (x, y, conf), or None without a pose model. The full
    detection tuple (boxes, scores, class ids) is returned so the held-object
    stage can take the COCO objects from the same pass."""

    def __init__(self, spec: str = DEFAULT_MODEL, device: str = "cpu", det_conf: float = TRACK_LOW_THRESH,
                 tracker: dict | None = None) -> None:
        det_name, pose_name = parse_model_spec(spec)
        self.det = Yolox(det_name, device)
        self.pose = RtmPose(pose_name, device) if pose_name else None
        self.det_conf = det_conf
        self.tracker = ByteTracker(**(tracker or {}))
        self.spec = spec

    @property
    def imgsz(self) -> int:
        return self.det.input_w

    def __call__(self, frame, run_conf: float | None = None):
        boxes, scores, cls = self.det(frame, min(run_conf or self.det_conf, self.det_conf))
        m = (cls == PERSON_CLASS) & (scores >= self.det_conf)
        pboxes = boxes[m]
        tracks = self.tracker.update(pboxes, scores[m])
        kps = [None] * len(tracks)
        if self.pose is not None and tracks:
            kp, kv = self.pose(frame, [pboxes[t.det_index] for t in tracks])
            kps = [[(float(x), float(y), float(c)) for (x, y), c in zip(pk, pv)] for pk, pv in zip(kp, kv)]
        tracked = [{"raw_id": t.id, "box": t.xyxy, "conf": t.score, "kp": k} for t, k in zip(tracks, kps)]
        return tracked, (boxes, scores, cls)


# ── foot-point estimation from pose keypoints ─────────────────────────────
KP_LSH, KP_RSH, KP_LHIP, KP_RHIP, KP_LKNEE, KP_RKNEE, KP_LANK, KP_RANK = 5, 6, 11, 12, 13, 14, 15, 16

# Vertical body proportions as fractions of standing height (head top → sole).
# Measured on 271 full-body detections in the sample video; they agree with
# anthropometric tables to within a few percent.
BODY_HEAD_TO_SHOULDER = 0.211
BODY_SHOULDER_TO_HIP = 0.311
BODY_HIP_TO_KNEE = 0.229
BODY_KNEE_TO_FOOT = 1 - BODY_HEAD_TO_SHOULDER - BODY_SHOULDER_TO_HIP - BODY_HIP_TO_KNEE
MIN_HEIGHT_SAMPLES = 20   # full-body samples needed before fitting the position model


def _mid(kp, a, b):
    """Midpoint of two keypoints, using whichever of the two are visible."""
    pts = [kp[i] for i in (a, b) if kp[i][2] >= KP_VISIBLE]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def estimate_foot(box, kp) -> tuple[float, float, str, bool]:
    """Where does this person touch the floor?

    ``box`` is (x1, y1, x2, y2); ``kp`` is a list of 17 (x, y, conf) COCO
    keypoints or None. Returns (foot_x, foot_y, source, full_body) where
    ``full_body`` marks detections usable for fitting the position model.

    Tested by hiding the lower body of full-body detections in the sample video:
    median error 8 px (knees) / 10 px (torso) against 33 px for the head-only
    position model, on people 140–430 px tall.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    if kp is None:
        return cx, y2, "box", False
    ank = _mid(kp, KP_LANK, KP_RANK)
    hip = _mid(kp, KP_LHIP, KP_RHIP)
    sh = _mid(kp, KP_LSH, KP_RSH)
    knee = _mid(kp, KP_LKNEE, KP_RKNEE)
    if ank is not None:                      # the box reaches the feet
        both = kp[KP_LANK][2] >= KP_VISIBLE and kp[KP_RANK][2] >= KP_VISIBLE
        return cx, y2, "box", both and sh is not None and hip is not None
    fx = hip[0] if hip else cx              # hips sit above the feet better than the head does
    if knee and hip:
        fy = knee[1] + (knee[1] - hip[1]) * BODY_KNEE_TO_FOOT / BODY_HIP_TO_KNEE
        return fx, max(fy, y2), "knees", False
    if sh and hip:
        fy = y1 + (hip[1] - sh[1]) / BODY_SHOULDER_TO_HIP
        return fx, max(fy, y2), "torso", False
    return fx, y2, "position", False        # refined after the run, see _fit_position_model


def _fit_position_model(samples: list[tuple[float, float]]):
    """Standing pixel height as a linear function of head-top y for this camera.
    People lower in the image are closer and taller; ``samples`` are (y1, height)
    from full-body detections. Returns (a, b) for height = a*y1 + b, or None."""
    if len(samples) < MIN_HEIGHT_SAMPLES:
        return None
    import numpy as np
    ys, hs = np.array(samples).T
    a, b = np.polyfit(ys, hs, 1)
    return float(a), float(b)


# Floor-point smoothing (post-pass, per visitor). The detector's box wobbles a
# few pixels frame to frame and ``foot_src`` can flip between box / knees /
# torso, which jumps ``foot_y`` by 10–30 px; the homography then turns that
# into a zig-zag on the plan for someone who is standing still.
SMOOTH_WINDOW_S = 0.5      # centred rolling median over this much video time
SMOOTH_DEADBAND = 0.03     # ignore motion smaller than this × box height


def smooth_feet(rows: list[dict], analysed_fps: float, window_s: float = SMOOTH_WINDOW_S,
                deadband_frac: float = SMOOTH_DEADBAND) -> dict:
    """Smooth ``foot_x`` / ``foot_y`` in place, per ``track_id`` in frame order.

    Two stages. A centred rolling median over ``window_s`` seconds of analysed
    frames removes single-frame spikes (a missed ankle, a flipped estimator).
    Then a *leaky* dead-band: the output only moves by the part of the
    displacement that exceeds ``deadband_frac`` of the person's box height, so
    a standing shopper is one fixed point, while a walking one follows the
    median continuously (trailing it by at most the dead-band, a few pixels).
    Returns a small stats dict.
    """
    window = int(round(window_s * analysed_fps)) | 1
    if window < 3 or not rows:
        return {"enabled": False}
    half = window // 2
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_track[r["track_id"]].append(r)
    moved = held = 0
    for rs in by_track.values():
        rs.sort(key=lambda r: r["frame"])
        xs = [r["foot_x"] for r in rs]
        ys = [r["foot_y"] for r in rs]
        n = len(rs)
        med = []
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            wx, wy = sorted(xs[lo:hi]), sorted(ys[lo:hi])
            med.append((wx[len(wx) // 2], wy[len(wy) // 2]))
        out_x, out_y = med[0]
        for r, (mx, my) in zip(rs, med):
            tol = deadband_frac * max(r["h"], 1.0)
            dx, dy = mx - out_x, my - out_y
            d = (dx * dx + dy * dy) ** 0.5
            if d > tol:
                k = (d - tol) / d              # move by the excess only
                out_x, out_y = out_x + dx * k, out_y + dy * k
                moved += 1
            else:
                held += 1
            r["foot_x"], r["foot_y"] = round(out_x, 1), round(out_y, 1)
    return {"enabled": True, "window_frames": window, "deadband_frac": deadband_frac,
            "held_pct": round(100 * held / max(held + moved, 1), 1)}


class FootSmoother:
    """Causal version of :func:`smooth_feet` for the live pipeline: per raw id,
    a running median over the last ``window`` points then the same leaky
    dead-band. Lags the true position by about half the window."""

    def __init__(self, analysed_fps: float, window_s: float = SMOOTH_WINDOW_S,
                 deadband_frac: float = SMOOTH_DEADBAND, forget_s: float = 30.0):
        self.window = max(1, int(round(window_s * analysed_fps)))
        self.deadband_frac = deadband_frac
        self.forget_s = forget_s
        self._hist: dict[int, list[tuple[float, float]]] = {}
        self._out: dict[int, tuple[float, float]] = {}
        self._seen: dict[int, float] = {}

    def update(self, raw_id: int, foot: tuple[float, float], h: float, t: float) -> tuple[float, float]:
        hist = self._hist.setdefault(raw_id, [])
        hist.append(foot)
        del hist[:-self.window]
        xs, ys = sorted(x for x, _ in hist), sorted(y for _, y in hist)
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        ox, oy = self._out.get(raw_id, (mx, my))
        tol = self.deadband_frac * max(h, 1.0)
        dx, dy = mx - ox, my - oy
        d = (dx * dx + dy * dy) ** 0.5
        if d > tol:
            k = (d - tol) / d
            ox, oy = ox + dx * k, oy + dy * k
        self._out[raw_id] = (ox, oy)
        self._seen[raw_id] = t
        if len(self._seen) > 64:                       # forget ids not seen for a while
            for rid in [r for r, ts in self._seen.items() if t - ts > self.forget_s]:
                self._hist.pop(rid, None); self._out.pop(rid, None); self._seen.pop(rid, None)
        return ox, oy


def _kp_str(kp) -> str:
    return "" if kp is None else ";".join(f"{x:.0f}:{y:.0f}:{c:.2f}" for x, y, c in kp)


# What to look for in people's hands. With the default ``yolox`` object model
# only the COCO classes among these are found (handbag, backpack, bottle, cell
# phone — ``COCO_ALIASES`` maps the everyday names); ``owlv2`` takes them all
# as free-text prompts — "cardboard box" and "shopping basket" are not COCO
# classes at all.
DEFAULT_OBJECT_CLASSES = [
    "shopping bag", "handbag", "backpack", "cardboard box",
    "shopping basket", "shopping cart", "bottle", "phone",
]
DEFAULT_OBJECT_MODEL = "yolox"
OBJECT_MODELS = ("yolox", "owlv2", "owlv2-large")
COCO_ALIASES = {"phone": "cell phone", "mobile phone": "cell phone", "smartphone": "cell phone",
                "bag": "handbag", "purse": "handbag", "trolley": "suitcase"}

# An object counts as "held" when at least this fraction of its box lies inside
# the person's box.
HELD_MIN_OVERLAP = 0.4


class CocoObjects:
    """Held-object "detector" that costs nothing: it picks the wanted COCO
    classes out of the YOLOX pass that found the people."""

    def __init__(self, object_classes: list[str]) -> None:
        wanted = [COCO_ALIASES.get(c.lower(), c.lower()) for c in object_classes]
        self.class_ids = sorted({COCO_CLASSES.index(c) for c in wanted if c in COCO_CLASSES})
        self.names = COCO_CLASSES
        self.missing = sorted({c for c in wanted if c not in COCO_CLASSES})
        if self.missing:
            print(f"warning: not COCO classes, skipped: {self.missing} — "
                  f"use --object-model owlv2 for free-text classes")

    def __call__(self, frame, conf: float, frame_dets=None):
        if frame_dets is None:
            raise ValueError("CocoObjects needs the frame's YOLOX detections")
        boxes, scores, cls = frame_dets
        m = np.isin(cls, self.class_ids) & (scores >= conf)
        boxes, scores, cls = boxes[m], scores[m], cls[m]
        idx = nms_boxes(boxes, scores, 0.45)               # agnostic: one box per object
        return boxes[idx], scores[idx], cls[idx]


class Owlv2Objects:
    """Open-vocabulary detector — Google OWLv2 (Apache-2.0) through
    ``transformers``: the classes are free-text prompts, embedded with the
    model's own CLIP text tower. Accurate and slow on a CPU (seconds per
    frame), so it is the option, not the default. Needs
    ``pip install -r vision/requirements-owl.txt``."""

    HF = {"owlv2": "google/owlv2-base-patch16-ensemble",
          "owlv2-large": "google/owlv2-large-patch14-ensemble"}

    def __init__(self, name: str, object_classes: list[str], device: str = "cpu") -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as e:
            raise RuntimeError("OWLv2 needs torch + transformers: "
                               "pip install -r vision/requirements-owl.txt") from e
        self.torch = torch
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(self.HF[name])
        self.model = Owlv2ForObjectDetection.from_pretrained(self.HF[name]).eval().to(device)
        self.names = list(object_classes)

    def __call__(self, frame, conf: float, frame_dets=None):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(text=[self.names], images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        # the processor pads the image to a square before resizing, so the
        # boxes come back relative to that square, not to the frame
        side = max(h, w)
        post = getattr(self.processor, "post_process_grounded_object_detection", None) \
            or self.processor.post_process_object_detection          # older transformers
        res = post(out, threshold=conf, target_sizes=self.torch.tensor([[side, side]]))[0]
        boxes = res["boxes"].cpu().numpy().astype(np.float32)
        scores = res["scores"].cpu().numpy().astype(np.float32)
        cls = res["labels"].cpu().numpy().astype(int)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        idx = nms_boxes(boxes, scores, 0.45)               # agnostic: not "bag" + "handbag" twice
        return boxes[idx], scores[idx], cls[idx]


def _overlap_fraction(obj, person) -> float:
    """Fraction of the object box's area that lies inside the person box."""
    ox1, oy1, ox2, oy2 = obj
    px1, py1, px2, py2 = person
    iw = max(0.0, min(ox2, px2) - max(ox1, px1))
    ih = max(0.0, min(oy2, py2) - max(oy1, py1))
    area = max((ox2 - ox1) * (oy2 - oy1), 1e-6)
    return iw * ih / area


def assign_holder(obj_box, people: list[tuple[int, tuple]]) -> int | None:
    """Which tracked person is holding this object? ``people`` is
    [(track_id, (x1,y1,x2,y2)), ...]. Picks the person box containing the
    biggest share of the object; on a tie (two overlapping people), the one whose
    centre is closer to the object's — a bag hangs next to its owner's body."""
    ocx = (obj_box[0] + obj_box[2]) / 2
    ocy = (obj_box[1] + obj_box[3]) / 2
    best, best_key = None, None
    for tid, pb in people:
        frac = _overlap_fraction(obj_box, pb)
        if frac < HELD_MIN_OVERLAP:
            continue
        pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        dist = ((ocx - pcx) ** 2 + (ocy - pcy) ** 2) ** 0.5
        key = (-round(frac, 1), dist)
        if best_key is None or key < best_key:
            best, best_key = tid, key
    return best


def extract_people(tracked: list[dict], frame, height: int, appearance_on: bool = True) -> list[dict]:
    """One dict per tracked person from :class:`PeopleModel`: raw id, box,
    conf, keypoints, floor point (+ how it was found), and — with
    ``appearance_on`` — the clothing descriptors for stitching. Shared by the
    offline analyser and the live pipeline."""
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if (appearance_on and tracked) else None
    out = []
    for t in tracked:
        x1, y1, x2, y2 = t["box"]
        kp = t["kp"]
        fx, fy, src, full = estimate_foot((x1, y1, x2, y2), kp)
        torso = legs = None
        if frame_hsv is not None:
            torso, legs = appearance(frame_hsv, (x1, y1, x2, y2), kp)
        out.append({
            "raw_id": t["raw_id"], "box": (x1, y1, x2, y2), "conf": t["conf"], "kp": kp,
            "w": x2 - x1, "h": y2 - y1,
            "foot": (fx, min(fy, height)), "foot_src": src, "full_body": full,
            "head": head_point((x1, y1, x2, y2), kp),  # stays visible behind a shelf
            "torso": torso, "legs": legs,
        })
    return out


def person_row(p: dict, frame_idx: int, t_sec: float) -> dict:
    """CSV/JSON row for one person from :func:`extract_people`. ``track_id``
    starts equal to ``raw_id``; stitching rewrites it."""
    x1, y1, x2, y2 = p["box"]
    return {
        "frame": frame_idx,
        "time_s": round(t_sec, 3),
        "track_id": p["raw_id"],
        "raw_id": p["raw_id"],
        "conf": round(p["conf"], 3),
        "x1": round(x1, 1), "y1": round(y1, 1),
        "x2": round(x2, 1), "y2": round(y2, 1),
        "cx": round(x1 + p["w"] / 2, 1), "cy": round(y1 + p["h"] / 2, 1),
        "foot_x": round(p["foot"][0], 1), "foot_y": round(p["foot"][1], 1),
        "foot_src": p["foot_src"],
        "w": round(p["w"], 1), "h": round(p["h"], 1),
        "keypoints": _kp_str(p["kp"]),
    }


def detect_objects(obj_model, frame, people_here, object_conf, frame_dets=None) -> list[dict]:
    """Run the held-object detector on one frame and attribute each object to
    the person holding it. ``frame_dets`` is the YOLOX output of the same
    frame (what :class:`CocoObjects` reads; ignored by OWLv2)."""
    boxes, scores, cls = obj_model(frame, object_conf, frame_dets)
    found = []
    for (x1, y1, x2, y2), c, k in zip(boxes.tolist(), scores.tolist(), cls.tolist()):
        holder = assign_holder((x1, y1, x2, y2), people_here)
        w, h = x2 - x1, y2 - y1
        found.append({
            "label": obj_model.names[k], "conf": round(c, 3),
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "cx": round(x1 + w / 2, 1), "cy": round(y1 + h / 2, 1),
            "w": round(w, 1), "h": round(h, 1),
            "person_id": "" if holder is None else holder,
        })
    return found


def load_object_model(object_model: str, object_classes: list[str], device: str = "cpu"):
    """``yolox`` (COCO classes from the people pass, free) or ``owlv2`` /
    ``owlv2-large`` (free-text classes, slow)."""
    if object_model == "yolox":
        return CocoObjects(object_classes)
    if object_model in Owlv2Objects.HF:
        return Owlv2Objects(object_model, object_classes, device)
    raise ValueError(f"object model must be one of {OBJECT_MODELS}, not {object_model!r}")


def analyze(
    video: Path,
    model_name: str = DEFAULT_MODEL,
    conf: float = 0.35,
    stride: int | None = None,
    max_seconds: float | None = None,
    start_seconds: float = 0.0,
    preview: Path | None = None,
    out_csv: Path | None = None,
    device: str = "cpu",
    track_buffer_s: float = TRACK_BUFFER_S,
    stitch_tracks: bool = True,
    stitch_gap_s: float = 8.0,
    stitch_min_sim: float = 0.6,
    smooth: bool = True,
    smooth_window_s: float = SMOOTH_WINDOW_S,
    smooth_deadband: float = SMOOTH_DEADBAND,
    progress_every: int = 10,
    on_progress=None,
    objects: bool = True,
    object_model: str = DEFAULT_OBJECT_MODEL,
    object_classes: list[str] | None = None,
    object_conf: float = 0.25,
    objects_csv: Path | None = None,
) -> dict:
    """Detect + track people (and, optionally, what they carry). Returns a
    small stats dict.

    ``stride`` None → :func:`auto_stride` (about ``TARGET_TRACK_FPS`` analysed
    frames per second). ``model_name`` is a ``detector+pose`` spec (see
    ``detector.py``), ``track_buffer_s`` → :func:`tracker_settings`. With
    ``stitch_tracks`` the raw tracker IDs are
    merged after the run by clothing colour + position + time (``stitch.py``);
    the CSV then carries the merged id as ``track_id`` and the tracker's own id
    as ``raw_id``.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    obj_model = None
    if objects:
        object_classes = object_classes or DEFAULT_OBJECT_CLASSES
        obj_model = load_object_model(object_model, object_classes, device)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = stride or auto_stride(fps)
    tracker_cfg = tracker_settings(fps, stride, conf, track_buffer_s)
    # Let weak detections reach the tracker's second stage; the tracker itself
    # applies ``conf`` (track_high) and only ever outputs boxes that belong to
    # an established track, so the CSV stays clean.
    det_conf = min(conf, TRACK_LOW_THRESH)
    model = PeopleModel(model_name, device, det_conf, tracker_cfg)
    # the COCO-objects path reads the same YOLOX pass, so run it low enough for both
    run_conf = min(det_conf, object_conf) if isinstance(obj_model, CocoObjects) else det_conf

    if start_seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_seconds * fps))

    last_frame = total_frames
    if max_seconds is not None:
        last_frame = int((start_seconds + max_seconds) * fps)

    out_csv = out_csv or TRACK_DIR / f"{video.stem}_tracks.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    objects_csv = objects_csv or out_csv.with_name(
        out_csv.name.replace("_tracks.csv", "_objects.csv") if out_csv.name.endswith("_tracks.csv")
        else out_csv.stem + "_objects.csv")

    writer = None
    if preview:
        preview.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(preview), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height)
        )

    rows: list[dict] = []
    height_samples: list[tuple[float, float]] = []   # (head-top y, pixel height) of full bodies
    foot_src_counts: dict[str, int] = defaultdict(int)
    obj_rows: list[dict] = []
    obj_by_label: dict[str, int] = defaultdict(int)
    held_by_track: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_frame_counts: dict[int, int] = {}
    track_frames: dict[int, int] = defaultdict(int)
    frags: dict[int, Fragment] = {}                    # raw track id → summary for stitching
    frame_idx = int(start_seconds * fps)
    processed = 0
    t0 = time.time()

    print(f"video   : {video.name}  ({width}x{height}, {fps:.2f} fps, {total_frames} frames)")
    print(f"model   : {model_name}  conf={conf} (detector {det_conf}) imgsz={model.imgsz} device={device}")
    print(f"sampling: every {stride} frame(s) → {fps / stride:.2f} analysed fps")
    print(f"tracker : bytetrack {tracker_cfg}")
    if stitch_tracks:
        print(f"stitch  : gap ≤ {stitch_gap_s}s, clothing similarity ≥ {stitch_min_sim}")
    if obj_model is not None:
        print(f"objects : {object_model}  conf={object_conf}  {object_classes}")
    print()

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= last_frame:
            break

        if (frame_idx - int(start_seconds * fps)) % stride != 0:
            frame_idx += 1
            continue

        tracked, frame_dets = model(frame, run_conf)

        t_sec = frame_idx / fps
        people = extract_people(tracked, frame, height, appearance_on=stitch_tracks)
        n = len(people)
        people_here: list[tuple[int, tuple]] = [(p["raw_id"], p["box"]) for p in people]
        for p in people:
            if p["full_body"]:
                height_samples.append((p["box"][1], p["h"]))
            foot_src_counts[p["foot_src"]] += 1
            if p["torso"] is not None or p["legs"] is not None:
                frags.setdefault(p["raw_id"], Fragment(p["raw_id"])).add(
                    t_sec, p["foot"], p["h"], p["torso"], p["legs"], head=p["head"])
            rows.append(person_row(p, frame_idx, t_sec))
            track_frames[p["raw_id"]] += 1

        per_frame_counts[frame_idx] = n
        processed += 1

        found: list[dict] = []
        n_obj = 0
        if obj_model is not None:
            found = detect_objects(obj_model, frame, people_here, object_conf, frame_dets)
            for o in found:
                obj_rows.append({"frame": frame_idx, "time_s": round(t_sec, 3), **o})
                obj_by_label[o["label"]] += 1
                if o["person_id"] != "":
                    held_by_track[o["person_id"]][o["label"]] += 1
                n_obj += 1

        if writer is not None:
            writer.write(draw_objects(draw_people(frame.copy(), people), found))

        if processed % progress_every == 0:
            elapsed = time.time() - t0
            done_s = t_sec - start_seconds
            print(f"  {processed:5d} frames | video t={t_sec:7.1f}s | "
                  f"{n} people, {n_obj} objects now | {elapsed:5.1f}s elapsed "
                  f"({done_s / max(elapsed, 1e-6):.2f}x realtime)")
            if on_progress is not None:
                on_progress({
                    "frames_processed": processed,
                    "video_time_s": round(t_sec, 1),
                    "people_now": n,
                    "objects_now": n_obj,
                    "unique_tracks": len(track_frames),
                    "elapsed_s": round(elapsed, 1),
                    "percent": round(100 * (frame_idx - int(start_seconds * fps))
                                     / max(last_frame - int(start_seconds * fps), 1), 1),
                })

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Head-only detections: now that the whole video has been seen, fit this
    # camera's height-vs-y model on the full bodies and fill their feet in.
    pos_model = _fit_position_model(height_samples)
    if pos_model is not None:
        a, b = pos_model
        for r in rows:
            if r["foot_src"] == "position":
                fy = r["y1"] + a * r["y1"] + b
                r["foot_y"] = round(min(max(fy, r["y2"]), height), 1)

    # Re-join fragmented IDs: the tracker's id becomes ``raw_id``, ``track_id``
    # is the merged visitor (the earliest raw id of the chain).
    raw_track_count = len(track_frames)
    stitch_log: list[dict] = []
    id_map: dict[int, int] = {}
    if stitch_tracks and frags:
        id_map = stitch(frags, max_gap_s=stitch_gap_s, min_sim=stitch_min_sim, log=stitch_log)
    for r in rows:
        r["track_id"] = id_map.get(r["raw_id"], r["raw_id"])
    for r in obj_rows:
        if r["person_id"] != "":
            r["person_id"] = id_map.get(r["person_id"], r["person_id"])
    if id_map:
        merged_frames: dict[int, int] = defaultdict(int)
        for tid, n_frames in track_frames.items():
            merged_frames[id_map.get(tid, tid)] += n_frames
        track_frames = merged_frames
        merged_held: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for tid, d in held_by_track.items():
            for label, n_seen in d.items():
                merged_held[id_map.get(tid, tid)][label] += n_seen
        held_by_track = merged_held

    smooth_stats = (smooth_feet(rows, fps / stride, smooth_window_s, smooth_deadband)
                    if smooth else {"enabled": False})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["frame", "time_s", "track_id", "raw_id", "conf", "x1", "y1", "x2", "y2",
                      "cx", "cy", "foot_x", "foot_y", "foot_src", "w", "h", "keypoints"]
        cw = csv.DictWriter(f, fieldnames=fieldnames)
        cw.writeheader()
        cw.writerows(rows)

    if obj_model is not None:
        with objects_csv.open("w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=["frame", "time_s", "label", "conf",
                                               "x1", "y1", "x2", "y2", "cx", "cy",
                                               "w", "h", "person_id"])
            cw.writeheader()
            cw.writerows(obj_rows)

    counts = list(per_frame_counts.values()) or [0]
    stats = {
        "video": str(video),
        "csv": str(out_csv),
        "preview": str(preview) if preview else None,
        "fps": round(fps, 3),
        "stride": stride,
        "analysed_fps": round(fps / stride, 2),
        "frames_processed": processed,
        "detections": len(rows),
        "model": model_name,
        "imgsz": model.imgsz,
        # tracker settings actually used
        "tracker": {"type": "bytetrack", "detector_conf": det_conf, **tracker_cfg},
        # ids straight from the tracker, before stitching
        "raw_tracks": raw_track_count,
        # after stitching: what the CSV's track_id column holds
        "unique_tracks": len(track_frames),
        # tracks seen in at least 3 sampled frames — filters one-frame false positives
        "stable_tracks": sum(1 for v in track_frames.values() if v >= 3),
        "smooth": smooth_stats,
        "stitch": ({"enabled": True, "max_gap_s": stitch_gap_s, "min_sim": stitch_min_sim,
                    "links": len(stitch_log), "log": stitch_log}
                   if stitch_tracks else {"enabled": False}),
        "max_people_in_frame": max(counts),
        "avg_people_in_frame": round(sum(counts) / len(counts), 2),
        "seconds_elapsed": round(time.time() - t0, 1),
        # how the floor point was found: box = feet visible, the rest estimated
        "foot_source": dict(foot_src_counts),
        "feet_estimated_pct": round(100 * (len(rows) - foot_src_counts["box"]) / max(len(rows), 1), 1),
        "height_model": (None if pos_model is None else
                         {"a": round(pos_model[0], 4), "b": round(pos_model[1], 1),
                          "samples": len(height_samples)}),
    }
    if obj_model is not None:
        held = sum(1 for r in obj_rows if r["person_id"] != "")
        stats.update({
            "objects_csv": str(objects_csv),
            "object_model": object_model,
            "object_classes": object_classes,
            "object_detections": len(obj_rows),
            "objects_held": held,                       # inside someone's box
            "objects_by_label": dict(sorted(obj_by_label.items(),
                                            key=lambda kv: -kv[1])),
            # tracks seen holding something in at least 3 sampled frames
            "tracks_with_objects": sum(1 for d in held_by_track.values()
                                       if sum(d.values()) >= 3),
        })

    print("\n── summary ──")
    for k, v in stats.items():
        print(f"{k:22s}: {v}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path, help="path to video file")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"detector+pose, e.g. {DEFAULT_MODEL} (default; keypoints → feet estimated "
                         "when hidden) / yolox_tiny+rtmpose-s (fast) / yolox_m+rtmpose-m (accurate) "
                         "/ yolox_s (boxes only)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=None,
                    help=f"process every Nth frame (default: auto, ≈{TARGET_TRACK_FPS:.0f} analysed fps)")
    ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    ap.add_argument("--seconds", type=float, default=None, help="how many seconds to analyse")
    ap.add_argument("--preview", type=Path, default=None, help="write annotated mp4 here")
    ap.add_argument("--out", type=Path, default=None, help="output tracks CSV")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda (needs onnxruntime-gpu)")
    ap.add_argument("--track-buffer", type=float, default=TRACK_BUFFER_S,
                    help="seconds a lost track is kept alive inside the tracker")
    ap.add_argument("--no-stitch", action="store_true",
                    help="keep the tracker's raw ids; skip the clothing/position stitching pass")
    ap.add_argument("--stitch-gap", type=float, default=8.0,
                    help="max seconds between a fragment ending and its continuation starting")
    ap.add_argument("--stitch-sim", type=float, default=0.6,
                    help="min clothing-colour similarity (0–1) to join two fragments")
    ap.add_argument("--no-smooth", action="store_true",
                    help="keep the raw per-frame floor points (no median / dead-band)")
    ap.add_argument("--smooth-window", type=float, default=SMOOTH_WINDOW_S,
                    help=f"rolling-median window in seconds (default {SMOOTH_WINDOW_S})")
    ap.add_argument("--smooth-deadband", type=float, default=SMOOTH_DEADBAND,
                    help=f"hold the point while it moves < this x box height (default {SMOOTH_DEADBAND})")
    ap.add_argument("--no-objects", action="store_true",
                    help="skip the held-object stage")
    ap.add_argument("--object-model", default=DEFAULT_OBJECT_MODEL, choices=OBJECT_MODELS,
                    help="yolox (default: COCO classes from the people pass, free) / "
                         "owlv2, owlv2-large (free-text classes, seconds per frame on a CPU)")
    ap.add_argument("--object-classes", default=",".join(DEFAULT_OBJECT_CLASSES),
                    help="comma-separated classes: COCO names for yolox, free text for owlv2")
    ap.add_argument("--object-conf", type=float, default=0.25)
    args = ap.parse_args()

    analyze(
        video=args.video, model_name=args.model, conf=args.conf, stride=args.stride, start_seconds=args.start, max_seconds=args.seconds,
        preview=args.preview, out_csv=args.out, device=args.device,
        track_buffer_s=args.track_buffer,
        stitch_tracks=not args.no_stitch, stitch_gap_s=args.stitch_gap,
        stitch_min_sim=args.stitch_sim,
        smooth=not args.no_smooth, smooth_window_s=args.smooth_window,
        smooth_deadband=args.smooth_deadband,
        objects=not args.no_objects, object_model=args.object_model,
        object_classes=[c.strip() for c in args.object_classes.split(",") if c.strip()],
        object_conf=args.object_conf,
    )


if __name__ == "__main__":
    main()
