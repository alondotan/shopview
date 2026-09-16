"""Track stitching — re-join fragmented track IDs into one visitor.

The tracker (ByteTrack / BoT-SORT) restarts an ID whenever it loses someone
for longer than its buffer: behind a shelf, behind another shopper, or simply
because a low-confidence frame broke the chain. This module runs *after* the
video has been processed and merges fragments that are plausibly the same
person, using three cues:

* **time** — fragment B starts shortly after fragment A ends (no overlap);
* **position** — B starts about where A was heading (A's last floor point,
  extrapolated with its velocity, allowing walking speed for the gap);
* **clothes** — the colour of A's torso/legs matches B's;
* **head continuity** — for a very short gap, B's head starts where A's head
  was heading; then the clothes only have to not contradict (see
  :func:`link_cost`, "quick re-appearance").

The clothing descriptor (:func:`appearance`) is an HSV histogram of the torso
(between the shoulders and hips, from the pose keypoints, or the upper-middle of
the box when there are no keypoints) plus one of the legs when the hips and
knees are visible. It is deliberately simple — no extra network, runs in well
under a millisecond per person — and it is what a human does when re-finding
someone on CCTV: "the one in the red top".

:func:`stitch` does a greedy assignment over all (A ends → B starts) pairs
ordered by a combined cost, so each fragment gets at most one predecessor and
one successor, and returns ``raw_id → merged_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ── clothing descriptor ───────────────────────────────────────────────────
# Pixels are split into *chromatic* (a real colour: red top, blue jacket) and
# *achromatic* (black, grey, white — most clothing on CCTV). Chromatic pixels go
# into hue×saturation bins; achromatic ones into brightness bins only, because
# their hue is noise. Without this split a black coat and a white hoodie look
# alike: both spread their (meaningless) hues over every bin. One histogram of
# unit mass, so "how colourful" counts too.
H_BINS, S_BINS, V_BINS = 12, 3, 8
SAT_MIN = 50                                    # below: achromatic (0–255 scale)
FEAT_LEN = H_BINS * S_BINS + V_BINS
KP_LSH, KP_RSH, KP_LHIP, KP_RHIP, KP_LKNEE, KP_RKNEE = 5, 6, 11, 12, 13, 14
from detector import KP_VISIBLE  # noqa: E402  (0.3 — see detector.py)
MIN_REGION_PX = 6                               # skip regions thinner than this


def _region_from_kp(kp, top_ids, bottom_ids, widen=0.25):
    """Rectangle spanning two keypoint pairs (e.g. shoulders → hips), widened a
    bit sideways so a turned body still fills it. None if either pair is hidden."""
    top = [kp[i] for i in top_ids if kp[i][2] >= KP_VISIBLE]
    bot = [kp[i] for i in bottom_ids if kp[i][2] >= KP_VISIBLE]
    if not top or not bot:
        return None
    xs = [p[0] for p in top + bot]
    y1 = sum(p[1] for p in top) / len(top)
    y2 = sum(p[1] for p in bot) / len(bot)
    if y2 - y1 < MIN_REGION_PX:
        return None
    w = max(max(xs) - min(xs), (y2 - y1) * 0.5)
    cx = (max(xs) + min(xs)) / 2
    return (cx - w * (0.5 + widen), y1, cx + w * (0.5 + widen), y2)


def _hist(hsv_crop) -> np.ndarray | None:
    """Unit-mass [hue×sat of colourful pixels | brightness of grey pixels]
    histogram of an HSV crop, or None if the crop is empty."""
    if hsv_crop is None or hsv_crop.size == 0 or min(hsv_crop.shape[:2]) < MIN_REGION_PX:
        return None
    chroma = (hsv_crop[:, :, 1] >= SAT_MIN).astype(np.uint8)
    hs = cv2.calcHist([hsv_crop], [0, 1], chroma, [H_BINS, S_BINS],
                      [0, 180, SAT_MIN, 256]).ravel()
    v = cv2.calcHist([hsv_crop], [2], 1 - chroma, [V_BINS], [0, 256]).ravel()
    feat = np.concatenate([hs, v]).astype(np.float32)
    return feat / max(feat.sum(), 1e-6)


def _crop(frame_hsv, region):
    H, W = frame_hsv.shape[:2]
    x1, y1, x2, y2 = region
    x1, x2 = int(max(0, x1)), int(min(W, x2))
    y1, y2 = int(max(0, y1)), int(min(H, y2))
    if x2 - x1 < MIN_REGION_PX or y2 - y1 < MIN_REGION_PX:
        return None
    return frame_hsv[y1:y2, x1:x2]


def appearance(frame_hsv, box, kp) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(torso, legs) clothing histograms for one detection. ``frame_hsv`` is the
    full frame already converted to HSV; ``box`` (x1,y1,x2,y2); ``kp`` 17 COCO
    keypoints as (x,y,conf) or None. Either part may be None when not visible."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    torso_r = legs_r = None
    if kp is not None:
        torso_r = _region_from_kp(kp, (KP_LSH, KP_RSH), (KP_LHIP, KP_RHIP))
        legs_r = _region_from_kp(kp, (KP_LHIP, KP_RHIP), (KP_LKNEE, KP_RKNEE), widen=0.1)
    if torso_r is None:
        # no usable keypoints: upper-middle of the box is almost always the torso
        torso_r = (x1 + 0.2 * w, y1 + 0.15 * h, x2 - 0.2 * w, y1 + 0.5 * h)
    else:
        # the box is the visible extent: when a shelf hides the hips the pose
        # model still guesses them, below the box — that strip is shelf, not shirt
        torso_r = (torso_r[0], max(torso_r[1], y1), torso_r[2], min(torso_r[3], y2))
    return _hist(_crop(frame_hsv, torso_r)), _hist(_crop(frame_hsv, legs_r)) if legs_r else None


def head_point(box, kp) -> tuple[float, float]:
    """Where the head is: x from the visible face keypoints (nose, eyes, ears),
    y the top of the box. The box centre would do for x, but it shifts by half
    a body width when a full-body box becomes a head-and-shoulders one — the
    face does not. The head is the part a shelf hides last, so it is the anchor
    for the quick re-appearance rule in :func:`link_cost`."""
    x1, y1, x2, _ = box
    face = [p[0] for p in (kp[:5] if kp else []) if p[2] >= KP_VISIBLE]
    return (sum(face) / len(face) if face else (x1 + x2) / 2), y1


# Weight of the learned embedding (when both fragments have one) against the
# colour histogram. Cosine similarity of ImageNet features is high for any two
# people, so it is rescaled from [EMB_COS_LO, 1] to [0, 1] first.
EMB_WEIGHT = 0.5
EMB_COS_LO = 0.5


def similarity(a: "Fragment", b: "Fragment") -> float:
    """Clothing similarity in [0, 1]: histogram intersection of the torso, blended
    with the legs when both fragments have seen them, and with the learned
    embedding when both carry one (see :class:`Embedder`)."""
    if a.torso is None or b.torso is None:
        return 0.0
    s = float(np.minimum(a.torso, b.torso).sum())
    if a.legs is not None and b.legs is not None:
        s_legs = float(np.minimum(a.legs, b.legs).sum())
        s = 0.7 * s + 0.3 * s_legs
    ea, eb = a.emb, b.emb
    if ea is not None and eb is not None:
        cos = float(np.dot(ea, eb))
        e = min(1.0, max(0.0, (cos - EMB_COS_LO) / (1 - EMB_COS_LO)))
        s = (1 - EMB_WEIGHT) * s + EMB_WEIGHT * e
    return s


class Embedder:
    """Learned appearance vector per person crop — OpenAI CLIP's image tower
    (``openai/clip-vit-base-patch32``, MIT, through ``transformers``) run on
    the torso region, ~20 ms per crop on a CPU. Optional: the colour histogram
    alone is the default; this is the upgrade path for crowded scenes where
    colour is not enough. Needs ``pip install -r vision/requirements-owl.txt``.
    Returns L2-normalised vectors, so their dot product is the cosine."""

    HF = {"clip": "openai/clip-vit-base-patch32", "clip-large": "openai/clip-vit-large-patch14"}

    def __init__(self, model: str = "clip", device: str = "cpu"):
        try:
            import torch
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError as e:
            raise RuntimeError("the CLIP embedder needs torch + transformers: "
                               "pip install -r vision/requirements-owl.txt") from e
        name = self.HF.get(model, model)
        self.torch = torch
        self.processor = CLIPImageProcessor.from_pretrained(name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(name).eval().to(device)
        self.device = device

    def __call__(self, frame_bgr, boxes: list[tuple], kps: list | None = None) -> list:
        crops, idx = [], []
        H, W = frame_bgr.shape[:2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            r = None
            if kps is not None and kps[i] is not None:
                r = _region_from_kp(kps[i], (KP_LSH, KP_RSH), (KP_LHIP, KP_RHIP))
            if r is None:
                r = (x1 + 0.15 * w, y1 + 0.1 * h, x2 - 0.15 * w, y1 + 0.6 * h)
            cx1, cy1 = int(max(0, r[0])), int(max(0, r[1]))
            cx2, cy2 = int(min(W, r[2])), int(min(H, r[3]))
            if cx2 - cx1 < MIN_REGION_PX or cy2 - cy1 < MIN_REGION_PX:
                continue
            crops.append(frame_bgr[cy1:cy2, cx1:cx2])
            idx.append(i)
        out: list = [None] * len(boxes)
        if not crops:
            return out
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self.model(**inputs).image_embeds.cpu().numpy().astype(np.float32)
        for i, f in zip(idx, feats):
            out[i] = f / max(float(np.linalg.norm(f)), 1e-6)
        return out


# ── fragments ─────────────────────────────────────────────────────────────
@dataclass
class Fragment:
    """One raw track ID, summarised for stitching."""
    id: int
    t: list[float] = field(default_factory=list)          # sampled times (s)
    foot: list[tuple[float, float]] = field(default_factory=list)
    head: list[tuple[float, float] | None] = field(default_factory=list)  # see head_point()
    heights: list[float] = field(default_factory=list)
    torso_sum: np.ndarray | None = None
    torso_n: int = 0
    legs_sum: np.ndarray | None = None
    legs_n: int = 0
    emb_sum: np.ndarray | None = None
    emb_n: int = 0

    def add(self, t, foot, h, torso, legs, emb=None, head=None):
        self.t.append(t)
        self.foot.append(foot)
        self.head.append(head)
        self.heights.append(h)
        if torso is not None:
            self.torso_sum = torso if self.torso_sum is None else self.torso_sum + torso
            self.torso_n += 1
        if legs is not None:
            self.legs_sum = legs if self.legs_sum is None else self.legs_sum + legs
            self.legs_n += 1
        if emb is not None:
            self.emb_sum = emb if self.emb_sum is None else self.emb_sum + emb
            self.emb_n += 1

    @property
    def emb(self):
        if self.emb_n == 0:
            return None
        v = self.emb_sum / self.emb_n
        return v / max(float(np.linalg.norm(v)), 1e-6)

    @property
    def torso(self):
        return None if self.torso_n == 0 else self.torso_sum / self.torso_n

    @property
    def legs(self):
        return None if self.legs_n == 0 else self.legs_sum / self.legs_n

    @property
    def t0(self): return self.t[0]

    @property
    def t1(self): return self.t[-1]

    @property
    def n(self): return len(self.t)

    @property
    def height(self) -> float:
        return float(np.median(self.heights))

    def velocity(self, window_s: float = 1.5) -> tuple[float, float]:
        """Mean floor-point velocity (px/s) over the last ``window_s`` seconds."""
        return self._velocity(self.foot, window_s)

    def head_velocity(self, window_s: float = 1.0) -> tuple[float, float] | None:
        """Mean head-point velocity (px/s) over the last ``window_s``
        seconds; None when the heads were not recorded."""
        if self.head[-1] is None:
            return None
        return self._velocity(self.head, window_s)

    def _velocity(self, pts, window_s: float) -> tuple[float, float]:
        i = len(self.t) - 1
        while i > 0 and self.t[-1] - self.t[i - 1] <= window_s and pts[i - 1] is not None:
            i -= 1
        dt = self.t[-1] - self.t[i]
        if dt <= 0:
            return 0.0, 0.0
        return ((pts[-1][0] - pts[i][0]) / dt, (pts[-1][1] - pts[i][1]) / dt)


# ── stitching ─────────────────────────────────────────────────────────────
# Walking speed in body heights per second: 1.4 m/s over ~1.7 m ≈ 0.8. The
# allowed start-of-B distance grows with the gap so a person can walk across
# the occluded stretch, capped so a long gap does not accept anyone in the shop.
WALK_HEIGHTS_PER_S = 0.8
BASE_SLACK_HEIGHTS = 0.6
MAX_SLACK_HEIGHTS = 3.5
EXTRAPOLATE_MAX_S = 1.0     # trust A's velocity for at most this long

# Quick re-appearance. The commonest break is a shelf: the legs go, the box
# shrinks to head and shoulders, the tracker's IoU match fails and a new id is
# born a few frames later — with the head still where it was heading. The
# clothing descriptor is at its worst just then (torso half hidden, background
# in the crop), so it is not asked to *prove* the match, only not to contradict
# it. The rule is deliberately tight on time and head position: in a crowded
# till a different person can appear at the same spot a second or two later.
QUICK_GAP_S = 1.0           # B starts this soon after A ends …
QUICK_HEAD_HEIGHTS = 0.3    # … with its head this close (in body heights) to A's …
QUICK_FOOT_FRACTION = 0.7   # … and its feet well inside the walking slack, not at its edge
QUICK_MIN_SIM = 0.35        # then the clothes need only this much similarity


def head_distance(a: Fragment, b: Fragment, gap: float) -> float | None:
    """Distance (px) from A's last head, carried forward with its velocity
    over the gap, to B's first head; None when heads were not recorded."""
    ha, hb = a.head[-1], b.head[0]
    v = a.head_velocity()
    if ha is None or hb is None or v is None:
        return None
    dt = min(gap, EXTRAPOLATE_MAX_S)
    return ((ha[0] + v[0] * dt - hb[0]) ** 2 + (ha[1] + v[1] * dt - hb[1]) ** 2) ** 0.5


def link_cost(a: Fragment, b: Fragment, max_gap_s: float, min_sim: float):
    """Can fragment B be the continuation of fragment A? Returns (cost, info)
    or None. Same gate for the offline pass and the live stitcher: B starts
    after A ends, within ``max_gap_s``; about where A was heading (last floor
    point + velocity, plus walking-speed slack growing with the gap); clothes
    at least ``min_sim`` alike — or, for a *quick re-appearance* (short gap,
    head continuous), at least ``QUICK_MIN_SIM`` alike."""
    gap = b.t0 - a.t1
    if gap <= 0 or gap > max_gap_s:
        return None
    h = (a.height + b.height) / 2
    vx, vy = a.velocity()
    dt = min(gap, EXTRAPOLATE_MAX_S)
    px, py = a.foot[-1][0] + vx * dt, a.foot[-1][1] + vy * dt
    d = ((px - b.foot[0][0]) ** 2 + (py - b.foot[0][1]) ** 2) ** 0.5
    allowed = h * min(BASE_SLACK_HEIGHTS + WALK_HEIGHTS_PER_S * gap, MAX_SLACK_HEIGHTS)
    if d > allowed:
        return None
    hd = head_distance(a, b, gap)
    quick = (hd is not None and gap <= QUICK_GAP_S and hd <= QUICK_HEAD_HEIGHTS * h
             and d <= QUICK_FOOT_FRACTION * allowed)
    sim = similarity(a, b)
    if sim < (min(min_sim, QUICK_MIN_SIM) if quick else min_sim):
        return None
    cost = 0.45 * (d / allowed) + 0.45 * (1 - sim) + 0.10 * (gap / max_gap_s)
    info = {"from": a.id, "to": b.id, "gap_s": round(gap, 2),
            "dist_px": round(d, 1), "allowed_px": round(allowed, 1),
            "sim": round(sim, 3), "cost": round(cost, 3)}
    if hd is not None:
        info["head_px"] = round(hd, 1)
    if quick:
        info["quick"] = True
    return cost, info


def stitch(frags: dict[int, Fragment], max_gap_s: float = 8.0, min_sim: float = 0.6,
           log: list | None = None) -> dict[int, int]:
    """Offline greedy fragment linking over a whole video. Returns
    raw_id → merged_id (the earliest id in the chain). ``log`` collects one
    dict per accepted link, for the stats."""
    ids = sorted(frags, key=lambda i: frags[i].t0)
    cands: list[tuple[float, int, int, dict]] = []
    for ia, a_id in enumerate(ids):
        a = frags[a_id]
        for b_id in ids[ia + 1:]:
            b = frags[b_id]
            if b.t0 - a.t1 > max_gap_s:
                break                     # ids are sorted by t0; nothing later fits either
            r = link_cost(a, b, max_gap_s, min_sim)
            if r is not None:
                cands.append((r[0], a_id, b_id, r[1]))

    cands.sort(key=lambda c: c[0])
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for _, a_id, b_id, info in cands:
        if a_id in succ or b_id in pred:
            continue
        succ[a_id] = b_id
        pred[b_id] = a_id
        if log is not None:
            log.append(info)

    merged: dict[int, int] = {}
    for i in ids:
        if i in pred:
            continue
        root, cur = i, i
        while True:
            merged[cur] = root
            if cur not in succ:
                break
            cur = succ[cur]
    return merged


class OnlineStitcher:
    """The same linking, frame by frame, for a live stream.

    Every raw tracker id becomes a :class:`Fragment` that keeps growing while
    the id is seen. When an id has not been seen for ``lost_after_s`` it moves
    to the *lost pool*, where it stays for ``max_gap_s``. A *new* id is not
    judged at birth (one frame of clothing is not enough): it is *pending* for
    ``decide_after_s``, then matched against the pool with :func:`link_cost`,
    taking the cheapest candidate that is not already continued by someone
    else. If the pool fragment comes back to life meanwhile (the tracker
    re-activated it), it is no longer a candidate — the two co-exist.

    Consumers therefore see a stable mapping only for rows older than
    ``decide_after_s``; :meth:`release` hands out rows once they are that old,
    already carrying the final ``track_id``. Merges are reported as events so a
    consumer that cannot wait can re-key instead.
    """

    def __init__(self, max_gap_s: float = 8.0, min_sim: float = 0.6,
                 decide_after_s: float = 2.0, lost_after_s: float = 0.5):
        self.max_gap_s = max_gap_s
        self.min_sim = min_sim
        self.decide_after_s = decide_after_s
        self.lost_after_s = lost_after_s
        self.frags: dict[int, Fragment] = {}
        self.last_seen: dict[int, float] = {}
        self.pending: dict[int, float] = {}          # new id → birth time
        self.succ: dict[int, int] = {}
        self.pred: dict[int, int] = {}
        self.root: dict[int, int] = {}               # raw id → visitor id
        self.merges: list[dict] = []                 # every accepted link
        self._buffer: list[dict] = []                # rows waiting for their id to settle

    # ── per frame ──
    def observe(self, raw_id: int, t: float, foot, h, torso, legs, emb=None, head=None) -> None:
        f = self.frags.get(raw_id)
        if f is None:
            f = self.frags[raw_id] = Fragment(raw_id)
            self.pending[raw_id] = t
            self.root[raw_id] = raw_id
        f.add(t, foot, h, torso, legs, emb, head)
        self.last_seen[raw_id] = t

    def step(self, t: float) -> list[dict]:
        """Call once per frame after all ``observe`` calls; returns the merge
        events decided at this step."""
        events = []
        for b_id, born in list(self.pending.items()):
            b = self.frags[b_id]
            ended = t - self.last_seen[b_id] > self.lost_after_s
            if t - born < self.decide_after_s and not ended:
                continue
            del self.pending[b_id]
            best = None
            for a_id, a in self.frags.items():
                if a_id == b_id or a_id in self.succ or a_id in self.pending:
                    continue
                if t - self.last_seen[a_id] <= self.lost_after_s:
                    continue                          # still alive → not the same person
                if a.t1 >= b.t0:
                    continue
                r = link_cost(a, b, self.max_gap_s, self.min_sim)
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], a_id, r[1])
            if best is not None:
                _, a_id, info = best
                self.succ[a_id] = b_id
                self.pred[b_id] = a_id
                root = self.root[a_id]
                self.root[b_id] = root
                info = {**info, "visitor": root, "t": round(t, 3)}
                self.merges.append(info)
                events.append(info)
        # forget fragments that can no longer be continued
        for a_id in [i for i, ls in self.last_seen.items()
                     if t - ls > self.max_gap_s + self.lost_after_s and i not in self.pending]:
            self.frags.pop(a_id, None)
            self.last_seen.pop(a_id, None)
        return events

    def visitor(self, raw_id: int) -> int:
        return self.root.get(raw_id, raw_id)

    # ── delayed output ──
    def push(self, row: dict) -> None:
        """Queue a per-detection row (must carry ``raw_id`` and ``time_s``)."""
        self._buffer.append(row)

    def release(self, t: float, flush: bool = False) -> list[dict]:
        """Rows whose ids have settled (older than ``decide_after_s``), with
        ``track_id`` filled in. ``flush`` releases everything (end of stream)."""
        out, keep = [], []
        for r in self._buffer:
            if flush or t - r["time_s"] >= self.decide_after_s:
                r["track_id"] = self.visitor(r["raw_id"])
                out.append(r)
            else:
                keep.append(r)
        self._buffer = keep
        return out


# ── offline: re-stitch an existing tracks CSV ─────────────────────────────
def fragments_from_csv(csv_path, video_path, embedder: "Embedder | None" = None
                       ) -> tuple[list[dict], dict[int, Fragment]]:
    """Rebuild the per-raw-id fragments from a tracks CSV, re-reading the video
    frames it references for the clothing descriptor. Rows use ``raw_id`` when
    present (a CSV that was stitched before), otherwise ``track_id``."""
    import csv
    from pathlib import Path

    rows = list(csv.DictReader(Path(csv_path).open(encoding="utf-8")))
    key = "raw_id" if rows and "raw_id" in rows[0] else "track_id"
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame"]), []).append(r)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video_path}")
    frags: dict[int, Fragment] = {}
    frame_idx, want = 0, sorted(by_frame)
    wi = 0
    while wi < len(want):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx == want[wi]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            group = by_frame[frame_idx]
            boxes, kps = [], []
            for r in group:
                boxes.append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
                kps.append([tuple(map(float, s.split(":"))) for s in r["keypoints"].split(";")]
                           if r.get("keypoints") else None)
            embs = embedder(frame, boxes, kps) if embedder is not None else [None] * len(group)
            for r, box, kp, emb in zip(group, boxes, kps, embs):
                torso, legs = appearance(hsv, box, kp)
                frags.setdefault(int(r[key]), Fragment(int(r[key]))).add(
                    float(r["time_s"]), (float(r["foot_x"]), float(r["foot_y"])),
                    float(r["h"]), torso, legs, emb, head=head_point(box, kp))
            wi += 1
        frame_idx += 1
    cap.release()
    for r in rows:
        r["raw_id"] = int(r[key])
    return rows, frags


def restitch(csv_path, video_path, max_gap_s=8.0, min_sim=0.6, out_csv=None,
             objects_csv=None, embed: str | None = None) -> dict:
    """Re-run the stitching pass on a tracks CSV with new parameters, without
    re-running detection. Rewrites ``track_id`` (keeping ``raw_id``), and
    ``person_id`` in the matching objects CSV when given."""
    import csv
    from pathlib import Path

    rows, frags = fragments_from_csv(csv_path, video_path,
                                     Embedder(embed) if embed else None)
    log: list[dict] = []
    id_map = stitch(frags, max_gap_s=max_gap_s, min_sim=min_sim, log=log)
    for r in rows:
        r["track_id"] = id_map.get(r["raw_id"], r["raw_id"])
    out = Path(out_csv or csv_path)
    fields = list(rows[0].keys()) if rows else ["frame", "time_s", "track_id", "raw_id"]
    if "raw_id" not in fields:
        fields.insert(fields.index("track_id") + 1, "raw_id")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if objects_csv and Path(objects_csv).exists():
        orows = list(csv.DictReader(Path(objects_csv).open(encoding="utf-8")))
        for r in orows:
            if r.get("person_id"):
                r["person_id"] = id_map.get(int(r["person_id"]), int(r["person_id"]))
        with Path(objects_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(orows[0].keys()) if orows else ["person_id"])
            w.writeheader()
            w.writerows(orows)
    return {"raw_tracks": len(frags), "tracks": len(set(id_map.values())),
            "links": len(log), "log": log, "csv": str(out)}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Re-stitch an existing tracks CSV by clothes + position + time.")
    ap.add_argument("csv")
    ap.add_argument("video")
    ap.add_argument("--gap", type=float, default=8.0, help="max seconds between fragments")
    ap.add_argument("--sim", type=float, default=0.6, help="min clothing similarity 0–1")
    ap.add_argument("--out", default=None, help="write here instead of overwriting the CSV")
    ap.add_argument("--objects", default=None, help="objects CSV whose person_id to remap")
    ap.add_argument("--embed", default=None,
                    help="also use a learned embedding: clip (slower, for crowded scenes; needs requirements-owl.txt)")
    a = ap.parse_args()
    res = restitch(a.csv, a.video, a.gap, a.sim, a.out, a.objects, a.embed)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1))
    for l in res["log"]:
        print(f"  {l['from']:4d} -> {l['to']:4d}  gap {l['gap_s']:5.2f}s  dist {l['dist_px']:6.1f}px  "
              f"sim {l['sim']:.2f}{'  quick' if l.get('quick') else ''}")
