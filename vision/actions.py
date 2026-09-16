"""Stage — what did each visitor *do* at the fixtures (fridge, shelf, counter)?

Input: the tracks CSV of a video and a fixtures file — polygons drawn in
**camera pixels** (a fridge door is a vertical surface, so it cannot live on
the floor plan like the zones do):

    data/fixtures/<video_id>.json
    {"fixtures": [{"id": "fridge", "name": "drinks fridge", "pts": [[x, y], …]}, …]}

Two steps:

1. **Contact segments** — per visitor, the frames in which a wrist keypoint
   lies inside a fixture polygon, or (for a small / far-away person whose
   wrists the pose model cannot place) the person's box mostly overlaps it.
   Contacts closer than ``gap_s`` are merged; segments shorter than ``min_s``
   are dropped. This is cheap and runs on every frame.
2. **Description** — for each segment, a handful of crops of the fixture with
   the person (before, during, after) go to Claude with a fixed JSON schema:
   which actions happened (opened door, took item, returned item, closed
   door, browsed…), what item, a one-line summary. One API call per segment,
   not per frame, so a whole video costs a few dozen calls.

Output: ``data/tracks/<video_id>_actions.csv`` —

    track_id,fixture,t_start,t_end,duration_s,took_item,actions,summary,confidence

``actions`` is JSON: ``[{"t": 33.2, "action": "opened_door", "item": ""}, …]``.

    python vision/actions.py KMJS66jBtVQ                          # all visitors
    python vision/actions.py --track 11 -- -1bRhYjw1qE            # one, for a look
    python vision/actions.py --no-describe -- -1bRhYjw1qE         # segments only, no API

(an id that starts with ``-`` goes after ``--``, or argparse reads it as a flag)

Needs ``ANTHROPIC_API_KEY`` in ``.env`` (same as the chat tab).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detector import KP_VISIBLE

ROOT = Path(__file__).resolve().parent.parent
TRACK_DIR = ROOT / "data" / "tracks"
VIDEO_DIR = ROOT / "data" / "videos"
FIXTURE_DIR = ROOT / "data" / "fixtures"

KP_LWRIST, KP_RWRIST = 9, 10
BOX_OVERLAP_MIN = 0.5      # fraction of the person box inside the fixture → contact (no wrists)
MODEL = "claude-opus-5"

ACTIONS = ["opened_door", "closed_door", "took_item", "returned_item", "touched_item",
           "browsed", "paid", "nothing"]
SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "number", "description": "timestamp in seconds, from the frame labels"},
                    "action": {"type": "string", "enum": ACTIONS},
                    "item": {"type": "string", "description": "what was taken/returned/touched, '' if none or unclear"},
                },
                "required": ["t", "action", "item"],
                "additionalProperties": False,
            },
        },
        "took_item": {"type": "boolean"},
        "summary": {"type": "string", "description": "one sentence, past tense, what this person did here"},
        "confidence": {"type": "number", "description": "0-1, how sure you are of the summary"},
    },
    "required": ["actions", "took_item", "summary", "confidence"],
    "additionalProperties": False,
}


# ── fixtures ──────────────────────────────────────────────────────────────
def load_fixtures(video_id: str, path: Path | None = None) -> list[dict]:
    path = path or FIXTURE_DIR / f"{video_id}.json"
    if not path.exists():
        raise SystemExit(f"no fixtures file: {path} — draw the fixture polygons (camera pixels) first")
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in data["fixtures"]:
        pts = np.array(f["pts"], dtype=np.float32)
        out.append({"id": f["id"], "name": f.get("name", f["id"]), "pts": pts,
                    "bbox": (float(pts[:, 0].min()), float(pts[:, 1].min()),
                             float(pts[:, 0].max()), float(pts[:, 1].max()))})
    return out


def _inside(pts: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def _box_overlap(pts: np.ndarray, box, frame_shape) -> float:
    """Fraction of the person box's area that lies inside the polygon."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_shape[1]), min(y2, frame_shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(mask, [np.round(pts - [x1, y1]).astype(np.int32)], 1)
    return float(mask.mean())


# ── contact segments ──────────────────────────────────────────────────────
def load_tracks(csv_path: Path) -> dict[int, list[dict]]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            kp = None
            if r.get("keypoints"):
                kp = [tuple(float(v) for v in p.split(":")) for p in r["keypoints"].split(";")]
            by_track[int(r["track_id"])].append({
                "t": float(r["time_s"]), "frame": int(r["frame"]),
                "box": (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])), "kp": kp,
            })
    for rows in by_track.values():
        rows.sort(key=lambda r: r["t"])
    return by_track


def contact(row: dict, fixture: dict, frame_shape) -> str | None:
    """'wrist' / 'box' when this detection touches the fixture, else None."""
    kp = row["kp"]
    if kp:
        for i in (KP_LWRIST, KP_RWRIST):
            if kp[i][2] >= KP_VISIBLE and _inside(fixture["pts"], kp[i][0], kp[i][1]):
                return "wrist"
    if _box_overlap(fixture["pts"], row["box"], frame_shape) >= BOX_OVERLAP_MIN:
        return "box"
    return None


def find_segments(by_track: dict[int, list[dict]], fixtures: list[dict], frame_shape,
                  min_s: float = 1.0, gap_s: float = 1.0) -> list[dict]:
    segs = []
    for tid, rows in by_track.items():
        for fx in fixtures:
            cur = None
            for r in rows:
                how = contact(r, fx, frame_shape)
                if how is None:
                    continue
                if cur is not None and r["t"] - cur["t_end"] <= gap_s:
                    cur["t_end"] = r["t"]
                    cur["rows"].append(r)
                    cur["wrist"] += how == "wrist"
                else:
                    if cur is not None:
                        segs.append(cur)
                    cur = {"track_id": tid, "fixture": fx["id"], "fixture_name": fx["name"],
                           "t_start": r["t"], "t_end": r["t"], "rows": [r], "wrist": int(how == "wrist")}
            if cur is not None:
                segs.append(cur)
    segs = [s for s in segs if s["t_end"] - s["t_start"] >= min_s]
    segs.sort(key=lambda s: (s["t_start"], s["track_id"]))
    return segs


# ── crops for the model ───────────────────────────────────────────────────
def _frame_at(cap, fps: float, t: float):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, img = cap.read()
    return img if ok else None


def _nearest_box(seg: dict, t: float):
    r = min(seg["rows"], key=lambda r: abs(r["t"] - t))
    return r["box"] if abs(r["t"] - t) <= 0.6 else None


def segment_crops(cap, fps: float, seg: dict, fixture: dict, n_during: int = 6,
                  pad_s: float = 1.2, min_side: int = 640) -> list[tuple[float, np.ndarray]]:
    """(t, jpeg-ready BGR crop) for before / during / after the segment: the
    fixture plus the person's boxes over the segment, padded, and upscaled so
    a far-away person is not a 40-px smudge. The person is outlined."""
    t0, t1 = seg["t_start"], seg["t_end"]
    ts = [t0 - pad_s] + [t0 + (t1 - t0) * k / (n_during - 1) for k in range(n_during)] + [t1 + pad_s]
    x1, y1, x2, y2 = fixture["bbox"]
    for r in seg["rows"]:
        x1, y1 = min(x1, r["box"][0]), min(y1, r["box"][1])
        x2, y2 = max(x2, r["box"][2]), max(y2, r["box"][3])
    w, h = x2 - x1, y2 - y1
    x1, y1, x2, y2 = x1 - 0.15 * w, y1 - 0.15 * h, x2 + 0.15 * w, y2 + 0.15 * h
    out = []
    for t in ts:
        img = _frame_at(cap, fps, t)
        if img is None:
            continue
        H, W = img.shape[:2]
        cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
        crop = img[cy1:cy2, cx1:cx2].copy()
        box = _nearest_box(seg, t)
        if box is not None:
            bx1, by1, bx2, by2 = (int(v) for v in box)
            cv2.rectangle(crop, (bx1 - cx1, by1 - cy1), (bx2 - cx1, by2 - cy1), (0, 255, 0), 1)
        scale = max(1.0, min_side / max(1, min(crop.shape[:2])))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"t={t:.1f}s", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out.append((round(t, 1), crop))
    return out


def _b64_jpeg(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


# ── Claude ────────────────────────────────────────────────────────────────
PROMPT = """These are crops from a shop's CCTV camera showing the fixture "{name}" ({kind}), \
in time order, each labelled with its timestamp. The person of interest is outlined in green \
(the outline may be missing in the first and last frame, before they arrive / after they leave). \
Other people may appear; ignore them.

Describe what the outlined person did with this fixture between t={t0}s and t={t1}s: \
did they open or close a door, take an item out, put an item back, just look? \
Name the item if you can see it (bottle, can, snack…). If you cannot tell, say so with a low \
confidence rather than guessing. Use the frame timestamps for the times."""


def describe_segment(client, crops: list[tuple[float, np.ndarray]], seg: dict, fixture: dict,
                     model: str = MODEL) -> dict:
    content = []
    for t, img in crops:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": _b64_jpeg(img)}})
    content.append({"type": "text", "text": PROMPT.format(
        name=fixture["name"], kind=fixture.get("kind", "fixture"),
        t0=round(seg["t_start"], 1), t1=round(seg["t_end"], 1))})
    response = client.beta.messages.create(
        model=model,
        max_tokens=2048,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        return {"actions": [], "took_item": False, "summary": "(refused)", "confidence": 0.0}
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    data["usage"] = {"in": response.usage.input_tokens, "out": response.usage.output_tokens}
    return data


# ── run ───────────────────────────────────────────────────────────────────
def run(video_id: str, tracks_csv: Path | None = None, fixtures_path: Path | None = None,
        out_csv: Path | None = None, only_track: int | None = None, describe: bool = True,
        model: str = MODEL, min_s: float = 1.0, gap_s: float = 1.0, debug_dir: Path | None = None) -> list[dict]:
    video = VIDEO_DIR / f"{video_id}.mp4"
    tracks_csv = tracks_csv or TRACK_DIR / f"{video_id}_tracks.csv"
    out_csv = out_csv or TRACK_DIR / f"{video_id}_actions.csv"
    fixtures = load_fixtures(video_id, fixtures_path)
    by_id = {f["id"]: f for f in fixtures}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    by_track = load_tracks(tracks_csv)
    if only_track is not None:
        by_track = {only_track: by_track.get(only_track, [])}
    segs = find_segments(by_track, fixtures, shape, min_s, gap_s)
    print(f"{video_id}: {len(fixtures)} fixtures, {len(by_track)} visitors, {len(segs)} contact segments")

    client = None
    if describe and segs:
        from dotenv import load_dotenv
        import anthropic
        load_dotenv(ROOT / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set (put it in .env) — or use --no-describe")
        client = anthropic.Anthropic()

    results = []
    t0 = time.time()
    for i, seg in enumerate(segs):
        fx = by_id[seg["fixture"]]
        row = {"track_id": seg["track_id"], "fixture": seg["fixture"],
               "t_start": round(seg["t_start"], 2), "t_end": round(seg["t_end"], 2),
               "duration_s": round(seg["t_end"] - seg["t_start"], 2),
               "contact": "wrist" if seg["wrist"] else "box",
               "took_item": "", "actions": "[]", "summary": "", "confidence": ""}
        if client is not None:
            crops = segment_crops(cap, fps, seg, fx)
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                for t, img in crops:
                    cv2.imwrite(str(debug_dir / f"{seg['track_id']}_{seg['fixture']}_{t:.1f}.jpg"), img)
            try:
                d = describe_segment(client, crops, seg, fx, model)
            except Exception as e:                                # noqa: BLE001
                d = {"actions": [], "took_item": False, "summary": f"(error: {type(e).__name__}: {e})",
                     "confidence": 0.0}
            row.update({"took_item": d["took_item"], "actions": json.dumps(d["actions"]),
                        "summary": d["summary"], "confidence": d["confidence"]})
            acts = ", ".join(f"{a['action']}@{a['t']}s" + (f" ({a['item']})" if a.get("item") else "")
                             for a in d["actions"])
            print(f"  #{seg['track_id']:<3d} {fx['name']:<14s} {seg['t_start']:6.1f}–{seg['t_end']:6.1f}s  "
                  f"{d['summary']}  [{acts}]  conf {d['confidence']}")
        else:
            print(f"  #{seg['track_id']:<3d} {fx['name']:<14s} {seg['t_start']:6.1f}–{seg['t_end']:6.1f}s  "
                  f"({row['contact']} contact)")
        results.append(row)

    cap.release()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "fixture", "t_start", "t_end", "duration_s",
                                          "contact", "took_item", "actions", "summary", "confidence"])
        w.writeheader()
        w.writerows(results)
    print(f"→ {out_csv}  ({len(results)} rows, {time.time() - t0:.0f}s)")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--tracks", type=Path, default=None, help="tracks CSV (default data/tracks/<id>_tracks.csv)")
    ap.add_argument("--fixtures", type=Path, default=None, help="fixtures JSON (default data/fixtures/<id>.json)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--track", type=int, default=None, help="only this visitor id")
    ap.add_argument("--no-describe", action="store_true", help="segments only, no API calls")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--min-seconds", type=float, default=1.0, help="drop contacts shorter than this")
    ap.add_argument("--gap", type=float, default=1.0, help="merge contacts closer than this (s)")
    ap.add_argument("--debug-crops", type=Path, default=None, help="save the crops sent to the model here")
    a = ap.parse_args()
    run(a.video_id, a.tracks, a.fixtures, a.out, a.track, not a.no_describe, a.model,
        a.min_seconds, a.gap, a.debug_crops)


if __name__ == "__main__":
    main()
