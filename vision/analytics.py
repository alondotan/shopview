"""Stage — visit statistics: dwell, occupancy, and the checkout queue.

This is the "zones / events" stage the fusion module feeds. It takes visitor
trails **in store-map pixels** and the zones drawn on that map, and turns them
into the numbers a store manager asks for:

* how long was each person in the store (``dwell_s``),
* how many people came in over the window, and what that is per hour,
* how many were inside *at the same time* — the occupancy curve, its peak and
  its average,
* how long people waited at the checkout, who reached the till and who gave up.

Two sources, same maths:

* **one camera** — the tracks CSV of a video, with its ``data/calibration/
  <video>.json`` homography applied to the foot point of every detection;
* **a scene** — ``data/fusion/<scene>_tracks.csv``, already on the map and
  already one row per *visitor*, whichever camera(s) saw them.

Zones come from ``data/zones/<map>.json`` and carry a ``role``:

    entrance   the door — a visit that starts here really is an arrival
    queue      where people wait to pay
    counter    the till itself: reaching it ends the wait
    (empty)    an ordinary zone; time in it is still reported

**Waiting** is time inside a queue zone before the first moment inside a
counter zone. A person merely walking through the queue area is not waiting,
so a queue segment shorter than ``min_queue_s`` is dropped. Someone who was in
a queue zone and never reached the till *abandoned* it.

**Censoring matters on short clips.** A visitor already on screen in the first
frame, or still there in the last, has a dwell that is only a lower bound.
Those visits are flagged ``truncated_in`` / ``truncated_out``, and the dwell
averages are reported twice: over complete visits and over all of them.

Output: ``data/analytics/<name>.json`` — ``totals``, ``occupancy`` (the curve),
``queue``, ``zones`` and one record per ``visit``.

    python vision/analytics.py --video=-1bRhYjw1qE      # an id starting with
    python vision/analytics.py --scene checkout          #   "-" needs the "="
    python vision/analytics.py --video KMJS66jBtVQ --min-visit 2 --json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACK_DIR = ROOT / "data" / "tracks"
VIDEO_DIR = ROOT / "data" / "videos"
ZONE_DIR = ROOT / "data" / "zones"
FUSION_DIR = ROOT / "data" / "fusion"
SCENE_DIR = ROOT / "data" / "scenes"
CALIB_DIR = ROOT / "data" / "calibration"
OUT_DIR = ROOT / "data" / "analytics"

ZONE_ROLES = ("", "entrance", "queue", "counter")
ENTRANCE_WINDOW_S = 2.0   # in an entrance zone this soon after appearing → walked in

DEFAULT_PARAMS = {
    "min_visit_s": 1.0,    # a shorter track is a detection blip, not a visitor
    "max_gap_s": 2.0,      # a hole in a trail up to this long is an occlusion, not an exit
    "min_queue_s": 2.0,    # shorter than this inside a queue zone is walking through
    "edge_s": 0.5,         # first/last seconds: a visit touching them is censored
    "grid_s": 0.5,         # occupancy curve sampling step
    "trail_step_s": 0.5,   # how finely each visit's path is kept for the UI
}

TRAIL_MAX_POINTS = 400     # a long visit is thinned further rather than shipped whole


# ── geometry ──────────────────────────────────────────────────────────────
def point_in_polygon(x: float, y: float, pts: list) -> bool:
    """Ray casting. ``pts`` is [[x, y], ...]; the polygon closes itself."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i][0], pts[i][1]
        xj, yj = pts[j][0], pts[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


# ── sources: visitor trails on the store map ──────────────────────────────
def _find_tracks_csv(video_id: str, job: str | None = None) -> Path | None:
    if job:
        path = TRACK_DIR / f"{video_id}_{job}_tracks.csv"
        return path if path.exists() else None
    cands = sorted(TRACK_DIR.glob(f"{video_id}*_tracks.csv"),
                   key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return cands[0] if cands else None


def _video_duration_s(video_id: str) -> float | None:
    path = VIDEO_DIR / f"{video_id}.mp4"
    if not path.exists():
        return None
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        return round(n / fps, 2) if fps > 0 and n > 0 else None
    finally:
        cap.release()


def _held_items(objects_csv: Path | None) -> dict[str, dict[str, int]]:
    """person_id → {label: frames seen holding it}."""
    items: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if objects_csv and objects_csv.exists():
        with objects_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("person_id"):
                    items[str(row["person_id"])][row["label"]] += 1
    return dict(items)


def load_video_source(video_id: str, job: str | None = None) -> dict:
    """One camera: tracks CSV + its homography → trails in map pixels."""
    import numpy as np

    from homography import load_calibration, to_map

    tracks_csv = _find_tracks_csv(video_id, job)
    if tracks_csv is None:
        raise FileNotFoundError(
            f"no tracks CSV for {video_id} — run the Video tab (or detect_people.py) first")
    calib = load_calibration(video_id)

    rows = list(csv.DictReader(tracks_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{tracks_csv.name} is empty")

    feet = np.array([[float(r["foot_x"]), float(r["foot_y"])] for r in rows], dtype=float)
    mapped = to_map(calib["H"], feet)

    trails: dict[str, list[tuple]] = defaultdict(list)
    for row, (mx, my) in zip(rows, mapped):
        trails[str(row["track_id"])].append((float(row["time_s"]), float(mx), float(my)))

    objects_csv = tracks_csv.with_name(tracks_csv.name.replace("_tracks.csv", "_objects.csv"))
    return {
        "kind": "video",
        "name": video_id,
        "map": calib.get("map"),
        "map_size": calib.get("map_size"),
        "trails": {k: sorted(v) for k, v in trails.items()},
        "items": _held_items(objects_csv),
        "duration_s": _video_duration_s(video_id),
        "source_file": str(tracks_csv.relative_to(ROOT)),
        "calibration_rms_px": calib.get("rms_px"),
    }


def load_scene_source(name: str) -> dict:
    """A fused scene: one trail per visitor, already in map pixels."""
    fused = FUSION_DIR / f"{name}_tracks.csv"
    if not fused.exists():
        raise FileNotFoundError(f"scene {name!r} is not fused yet — run the Multi-cam tab first")
    scene_doc = SCENE_DIR / f"{name}.json"
    doc = json.loads(scene_doc.read_text(encoding="utf-8")) if scene_doc.exists() else {}

    trails: dict[str, list[tuple]] = defaultdict(list)
    with fused.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trails[str(row["visitor_id"])].append(
                (float(row["time_s"]), float(row["map_x"]), float(row["map_y"])))

    durations = [d for d in (_video_duration_s(c["video"]) for c in doc.get("cameras", [])) if d]
    return {
        "kind": "scene",
        "name": name,
        "map": doc.get("map"),
        "map_size": doc.get("map_size"),
        "trails": {k: sorted(v) for k, v in trails.items()},
        "items": {},
        "duration_s": max(durations) if durations else None,
        "source_file": str(fused.relative_to(ROOT)),
        "calibration_rms_px": None,
    }


def load_zones(map_name: str | None) -> list[dict]:
    if not map_name:
        return []
    path = ZONE_DIR / f"{map_name}.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for z in doc.get("zones", []):
        role = str(z.get("role") or "")
        out.append({"id": z["id"], "name": z.get("name") or z["id"],
                    "color": z.get("color"), "role": role if role in ZONE_ROLES else "",
                    "pts": z["pts"]})
    return out


# ── per-visit maths ───────────────────────────────────────────────────────
def _intervals(samples: list[tuple], max_gap_s: float) -> list[list[float]]:
    """Merge the sample times into [start, end] runs, breaking on long holes."""
    runs = [[samples[0][0], samples[0][0]]]
    for t, _, _ in samples[1:]:
        if t - runs[-1][1] > max_gap_s:
            runs.append([t, t])
        else:
            runs[-1][1] = t
    return runs


def _sample_weights(samples: list[tuple], max_gap_s: float) -> list[float]:
    """Seconds each sample stands for — half the gap on either side, capped so a
    hole in the trail is not billed as time spent standing in a zone."""
    n = len(samples)
    if n == 1:
        return [0.0]
    ts = [s[0] for s in samples]
    w = []
    for i in range(n):
        before = min(ts[i] - ts[i - 1], max_gap_s) / 2 if i > 0 else 0.0
        after = min(ts[i + 1] - ts[i], max_gap_s) / 2 if i < n - 1 else 0.0
        w.append(before + after)
    return w


def _segments(samples: list[tuple], flags: list[bool], weights: list[float],
              max_gap_s: float) -> list[dict]:
    """Contiguous runs where ``flags`` is true, merged over short holes."""
    segs: list[dict] = []
    for (t, _, _), flag, w in zip(samples, flags, weights):
        if not flag:
            continue
        if segs and t - segs[-1]["t_end"] <= max_gap_s:
            segs[-1]["t_end"] = t
            segs[-1]["seconds"] += w
        else:
            segs.append({"t_start": t, "t_end": t, "seconds": w})
    for s in segs:
        s["span_s"] = round(s["t_end"] - s["t_start"], 2)
        s["seconds"] = round(s["seconds"], 2)
        s["t_start"] = round(s["t_start"], 2)
        s["t_end"] = round(s["t_end"], 2)
    return segs


def _trail(samples: list[tuple], step_s: float) -> list[list[float]]:
    """The path, thinned for drawing: one point per ``step_s``, and never more
    than ``TRAIL_MAX_POINTS`` of them."""
    span = samples[-1][0] - samples[0][0]
    step = max(step_s, span / TRAIL_MAX_POINTS) if span > 0 else step_s
    out, last = [], None
    for t, x, y in samples:
        if last is None or t - last >= step:
            out.append([round(t, 2), round(x, 1), round(y, 1)])
            last = t
    if out[-1][0] != round(samples[-1][0], 2):
        out.append([round(samples[-1][0], 2), round(samples[-1][1], 1), round(samples[-1][2], 1)])
    return out


def _visit(vid: str, samples: list[tuple], zones: list[dict], window: tuple[float, float],
           items: dict[str, int], p: dict) -> dict:
    t0, t1 = samples[0][0], samples[-1][0]
    weights = _sample_weights(samples, p["max_gap_s"])
    runs = _intervals(samples, p["max_gap_s"])

    inside = {z["id"]: [point_in_polygon(x, y, z["pts"]) for _, x, y in samples] for z in zones}
    zone_seconds = {
        zid: round(sum(w for w, hit in zip(weights, flags) if hit), 2)
        for zid, flags in inside.items()
    }

    queue_ids = [z["id"] for z in zones if z["role"] == "queue"]
    counter_ids = [z["id"] for z in zones if z["role"] == "counter"]
    entrance_ids = [z["id"] for z in zones if z["role"] == "entrance"]

    in_queue = [any(inside[z][i] for z in queue_ids) for i in range(len(samples))]
    in_counter = [any(inside[z][i] for z in counter_ids) for i in range(len(samples))]

    counter_first = next((samples[i][0] for i, hit in enumerate(in_counter) if hit), None)
    counter_s = round(sum(w for w, hit in zip(weights, in_counter) if hit), 2)

    # queue runs long enough to be waiting rather than walking through
    queue_segs = [s for s in _segments(samples, in_queue, weights, p["max_gap_s"])
                  if s["span_s"] >= p["min_queue_s"]]
    joined = bool(queue_segs)
    before_till = [s for s in queue_segs
                   if counter_first is None or s["t_start"] < counter_first]
    wait_s = round(sum(min(s["t_end"], counter_first if counter_first is not None else s["t_end"])
                       - s["t_start"] for s in before_till), 2) if before_till else 0.0

    # an arrival through the door: in an entrance zone within the first seconds
    entered_here = bool(entrance_ids) and any(
        any(inside[z][i] for z in entrance_ids)
        for i, (t, _, _) in enumerate(samples) if t - t0 <= ENTRANCE_WINDOW_S)

    return {
        "id": vid,
        "t_in": round(t0, 2),
        "t_out": round(t1, 2),
        "dwell_s": round(t1 - t0, 2),
        "tracked_s": round(sum(b - a for a, b in runs), 2),
        "n_points": len(samples),
        "truncated_in": t0 <= window[0] + p["edge_s"],
        "truncated_out": t1 >= window[1] - p["edge_s"],
        "entered_via_entrance": entered_here,
        "zone_seconds": zone_seconds,
        "queue_joined": joined,
        "queue_wait_s": wait_s,
        "queue_segments": queue_segs,
        "reached_counter": counter_first is not None,
        "counter_at_s": round(counter_first, 2) if counter_first is not None else None,
        "counter_s": counter_s,
        "abandoned_queue": joined and counter_first is None,
        "items": dict(sorted(items.items(), key=lambda kv: -kv[1])) if items else {},
        "intervals": [[round(a, 2), round(b, 2)] for a, b in runs],
        "trail": _trail(samples, p["trail_step_s"]),
    }


# ── aggregation ───────────────────────────────────────────────────────────
def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return round(s[0], 2)
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 2)


def _spread(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 2) if values else None,
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "max": round(max(values), 2) if values else None,
        "min": round(min(values), 2) if values else None,
    }


def _curve(visits: list[dict], window: tuple[float, float], grid_s: float,
           key: str = "intervals") -> list[list[float]]:
    """Head count on a regular grid: [[t, n], ...]."""
    start, end = window
    n_steps = max(1, int((end - start) / grid_s) + 1)
    counts = [0] * n_steps
    for v in visits:
        for a, b in v[key]:
            i0 = max(0, int((a - start) / grid_s))
            i1 = min(n_steps - 1, int(math.ceil((b - start) / grid_s)))
            for i in range(i0, i1 + 1):
                counts[i] += 1
    return [[round(start + i * grid_s, 2), c] for i, c in enumerate(counts)]


def compute(source: dict, zones: list[dict] | None = None, params: dict | None = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    zones = zones if zones is not None else load_zones(source.get("map"))

    trails = source["trails"]
    all_times = [t for s in trails.values() for t, _, _ in s]
    if not all_times:
        raise ValueError("no trail points to analyse")
    window = (min(all_times), max(all_times))
    if source.get("duration_s") and source["duration_s"] > window[1]:
        window = (window[0], source["duration_s"])
    window_s = max(1e-6, window[1] - window[0])

    visits, dropped = [], 0
    for vid, samples in trails.items():
        if samples[-1][0] - samples[0][0] < p["min_visit_s"]:
            dropped += 1
            continue
        visits.append(_visit(vid, samples, zones, window,
                             source.get("items", {}).get(vid, {}), p))
    visits.sort(key=lambda v: (v["t_in"], v["id"]))

    occupancy = _curve(visits, window, p["grid_s"])
    peak = max((c for _, c in occupancy), default=0)
    peak_at = next((t for t, c in occupancy if c == peak), None)
    person_seconds = sum(v["tracked_s"] for v in visits)

    complete = [v for v in visits if not v["truncated_in"] and not v["truncated_out"]]
    arrivals = [v for v in visits if not v["truncated_in"]]

    # queue: only the qualifying waiting segments count towards the curve
    queued = [v for v in visits if v["queue_joined"]]
    q_visits = [{"intervals": [[s["t_start"], s["t_end"]] for s in v["queue_segments"]]}
                for v in queued]
    q_curve = _curve(q_visits, window, p["grid_s"]) if q_visits else []
    q_peak = max((c for _, c in q_curve), default=0)
    waits = [v["queue_wait_s"] for v in queued if v["queue_wait_s"] > 0]

    zone_rollup = []
    for z in zones:
        secs = [v["zone_seconds"].get(z["id"], 0.0) for v in visits]
        visited = [s for s in secs if s > 0]
        zone_rollup.append({
            "id": z["id"], "name": z["name"], "role": z["role"], "color": z.get("color"),
            "visitors": len(visited),
            "total_s": round(sum(visited), 2),
            "mean_s": round(sum(visited) / len(visited), 2) if visited else None,
            "share_of_visitors": round(len(visited) / len(visits), 3) if visits else None,
        })

    has_queue = any(z["role"] == "queue" for z in zones)
    has_counter = any(z["role"] == "counter" for z in zones)

    return {
        "source": {k: source[k] for k in
                   ("kind", "name", "map", "map_size", "source_file",
                    "duration_s", "calibration_rms_px") if k in source},
        "params": p,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {"start_s": round(window[0], 2), "end_s": round(window[1], 2),
                   "duration_s": round(window_s, 2)},
        "totals": {
            "visitors": len(visits),
            "dropped_short": dropped,
            "arrivals": len(arrivals),
            "visitors_per_hour": round(len(arrivals) * 3600 / window_s, 1),
            "complete_visits": len(complete),
            "truncated_visits": len(visits) - len(complete),
            "dwell_all_s": _spread([v["dwell_s"] for v in visits]),
            "dwell_complete_s": _spread([v["dwell_s"] for v in complete]),
            "person_seconds": round(person_seconds, 1),
            "avg_concurrent": round(person_seconds / window_s, 2),
            "peak_concurrent": peak,
            "peak_at_s": peak_at,
        },
        "occupancy": {"grid_s": p["grid_s"], "curve": occupancy},
        "queue": {
            "configured": has_queue,
            "counter_configured": has_counter,
            "joined": len(queued),
            "share_of_visitors": round(len(queued) / len(visits), 3) if visits else None,
            "served": sum(1 for v in queued if v["reached_counter"]),
            "abandoned": sum(1 for v in queued if v["abandoned_queue"]),
            "wait_s": _spread(waits),
            "total_wait_s": round(sum(waits), 1),
            "peak_length": q_peak,
            "avg_length": round(sum(c for _, c in q_curve) * p["grid_s"] / window_s, 2) if q_curve else 0.0,
            "curve": q_curve,
            "counter_s": _spread([v["counter_s"] for v in visits if v["counter_s"] > 0]),
        },
        "zones": zone_rollup,
        "visits": visits,
    }


def brief(res: dict) -> str:
    """The whole result as text, for a model to reason over.

    Small enough to sit in a system prompt — a busy hour is a few hundred
    visits — so there is no retrieval step and no summarising away of the rows
    the answer should cite. The caveats travel with the numbers on purpose: a
    model handed a censored dwell without being told it is censored will report
    it as a fact.
    """
    src, t, q, w, p = res["source"], res["totals"], res["queue"], res["window"], res["params"]
    zones = res["zones"]
    out = [
        f"Source: {src['kind']} {src['name']} ({src['source_file']})",
        f"Analysed window: {w['start_s']}–{w['end_s']} s ({w['duration_s']} s)."
        + (f" The video itself is {src['duration_s']} s." if src.get("duration_s") else ""),
        "Times are seconds from the start of the clip, not wall-clock time, so there is no"
        " hour of day and no date.",
        "",
        "TOTALS",
        f"  visitors: {t['visitors']} ({t['dropped_short']} tracks shorter than"
        f" {p['min_visit_s']}s were dropped as detection noise)",
        f"  arrivals inside the window: {t['arrivals']}  → {t['visitors_per_hour']} per hour at this rate",
        f"  time in store, complete visits only (n={t['dwell_complete_s']['n']}):"
        f" mean {t['dwell_complete_s']['mean']}s, median {t['dwell_complete_s']['median']}s,"
        f" p90 {t['dwell_complete_s']['p90']}s, max {t['dwell_complete_s']['max']}s",
        f"  time in store, all {t['visitors']} visits: mean {t['dwell_all_s']['mean']}s,"
        f" median {t['dwell_all_s']['median']}s",
        f"  in the store at the same time: {t['avg_concurrent']} on average,"
        f" peak {t['peak_concurrent']} at {t['peak_at_s']}s",
    ]

    if q["configured"]:
        out += [
            "",
            "CHECKOUT",
            f"  queued: {q['joined']} of {t['visitors']} visitors"
            f" ({q['served']} reached the till, {q['abandoned']} left without paying)",
            f"  wait, over the {q['wait_s']['n']} who actually waited:"
            f" mean {q['wait_s']['mean']}s, median {q['wait_s']['median']}s, max {q['wait_s']['max']}s",
            f"  queue length: {q['avg_length']} on average, {q['peak_length']} at its longest",
            f"  time at the till: mean {q['counter_s']['mean']}s over {q['counter_s']['n']} people",
        ]
    else:
        out += ["", "CHECKOUT: no zone on this store plan is marked 'queue', so there are no"
                    " wait times. Say so rather than inferring them."]

    if zones:
        out += ["", "ZONES (id | name | role | visitors | mean seconds each | total seconds)"]
        out += [f"  {z['id']} | {z['name']} | {z['role'] or '-'} | {z['visitors']}"
                f" | {z['mean_s']} | {z['total_s']}" for z in zones]

    cols = ["id", "enters_s", "leaves_s", "dwell_s", "censored", "walked_in",
            "queue_wait_s", "reached_till", "till_s", "carrying"]
    cols[9:9] = [f"zone_{z['id']}_s" for z in zones]
    out += ["", f"VISITS ({len(res['visits'])} rows, CSV)", ",".join(cols)]
    for v in res["visits"]:
        censored = ("start" if v["truncated_in"] else "") + ("end" if v["truncated_out"] else "")
        row = [v["id"], v["t_in"], v["t_out"], v["dwell_s"], censored or "no",
               "yes" if v["entered_via_entrance"] else "no",
               v["queue_wait_s"], "yes" if v["reached_counter"] else "no", v["counter_s"]]
        row += [v["zone_seconds"].get(z["id"], 0) for z in zones]
        row.append(";".join(v["items"]) or "")
        out.append(",".join(str(c) for c in row))

    out += [
        "",
        "HOW TO READ THIS",
        f"- A visit is 'censored' when the person was already on screen in the first frame"
        f" (start) or still there in the last (end). Its dwell_s is a lower bound, not a"
        f" measurement. {t['truncated_visits']} of {t['visitors']} visits are censored here."
        " Never average a censored dwell in with a complete one without saying so.",
        f"- A hole in a trail up to {p['max_gap_s']}s is treated as an occlusion, not an exit.",
        f"- 'queue_wait_s' is time in a queue zone before first reaching a till zone."
        f" A stay under {p['min_queue_s']}s counts as walking through, so 0 means"
        " 'did not wait', not 'was never near the checkout' — check the zone column too.",
        "- 'carrying' is what the object detector saw in the person's hands, sampled over"
        " the visit. It is evidence of handling, not of buying.",
    ]
    if src.get("calibration_rms_px"):
        out.append(f"- The camera-to-plan calibration has an RMS error of"
                   f" {src['calibration_rms_px']} map pixels, so zone boundaries are fuzzy to"
                   f" roughly the width of a person. Do not read a one-second zone total as"
                   f" meaningful.")
    if w["duration_s"] < 300:
        out.append(f"- This window is only {w['duration_s']} seconds. Per-hour figures are"
                   " extrapolations from a very short sample; say so when you quote one.")
    return "\n".join(out)


def analyse(kind: str, name: str, job: str | None = None, params: dict | None = None) -> dict:
    source = load_video_source(name, job) if kind == "video" else load_scene_source(name)
    return compute(source, params=params)


def save(result: dict, name: str | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name or result['source']['name']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────
def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video id, e.g. KMJS66jBtVQ "
                                     "(an id starting with '-' needs --video=<id>)")
    src.add_argument("--scene", help="fused scene name, e.g. checkout")
    ap.add_argument("--job", help="a specific analysis run of that video")
    ap.add_argument("--min-visit", type=float, default=DEFAULT_PARAMS["min_visit_s"])
    ap.add_argument("--min-queue", type=float, default=DEFAULT_PARAMS["min_queue_s"])
    ap.add_argument("--max-gap", type=float, default=DEFAULT_PARAMS["max_gap_s"])
    ap.add_argument("--grid", type=float, default=DEFAULT_PARAMS["grid_s"])
    ap.add_argument("--json", action="store_true", help="print the whole result instead of a summary")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    kind = "video" if args.video else "scene"
    res = analyse(kind, args.video or args.scene, args.job, {
        "min_visit_s": args.min_visit, "min_queue_s": args.min_queue,
        "max_gap_s": args.max_gap, "grid_s": args.grid,
    })

    if args.json:
        print(json.dumps(res, indent=1))
    else:
        t, q, w = res["totals"], res["queue"], res["window"]
        print(f"{res['source']['name']}  ({res['source']['source_file']})")
        print(f"  window            {w['duration_s']:.1f} s"
              f"   map {res['source'].get('map') or '–'}")
        print(f"  visitors          {t['visitors']}"
              f"  ({t['dropped_short']} short tracks dropped,"
              f" {t['truncated_visits']} cut off by the clip edges)")
        print(f"  arrivals          {t['arrivals']}  → {t['visitors_per_hour']}/hour at this rate")
        print(f"  dwell (complete)  mean {_fmt(t['dwell_complete_s']['mean'])}"
              f"   median {_fmt(t['dwell_complete_s']['median'])}"
              f"   max {_fmt(t['dwell_complete_s']['max'])}"
              f"   (n={t['dwell_complete_s']['n']})")
        print(f"  dwell (all)       mean {_fmt(t['dwell_all_s']['mean'])}"
              f"   median {_fmt(t['dwell_all_s']['median'])}")
        print(f"  in store at once  avg {t['avg_concurrent']}   peak {t['peak_concurrent']}"
              f" at {t['peak_at_s']}s")
        if q["configured"]:
            print(f"  checkout          {q['joined']} queued"
                  f" ({q['served']} served, {q['abandoned']} left without paying)")
            print(f"    wait            mean {_fmt(q['wait_s']['mean'])}"
                  f"   median {_fmt(q['wait_s']['median'])}"
                  f"   max {_fmt(q['wait_s']['max'])}")
            print(f"    queue length    avg {q['avg_length']}   peak {q['peak_length']}")
        else:
            print("  checkout          no zone with role 'queue' on this map — "
                  "draw one in the Zones tab")
        for z in res["zones"]:
            role = f" [{z['role']}]" if z["role"] else ""
            print(f"    {z['name']}{role}: {z['visitors']} visitors,"
                  f" {_fmt(z['mean_s'])} each on average")

    if not args.no_save:
        print(f"\nsaved → {save(res).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
