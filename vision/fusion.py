"""Multi-camera fusion — several cameras on one store, one path per shopper.

A *scene* (``data/scenes/<name>.json``) is a set of videos that look at the
same floor at the same time, share one store map and are calibrated against
it (each camera keeps its own ``data/calibration/<video>.json``, fitted
jointly in the Multi-cam calibration tab). This module takes every camera's
tracks, puts them on the map through its homography, and joins the tracks
that are the same person into one *visitor* with one fused trail.

Two kinds of evidence link a track in camera A to a track in camera B:

* **overlap** — both cameras see the person at the same time and the two
  mapped floor points stay close (median distance over the shared seconds
  under ``max_dist_px``, and most samples within it);
* **hand-off** — B starts shortly after A ends (``max_gap_s``), about where A
  was heading (A's last point extrapolated with its velocity, with slack that
  grows with the gap) — the person walked out of one view into the other.

Either way the **clothes must agree**: each track carries the clothing
histogram from ``stitch.appearance`` (torso between the shoulders and hips,
legs when visible), averaged over its frames, and a link needs at least
``min_sim`` similarity (histogram intersection). Measured on the sample pair
— a steep view against a shallow one — the same person scores 0.46–0.68 and
different people 0.24–0.38, so the gate is what stops two shoppers standing
side by side at the till from being merged when the geometry alone cannot
tell them apart (a floor point estimated behind a counter is easily 50–100
map px off). Building the histograms means re-reading the frames the CSV
references — a few seconds per video; they are cached next to the fusion
output.

Links are taken greedily, cheapest first, with union-find; a merge is refused
when it would put two tracks *of the same camera* that overlap in time into
one visitor (one person cannot be two boxes in one frame). Same-camera
fragments are never linked directly — the per-camera stitching pass already
did that with clothing — but they do end up together when a track in the
other camera bridges them (A1 ↔ B ↔ A2), which is where the second camera
earns its keep: it covers the occlusions of the first.

The fused trail samples every ``GRID_S`` seconds and averages the cameras
that see the person at that moment (weighted by detection confidence, less
when the floor point was estimated rather than seen). Gaps longer than
``BREAK_S`` are left as gaps.

Outputs ``data/fusion/<scene>.json`` (visitors, members, trails, links, stats)
and ``data/fusion/<scene>_tracks.csv``:

    time_s,visitor_id,map_x,map_y,n_cams,cams

which is the input for the zones / events stage — one row per visitor per
grid step, in store-map pixels, whichever camera(s) saw them.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from homography import CALIB_DIR, TRACK_DIR, load_calibration, to_map
from stitch import Fragment, fragments_from_csv, similarity

ROOT = Path(__file__).resolve().parent.parent
SCENE_DIR = ROOT / "data" / "scenes"
FUSION_DIR = ROOT / "data" / "fusion"
VIDEO_DIR = ROOT / "data" / "videos"

GRID_S = 0.2            # fused trail sampling step
BREAK_S = 1.5           # a hole longer than this is a gap in the trail, not a line
SAME_CAM_TOL_S = 0.5    # two tracks of one camera may overlap this much and still be one person
VEL_WINDOW_S = 1.5      # velocity for the hand-off extrapolation
EXTRAP_MAX_S = 1.0
HANDOFF_PENALTY = 0.5   # hand-offs are weaker evidence than overlaps → merged after them
FULL_OVERLAP_S = 15.0   # co-location this long counts as full evidence; a brief brush counts little

DEFAULT_PARAMS = {
    "max_dist_px": 100.0,   # two cameras agree if their floor points are this close (map px)
    "min_overlap_s": 1.0,   # shared seconds needed for an overlap link
    "max_gap_s": 6.0,       # longest hand-off gap
    "min_sim": 0.4,         # clothing similarity a link needs (0 = ignore clothes)
    "min_track_s": 0.6,     # shorter tracks are noise
}


# ── scenes ────────────────────────────────────────────────────────────────
def scene_path(name: str) -> Path:
    return SCENE_DIR / f"{name}.json"


def load_scene(name: str) -> dict:
    path = scene_path(name)
    if not path.exists():
        raise FileNotFoundError(f"scene {name!r} not found — create it in the Multi-cam calibration tab")
    return json.loads(path.read_text(encoding="utf-8"))


def find_tracks_csv(video_id: str) -> Path | None:
    """The run covering most of the video — the biggest tracks CSV for it."""
    cands = sorted(TRACK_DIR.glob(f"{video_id}*_tracks.csv"),
                   key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return cands[0] if cands else None


# ── tracks on the map ─────────────────────────────────────────────────────
@dataclass
class CamTrack:
    """One track of one camera, in map coordinates and scene time."""
    cam: int
    video: str
    tid: int
    t: np.ndarray                 # scene seconds, ascending
    xy: np.ndarray                # (N,2) map px
    w: np.ndarray                 # per-sample weight: conf, halved when the floor point was estimated
    est_frac: float = 0.0         # share of samples whose floor point was estimated (feet hidden)
    look: Fragment | None = None  # clothing histograms (torso / legs), averaged over the track

    @property
    def t0(self) -> float: return float(self.t[0])

    @property
    def t1(self) -> float: return float(self.t[-1])

    @property
    def dur(self) -> float: return self.t1 - self.t0

    def at(self, ts: np.ndarray) -> np.ndarray:
        """Linear interpolation of the map position at ``ts`` (inside [t0, t1])."""
        return np.stack([np.interp(ts, self.t, self.xy[:, 0]),
                         np.interp(ts, self.t, self.xy[:, 1])], axis=1)

    def weight_at(self, ts: np.ndarray) -> np.ndarray:
        return np.interp(ts, self.t, self.w)

    def velocity(self) -> tuple[float, float]:
        i = len(self.t) - 1
        while i > 0 and self.t[-1] - self.t[i - 1] <= VEL_WINDOW_S:
            i -= 1
        dt = self.t[-1] - self.t[i]
        if dt <= 0:
            return 0.0, 0.0
        return (float((self.xy[-1, 0] - self.xy[i, 0]) / dt),
                float((self.xy[-1, 1] - self.xy[i, 1]) / dt))


def track_appearance(video: str, tracks_csv: Path) -> dict[int, Fragment]:
    """track_id → a Fragment holding the summed clothing histograms of every
    detection of that track (all its raw ids), re-reading the video frames the
    CSV references. Cached in data/fusion/ keyed by the CSV's size and mtime."""
    st = tracks_csv.stat()
    cache = FUSION_DIR / f"appearance_{tracks_csv.stem}_{st.st_size}_{int(st.st_mtime)}.npz"
    if cache.exists():
        z = np.load(cache)
        out = {}
        for tid in z["ids"]:
            f = Fragment(int(tid))
            f.torso_sum, f.torso_n = z[f"t{tid}"], int(z[f"tn{tid}"])
            if f.torso_n == 0:
                f.torso_sum = None
            if int(z[f"ln{tid}"]):
                f.legs_sum, f.legs_n = z[f"l{tid}"], int(z[f"ln{tid}"])
            out[int(tid)] = f
        return out
    video_path = VIDEO_DIR / f"{video}.mp4"
    if not video_path.exists():
        return {}
    rows, frags = fragments_from_csv(tracks_csv, video_path)
    raw_ids: dict[int, set[int]] = {}
    for r in rows:
        raw_ids.setdefault(int(r["track_id"]), set()).add(int(r["raw_id"]))
    out: dict[int, Fragment] = {}
    for tid, rids in raw_ids.items():
        m = out[tid] = Fragment(tid)
        for f in (frags[rid] for rid in rids if rid in frags):
            if f.torso_sum is not None:
                m.torso_sum = f.torso_sum.copy() if m.torso_sum is None else m.torso_sum + f.torso_sum
                m.torso_n += f.torso_n
            if f.legs_sum is not None:
                m.legs_sum = f.legs_sum.copy() if m.legs_sum is None else m.legs_sum + f.legs_sum
                m.legs_n += f.legs_n
    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    blob = {"ids": np.array(sorted(out))}
    for tid, f in out.items():
        blob[f"t{tid}"] = f.torso_sum if f.torso_sum is not None else np.zeros(1)
        blob[f"tn{tid}"] = f.torso_n
        blob[f"l{tid}"] = f.legs_sum if f.legs_sum is not None else np.zeros(1)
        blob[f"ln{tid}"] = f.legs_n
    np.savez(cache, **blob)
    return out


def load_camera_tracks(cam: int, video: str, offset_s: float, min_track_s: float,
                       tracks_csv: Path | None = None, appearance: bool = True) -> tuple[list[CamTrack], dict]:
    """Read a camera's tracks CSV, map the floor points through its saved
    calibration and split into :class:`CamTrack` s. ``offset_s`` is added to
    the video's own clock to get scene time."""
    calib = load_calibration(video)
    path = tracks_csv or find_tracks_csv(video)
    if path is None:
        raise FileNotFoundError(f"no tracks CSV for {video} — run detect_people.py first")
    looks = track_appearance(video, path) if appearance else {}
    by_id: dict[int, list[tuple[float, float, float, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            est = r.get("foot_src", "box") != "box"
            w = float(r["conf"]) * (0.5 if est else 1.0)
            by_id.setdefault(int(r["track_id"]), []).append(
                (float(r["time_s"]) + offset_s, float(r["foot_x"]), float(r["foot_y"]), w, float(est)))
    tracks, dropped = [], 0
    for tid, rows in by_id.items():
        rows.sort()
        arr = np.asarray(rows, dtype=np.float64)
        # one sample per instant (a duplicated frame would break the interpolation)
        _, keep = np.unique(arr[:, 0], return_index=True)
        arr = arr[keep]
        if arr[-1, 0] - arr[0, 0] < min_track_s:
            dropped += 1
            continue
        tracks.append(CamTrack(cam, video, tid, arr[:, 0], to_map(calib["H"], arr[:, 1:3]), arr[:, 3],
                               float(arr[:, 4].mean()), looks.get(tid)))
    info = {"video": video, "tracks_csv": str(path.relative_to(ROOT)), "calibration_rms_px": calib.get("rms_px"),
            "tracks": len(tracks), "dropped_short": dropped, "offset_s": offset_s,
            "with_appearance": sum(1 for t in tracks if t.look is not None and t.look.torso is not None)}
    return tracks, info


# ── linking ───────────────────────────────────────────────────────────────
def clothes(a: CamTrack, b: CamTrack, min_sim: float) -> tuple[float | None, bool]:
    """(similarity or None when either side has no descriptor, passes-the-gate)."""
    if min_sim <= 0 or a.look is None or b.look is None or a.look.torso is None or b.look.torso is None:
        return None, True
    sim = similarity(a.look, b.look)
    return sim, sim >= min_sim


EST_SLACK = 0.5         # an estimated floor point (feet hidden) may be this much further off
WALK_PER_S = 0.7        # a hand-off may cover this many max_dist per second of gap …
WALK_CAP = 3.0          # … up to this many in total


def handoff_distance(first: CamTrack, second: CamTrack, max_dist: float) -> tuple[float, float]:
    """(distance, allowed): from ``first``'s last point carried forward with its
    velocity to ``second``'s first point, against the walking-speed slack."""
    gap = second.t0 - first.t1
    vx, vy = first.velocity()
    dt = min(gap, EXTRAP_MAX_S)
    px, py = first.xy[-1, 0] + vx * dt, first.xy[-1, 1] + vy * dt
    d = float(np.hypot(px - second.xy[0, 0], py - second.xy[0, 1]))
    return d, max_dist * min(1.0 + WALK_PER_S * gap, WALK_CAP)


def pair_cost(a: CamTrack, b: CamTrack, p: dict) -> tuple[float, dict] | None:
    """Can A (camera i) and B (camera j) be the same person? (cost, info) or None."""
    min_overlap, max_gap = p["min_overlap_s"], p["max_gap_s"]
    # a floor point guessed from the torso because a counter hides the legs is
    # a few times less accurate than one seen — give such tracks more room
    max_dist = p["max_dist_px"] * (1 + EST_SLACK * max(a.est_frac, b.est_frac))
    lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
    overlap = hi - lo
    if overlap >= min_overlap:
        ts = np.arange(lo, hi + 1e-9, GRID_S)
        d = np.linalg.norm(a.at(ts) - b.at(ts), axis=1)
        d_med = float(np.median(d))
        close = float(np.mean(d <= max_dist))
        if d_med > max_dist or close < 0.5:
            return None
        sim, ok = clothes(a, b, p["min_sim"])
        if not ok:
            return None
        # a long co-location at some distance beats a brief brush at none: two
        # shoppers pass within a metre of each other all the time, they do not
        # stand together for a quarter of a minute
        cost = (0.4 * d_med / max_dist + 0.2 * (1 - close) + 0.4 * (1 - (sim if sim is not None else 0.5))
                + 0.4 * (1 - min(overlap, FULL_OVERLAP_S) / FULL_OVERLAP_S))
        info = {"kind": "overlap", "overlap_s": round(overlap, 2),
                "d_med_px": round(d_med, 1), "close_frac": round(close, 2), "allowed_px": round(max_dist)}
        if sim is not None:
            info["sim"] = round(sim, 2)
        return cost, info
    if overlap >= 0:
        return None                                  # overlapping but too briefly to judge
    # no overlap: hand-off from whichever ends first
    first, second = (a, b) if a.t1 <= b.t0 else (b, a)
    gap = second.t0 - first.t1
    if gap > max_gap:
        return None
    d, allowed = handoff_distance(first, second, max_dist)
    if d > allowed:
        return None
    sim, ok = clothes(a, b, p["min_sim"])
    if not ok:
        return None
    cost = 0.4 * d / allowed + 0.2 * gap / max_gap + 0.4 * (1 - (sim if sim is not None else 0.5)) + HANDOFF_PENALTY
    info = {"kind": "handoff", "gap_s": round(gap, 2), "dist_px": round(d, 1), "allowed_px": round(allowed, 1)}
    if sim is not None:
        info["sim"] = round(sim, 2)
    return cost, info


class _Groups:
    """Union-find over tracks, keeping per camera the time spans in each group
    so a merge that puts two simultaneous same-camera tracks together is refused."""

    def __init__(self, tracks: list[CamTrack]):
        self.tracks = tracks
        self.parent = list(range(len(tracks)))
        self.spans: list[dict[int, list[tuple[float, float]]]] = [
            {t.cam: [(t.t0, t.t1)]} for t in tracks]
        self.members: list[list[int]] = [[i] for i in range(len(tracks))]

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def conflict(self, ra: int, rb: int) -> bool:
        sa, sb = self.spans[ra], self.spans[rb]
        for cam in sa.keys() & sb.keys():
            for (a0, a1) in sa[cam]:
                for (b0, b1) in sb[cam]:
                    if min(a1, b1) - max(a0, b0) > SAME_CAM_TOL_S:
                        return True
        return False

    def implausible_walk(self, ra: int, rb: int, max_dist: float) -> bool:
        """A hand-off is judged between two tracks, but the visitor may have
        been seen later than track A by another camera: the walk must be
        possible from the group's *latest* sighting before B starts, else a
        shopper who left on the left is glued to one who entered on the right."""
        ta, tb = self.tracks, self.tracks
        for ga, gb in ((ra, rb), (rb, ra)):
            starts = min(tb[k].t0 for k in self.members[gb])
            before = [ta[k] for k in self.members[ga] if ta[k].t1 <= starts]
            if not before:
                continue
            last = max(before, key=lambda t: t.t1)
            first_b = min((tb[k] for k in self.members[gb]), key=lambda t: t.t0)
            if first_b.t0 - last.t1 > 0:
                d, allowed = handoff_distance(last, first_b, max_dist)
                if d > allowed:
                    return True
        return False

    def union(self, i: int, j: int, max_dist: float | None = None) -> bool:
        ra, rb = self.find(i), self.find(j)
        if ra == rb:
            return True
        if self.conflict(ra, rb):
            return False
        if max_dist is not None and self.implausible_walk(ra, rb, max_dist):
            return False
        self.parent[rb] = ra
        for cam, spans in self.spans[rb].items():
            self.spans[ra].setdefault(cam, []).extend(spans)
        self.spans[rb] = {}
        self.members[ra].extend(self.members[rb])
        self.members[rb] = []
        return True


def fuse_tracks(tracks: list[CamTrack], params: dict) -> tuple[list[list[int]], list[dict]]:
    """Greedy cross-camera linking. Returns the groups (lists of track indexes,
    ordered by first appearance) and the accepted links."""
    by_t0 = sorted(range(len(tracks)), key=lambda i: tracks[i].t0)
    max_gap = params["max_gap_s"]
    cands: list[tuple[float, int, int, dict]] = []
    for ii, i in enumerate(by_t0):
        a = tracks[i]
        for j in by_t0[ii + 1:]:
            b = tracks[j]
            if b.t0 - a.t1 > max_gap:
                break                                # sorted by t0: nothing later fits either
            if b.cam == a.cam:
                continue
            r = pair_cost(a, b, params)
            if r is not None:
                cands.append((r[0], i, j, r[1]))
    cands.sort(key=lambda c: c[0])

    groups = _Groups(tracks)
    links: list[dict] = []
    for cost, i, j, info in cands:
        if groups.find(i) == groups.find(j):
            continue                                 # already together (via a third track)
        est = max(tracks[i].est_frac, tracks[j].est_frac)
        if groups.union(i, j, params["max_dist_px"] * (1 + EST_SLACK * est)):
            a, b = tracks[i], tracks[j]
            links.append({**info, "cost": round(cost, 3),
                          "a": {"cam": a.cam, "video": a.video, "track_id": a.tid},
                          "b": {"cam": b.cam, "video": b.video, "track_id": b.tid}})

    members: dict[int, list[int]] = {}
    for i in by_t0:
        members.setdefault(groups.find(i), []).append(i)
    return sorted(members.values(), key=lambda g: tracks[g[0]].t0), links


def fused_trail(members: list[CamTrack]) -> list[list]:
    """[[t, x, y, n_cams], …] every GRID_S over the visitor's life: the weighted
    mean of every camera that sees them at that instant; instants nobody covers
    are skipped, so a hole stays a hole."""
    t0 = min(m.t0 for m in members)
    t1 = max(m.t1 for m in members)
    ts = np.arange(round(t0 / GRID_S) * GRID_S, t1 + 1e-9, GRID_S)
    acc = np.zeros((len(ts), 2))
    wsum = np.zeros(len(ts))
    n = np.zeros(len(ts), dtype=int)
    for m in members:
        inside = (ts >= m.t0 - 1e-9) & (ts <= m.t1 + 1e-9)
        if not inside.any():
            continue
        w = np.maximum(m.weight_at(ts[inside]), 0.05)
        acc[inside] += m.at(ts[inside]) * w[:, None]
        wsum[inside] += w
        n[inside] += 1
    out = []
    for k in np.nonzero(n)[0]:
        x, y = acc[k] / wsum[k]
        out.append([round(float(ts[k]), 2), round(float(x), 1), round(float(y), 1), int(n[k])])
    return out


def fuse_scene(scene: dict, params: dict | None = None, name: str | None = None) -> dict:
    """Run the whole thing for a scene doc. Returns the result document."""
    p = {**DEFAULT_PARAMS, **{k: float(v) for k, v in (params or {}).items() if v is not None}}
    tracks: list[CamTrack] = []
    cameras: list[dict] = []
    for ci, cam in enumerate(scene["cameras"]):
        ct, info = load_camera_tracks(ci, cam["video"], float(cam.get("offset_s") or 0.0), p["min_track_s"])
        tracks.extend(ct)
        cameras.append(info)

    groups, links = fuse_tracks(tracks, p)
    visitors = []
    for vid, g in enumerate(groups, start=1):
        mem = [tracks[i] for i in g]
        cams = sorted({m.cam for m in mem})
        trail = fused_trail(mem)
        visitors.append({
            "id": vid,
            "cams": cams,
            "members": [{"cam": m.cam, "video": m.video, "track_id": m.tid,
                         "t0": round(m.t0, 2), "t1": round(m.t1, 2)} for m in mem],
            "t0": trail[0][0], "t1": trail[-1][0],
            "seen_by_both_s": round(sum(GRID_S for pt in trail if pt[3] > 1), 1),
            "trail": trail,
        })

    n_multi = sum(1 for v in visitors if len(v["cams"]) > 1)
    bridged = sum(1 for v in visitors
                  if any(sum(1 for m in v["members"] if m["cam"] == c) > 1 for c in v["cams"]))
    return {
        "scene": name or scene.get("name"),
        "map": scene.get("map"),
        "params": p,
        "cameras": cameras,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": {
            "camera_tracks": [c["tracks"] for c in cameras],
            "tracks_total": len(tracks),
            "visitors": len(visitors),
            "visitors_multi_cam": n_multi,
            "visitors_bridged": bridged,      # same-camera fragments joined through the other camera
            "links": len(links),
            "links_overlap": sum(1 for l in links if l["kind"] == "overlap"),
            "links_handoff": sum(1 for l in links if l["kind"] == "handoff"),
        },
        "links": links,
        "visitors": visitors,
    }


def save_fusion(result: dict, name: str) -> dict[str, Path]:
    FUSION_DIR.mkdir(parents=True, exist_ok=True)
    js = FUSION_DIR / f"{name}.json"
    js.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    cs = FUSION_DIR / f"{name}_tracks.csv"
    with cs.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "visitor_id", "map_x", "map_y", "n_cams", "cams"])
        for v in result["visitors"]:
            cams = ";".join(str(c) for c in v["cams"])
            for t, x, y, n in v["trail"]:
                w.writerow([t, v["id"], x, y, n, cams])
    return {"json": js, "csv": cs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", help="scene name (data/scenes/<name>.json)")
    ap.add_argument("--max-dist", type=float, default=DEFAULT_PARAMS["max_dist_px"], help="map px")
    ap.add_argument("--min-overlap", type=float, default=DEFAULT_PARAMS["min_overlap_s"])
    ap.add_argument("--max-gap", type=float, default=DEFAULT_PARAMS["max_gap_s"])
    ap.add_argument("--min-sim", type=float, default=DEFAULT_PARAMS["min_sim"], help="clothing similarity gate, 0 = off")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    res = fuse_scene(load_scene(a.scene), {"max_dist_px": a.max_dist, "min_overlap_s": a.min_overlap,
                                           "max_gap_s": a.max_gap, "min_sim": a.min_sim}, name=a.scene)
    print(json.dumps({"cameras": res["cameras"], "stats": res["stats"]}, indent=1))
    for l in res["links"]:
        extra = (f"overlap {l['overlap_s']:5.1f}s  d {l['d_med_px']:5.1f}px  close {l['close_frac']:.2f}"
                 if l["kind"] == "overlap" else
                 f"gap {l['gap_s']:5.2f}s  d {l['dist_px']:5.1f}px / {l['allowed_px']:.0f}")
        print(f"  cam{l['a']['cam']} #{l['a']['track_id']:<4} ↔ cam{l['b']['cam']} #{l['b']['track_id']:<4}"
              f"  {l['kind']:8s} {extra}  sim {l.get('sim', float('nan')):.2f}  cost {l['cost']:.2f}")
    if not a.no_save:
        paths = save_fusion(res, a.scene)
        print("→", paths["json"].relative_to(ROOT), "and", paths["csv"].relative_to(ROOT))


if __name__ == "__main__":
    main()
