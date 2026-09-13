"""Flask blueprint: run YOLO person-detection jobs over store videos.

Endpoints (all under /api/vision):
    GET  /videos                 – videos already downloaded locally
    POST /download  {url}        – download a video (yt-dlp), returns its id
    POST /jobs      {video, ...} – start an analysis job (runs in a thread)
    GET  /jobs                   – all jobs with status/progress
    GET  /jobs/<id>              – one job
    GET  /jobs/<id>/tracks       – tracks CSV produced by the job
    GET  /jobs/<id>/tracks.json  – same data grouped per track (for the UI),
                                   with the objects each track was seen holding
    GET  /jobs/<id>/objects      – held-objects CSV produced by the job
    GET  /jobs/<id>/preview      – annotated mp4 (if requested)
    GET  /objects/<video_id>     – objects CSV matching /tracks/<video_id>
    GET  /zones/<map>            – zone polygons drawn on a store map
    POST /zones/<map>            – save them
    POST /live/start {source,…}  – run the live pipeline on a camera / RTSP url / video
    POST /live/stop
    GET  /live/status
    GET  /live/frame.jpg         – latest annotated frame
    GET  /live/events?since=N    – poll events (people / objects / merge / status)
    GET  /live/stream            – the same as server-sent events
"""

from __future__ import annotations

import csv
import json
import re
import threading
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from detect_people import DEFAULT_OBJECT_CLASSES, DEFAULT_OBJECT_MODEL, TRACK_BUFFER_S, analyze
from download_video import download
from homography import CALIB_DIR, solve_homography

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
TRACK_DIR = ROOT / "data" / "tracks"
MAP_DIR = ROOT / "data" / "map"
ZONE_DIR = ROOT / "data" / "zones"

bp = Blueprint("vision", __name__, url_prefix="/api/vision")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_video(name: str) -> Path:
    """Accept a bare id ('KMJS66jBtVQ'), a filename, or a path under data/videos."""
    cand = Path(name)
    if cand.is_file():
        return cand
    for p in (VIDEO_DIR / name, VIDEO_DIR / f"{name}.mp4"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"video not found: {name}")


def _run_job(job_id: str, video: Path, params: dict) -> None:
    def on_progress(p: dict) -> None:
        with _lock:
            _jobs[job_id]["progress"] = p

    try:
        stats = analyze(video=video, on_progress=on_progress, **params)
        with _lock:
            _jobs[job_id].update(status="done", stats=stats, finished_at=_now(),
                                 progress={**_jobs[job_id].get("progress", {}), "percent": 100.0})
    except Exception as e:                                   # noqa: BLE001
        with _lock:
            _jobs[job_id].update(status="error", error=str(e),
                                 traceback=traceback.format_exc(), finished_at=_now())


@bp.get("/videos")
def list_videos():
    """Videos on disk, the ones with detections first — so a UI can default to one
    that actually has something to show."""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(VIDEO_DIR.glob("*.mp4")):
        out.append({
            "id": p.stem,
            "file": p.name,
            "size_mb": round(p.stat().st_size / 1e6, 1),
            "has_tracks": bool(list(TRACK_DIR.glob(f"{p.stem}*_tracks.csv"))),
            "has_objects": bool(list(TRACK_DIR.glob(f"{p.stem}*_objects.csv"))),
            "has_calibration": (CALIB_DIR / f"{p.stem}.json").exists(),
        })
    out.sort(key=lambda v: (not v["has_tracks"], not v["has_calibration"], v["id"]))
    return jsonify(out)


@bp.post("/download")
def download_video():
    url = (request.json or {}).get("url")
    if not url:
        return {"error": "url is required"}, 400
    try:
        path = download(url, max_height=(request.json or {}).get("max_height", 720))
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}, 500
    return {"id": path.stem, "file": path.name}


def _class_list(value) -> list[str]:
    """Accept a list or a comma-separated string; empty → the defaults."""
    if isinstance(value, str):
        value = value.split(",")
    value = [str(v).strip() for v in (value or []) if str(v).strip()]
    return value or list(DEFAULT_OBJECT_CLASSES)


@bp.post("/jobs")
def create_job():
    body = request.json or {}
    try:
        video = _resolve_video(body.get("video", ""))
    except FileNotFoundError as e:
        return {"error": str(e)}, 404

    job_id = uuid.uuid4().hex[:8]
    preview = TRACK_DIR / f"{video.stem}_{job_id}_preview.mp4" if body.get("preview") else None
    params = {
        "model_name": body.get("model", "yolo11n-pose.pt"),
        "conf": float(body.get("conf", 0.35)),
        "imgsz": int(body.get("imgsz", 640)),
        # 0 / missing → auto (≈6 analysed fps, what the tracker needs)
        "stride": int(body.get("stride") or 0) or None,
        "start_seconds": float(body.get("start", 0.0)),
        "max_seconds": (float(body["seconds"]) if body.get("seconds") else None),
        "device": body.get("device", "cpu"),
        "preview": preview,
        "out_csv": TRACK_DIR / f"{video.stem}_{job_id}_tracks.csv",
        # tracking: botsort + appearance features, then clothing/position stitching
        "tracker": body.get("tracker", "botsort"),
        "reid": bool(body.get("reid", True)),
        "track_buffer_s": float(body.get("track_buffer", TRACK_BUFFER_S)),
        "stitch_tracks": bool(body.get("stitch", True)),
        "stitch_gap_s": float(body.get("stitch_gap", 8.0)),
        "stitch_min_sim": float(body.get("stitch_sim", 0.6)),
        # held-object detection (YOLO-World, free-text classes) — on by default
        "objects": bool(body.get("objects", True)),
        "object_model": body.get("object_model", DEFAULT_OBJECT_MODEL),
        "object_classes": _class_list(body.get("object_classes")),
        "object_conf": float(body.get("object_conf", 0.25)),
        "objects_csv": TRACK_DIR / f"{video.stem}_{job_id}_objects.csv",
    }

    with _lock:
        _jobs[job_id] = {
            "id": job_id, "video": video.name, "status": "running",
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
            "started_at": _now(), "progress": {"percent": 0.0},
        }

    threading.Thread(target=_run_job, args=(job_id, video, params), daemon=True).start()
    return {"id": job_id, "status": "running"}, 202


@bp.get("/jobs")
def list_jobs():
    with _lock:
        return jsonify(sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True))


@bp.get("/jobs/<job_id>")
def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    return (jsonify(job), 200) if job else ({"error": "no such job"}, 404)


def _job_csv(job_id: str) -> Path | None:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return None
    path = Path(job["params"]["out_csv"])
    return path if path.exists() else None


@bp.get("/jobs/<job_id>/tracks")
def job_tracks(job_id: str):
    path = _job_csv(job_id)
    if not path:
        return {"error": "tracks not available yet"}, 404
    return send_file(path, mimetype="text/csv")


def _job_objects_csv(job_id: str) -> Path | None:
    with _lock:
        job = _jobs.get(job_id)
    if not job or not job["params"].get("objects_csv"):
        return None
    path = Path(job["params"]["objects_csv"])
    return path if path.exists() else None


@bp.get("/jobs/<job_id>/objects")
def job_objects(job_id: str):
    path = _job_objects_csv(job_id)
    if not path:
        return {"error": "objects not available (yet, or job ran without them)"}, 404
    return send_file(path, mimetype="text/csv")


def _held_items(objects_csv: Path | None) -> dict[int, dict[str, int]]:
    """track_id → {label: frames seen holding it}."""
    items: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if objects_csv and objects_csv.exists():
        with objects_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("person_id"):
                    items[int(row["person_id"])][row["label"]] += 1
    return items


@bp.get("/jobs/<job_id>/tracks.json")
def job_tracks_json(job_id: str):
    """Per-track summary: first/last seen, duration, point trail (foot position),
    and the objects the person was seen holding (label → sampled frames)."""
    path = _job_csv(job_id)
    if not path:
        return {"error": "tracks not available yet"}, 404
    held = _held_items(_job_objects_csv(job_id))

    trails: dict[int, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trails[int(row["track_id"])].append({
                "t": float(row["time_s"]),
                "x": float(row["foot_x"]),
                "y": float(row["foot_y"]),
                "conf": float(row["conf"]),
            })

    tracks = []
    for tid, pts in sorted(trails.items()):
        tracks.append({
            "track_id": tid,
            "first_seen_s": pts[0]["t"],
            "last_seen_s": pts[-1]["t"],
            "duration_s": round(pts[-1]["t"] - pts[0]["t"], 2),
            "n_points": len(pts),
            "items": dict(sorted(held.get(tid, {}).items(), key=lambda kv: -kv[1])),
            "trail": pts,
        })
    return jsonify({"n_tracks": len(tracks), "tracks": tracks})


@bp.get("/video/<video_id>")
def raw_video(video_id: str):
    """Serve the source mp4 itself (Range requests supported — needed for seeking)."""
    try:
        path = _resolve_video(video_id)
    except FileNotFoundError as e:
        return {"error": str(e)}, 404
    return send_file(path, mimetype="video/mp4", conditional=True)


@bp.get("/tracks/<video_id>")
def tracks_for_video(video_id: str):
    """Latest tracks CSV for a video — job output if there is one, else the CLI output."""
    try:
        video = _resolve_video(video_id)
    except FileNotFoundError as e:
        return {"error": str(e)}, 404

    job = request.args.get("job")
    path = _tracks_path(video, job)
    if path is None:
        return ({"error": f"no tracks for job {job}"} if job else
                {"error": f"no tracks for {video.stem} — run detect_people.py first"}), 404
    return send_file(path, mimetype="text/csv")


def _tracks_path(video: Path, job: str | None) -> Path | None:
    if job:
        path = TRACK_DIR / f"{video.stem}_{job}_tracks.csv"
        return path if path.exists() else None
    # No job asked for → the run covering most of the video, i.e. the biggest file
    candidates = sorted(TRACK_DIR.glob(f"{video.stem}*_tracks.csv"),
                        key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return candidates[0] if candidates else None


@bp.get("/objects/<video_id>")
def objects_for_video(video_id: str):
    """The objects CSV that belongs to the tracks CSV /tracks/<video_id> serves —
    the same run, so track ids line up."""
    try:
        video = _resolve_video(video_id)
    except FileNotFoundError as e:
        return {"error": str(e)}, 404

    tracks = _tracks_path(video, request.args.get("job"))
    if tracks is None:
        return {"error": f"no tracks for {video.stem}"}, 404
    path = tracks.with_name(tracks.name.replace("_tracks.csv", "_objects.csv"))
    if not path.exists():
        return {"error": f"no objects for {video.stem} — re-run detect_people.py "
                         f"(objects are on by default)"}, 404
    return send_file(path, mimetype="text/csv")


@bp.get("/jobs/<job_id>/preview")
def job_preview(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job or not job["params"].get("preview"):
        return {"error": "no preview for this job"}, 404
    path = Path(job["params"]["preview"])
    if not path.exists():
        return {"error": "preview not ready"}, 404
    return send_file(path, mimetype="video/mp4")


# ── store map + camera→map calibration ────────────────────────────────────

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str) -> str:
    return _SAFE_NAME.sub("_", Path(name).name)


@bp.get("/maps")
def list_maps():
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    return jsonify([
        {"name": p.name, "size_kb": round(p.stat().st_size / 1e3, 1)}
        for p in sorted(MAP_DIR.iterdir())
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".svg"}
    ])


@bp.post("/maps")
def upload_map():
    f = request.files.get("file")
    if not f or not f.filename:
        return {"error": "no file uploaded"}, 400
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe(f.filename)
    f.save(MAP_DIR / name)
    return {"name": name}


@bp.get("/maps/<name>")
def get_map(name: str):
    path = MAP_DIR / _safe(name)
    if not path.exists():
        return {"error": "no such map"}, 404
    return send_file(path)


@bp.get("/calibration/<video_id>")
def get_calibration(video_id: str):
    path = CALIB_DIR / f"{_safe(video_id)}.json"
    if not path.exists():
        return {"error": "not calibrated yet"}, 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@bp.post("/calibration/<video_id>")
def post_calibration(video_id: str):
    """Fit a camera→map homography from clicked point pairs.

    Body: {"points": [{"img": [x, y], "map": [x, y]}, ...],   (>= 4 pairs)
           "map": "store.png", "map_size": [w, h],
           "video_size": [w, h], "save": bool}
    """
    body = request.json or {}
    pairs = body.get("points") or []
    if len(pairs) < 4:
        return {"error": "at least 4 point pairs are required"}, 400

    try:
        result = solve_homography(
            [p["img"] for p in pairs],
            [p["map"] for p in pairs],
        )
    except Exception as e:                                   # noqa: BLE001
        return {"error": str(e)}, 400

    calib = {
        "video_id": video_id,
        "map": body.get("map"),
        "map_size": body.get("map_size"),
        "video_size": body.get("video_size"),
        "points": pairs,
        **result,
    }

    if body.get("save"):
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        path = CALIB_DIR / f"{_safe(video_id)}.json"
        path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        calib["saved_to"] = str(path.relative_to(ROOT))

    return jsonify(calib)


# ── zones: polygons drawn on the store map ─────────────────────────────────
# Zones live in map pixels, so they belong to the map image, not to a video —
# every camera calibrated against the same plan shares them.

def _zone_path(map_name: str) -> Path:
    return ZONE_DIR / f"{_safe(map_name)}.json"


@bp.get("/zones/<map_name>")
def get_zones(map_name: str):
    path = _zone_path(map_name)
    if not path.exists():
        return jsonify({"map": map_name, "zones": []})
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@bp.post("/zones/<map_name>")
def post_zones(map_name: str):
    """Body: {"zones": [{"id", "name", "color", "pts": [[x, y], ...], "visible"}, ...],
              "map_size": [w, h]}  — pts in map pixels, >= 3 per zone."""
    body = request.json or {}
    zones = body.get("zones")
    if not isinstance(zones, list):
        return {"error": "zones must be a list"}, 400
    clean = []
    for i, z in enumerate(zones):
        pts = z.get("pts") if isinstance(z, dict) else None
        if not pts or len(pts) < 3:
            return {"error": f"zone {i} needs at least 3 points"}, 400
        try:
            pts = [[float(p[0]), float(p[1])] for p in pts]
        except (TypeError, ValueError, IndexError):
            return {"error": f"zone {i} has malformed points"}, 400
        clean.append({
            "id": str(z.get("id") or f"z{i}"),
            "name": str(z.get("name") or f"Zone {i + 1}"),
            "color": str(z.get("color") or "#4cc9f0"),
            "pts": pts,
            "visible": bool(z.get("visible", True)),
        })

    ZONE_DIR.mkdir(parents=True, exist_ok=True)
    path = _zone_path(map_name)
    doc = {"map": map_name, "map_size": body.get("map_size"), "zones": clean,
           "saved_at": _now()}
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return jsonify({**doc, "saved_to": str(path.relative_to(ROOT))})


# ── live stream ───────────────────────────────────────────────────────────
# One pipeline per server (the demo has one camera). See live.py.
_live: dict = {"pipe": None}


def _live_pipe():
    return _live["pipe"]


@bp.post("/live/start")
def live_start():
    """Body: {"source": "rtsp://…" | "0" | "<video id or path>", "fps", "conf",
    "model", "tracker", "reid", "track_buffer", "stitch", "stitch_gap",
    "stitch_sim", "decide_after", "embed", "objects", "object_classes",
    "object_model", "object_conf", "object_interval", "fast", "seconds",
    "jsonl": true|false}. A video id plays at its own speed, like a camera."""
    from live import LIVE_DIR, LivePipeline

    body = request.json or {}
    pipe = _live_pipe()
    if pipe is not None and pipe.state in ("loading", "running"):
        return {"error": "already running — POST /live/stop first", "status": pipe.status()}, 409
    source = str(body.get("source", "")).strip()
    if not source:
        return {"error": "source is required"}, 400
    if not (source.isdigit() or "://" in source):
        try:
            source = str(_resolve_video(source))
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
    jsonl = None
    if body.get("jsonl", True):
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        jsonl = LIVE_DIR / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    pipe = LivePipeline(
        source,
        model_name=body.get("model", "yolo11n-pose.pt"),
        conf=float(body.get("conf", 0.35)),
        imgsz=int(body.get("imgsz", 640)),
        device=body.get("device", "cpu"),
        target_fps=float(body.get("fps", 6.0)),
        tracker=body.get("tracker", "botsort"),
        reid=bool(body.get("reid", True)),
        reid_model=body.get("reid_model", "auto"),
        track_buffer_s=float(body.get("track_buffer", TRACK_BUFFER_S)),
        stitch=bool(body.get("stitch", True)),
        stitch_gap_s=float(body.get("stitch_gap", 8.0)),
        stitch_min_sim=float(body.get("stitch_sim", 0.6)),
        decide_after_s=float(body.get("decide_after", 2.0)),
        embed=body.get("embed") or None,
        objects=bool(body.get("objects", True)),
        object_model=body.get("object_model", DEFAULT_OBJECT_MODEL),
        object_classes=_class_list(body.get("object_classes")),
        object_conf=float(body.get("object_conf", 0.25)),
        object_interval_s=float(body.get("object_interval", 1.0)),
        fast=bool(body.get("fast", False)),
        max_seconds=(float(body["seconds"]) if body.get("seconds") else None),
        jsonl=jsonl,
    ).start()
    _live["pipe"] = pipe
    return {"status": pipe.status(), "jsonl": (str(jsonl.relative_to(ROOT)) if jsonl else None)}, 202


@bp.post("/live/stop")
def live_stop():
    pipe = _live_pipe()
    if pipe is None:
        return {"error": "nothing running"}, 404
    pipe.stop(wait=15.0)
    return jsonify(pipe.status())


@bp.get("/live/status")
def live_status():
    pipe = _live_pipe()
    return jsonify(pipe.status() if pipe is not None else {"state": "idle"})


@bp.get("/live/frame.jpg")
def live_frame():
    """Latest processed frame with visitor ids drawn on it (for the demo page)."""
    pipe = _live_pipe()
    jpg = pipe.snapshot_jpeg() if pipe is not None else None
    if not jpg:
        return {"error": "no frame yet"}, 404
    from flask import Response
    return Response(jpg, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@bp.get("/live/events")
def live_events():
    """Poll: events with seq > ?since (default 0), at most ?limit (500)."""
    pipe = _live_pipe()
    if pipe is None:
        return jsonify([])
    since = int(request.args.get("since", 0))
    limit = int(request.args.get("limit", 500))
    return jsonify(pipe.events_since(since, limit))


@bp.get("/live/stream")
def live_stream():
    """Server-sent events: every pipeline event as it happens (people rows are
    ``decide_after`` seconds late by design, once their visitor id is final)."""
    import queue as _queue

    from flask import Response, stream_with_context

    pipe = _live_pipe()
    if pipe is None:
        return {"error": "nothing running"}, 404
    q = pipe.subscribe()

    def gen():
        try:
            yield "event: status\ndata: " + json.dumps(pipe.status()) + "\n\n"
            while pipe.state not in ("stopped", "error"):
                try:
                    ev = q.get(timeout=15)
                except _queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                if ev["type"] == "status" and ev.get("state") in ("stopped", "error"):
                    break
        finally:
            pipe.unsubscribe(q)

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
