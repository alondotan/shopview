"""Live pipeline — camera / RTSP / file → detect + track → online stitching → events.

The offline analyser (``detect_people.py``) sees the whole video and stitches at
the end; a stream has no end. This module runs the same detector, tracker and
clothing descriptor frame by frame, and replaces the final stitching pass with
:class:`stitch.OnlineStitcher` — a lost-track pool that new ids are matched
against after a short decision window, so a visitor keeps one id across a
shelf or a crossing without waiting for the stream to finish.

Three things are done differently from the offline path because the source is
live:

* **Ingest** (:class:`FrameSource`): a reader thread always keeps only the
  *latest* frame and drops the backlog, so a slow moment never turns into a
  growing delay; timestamps come from the wall clock, not frame counts;
  the capture reconnects with back-off when the stream drops.
* **Rate**: instead of "every Nth frame", the pipeline processes a frame
  whenever ``1 / target_fps`` seconds have passed (about 6 per second, what
  IoU tracking needs); the tracker's lost buffer is derived from that.
* **Held objects** (:class:`ObjectWorker`): YOLO-World is the expensive part
  and bags do not change every frame, so it runs in its own thread about once
  a second on the most recent frame, and its results are attributed to the
  visitor ids current at that moment.

Output is a stream of events (dicts), each also appended to a JSONL file when
asked, and kept in a ring buffer for the HTTP API:

    {"type": "people",  "seq", "t", "ts", "rows": [<person rows>]}
    {"type": "objects", "seq", "t", "ts", "objects": [<object rows>]}
    {"type": "merge",   "seq", "t", "from", "to", "visitor", "sim", ...}
    {"type": "status",  ...}

Person rows are the same fields as the offline CSV (``track_id`` = visitor
after stitching, ``raw_id`` = tracker id) and are released ``decide_after_s``
seconds late, once the id is final. ``t`` is seconds since the pipeline
started, ``ts`` the Unix time of the frame.

CLI (a file plays at its own speed, like a camera; ``--fast`` reads it as fast
as possible with no drops, for tests):

    python vision/live.py rtsp://user:pass@cam/stream --out data/live/cam1.jsonl
    python vision/live.py 0                                     # webcam
    python vision/live.py data/videos/KMJS66jBtVQ.mp4 --seconds 60 --fast
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from detect_people import (
    DEFAULT_OBJECT_CLASSES, DEFAULT_OBJECT_MODEL, PERSON_CLASS, TRACK_BUFFER_S,
    TRACK_LOW_THRESH, TARGET_TRACK_FPS, _load_model, detect_objects, extract_people,
    load_object_model, person_row, tracker_config,
)
from stitch import Embedder, OnlineStitcher

ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "data" / "live"


# ── ingest ────────────────────────────────────────────────────────────────
class FrameSource:
    """Reader thread that keeps only the newest frame.

    ``source``: camera index (``0``), URL (``rtsp://…``, ``http://…``) or a file
    path. Files are paced at their own fps so the pipeline sees them as a
    camera would; with ``fast=True`` every frame is handed over and nothing is
    dropped (offline-style test runs). On read failure the capture is reopened
    with exponential back-off (streams) or the source ends (files).
    """

    def __init__(self, source, fast: bool = False, reconnect_s: float = 1.0,
                 reconnect_max_s: float = 30.0):
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.fast = fast and self.is_file
        self.reconnect_s, self.reconnect_max_s = reconnect_s, reconnect_max_s
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._taken = 0           # seq of the last frame a consumer took (fast mode gate)
        self._stop = threading.Event()
        self.ended = False
        self.connected = False
        self.fps = 0.0
        self.size = (0, 0)
        self.frames_read = 0
        self.frames_dropped = 0
        self.reconnects = 0
        self.last_error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="frame-source")

    def start(self) -> "FrameSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # ask the backend not to queue frames
        except Exception:                             # noqa: BLE001 — backend-dependent
            pass
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cap

    def _run(self) -> None:
        backoff = self.reconnect_s
        cap = None
        t_file0 = None
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.connected = False
                    self.last_error = f"cannot open {self.source}"
                    if self.is_file:
                        break
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, self.reconnect_max_s)
                    self.reconnects += 1
                    continue
                self.connected = True
                backoff = self.reconnect_s
                t_file0 = time.time()
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                if self.is_file:
                    break
                self.connected = False
                self.last_error = "read failed — reconnecting"
                self.reconnects += 1
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.reconnect_max_s)
                continue
            self.frames_read += 1
            if self.is_file and not self.fast and self.fps > 0:
                # pace the file like a camera: frame k is due at k / fps
                due = t_file0 + self.frames_read / self.fps
                delay = due - time.time()
                if delay > 0:
                    self._stop.wait(delay)
            with self._cond:
                if self.fast:
                    # no drops: wait until the consumer took the previous frame
                    while self._seq != self._taken and not self._stop.is_set():
                        self._cond.wait(0.05)
                elif self._seq != self._taken:
                    self.frames_dropped += 1
                self._frame, self._ts = frame, time.time()
                self._seq += 1
                self._cond.notify_all()
        if cap is not None:
            cap.release()
        self.connected = False
        with self._cond:
            self.ended = True
            self._cond.notify_all()

    def latest(self, timeout: float = 1.0):
        """(frame, ts, seq) newer than the last one taken, or None on timeout /
        end of source."""
        with self._cond:
            while self._seq == self._taken and not self.ended and not self._stop.is_set():
                if not self._cond.wait(timeout):
                    return None
            if self._seq == self._taken:
                return None
            self._taken = self._seq
            self._cond.notify_all()
            return self._frame, self._ts, self._seq


# ── held objects, off the critical path ───────────────────────────────────
class ObjectWorker:
    """Runs the held-object detector in its own thread on the most recent
    frame it was given, at most once per ``interval_s``. ``results()`` drains
    what it found, each tagged with the frame's ``seq`` and ``t`` and the
    people (raw ids + boxes) that were in that frame."""

    def __init__(self, model, class_ids, object_conf, imgsz, device, interval_s=1.0):
        self.model, self.class_ids = model, class_ids
        self.object_conf, self.imgsz, self.device = object_conf, imgsz, device
        self.interval_s = interval_s
        self._job = None
        self._cond = threading.Condition()
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.runs = 0
        self.last_ms = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="object-worker")

    def start(self) -> "ObjectWorker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def submit(self, frame, people_here, t, seq) -> None:
        with self._cond:
            self._job = (frame, people_here, t, seq)     # newer frame replaces the old one
            self._cond.notify()

    def results(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def _run(self) -> None:
        last = 0.0
        while not self._stop.is_set():
            with self._cond:
                while self._job is None and not self._stop.is_set():
                    self._cond.wait(0.2)
                if self._stop.is_set():
                    return
                wait = self.interval_s - (time.time() - last)
                if wait > 0:
                    self._cond.wait(wait)
                    continue
                frame, people_here, t, seq = self._job
                self._job = None
            t0 = time.time()
            _, found = detect_objects(self.model, frame, people_here, self.object_conf,
                                      self.imgsz, self.device, self.class_ids)
            self.last_ms = (time.time() - t0) * 1000
            self.runs += 1
            last = time.time()
            self._out.put({"t": t, "seq": seq, "objects": found})


# ── the pipeline ──────────────────────────────────────────────────────────
class LivePipeline:
    """See the module docstring. Construct, then ``start()`` (own thread) or
    ``run()`` (blocking). ``events`` is a ring buffer of the latest events;
    ``subscribe()`` returns a queue that receives every new one (SSE)."""

    def __init__(
        self,
        source,
        model_name: str = "yolo11n-pose.pt",
        conf: float = 0.35,
        imgsz: int = 640,
        device: str = "cpu",
        target_fps: float = TARGET_TRACK_FPS,
        tracker: str = "botsort",
        reid: bool = True,
        reid_model: str = "auto",
        track_buffer_s: float = TRACK_BUFFER_S,
        stitch: bool = True,
        stitch_gap_s: float = 8.0,
        stitch_min_sim: float = 0.6,
        decide_after_s: float = 2.0,
        embed: str | None = None,
        objects: bool = True,
        object_model: str = DEFAULT_OBJECT_MODEL,
        object_classes: list[str] | None = None,
        object_conf: float = 0.25,
        object_interval_s: float = 1.0,
        fast: bool = False,
        max_seconds: float | None = None,
        jsonl: Path | str | None = None,
        on_event=None,
        ring: int = 2000,
    ):
        self.source_spec = source
        self.model_name, self.conf, self.imgsz, self.device = model_name, conf, imgsz, device
        self.target_fps = target_fps
        self.tracker_name, self.reid, self.reid_model = tracker, reid, reid_model
        self.track_buffer_s = track_buffer_s
        self.stitch_on = stitch
        self.stitcher = OnlineStitcher(stitch_gap_s, stitch_min_sim, decide_after_s) if stitch else None
        self.decide_after_s = decide_after_s if stitch else 0.0
        self.embed_name = embed
        self.objects_on = objects
        self.object_model_name = object_model
        self.object_classes = object_classes or list(DEFAULT_OBJECT_CLASSES)
        self.object_conf, self.object_interval_s = object_conf, object_interval_s
        self.fast, self.max_seconds = fast, max_seconds
        self.jsonl = Path(jsonl) if jsonl else None
        self.on_event = on_event
        self.events: deque = deque(maxlen=ring)
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.seq = 0
        self.started_at = None
        self.finished_at = None
        self.error = None
        self.state = "new"
        self.frames_processed = 0
        self.people_now = 0
        self.last_frame_ms = 0.0
        self._recent: deque = deque(maxlen=60)       # wall-clock times of recent processed frames
        self._snapshot = None
        self._snapshot_lock = threading.Lock()
        self.source: FrameSource | None = None
        self.objects_worker: ObjectWorker | None = None
        self._pending_objects: list[dict] = []

    # ── control ──
    def start(self) -> "LivePipeline":
        self._thread = threading.Thread(target=self.run, daemon=True, name="live-pipeline")
        self._thread.start()
        return self

    def stop(self, wait: float = 10.0) -> None:
        self._stop.set()
        if self.source is not None:
            self.source.stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(wait)

    def subscribe(self, maxsize: int = 500) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def status(self) -> dict:
        src = self.source
        return {
            "state": self.state, "source": str(self.source_spec), "error": self.error,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "frames_processed": self.frames_processed, "people_now": self.people_now,
            "frame_ms": round(self.last_frame_ms, 1), "target_fps": self.target_fps,
            "achieved_fps": self._achieved_fps(),
            "seq": self.seq,
            "ingest": (None if src is None else {
                "connected": src.connected, "ended": src.ended, "fps": round(src.fps, 2),
                "size": src.size, "frames_read": src.frames_read,
                "frames_dropped": src.frames_dropped, "reconnects": src.reconnects,
                "last_error": src.last_error}),
            "objects": (None if self.objects_worker is None else {
                "runs": self.objects_worker.runs, "last_ms": round(self.objects_worker.last_ms, 1),
                "interval_s": self.object_interval_s}),
            "stitch": (None if self.stitcher is None else {
                "merges": len(self.stitcher.merges), "pool": len(self.stitcher.frags),
                "pending": len(self.stitcher.pending), "decide_after_s": self.decide_after_s,
                "min_sim": self.stitcher.min_sim, "max_gap_s": self.stitcher.max_gap_s}),
        }

    def _achieved_fps(self) -> float:
        """Processed frames per second over the last ~10 s of wall clock."""
        now = time.time()
        recent = [x for x in self._recent if now - x <= 10.0]
        if len(recent) < 2:
            return 0.0
        return round((len(recent) - 1) / max(recent[-1] - recent[0], 1e-6), 2)

    def snapshot_jpeg(self) -> bytes | None:
        with self._snapshot_lock:
            return self._snapshot

    # ── events ──
    def _emit(self, ev: dict) -> None:
        self.seq += 1
        ev = {"seq": self.seq, **ev}
        self.events.append(ev)
        if getattr(self, "_jsonl_fh", None) is not None:
            with self._jsonl_fh_lock:
                self._jsonl_fh.write(json.dumps(ev) + "\n")
                self._jsonl_fh.flush()
        if self.on_event is not None:
            self.on_event(ev)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass                                   # slow consumer: it misses this one

    def events_since(self, seq: int, limit: int = 500) -> list[dict]:
        return [e for e in list(self.events) if e["seq"] > seq][:limit]

    # ── main loop ──
    def run(self) -> None:
        self._jsonl_fh_lock = threading.Lock()
        self._jsonl_fh = None
        try:
            self._run()
        except Exception as e:                         # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            import traceback
            traceback.print_exc()
        finally:
            if self.objects_worker is not None:
                self.objects_worker.stop()
            if self.source is not None:
                self.source.stop()
            self.finished_at = time.time()
            if self.state != "error":
                self.state = "stopped"
            self._emit({"type": "status", **self.status()})
            if self._jsonl_fh is not None:
                self._jsonl_fh.close()
                self._jsonl_fh = None

    def _run(self) -> None:
        self.state = "loading"
        model = _load_model(self.model_name)
        embedder = Embedder(self.embed_name, self.device) if self.embed_name else None
        if self.objects_on:
            obj_model, class_ids = load_object_model(self.object_model_name, self.object_classes)
            self.objects_worker = ObjectWorker(obj_model, class_ids, self.object_conf, self.imgsz,
                                               self.device, self.object_interval_s).start()
        tracker_path, tracker_cfg = tracker_config(
            self.tracker_name, self.reid, self.target_fps, 1, self.conf,
            self.track_buffer_s, self.reid_model)
        det_conf = min(self.conf, TRACK_LOW_THRESH)
        if self.jsonl is not None:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_fh = self.jsonl.open("a", encoding="utf-8")

        self.source = FrameSource(self.source_spec, fast=self.fast).start()
        self.started_at = time.time()
        self.state = "running"
        self._emit({"type": "status", "tracker": {"file": tracker_path, **tracker_cfg},
                    **self.status()})

        t_start = self.started_at
        t = 0.0
        next_due = 0.0
        last_status = time.time()
        min_dt = 1.0 / self.target_fps
        height = None
        # file in fast mode: use the file's own clock so results are comparable
        # with the offline analyser; otherwise the wall clock
        file_clock = self.fast and self.source.is_file

        while not self._stop.is_set():
            got = self.source.latest(timeout=1.0)
            if got is None:
                if self.source.ended:
                    break
                continue
            frame, ts, seq = got
            if file_clock:
                t = (self.source.frames_read - 1) / max(self.source.fps, 1e-6)
            else:
                t = ts - t_start
            if self.max_seconds is not None and t > self.max_seconds:
                break
            if file_clock:
                # every frame of the file is offered; keep one in round(fps/target)
                stride = max(1, round(self.source.fps / self.target_fps))
                if (self.source.frames_read - 1) % stride:
                    continue
            else:
                if t < next_due:
                    continue                           # rate limit to target_fps
                next_due = t + min_dt
            if height is None:
                height = frame.shape[0]

            t0 = time.time()
            results = model.track(frame, persist=True, classes=[PERSON_CLASS], conf=det_conf,
                                  imgsz=self.imgsz, device=self.device, tracker=tracker_path,
                                  verbose=False)[0]
            people = extract_people(results, frame, height, appearance_on=self.stitch_on)
            if embedder is not None and people:
                embs = embedder(frame, [p["box"] for p in people], [p["kp"] for p in people])
                for p, e in zip(people, embs):
                    p["emb"] = e
            self.frames_processed += 1
            self.people_now = len(people)

            rows = []
            for p in people:
                row = person_row(p, seq, t)
                row["ts"] = round(ts, 3)
                rows.append(row)
                if self.stitcher is not None:
                    self.stitcher.observe(p["raw_id"], t, p["foot"], p["h"], p["torso"],
                                          p["legs"], p.get("emb"))
                    self.stitcher.push(row)
            if self.stitcher is not None:
                for m in self.stitcher.step(t):
                    self._emit({"type": "merge", "ts": round(ts, 3), **m})
                released = self.stitcher.release(t)
            else:
                released = rows
            if released:
                self._emit({"type": "people", "t": round(released[-1]["time_s"], 3),
                            "ts": released[-1]["ts"], "rows": released})

            if self.objects_worker is not None:
                self.objects_worker.submit(frame, [(p["raw_id"], p["box"]) for p in people], t, seq)
                self._pending_objects.extend(self.objects_worker.results())
                self._flush_objects(t)

            self.last_frame_ms = (time.time() - t0) * 1000
            self._recent.append(time.time())
            self._draw_snapshot(frame, people, t)
            if time.time() - last_status >= 5.0:
                last_status = time.time()
                self._emit({"type": "status", **self.status()})

        # end of stream: release everything that is still waiting
        if self.stitcher is not None:
            released = self.stitcher.release(t, flush=True)
            if released:
                self._emit({"type": "people", "t": round(released[-1]["time_s"], 3),
                            "ts": released[-1]["ts"], "rows": released})
        self._flush_objects(float("inf"))

    def _flush_objects(self, t: float) -> None:
        """Objects wait like people rows do, so their holder id is final."""
        keep = []
        for job in self._pending_objects:
            if t - job["t"] >= self.decide_after_s:
                objs = []
                for o in job["objects"]:
                    o = dict(o)
                    if o["person_id"] != "" and self.stitcher is not None:
                        o["person_id"] = self.stitcher.visitor(o["person_id"])
                    objs.append(o)
                if objs:
                    self._emit({"type": "objects", "t": round(job["t"], 3), "frame": job["seq"],
                                "objects": objs})
            else:
                keep.append(job)
        self._pending_objects = keep

    def _draw_snapshot(self, frame, people, t) -> None:
        canvas = frame.copy()
        for p in people:
            x1, y1, x2, y2 = (int(v) for v in p["box"])
            vid = self.stitcher.visitor(p["raw_id"]) if self.stitcher is not None else p["raw_id"]
            col = _colour(vid)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
            label = f"#{vid}" + (f" ({p['raw_id']})" if vid != p["raw_id"] else "")
            cv2.putText(canvas, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            fx, fy = (int(v) for v in p["foot"])
            cv2.circle(canvas, (fx, fy), 4, col, -1 if p["foot_src"] == "box" else 1)
        cv2.putText(canvas, f"t={t:7.1f}s  people={len(people)}  {self.last_frame_ms:.0f} ms/frame",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._snapshot_lock:
                self._snapshot = buf.tobytes()


def _colour(i: int) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="camera index, rtsp/http URL, or video file")
    ap.add_argument("--out", type=Path, default=None, help="append events as JSONL here")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much stream time")
    ap.add_argument("--fast", action="store_true", help="file only: no pacing, no drops")
    ap.add_argument("--fps", type=float, default=TARGET_TRACK_FPS, help="analysed frames per second")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tracker", default="botsort")
    ap.add_argument("--no-reid", action="store_true")
    ap.add_argument("--reid-model", default="auto")
    ap.add_argument("--track-buffer", type=float, default=TRACK_BUFFER_S)
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--stitch-gap", type=float, default=8.0)
    ap.add_argument("--stitch-sim", type=float, default=0.6)
    ap.add_argument("--decide-after", type=float, default=2.0,
                    help="seconds a new id is watched before it is matched to a lost one (= output delay)")
    ap.add_argument("--embed", default=None, help="learned appearance model, e.g. yolo11n-cls.pt")
    ap.add_argument("--no-objects", action="store_true")
    ap.add_argument("--object-model", default=DEFAULT_OBJECT_MODEL)
    ap.add_argument("--object-classes", default=",".join(DEFAULT_OBJECT_CLASSES))
    ap.add_argument("--object-conf", type=float, default=0.25)
    ap.add_argument("--object-interval", type=float, default=1.0, help="seconds between object detections")
    ap.add_argument("--quiet", action="store_true", help="do not print events")
    a = ap.parse_args()

    def printer(ev: dict) -> None:
        if a.quiet:
            return
        if ev["type"] == "people":
            print(f"t={ev['t']:7.1f}s  {len(ev['rows'])} rows  visitors "
                  f"{sorted(set(r['track_id'] for r in ev['rows']))}")
        elif ev["type"] == "merge":
            print(f"t={ev['t']:7.1f}s  MERGE raw {ev['to']} -> visitor {ev['visitor']} "
                  f"(gap {ev['gap_s']}s, sim {ev['sim']})")
        elif ev["type"] == "objects":
            print(f"t={ev['t']:7.1f}s  objects: " +
                  ", ".join(f"{o['label']}@{o['person_id'] or '-'}" for o in ev["objects"]))
        else:
            s = {k: ev.get(k) for k in ("state", "frames_processed", "frame_ms", "ingest")}
            print(f"status: {s}")

    pipe = LivePipeline(
        a.source, model_name=a.model, conf=a.conf, imgsz=a.imgsz, device=a.device,
        target_fps=a.fps, tracker=a.tracker, reid=not a.no_reid, reid_model=a.reid_model,
        track_buffer_s=a.track_buffer, stitch=not a.no_stitch, stitch_gap_s=a.stitch_gap,
        stitch_min_sim=a.stitch_sim, decide_after_s=a.decide_after, embed=a.embed,
        objects=not a.no_objects, object_model=a.object_model,
        object_classes=[c.strip() for c in a.object_classes.split(",") if c.strip()],
        object_conf=a.object_conf, object_interval_s=a.object_interval,
        fast=a.fast, max_seconds=a.seconds, jsonl=a.out, on_event=printer,
    )
    try:
        pipe.run()
    except KeyboardInterrupt:
        pipe.stop()
    st = pipe.status()
    print("\n── summary ──")
    for k in ("state", "frames_processed", "ingest", "objects", "stitch", "error"):
        print(f"{k:18s}: {st[k]}")


if __name__ == "__main__":
    main()
