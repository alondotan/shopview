"""Serve viewer.html and proxy chat requests to Anthropic."""

from __future__ import annotations
import os
import sys
from pathlib import Path

from flask import Flask, Response, request, send_file, stream_with_context

import llm_stream

HERE = Path(__file__).parent

# Load .env manually — no dependency needed. Blank values are skipped on
# purpose: an empty ANTHROPIC_API_KEY= still wins its slot in the SDK's
# credential order and authenticates as nobody, shadowing a working
# `ant auth login`. Leaving the line blank should mean "use my login".
_env_file = HERE / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            if _v.strip():
                os.environ.setdefault(_k.strip(), _v.strip())

app = Flask(__name__)

# Video-analytics endpoints under /api/vision — optional: only mounted when
# the vision extras (onnxruntime, opencv, yt-dlp) are installed.
sys.path.insert(0, str(HERE / "vision"))
try:
    from vision_api import bp as vision_bp
    app.register_blueprint(vision_bp)
    VISION_ENABLED = True
except ImportError as _e:
    print(f"[vision] disabled ({_e}) — pip install -r vision/requirements.txt")
    VISION_ENABLED = False

SUMMARY_FILE = next(
    (HERE / f for f in ("week_summary_new.csv", "week_summary.csv") if (HERE / f).exists()),
    HERE / "week_summary.csv",
)
VIDEO_FILE = next(
    (HERE / f for f in ("week3_log_video.csv", "week2_log_video.csv", "week_log_video.csv", "day_log_video.csv") if (HERE / f).exists()),
    None,
)
MODEL = llm_stream.MODEL

FIELD_DOCS = """
day                  – day of week (Monday–Sunday)
shopper_id           – unique per day (resets each day)
gender / age_group   – estimated from video (may be unknown)
entry_time / exit_time / dwell_min – visit timing; dwell in minutes
sections_visited     – semicolon-separated sections browsed
n_sections           – distinct section count
browsing_min         – ticks spent at shelves
approached_fitting   – reached fitting-room area
used_fitting_room    – actually entered a room
fitting_wait_min     – ticks waited before entering
gave_up_fitting      – left without entering (queue too long)
joined_queue         – joined checkout queue
initial_queue_position – position when they joined
queue_wait_min       – ticks spent in queue
abandoned_queue      – left queue without buying
reached_counter      – reached payment counter
purchased            – completed a purchase
funnel_stage         – entered_only → browsed → approached_fitting →
                       used_fitting / gave_up_fitting → queued →
                       abandoned_queue → at_counter → purchased
""".strip()

def _build_system(csv_text: str) -> str:
    return f"""You are a retail analytics assistant helping a store manager understand shopper behavior.

The data below was captured through video analytics for one week in a clothing store \
(10:00–21:00 daily). Each row is one shopper visit. Shopper IDs reset each day, so the \
unique key is (day, shopper_id).

<data>
{csv_text}
</data>

Field guide:
{FIELD_DOCS}

Answer questions based on this data. Compute statistics, compare days, spot bottlenecks, \
and give concise actionable recommendations. Cite numbers from the data.

When a chart would help understanding, call the render_chart tool with a Chart.js v4 config. \
You may combine text explanation with a chart in the same response."""


@app.route("/")
def index():
    """Tab shell: Simulation | Video | Calibration | Live map | Zones | Multi-cam | Live."""
    return send_file(HERE / "vision" / "shell.html")


@app.route("/viewer")
def viewer():
    """The simulation viewer itself (shown in the Simulation tab)."""
    return send_file(HERE / "viewer.html")


@app.route("/livemap")
def livemap():
    """The real-data counterpart of the simulation viewer: the video and the plan
    side by side, plus analytics, paths and a chat over what the pipeline measured."""
    return send_file(HERE / "vision" / "livemap.html")


@app.route("/zones")
def zones():
    """Draw zone polygons on the store plan and see them projected into the camera."""
    return send_file(HERE / "vision" / "zones.html")


@app.route("/vision")
def vision_player():
    """Video player with live YOLO annotation overlay."""
    return send_file(HERE / "vision" / "player.html")


@app.route("/live")
def vision_live():
    """Live-stream demo: camera → visitors, with the annotated frame and event feed."""
    return send_file(HERE / "vision" / "live.html")


@app.route("/calibrate")
def vision_calibrate():
    """Click matching points on a camera frame and the store map."""
    return send_file(HERE / "vision" / "calibrate.html")


@app.route("/multical")
def vision_multical():
    """Joint calibration: one map, several cameras, shared landmarks."""
    return send_file(HERE / "vision" / "multical.html")


@app.route("/multiview")
def vision_multiview():
    """All cameras of a scene and the plan together, tracks fused into visitors."""
    return send_file(HERE / "vision" / "multiview.html")


@app.route("/api/video-log")
def video_log():
    if not VIDEO_FILE or not VIDEO_FILE.exists():
        return {"error": "No video log file found"}, 404
    return send_file(VIDEO_FILE, mimetype="text/csv")


@app.route("/api/summary-info")
def summary_info():
    if not SUMMARY_FILE.exists():
        return {"error": f"{SUMMARY_FILE.name} not found — run: python summarize.py"}, 404
    rows = SUMMARY_FILE.read_text(encoding="utf-8").count("\n") - 1
    return {"file": SUMMARY_FILE.name, "rows": rows}


@app.route("/api/chat", methods=["POST"])
def chat():
    if not SUMMARY_FILE.exists():
        return {"error": "week_summary.csv not found"}, 404

    system = _build_system(SUMMARY_FILE.read_text(encoding="utf-8"))
    messages = request.json.get("messages", [])
    return Response(
        stream_with_context(llm_stream.stream(system, messages, model=MODEL)),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.route("/api/auth-status")
def auth_status():
    """Which Claude credential the chats will use — so the UI can say "sign in"
    instead of surfacing a raw 401 after someone has typed out a question."""
    source, detail = llm_stream.credential_source()
    return {"signed_in": source is not None, "source": source, "detail": detail}


if __name__ == "__main__":
    # The chats need a Claude credential; the video analytics do not, so a
    # missing one is a warning rather than a refusal to start.
    _source, _detail = llm_stream.credential_source()
    print(f"[claude] {_detail}" if _source else f"[claude] {_detail} — the chat tabs will say so")
    print("http://localhost:5000")
    app.run(port=5000, debug=False)
