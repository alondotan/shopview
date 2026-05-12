"""Serve viewer.html and proxy chat requests to Anthropic."""

from __future__ import annotations
import json
import os
from pathlib import Path

from flask import Flask, Response, request, send_file, stream_with_context
import anthropic

HERE = Path(__file__).parent

# Load .env manually — no dependency needed
_env_file = HERE / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

app = Flask(__name__)
SUMMARY_FILE = next(
    (HERE / f for f in ("week_summary_new.csv", "week_summary.csv") if (HERE / f).exists()),
    HERE / "week_summary.csv",
)
VIDEO_FILE = next(
    (HERE / f for f in ("week3_log_video.csv", "week2_log_video.csv", "week_log_video.csv", "day_log_video.csv") if (HERE / f).exists()),
    None,
)
MODEL = "claude-sonnet-4-6"

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

RENDER_CHART_TOOL = {
    "name": "render_chart",
    "description": (
        "Render an interactive chart in the chat UI using Chart.js v4. "
        "Call this whenever a chart would be clearer than text. "
        "Always set options.plugins.title.display=true with a descriptive text. "
        "Use dark-friendly colors (semi-transparent rgba with sufficient brightness)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": (
                    "Complete Chart.js v4 config object: "
                    "{type, data:{labels, datasets:[{label, data, backgroundColor, ...}]}, "
                    "options:{plugins:{title:{display:true,text:'...'}}, scales:{...}}}"
                ),
            }
        },
        "required": ["config"],
    },
}


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


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.route("/")
def index():
    return send_file(HERE / "viewer.html")


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

    csv_text = SUMMARY_FILE.read_text(encoding="utf-8")
    messages  = request.json.get("messages", [])
    system    = _build_system(csv_text)
    client    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate():
        try:
            accumulated_text = ""
            current_messages = list(messages)

            # Allow up to 2 rounds (one tool call + follow-up)
            for _round in range(2):
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=system,
                    tools=[RENDER_CHART_TOOL],
                    messages=current_messages,
                )

                for block in response.content:
                    if block.type == "text":
                        accumulated_text += block.text
                        yield _sse({"type": "text", "chunk": block.text})

                    elif block.type == "tool_use" and block.name == "render_chart":
                        config = block.input.get("config", block.input)
                        yield _sse({"type": "chart", "config": config})

                if response.stop_reason != "tool_use":
                    break

                # Build tool-result turn and loop
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": "Chart rendered successfully in the UI.",
                    }
                    for b in response.content
                    if b.type == "tool_use"
                ]
                current_messages = current_messages + [
                    {"role": "assistant", "content": [b.model_dump() for b in response.content]},
                    {"role": "user",      "content": tool_results},
                ]

            # Tell the client what text to store in its history
            yield _sse({"type": "history", "text": accumulated_text})

        except Exception as e:
            yield _sse({"type": "text", "chunk": f"[Error: {e}]"})

        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")
    print("http://localhost:5000")
    app.run(port=5000, debug=False)
