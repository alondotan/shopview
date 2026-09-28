"""Streaming Claude chat over a fixed dataset, as server-sent events.

Both chats in the app are the same machinery with a different system prompt:

* `/api/chat` — the simulated weekly summary (`week_summary.csv`);
* `/api/vision/chat` — the statistics of a video the vision pipeline analysed.

Either works with a signed-in account (`ant auth login`) or a static API key —
see ``credential_source()`` below for how the choice is made.

Either way the whole dataset is small enough to sit in the system prompt, so
there is no retrieval step: the model sees every row and does the arithmetic
itself. `render_chart` lets it answer with a Chart.js figure instead of a
paragraph of numbers; the browser draws it.

The event stream the browser consumes:

    {"type": "text",    "chunk": "…"}   text as it arrives
    {"type": "chart",   "config": {…}}  a Chart.js v4 config to render
    {"type": "history", "text": "…"}    what to store as the assistant turn
    [DONE]
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

MODEL = "claude-sonnet-4-6"

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


# ── credentials ───────────────────────────────────────────────────────────
# The SDK resolves credentials itself, first match wins:
#
#   ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN → the OAuth profile left by
#   `ant auth login` → workload identity federation → the default profile
#
# so a bare ``anthropic.Anthropic()`` signs in with whichever of those exists.
# Never pass ``api_key=os.getenv(...)`` — an unset key becomes ``api_key=None``,
# which the SDK reads as "no key configured" rather than "look further down the
# chain", and an *empty* key wins its slot and authenticates as nobody. For the
# same reason the .env loader in server.py skips blank values: `ANTHROPIC_API_KEY=`
# with nothing after it would silently shadow a perfectly good login.

CONFIG_DIR = Path(os.environ.get("ANTHROPIC_CONFIG_DIR") or (Path.home() / ".config" / "anthropic"))


def credential_source() -> tuple[str | None, str]:
    """Best-effort report of what the SDK will authenticate with: a
    ``(source, human sentence)`` pair, with ``None`` when nothing is set up.

    Mirrors the SDK's order rather than asking it — the SDK has no public
    "am I signed in" call, and finding out by making a request costs money.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api_key", "API key from the environment (ANTHROPIC_API_KEY)"
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "auth_token", "bearer token from the environment (ANTHROPIC_AUTH_TOKEN)"

    profile = os.environ.get("ANTHROPIC_PROFILE")
    creds = CONFIG_DIR / "credentials"
    if creds.is_dir():
        files = sorted(creds.glob("*.json"))
        names = [f.stem for f in files]
        if profile and profile in names:
            return "oauth", f"signed in as the {profile!r} profile ({creds})"
        if names:
            shown = names[0] if len(names) == 1 else f"{len(names)} profiles, e.g. {names[0]!r}"
            return "oauth", f"signed in ({shown}, {creds})"
    return None, ("not signed in — run `ant auth login`, or put ANTHROPIC_API_KEY "
                  "in .env")


NOT_SIGNED_IN = (
    "Claude is not signed in on this machine. Run `ant auth login` "
    "(brew install anthropics/tap/ant) to sign in with your account, or put "
    "ANTHROPIC_API_KEY in the .env file next to server.py."
)


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream(system: str, messages: list[dict], *, model: str = MODEL,
           max_tokens: int = 4096, max_rounds: int = 2) -> Iterator[str]:
    """Yield the SSE lines for one exchange.

    ``max_rounds`` bounds the tool loop: the model may draw a chart and then
    write about it, which is two rounds, and that is as far as it needs to go.
    """
    import anthropic

    if credential_source()[0] is None:
        yield sse({"type": "text", "chunk": NOT_SIGNED_IN})
        yield sse({"type": "history", "text": NOT_SIGNED_IN})
        yield "data: [DONE]\n\n"
        return

    client = anthropic.Anthropic()          # resolves the chain described above
    accumulated = ""
    current = list(messages)

    try:
        for _round in range(max_rounds):
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                tools=[RENDER_CHART_TOOL], messages=current,
            )

            for block in response.content:
                if block.type == "text":
                    accumulated += block.text
                    yield sse({"type": "text", "chunk": block.text})
                elif block.type == "tool_use" and block.name == "render_chart":
                    yield sse({"type": "chart",
                               "config": block.input.get("config", block.input)})

            if response.stop_reason != "tool_use":
                break

            current = current + [
                {"role": "assistant", "content": [b.model_dump() for b in response.content]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id,
                     "content": "Chart rendered successfully in the UI."}
                    for b in response.content if b.type == "tool_use"
                ]},
            ]

        # the client stores this as the assistant turn, so a chart-only answer
        # does not leave an empty bubble in the history
        yield sse({"type": "history", "text": accumulated})

    except Exception as e:                                   # noqa: BLE001
        yield sse({"type": "text", "chunk": f"[Error: {e}]"})

    yield "data: [DONE]\n\n"
