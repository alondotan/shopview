"""Interactive research chat over week_summary.csv — video-analytics data only."""

from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv()

SUMMARY_FILE = Path(__file__).parent / "week_summary.csv"
MODEL = "claude-sonnet-4-6"

FIELD_DOCS = """
day                   — day of week (Monday–Sunday)
shopper_id            — unique per day (resets each day)
gender / age_group    — estimated from video (may be unknown)
entry_time / exit_time / dwell_min  — visit timing and duration in minutes
sections_visited      — semicolon-separated list of sections browsed
n_sections            — number of distinct sections visited
browsing_min          — ticks spent browsing at shelves
approached_fitting    — reached the fitting-room area
used_fitting_room     — actually entered a fitting room
fitting_wait_min      — ticks waited before entering fitting room
gave_up_fitting       — left fitting area without entering (queue too long)
joined_queue          — joined the checkout queue
initial_queue_position — position in queue when they joined
queue_wait_min        — ticks spent waiting in queue
abandoned_queue       — left the queue without buying
reached_counter       — reached the payment counter
purchased             — completed a purchase
funnel_stage          — categorical: entered_only → browsed → approached_fitting →
                        used_fitting / gave_up_fitting → queued → abandoned_queue →
                        at_counter → purchased
""".strip()


def load_csv(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def system_prompt(csv_text: str) -> str:
    return f"""You are a retail analytics assistant helping a store manager understand shopper behavior.

The data below comes from one week of a clothing store, captured entirely through video analytics \
(camera-observable events only — no private data). The store is open 10:00–21:00. \
Shopper IDs reset to 1 each day, so the unique key is (day, shopper_id).

<data>
{csv_text}
</data>

Field guide:
{FIELD_DOCS}

Answer questions based on this data. Compute statistics, compare days, spot bottlenecks, \
and give concise actionable recommendations. When you quote numbers, cite them from the data."""


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not found — add it to the .env file in this folder.")

    if not SUMMARY_FILE.exists():
        sys.exit(f"{SUMMARY_FILE.name} not found — run:  python summarize.py")

    csv_text = load_csv(SUMMARY_FILE)
    row_count = csv_text.count("\n") - 1  # subtract header

    client = anthropic.Anthropic(api_key=api_key)
    system = system_prompt(csv_text)
    history: list[dict] = []

    print(f"Store Analytics Chat  ({row_count} shoppers, {SUMMARY_FILE.name})")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        reply = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=2048,
            system=system,
            messages=history,
        ) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                reply += chunk

        print("\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
