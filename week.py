"""Generate a full week of store simulation data (10:00-21:00)."""

from __future__ import annotations
import csv

from simulation import Simulation, _to_video_row
from shopper import Shopper

WEEK_DAYS = [
    {"name": "Monday",    "arrival_prob": 0.16, "seed": 101, "label": "Quiet start"},
    {"name": "Tuesday",   "arrival_prob": 0.21, "seed": 102, "label": "Regular"},
    {"name": "Wednesday", "arrival_prob": 0.25, "seed": 103, "label": "Mid-week"},
    {"name": "Thursday",  "arrival_prob": 0.30, "seed": 104, "label": "Pre-weekend"},
    {"name": "Friday",    "arrival_prob": 0.42, "seed": 105, "label": "Busy"},
    {"name": "Saturday",  "arrival_prob": 0.56, "seed": 106, "label": "Peak weekend"},
    {"name": "Sunday",    "arrival_prob": 0.37, "seed": 107, "label": "Family day"},
]

FIELDNAMES = ["day", "timestamp", "shopper_id", "action", "row", "col", "details"]


def run_week(output: str = "week_log.csv") -> None:
    stem = output.removesuffix(".csv")
    video_out = f"{stem}_video.csv"

    all_truth: list[dict] = []
    all_video: list[dict] = []

    print(f"Week simulation — 7 days, 10:00–21:00 (11h)\n")

    for day in WEEK_DAYS:
        Shopper.reset_counter()
        sim = Simulation(
            duration_hours=11,
            arrival_prob=day["arrival_prob"],
            open_hour=10,
            seed=day["seed"],
            output_file=output,   # irrelevant — save=False
            day_name=day["name"],
        )
        sim.run(save=False, verbose=False)

        purchased = sum(1 for s in sim.shoppers if s.items_purchased)
        total = len(sim.shoppers)
        conv = f"{purchased / total * 100:.0f}%" if total else "—"
        print(f"  {day['name']:<12} ({day['label']:<14})  "
              f"{total:>4} shoppers   {purchased:>3} purchases  ({conv})")

        all_truth.extend(sim._log_rows)
        all_video.extend(
            v for r in sim._log_rows if (v := _to_video_row(r)) is not None
        )

    # Write combined files
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_truth)

    with open(video_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_video)

    total_all = sum(len(sim.shoppers) for sim in [])  # just for spacing
    print(f"\nGround-truth log : {output}")
    print(f"Video-analytics log: {video_out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a full week of store simulation data")
    parser.add_argument("--output", default="week_log.csv", help="Output CSV filename")
    args = parser.parse_args()
    run_week(output=args.output)
