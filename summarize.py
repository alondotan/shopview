"""
Build a one-row-per-shopper research summary from the video-analytics log.

Every field is derived exclusively from camera-observable events.
Nothing from the ground-truth simulation is used.
"""

from __future__ import annotations
import csv
import re
import argparse
from collections import defaultdict

# ── Day order for sorting ──────────────────────────────────────────────────
_DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

FIELDNAMES = [
    # Identity
    'day', 'shopper_id',
    # Estimated demographics (visible from video)
    'gender', 'age_group',
    # Store visit
    'entry_time', 'exit_time', 'dwell_min',
    # Browsing
    'sections_visited', 'n_sections', 'browsing_min',
    # Fitting room
    'approached_fitting', 'used_fitting_room', 'fitting_wait_min', 'gave_up_fitting',
    # Checkout queue
    'joined_queue', 'initial_queue_position', 'queue_wait_min', 'abandoned_queue',
    # Purchase
    'reached_counter', 'purchased',
    # Funnel stage (derived)
    'funnel_stage',
]


def _ts_to_min(ts: str) -> int:
    h, m = map(int, ts.split(':'))
    return h * 60 + m


def _funnel_stage(rec: dict, sections: list) -> str:
    if rec['purchased']:              return 'purchased'
    if rec['reached_counter']:        return 'at_counter'
    if rec['joined_queue']:           return 'queued'
    if rec['abandoned_queue']:        return 'abandoned_queue'
    if rec['used_fitting_room']:      return 'used_fitting'
    if rec['gave_up_fitting']:        return 'gave_up_fitting'
    if rec['approached_fitting']:     return 'approached_fitting'
    if sections:                      return 'browsed'
    return 'entered_only'


def build_summary(video_csv: str, out_csv: str) -> None:
    # Group events by (day, shopper_id)
    shopper_events: dict[tuple, list] = defaultdict(list)
    with open(video_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['day'], row['shopper_id'])
            shopper_events[key].append(row)

    records: list[dict] = []

    for (day, sid), events in shopper_events.items():
        rec: dict = dict.fromkeys(FIELDNAMES)
        rec['day']        = day
        rec['shopper_id'] = sid

        sections: list[str] = []
        browsing_ticks = 0
        queue_ticks    = 0
        entry_ts = exit_ts = None

        for e in events:
            action  = e['action']
            details = e['details']

            match action:
                case 'ENTERED':
                    entry_ts = e['timestamp']
                    gm = re.search(r'gender~(\w+)', details)
                    am = re.search(r'age~(\w+)', details)
                    rec['gender']    = gm.group(1) if gm else None
                    rec['age_group'] = am.group(1) if am else None

                case 'EXITED':
                    exit_ts = e['timestamp']

                case 'ARRIVED_SECTION':
                    if details and details not in sections:
                        sections.append(details)

                case 'BROWSING':
                    browsing_ticks += 1

                case 'IN_QUEUE':
                    queue_ticks += 1

                case 'JOINED_QUEUE':
                    rec['joined_queue'] = True
                    pm = re.search(r'position #(\d+)', details)
                    if pm and rec['initial_queue_position'] is None:
                        rec['initial_queue_position'] = int(pm.group(1))

                case 'ABANDONED_QUEUE':
                    rec['abandoned_queue'] = True

                case 'ARRIVED_FITTING_AREA':
                    rec['approached_fitting'] = True

                case 'ENTERED_FITTING':
                    rec['used_fitting_room'] = True
                    wm = re.search(r'waited (\d+) ticks', details)
                    if wm:
                        rec['fitting_wait_min'] = int(wm.group(1))

                case 'GAVE_UP_FITTING':
                    rec['gave_up_fitting'] = True

                case 'REACHED_COUNTER':
                    rec['reached_counter'] = True

                case 'CHECKOUT_COMPLETED':
                    rec['purchased'] = True

        # Derived fields
        rec['entry_time'] = entry_ts
        rec['exit_time']  = exit_ts
        if entry_ts and exit_ts:
            rec['dwell_min'] = _ts_to_min(exit_ts) - _ts_to_min(entry_ts)

        rec['sections_visited'] = ';'.join(sections)
        rec['n_sections']       = len(sections)
        rec['browsing_min']     = browsing_ticks
        rec['queue_wait_min']   = queue_ticks

        # Boolean defaults
        for field in ('approached_fitting', 'used_fitting_room', 'gave_up_fitting',
                      'joined_queue', 'abandoned_queue', 'reached_counter', 'purchased'):
            if rec[field] is None:
                rec[field] = False

        rec['funnel_stage'] = _funnel_stage(rec, sections)

        records.append(rec)

    # Sort by day order then shopper_id
    records.sort(key=lambda r: (
        _DAY_ORDER.index(r['day']) if r['day'] in _DAY_ORDER else 99,
        int(r['shopper_id']),
    ))

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(records)

    # Print quick summary
    from collections import Counter
    stages = Counter(r['funnel_stage'] for r in records)
    days   = Counter(r['day'] for r in records)
    print(f"Shoppers summarised: {len(records)}")
    print(f"Output: {out_csv}\n")
    print("Funnel stages:")
    for stage, n in stages.most_common():
        print(f"  {stage:<22} {n:>4}  ({n/len(records)*100:.0f}%)")
    print("\nPer day:")
    for d in _DAY_ORDER:
        n = days.get(d, 0)
        bought = sum(1 for r in records if r['day'] == d and r['purchased'])
        print(f"  {d:<12} {n:>4} shoppers  {bought:>3} purchases "
              f"({bought/n*100:.0f}%)" if n else f"  {d:<12}  —")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise video log to one row per shopper")
    parser.add_argument('--input',  default='week_log_video.csv', help='Video log CSV')
    parser.add_argument('--output', default='week_summary.csv',   help='Output summary CSV')
    args = parser.parse_args()
    build_summary(args.input, args.output)


if __name__ == '__main__':
    main()
