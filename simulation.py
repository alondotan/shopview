"""Main simulation loop."""

from __future__ import annotations
import csv
import math
import re
import random

from shopper import Shopper, State
from store import (SECTION_NAMES, FITTING_CAPACITY, CHECKOUT_LANES,
                   CHECKOUT_FIRST_CELLS, to_display)

# ── Video-log filter ───────────────────────────────────────────────────────
# Events the camera never sees (internal simulation decisions)
_VIDEO_SKIP = {
    'HEADING',    # intent, not visible
    'IN_FITTING', # fitting rooms are opaque
}
# Item-interaction events: camera sees person at shelf but not the action itself
_SHELF_TO_BROWSING = {'NO_PICK', 'PICKED_ITEM', 'PUT_BACK_SHELF'}


def _to_video_row(row: dict) -> dict | None:
    """Return a sanitised copy of row for the video log, or None to drop it."""
    action = row['action']
    if action in _VIDEO_SKIP:
        return None

    details = row['details']

    if action in _SHELF_TO_BROWSING:
        # Camera sees person still at the shelf — log as BROWSING with section name
        return {**row, 'action': 'BROWSING', 'details': details}

    if action == 'ENTERED':
        # Keep estimated demographics; strip profile, plan, speed (all internal)
        parts = []
        gm = re.search(r'gender=(\w+)', details)
        am = re.search(r'age=(\w+)',    details)
        if gm: parts.append(f'gender~{gm.group(1)}')
        if am: parts.append(f'age~{am.group(1)}')
        details = ' '.join(parts)

    elif action in ('MOVING', 'PAUSED'):
        # Direction reveals intent (→ checkout / → fitting room) — camera only sees position
        details = ''

    elif action == 'AT_CHECKOUT':
        # Internal tick progress N/M is not visible; camera sees person at counter
        details = ''

    elif action == 'ARRIVED_FITTING_AREA':
        # Item count not visible from camera
        details = ''

    elif action == 'JOINED_QUEUE':
        # Keep queue position; strip item count (camera can't count carried items)
        pm = re.search(r'position #(\d+)', details)
        details = f'position #{pm.group(1)}' if pm else ''

    elif action == 'ENTERED_FITTING':
        # Wait time observable; item count is not
        wm = re.search(r'waited (\d+) ticks', details)
        details = f'waited {wm.group(1)} ticks' if wm else ''

    elif action == 'LEFT_FITTING':
        # What happened inside is not visible
        details = ''

    elif action == 'ABANDONED_QUEUE':
        # Position visible; item count is not
        pm = re.search(r'position #(\d+)', details)
        details = f'position #{pm.group(1)}' if pm else ''

    elif action == 'CHECKOUT_CALLED':
        # Rename: camera sees person reach counter, not a staff "call"
        action  = 'REACHED_COUNTER'
        details = ''

    elif action == 'PURCHASED':
        # Rename + strip items: camera sees checkout completed, not what was bought
        action  = 'CHECKOUT_COMPLETED'
        details = ''

    return {**row, 'action': action, 'details': details}


# ── Simulation ─────────────────────────────────────────────────────────────

class Simulation:
    def __init__(
        self,
        duration_hours: int = 8,
        arrival_prob: float = 0.30,
        open_hour: int = 10,
        seed: int | None = None,
        output_file: str = "store_log.csv",
        day_name: str = "Day",
    ) -> None:
        if seed is not None:
            random.seed(seed)

        self.total_ticks = duration_hours * 60
        self.arrival_prob = arrival_prob
        self.open_hour = open_hour
        self.output_file = output_file
        # video log: same stem, _video suffix
        stem = output_file.removesuffix('.csv')
        self.video_file = f"{stem}_video.csv"

        self.shoppers: list[Shopper] = []
        self.active: list[Shopper] = []
        self.checkout_queue: list[Shopper] = []
        self.fitting_state: dict = {'occupied': set(), 'capacity': FITTING_CAPACITY}
        self._log_rows: list[dict] = []
        self.day_name = day_name

    # ------------------------------------------------------------------ #

    def _timestamp(self, tick: int) -> str:
        total_min = self.open_hour * 60 + tick
        return f"{total_min // 60:02d}:{total_min % 60:02d}"

    def _arrival_prob(self, tick: int) -> float:
        """Time-of-day arrival curve: slow morning, lunch peak, afternoon peak, quiet close."""
        hour = self.open_hour + tick / 60.0
        peak1 = math.exp(-0.5 * ((hour - 13.0) / 1.5) ** 2)
        peak2 = math.exp(-0.5 * ((hour - 16.5) / 1.2) ** 2)
        curve = 0.15 + 0.85 * max(peak1, peak2)
        return self.arrival_prob * curve

    def _log(self, timestamp: str, shopper_id: int, action: str,
             pos: tuple[int, int], details: str = "") -> None:
        row_label, col_num = to_display(*pos)
        self._log_rows.append({
            "day":        self.day_name,
            "timestamp":  timestamp,
            "shopper_id": shopper_id,
            "action":     action,
            "row":        row_label,
            "col":        col_num,
            "details":    details,
        })

    # ------------------------------------------------------------------ #

    def run(self, save: bool = True, verbose: bool = True) -> None:
        if verbose:
            print(f"Simulation start — {self.total_ticks // 60}h, "
                  f"arrival_prob={self.arrival_prob} (peak), open={self.open_hour:02d}:00, "
                  f"checkout_lanes={CHECKOUT_LANES}")

        for tick in range(self.total_ticks):
            ts = self._timestamp(tick)

            if random.random() < self._arrival_prob(tick):
                s = Shopper()
                self.shoppers.append(s)
                self.active.append(s)
                plan = ", ".join(SECTION_NAMES[sec] for sec in s.sections)
                self._log(ts, s.id, "ENTERED", s.pos,
                          f"profile={s.profile.value} gender={s.gender} age={s.age_group} "
                          f"speed={s.speed} plan=[{plan}]")

            occupied = {s.pos for s in self.checkout_queue if s.state == State.AT_CHECKOUT}
            free_counters = [c for c in CHECKOUT_FIRST_CELLS if c not in occupied]
            for s in self.checkout_queue:
                if not free_counters:
                    break
                if s.state == State.IN_QUEUE:
                    counter = free_counters.pop(0)
                    s.serve_checkout(counter)
                    self._log(ts, s.id, "REACHED_COUNTER", s.pos, "")

            for s in list(self.active):
                events = s.step(self.checkout_queue, self.fitting_state)
                for action, details in events:
                    self._log(ts, s.id, action, s.pos, details)
                if s.state == State.DONE:
                    self.active.remove(s)

        if save:
            self._save()

        if verbose:
            total = len(self.shoppers)
            purchased = sum(1 for s in self.shoppers if s.items_purchased)
            print(f"Done — {total} shoppers visited, {purchased} made a purchase.")
        if save:
            print(f"Ground-truth log : {self.output_file}")
            print(f"Video-analytics log: {self.video_file}")

    def _save(self) -> None:
        fieldnames = ["day", "timestamp", "shopper_id", "action", "row", "col", "details"]

        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self._log_rows)

        video_rows = [v for r in self._log_rows if (v := _to_video_row(r)) is not None]
        with open(self.video_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(video_rows)
