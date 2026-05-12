"""Shopper agent with profile-based behaviour."""

from __future__ import annotations
import random
from dataclasses import dataclass, replace as dc_replace
from enum import Enum, auto

from store import (
    Cell, SECTION_NAMES, SHELF_TYPES, ENTRANCE_POS,
    SECTION_ACCESS, DRESSING_ROOM_CELLS, CHECKOUT_WAIT_CELLS, CHECKOUT_FIRST_CELLS,
    FITTING_WAIT_AREA, bfs,
)


# ══════════════════════════════════════════════════
#  PROFILES
# ══════════════════════════════════════════════════

class ShopperProfile(Enum):
    QUICK_BUYER    = "quick_buyer"
    BROWSER        = "browser"
    INDECISIVE     = "indecisive"
    WINDOW_SHOPPER = "window_shopper"
    REGULAR        = "regular"


@dataclass(frozen=True)
class ProfileCfg:
    speed:               tuple[int, int]  # cells per tick (min, max)
    n_sections:          tuple[int, int]  # how many shelf sections to visit
    browse_time:         tuple[int, int]  # ticks spent at each section
    checkout_time:       tuple[int, int]  # ticks at checkout counter
    p_pick:              float  # prob of picking up an item at all
    p_put_back_shelf:    float  # prob of putting it back right there (picked up → decided against)
    use_fitting_room:    float  # prob of going to fitting room if holding items
    p_keep_fitting:      float  # prob of keeping each item after trying on
    p_leave_in_room:     float  # of rejected items: prob left in room vs returned to shelf
    p_abandon_queue:          float  # prob per tick of abandoning checkout line
    p_random_pause:           float  # prob per movement tick of stopping briefly
    fitting_wait_tolerance:   int    # max ticks to wait outside fitting room
    p_abandon_fitting_wait:   float  # prob per tick of giving up on fitting room


PROFILES: dict[ShopperProfile, ProfileCfg] = {
    ShopperProfile.QUICK_BUYER: ProfileCfg(
        speed=(2, 3), n_sections=(1, 3), browse_time=(3, 7), checkout_time=(2, 4),
        p_pick=0.85, p_put_back_shelf=0.05,
        use_fitting_room=0.25, p_keep_fitting=0.92, p_leave_in_room=0.10,
        p_abandon_queue=0.03, p_random_pause=0.01,
        fitting_wait_tolerance=3, p_abandon_fitting_wait=0.25,
    ),
    ShopperProfile.BROWSER: ProfileCfg(
        speed=(1, 2), n_sections=(4, 7), browse_time=(6, 15), checkout_time=(2, 5),
        p_pick=0.35, p_put_back_shelf=0.30,
        use_fitting_room=0.55, p_keep_fitting=0.60, p_leave_in_room=0.30,
        p_abandon_queue=0.12, p_random_pause=0.06,
        fitting_wait_tolerance=8, p_abandon_fitting_wait=0.08,
    ),
    ShopperProfile.INDECISIVE: ProfileCfg(
        speed=(1, 2), n_sections=(3, 6), browse_time=(8, 18), checkout_time=(3, 7),
        p_pick=0.65, p_put_back_shelf=0.45,
        use_fitting_room=0.80, p_keep_fitting=0.50, p_leave_in_room=0.40,
        p_abandon_queue=0.08, p_random_pause=0.08,
        fitting_wait_tolerance=12, p_abandon_fitting_wait=0.04,
    ),
    ShopperProfile.WINDOW_SHOPPER: ProfileCfg(
        speed=(1, 2), n_sections=(3, 6), browse_time=(4, 10), checkout_time=(2, 4),
        p_pick=0.12, p_put_back_shelf=0.88,
        use_fitting_room=0.15, p_keep_fitting=0.20, p_leave_in_room=0.60,
        p_abandon_queue=0.40, p_random_pause=0.07,
        fitting_wait_tolerance=2, p_abandon_fitting_wait=0.35,
    ),
    ShopperProfile.REGULAR: ProfileCfg(
        speed=(1, 2), n_sections=(2, 5), browse_time=(4, 10), checkout_time=(2, 5),
        p_pick=0.62, p_put_back_shelf=0.15,
        use_fitting_room=0.55, p_keep_fitting=0.75, p_leave_in_room=0.15,
        p_abandon_queue=0.07, p_random_pause=0.04,
        fitting_wait_tolerance=6, p_abandon_fitting_wait=0.10,
    ),
}

_PROFILE_LIST    = list(ShopperProfile)
_PROFILE_WEIGHTS = [0.15, 0.20, 0.15, 0.10, 0.40]

_GENDERS      = ["female", "male"]
_GENDER_W     = [0.58, 0.42]        # clothing stores skew female

_AGE_GROUPS   = ["young", "adult", "senior"]
_AGE_W        = [0.30, 0.50, 0.20]


# ══════════════════════════════════════════════════
#  FSM STATES
# ══════════════════════════════════════════════════

class State(Enum):
    TO_SECTION      = auto()
    AT_SECTION      = auto()
    TO_FITTING      = auto()
    WAITING_FITTING = auto()
    IN_FITTING      = auto()
    TO_CHECKOUT     = auto()
    IN_QUEUE        = auto()
    AT_CHECKOUT     = auto()
    TO_EXIT         = auto()
    DONE            = auto()


# ══════════════════════════════════════════════════
#  SHOPPER
# ══════════════════════════════════════════════════

class Shopper:
    _id_counter = 0

    @classmethod
    def reset_counter(cls) -> None:
        cls._id_counter = 0

    def __init__(self) -> None:
        Shopper._id_counter += 1
        self.id: int = Shopper._id_counter
        self.pos: tuple[int, int] = ENTRANCE_POS

        self.profile: ShopperProfile = random.choices(_PROFILE_LIST, _PROFILE_WEIGHTS)[0]
        self.gender: str             = random.choices(_GENDERS,   _GENDER_W)[0]
        self.age_group: str          = random.choices(_AGE_GROUPS, _AGE_W)[0]
        self.cfg: ProfileCfg         = self._apply_demographics(PROFILES[self.profile])

        self.speed: int = random.randint(*self.cfg.speed)

        n = min(random.randint(*self.cfg.n_sections), len(SHELF_TYPES))
        self.sections: list[Cell] = random.sample(SHELF_TYPES, n)

        self.items_picked: list[Cell] = []
        self.items_purchased: list[Cell] = []

        self.state: State = State.TO_SECTION
        self.path: list[tuple[int, int]] = []
        self.ticks_here: int = 0
        self.ticks_needed: int = 0
        self.pause_remaining: int = 0
        self.fitting_wait_ticks: int = 0

        self._navigate(SECTION_ACCESS[self.sections[0]])

    # ── demographics ──────────────────────────────────

    def _apply_demographics(self, base: ProfileCfg) -> ProfileCfg:
        mods: dict = {}

        def _scale_range(rng: tuple[int, int], factor: float) -> tuple[int, int]:
            return (max(1, int(rng[0] * factor)), max(1, int(rng[1] * factor)))

        # Gender effects
        if self.gender == "female":
            mods["browse_time"]      = _scale_range(base.browse_time, 1.25)
            mods["use_fitting_room"] = min(1.0, base.use_fitting_room * 1.30)
            mods["p_pick"]           = min(1.0, base.p_pick * 1.15)
            mods["n_sections"]       = (base.n_sections[0], min(len(SHELF_TYPES), base.n_sections[1] + 1))
        elif self.gender == "male":
            mods["browse_time"]      = _scale_range(base.browse_time, 0.80)
            mods["use_fitting_room"] = base.use_fitting_room * 0.50
            mods["p_put_back_shelf"] = min(1.0, base.p_put_back_shelf * 1.25)
            mods["n_sections"]       = (max(1, base.n_sections[0] - 1), max(1, base.n_sections[1] - 1))

        # Age effects
        if self.age_group == "young":
            mods["speed"]           = (min(3, base.speed[0] + 1), min(3, base.speed[1] + 1))
            mods["p_random_pause"]  = min(0.50, base.p_random_pause * 2.0)   # phone distractions
            mods["p_put_back_shelf"] = min(1.0, mods.get("p_put_back_shelf", base.p_put_back_shelf) * 1.20)
        elif self.age_group == "senior":
            mods["speed"]           = (max(1, base.speed[0] - 1), max(1, base.speed[1] - 1))
            mods["browse_time"]     = _scale_range(mods.get("browse_time", base.browse_time), 1.20)
            mods["checkout_time"]   = _scale_range(base.checkout_time, 1.40)
            mods["p_abandon_queue"] = base.p_abandon_queue * 0.50  # seniors are patient

        return dc_replace(base, **mods) if mods else base

    # ── navigation ────────────────────────────────────

    def _navigate(self, goals: list[tuple[int, int]]) -> None:
        path = bfs(self.pos, goals)
        self.path = path[1:] if path else []

    def _move(self) -> bool:
        """
        Attempt to move up to `speed` cells along the path.
        Returns True if the shopper paused (random distraction) instead of moving.
        """
        if not self.path:
            return False

        if self.pause_remaining > 0:
            self.pause_remaining -= 1
            return True

        if random.random() < self.cfg.p_random_pause:
            self.pause_remaining = random.randint(1, 3)
            return True

        for _ in range(min(self.speed, len(self.path))):
            if self.path:
                self.pos = self.path.pop(0)
        return False

    # ── main tick ─────────────────────────────────────

    def step(self, checkout_queue: list[Shopper],
             fitting_state: dict) -> list[tuple[str, str]]:
        """Execute one simulation tick. Returns (action, details) pairs to log."""
        ev: list[tuple[str, str]] = []

        match self.state:

            # ── walking to a shelf section ──
            case State.TO_SECTION:
                paused = self._move() if self.path else False
                if not self.path:
                    self.ticks_here  = 0
                    self.ticks_needed = random.randint(*self.cfg.browse_time)
                    self.state = State.AT_SECTION
                    ev.append(("ARRIVED_SECTION", SECTION_NAMES[self.sections[0]]))
                elif paused:
                    ev.append(("PAUSED", f"→ {SECTION_NAMES[self.sections[0]]}"))
                else:
                    ev.append(("MOVING", f"→ {SECTION_NAMES[self.sections[0]]}"))

            # ── browsing a shelf ──
            case State.AT_SECTION:
                self.ticks_here += 1
                if self.ticks_here < self.ticks_needed:
                    ev.append(("BROWSING", SECTION_NAMES[self.sections[0]]))
                else:
                    section = self.sections.pop(0)
                    if random.random() < self.cfg.p_pick:
                        if random.random() < self.cfg.p_put_back_shelf:
                            # Picked up, inspected, put back on the spot
                            ev.append(("PUT_BACK_SHELF", SECTION_NAMES[section]))
                        else:
                            self.items_picked.append(section)
                            ev.append(("PICKED_ITEM", SECTION_NAMES[section]))
                    else:
                        ev.append(("NO_PICK", SECTION_NAMES[section]))

                    # Decide next move
                    if self.sections:
                        self._navigate(SECTION_ACCESS[self.sections[0]])
                        self.state = State.TO_SECTION
                    elif self.items_picked and random.random() < self.cfg.use_fitting_room:
                        self._navigate(DRESSING_ROOM_CELLS)
                        self.state = State.TO_FITTING
                        ev.append(("HEADING", "fitting room"))
                    else:
                        self._go_to_checkout_or_exit(ev)

            # ── walking to fitting room wait area ──
            case State.TO_FITTING:
                paused = self._move() if self.path else False
                if not self.path:
                    self.fitting_wait_ticks = 0
                    self.state = State.WAITING_FITTING
                    ev.append(("ARRIVED_FITTING_AREA", f"{len(self.items_picked)} item(s)"))
                elif paused:
                    ev.append(("PAUSED", "→ fitting room"))
                else:
                    ev.append(("MOVING", "→ fitting room"))

            # ── waiting outside fitting rooms ──
            case State.WAITING_FITTING:
                free_rooms = [rc for rc in DRESSING_ROOM_CELLS
                              if rc not in fitting_state['occupied']]
                if free_rooms:
                    room = min(free_rooms,
                               key=lambda p: abs(p[0] - self.pos[0]) + abs(p[1] - self.pos[1]))
                    fitting_state['occupied'].add(room)
                    self.pos = room
                    self.ticks_here = 0
                    base = random.randint(5, 10)
                    self.ticks_needed = base + len(self.items_picked) * random.randint(2, 4)
                    self.state = State.IN_FITTING
                    waited = self.fitting_wait_ticks
                    self.fitting_wait_ticks = 0
                    ev.append(("ENTERED_FITTING",
                               f"{len(self.items_picked)} item(s) to try on, waited {waited} ticks"))
                elif (self.fitting_wait_ticks >= self.cfg.fitting_wait_tolerance
                      or random.random() < self.cfg.p_abandon_fitting_wait
                      or not free_rooms):
                    waited = self.fitting_wait_ticks
                    self.fitting_wait_ticks = 0
                    ev.append(("GAVE_UP_FITTING", f"waited {waited} ticks"))
                    self._go_to_checkout_or_exit(ev)
                else:
                    self.fitting_wait_ticks += 1
                    ev.append(("WAITING_FITTING",
                               f"tick {self.fitting_wait_ticks}/{self.cfg.fitting_wait_tolerance}"))

            # ── inside fitting room ──
            case State.IN_FITTING:
                self.ticks_here += 1
                if self.ticks_here < self.ticks_needed:
                    ev.append(("IN_FITTING", f"{self.ticks_here}/{self.ticks_needed} min"))
                else:
                    kept, left_in_room, returned_to_shelf = [], 0, 0
                    for item in self.items_picked:
                        if random.random() < self.cfg.p_keep_fitting:
                            kept.append(item)
                        elif random.random() < self.cfg.p_leave_in_room:
                            left_in_room += 1
                        else:
                            returned_to_shelf += 1
                    self.items_picked = kept
                    parts = []
                    if kept:              parts.append(f"keeping={len(kept)}")
                    if returned_to_shelf: parts.append(f"returned_to_shelf={returned_to_shelf}")
                    if left_in_room:      parts.append(f"left_in_room={left_in_room}")
                    ev.append(("LEFT_FITTING", ", ".join(parts) or "kept nothing"))
                    fitting_state['occupied'].discard(self.pos)
                    self._go_to_checkout_or_exit(ev)

            # ── walking to checkout ──
            case State.TO_CHECKOUT:
                paused = self._move() if self.path else False
                if not self.path:
                    checkout_queue.append(self)
                    self.state = State.IN_QUEUE
                    ev.append(("JOINED_QUEUE",
                               f"position #{len(checkout_queue)}, {len(self.items_picked)} item(s)"))
                elif paused:
                    ev.append(("PAUSED", "→ checkout"))
                else:
                    ev.append(("MOVING", "→ checkout"))

            # ── waiting in checkout line ──
            case State.IN_QUEUE:
                pos_in_queue = checkout_queue.index(self) + 1
                # Abandonment probability grows with queue depth
                p_abandon = min(self.cfg.p_abandon_queue * (1 + (pos_in_queue - 1) * 0.4), 0.6)
                if random.random() < p_abandon:
                    checkout_queue.remove(self)
                    n = len(self.items_picked)
                    self.items_picked.clear()
                    ev.append(("ABANDONED_QUEUE", f"position #{pos_in_queue}, left {n} item(s)"))
                    self._navigate([ENTRANCE_POS])
                    self.state = State.TO_EXIT
                else:
                    ev.append(("IN_QUEUE", f"position #{pos_in_queue}"))

            # ── at checkout counter ──
            case State.AT_CHECKOUT:
                self.ticks_here += 1
                if self.ticks_here < self.ticks_needed:
                    ev.append(("AT_CHECKOUT", f"{self.ticks_here}/{self.ticks_needed} min"))
                else:
                    self.items_purchased = list(self.items_picked)
                    names = ", ".join(SECTION_NAMES[s] for s in self.items_purchased)
                    ev.append(("PURCHASED",
                               f"{len(self.items_purchased)} item(s): {names}"))
                    if self in checkout_queue:
                        checkout_queue.remove(self)
                    self.items_picked.clear()
                    self._navigate([ENTRANCE_POS])
                    self.state = State.TO_EXIT

            # ── walking to exit ──
            case State.TO_EXIT:
                paused = self._move() if self.path else False
                if not self.path or self.pos == ENTRANCE_POS:
                    self.state = State.DONE
                    ev.append(("EXITED", ""))
                elif paused:
                    ev.append(("PAUSED", "→ exit"))
                else:
                    ev.append(("MOVING", "→ exit"))

        return ev

    # ── called by simulation when a Cf counter cell is free ──
    def serve_checkout(self, counter_pos: tuple[int, int]) -> None:
        self.pos = counter_pos          # shopper physically moves to the Cf cell
        self.ticks_here = 0
        base = random.randint(*self.cfg.checkout_time)
        per_item = random.randint(1, 2)
        self.ticks_needed = base + len(self.items_picked) * per_item
        self.state = State.AT_CHECKOUT

    # ── helpers ───────────────────────────────────────

    def _go_to_checkout_or_exit(self, ev: list[tuple[str, str]]) -> None:
        if self.items_picked:
            self._navigate(CHECKOUT_WAIT_CELLS)
            self.state = State.TO_CHECKOUT
            ev.append(("HEADING", "checkout"))
        else:
            self._navigate([ENTRANCE_POS])
            self.state = State.TO_EXIT
            ev.append(("HEADING", "exit"))
