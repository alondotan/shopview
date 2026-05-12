"""Store grid definition and BFS pathfinding."""

from __future__ import annotations
import collections
from enum import Enum

ROWS = 8
COLS = 14


class Cell(Enum):
    AISLE         = "AISLE"
    DRESSING_ROOM = "D"
    CHECKOUT_WAIT = "Cw"
    CHECKOUT_FIRST = "Cf"
    ENTRANCE      = "E"
    TOPS          = "T"
    HATS_HANDBAGS = "H"
    SKIRTS        = "Sk"
    PANTS         = "P"
    UNDERWEAR     = "U"
    STOCKINGS     = "St"
    JEANS         = "J"
    CHECKOUT      = "C"


WALKABLE = {Cell.AISLE, Cell.DRESSING_ROOM, Cell.CHECKOUT_WAIT, Cell.CHECKOUT_FIRST, Cell.ENTRANCE}

SHELF_TYPES: list[Cell] = [
    Cell.TOPS, Cell.HATS_HANDBAGS, Cell.SKIRTS,
    Cell.PANTS, Cell.UNDERWEAR, Cell.STOCKINGS, Cell.JEANS,
]

SECTION_NAMES: dict[Cell, str] = {
    Cell.TOPS:          "Tops",
    Cell.HATS_HANDBAGS: "Hats & Handbags",
    Cell.SKIRTS:        "Skirts & Dresses",
    Cell.PANTS:         "Pants",
    Cell.UNDERWEAR:     "Underwear",
    Cell.STOCKINGS:     "Stockings",
    Cell.JEANS:         "Jeans",
}

# fmt: off
# col index:   0     1     2     3     4     5     6     7     8     9    10    11    12    13
# orig col:   14    13    12    11    10     9     8     7     6     5     4     3     2     1
_RAW = [
    ["D",  "D",  "D",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  " "],  # row a (0)
    [" ",  " ",  " ",  " ",  " ",  " ", "Cw",  "Cf",  " ",  " ",  " ",  " ",  "U",  " "],  # row b (1)
    ["H",  " ",  "T",  "T",  " ",  " ",  "C",  "C",  " ",  "P",  " ",  " ",  "St", " "],  # row c (2)
    ["H",  " ",  "T",  "T",  " ",  " ",  " ",  " ",  " ",  "P",  " ",  " ",  "St", " "],  # row d (3)
    [" ",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  "P",  " ",  " ",  " ",  " "],  # row e (4)
    ["Sk", " ",  "T",  "T",  " ",  " ",  " ",  " ",  " ",  "P",  " ",  " ",  " ",  " "],  # row f (5)
    ["Sk", " ",  "T",  "T",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  " ",  "J",  " "],  # row g (6)
    ["Sk", " ",  " ",  " ",  " ",  " ",  "E",  " ",  " ",  " ",  " ",  " ",  "J",  " "],  # row h (7)
]
# fmt: on

_STR_TO_CELL: dict[str, Cell] = {
    " ": Cell.AISLE,  "D": Cell.DRESSING_ROOM, "Cw": Cell.CHECKOUT_WAIT,
    "Cf": Cell.CHECKOUT_FIRST,
    "E": Cell.ENTRANCE, "T": Cell.TOPS, "H": Cell.HATS_HANDBAGS,
    "Sk": Cell.SKIRTS, "P": Cell.PANTS, "U": Cell.UNDERWEAR,
    "St": Cell.STOCKINGS, "J": Cell.JEANS, "C": Cell.CHECKOUT,
}

GRID: list[list[Cell]] = [[_STR_TO_CELL[s] for s in row] for row in _RAW]

ENTRANCE_POS: tuple[int, int] = (7, 6)  # row h, original col 8


def to_display(r: int, c: int) -> tuple[str, int]:
    """Convert internal (row_idx, col_idx) to (row_letter, original_col_number)."""
    return chr(ord("a") + r), 14 - c


def is_walkable(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS and GRID[r][c] in WALKABLE


def cells_of(cell_type: Cell) -> list[tuple[int, int]]:
    return [(r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] == cell_type]


def adjacent_walkable(cell_type: Cell) -> list[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for r, c in cells_of(cell_type):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if is_walkable(nr, nc):
                result.add((nr, nc))
    return list(result)


def bfs(start: tuple[int, int], goals: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """BFS from start to nearest goal. Returns full path including start, or None."""
    if not goals:
        return None
    goal_set = set(goals)
    if start in goal_set:
        return [start]
    visited = {start}
    queue: collections.deque[list[tuple[int, int]]] = collections.deque([[start]])
    while queue:
        path = queue.popleft()
        r, c = path[-1]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            npos = (r + dr, c + dc)
            if npos in goal_set:
                return path + [npos]
            if is_walkable(*npos) and npos not in visited:
                visited.add(npos)
                queue.append(path + [npos])
    return None


# Pre-computed access points (computed once at import time)
SECTION_ACCESS: dict[Cell, list[tuple[int, int]]] = {
    s: adjacent_walkable(s) for s in SHELF_TYPES
}
DRESSING_ROOM_CELLS:   list[tuple[int, int]] = cells_of(Cell.DRESSING_ROOM)
CHECKOUT_WAIT_CELLS:   list[tuple[int, int]] = cells_of(Cell.CHECKOUT_WAIT)
CHECKOUT_FIRST_CELLS:  list[tuple[int, int]] = cells_of(Cell.CHECKOUT_FIRST)
FITTING_CAPACITY: int = len(DRESSING_ROOM_CELLS)    # = 3
CHECKOUT_LANES:   int = len(CHECKOUT_FIRST_CELLS)   # = 1

# Aisle cells adjacent to dressing rooms — shoppers wait here when rooms are full
FITTING_WAIT_AREA: list[tuple[int, int]] = [
    pos for pos in adjacent_walkable(Cell.DRESSING_ROOM)
    if GRID[pos[0]][pos[1]] != Cell.DRESSING_ROOM
]
