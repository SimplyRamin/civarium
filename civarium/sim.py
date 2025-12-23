# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import numpy as np

from .worldgen import MAP_W, MAP_H

LOG_H = 8
FORUM = 1
HOUSE = 2
ROAD = 3


# -------------------
# State
# -------------------
@dataclass
class Actor:
    x: int
    y: int
    glyph: str
    fg: tuple[int, int, int]
    home: tuple[int, int] | None = None


@dataclass
class GameState:
    seed: int
    tick: int = 0
    paused: bool = False
    tps: float = 10.0       # ticks per second
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_H))
    cursor: Tuple[int, int] = (MAP_W // 2, MAP_H // 2)
    actors: list["Actor"] = field(default_factory=list)
    buildings: dict[tuple[int, int], int] = field(default_factory=dict)
    food: float = 120.0
    morale: float = 0.75


def add_log(gs: GameState, msg: str) -> None:
    gs.log.appendleft(msg)


def get_forum_pos(gs: GameState) -> Tuple[int, int] | None:
    for (x, y), b in gs.buildings.items():
        if b == FORUM:
            return (x, y)
    return None


def is_water(world: np.ndarray, x: int, y: int) -> bool:
    return int(world[y, x]) == 2


def is_blocked(gs: GameState, world: np.ndarray, x: int, y: int) -> bool:
    if is_water(world, x, y):
        return True
    b = gs.buildings.get((x, y))
    return b == FORUM or b == HOUSE     # Roads are walkable


def place_near_center_nonwater(gs: GameState, world: np.ndarray) -> tuple[int, int]:
    cx, cy = MAP_W // 2, MAP_H // 2
    for r in range(10):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < MAP_W and 0 <= y < MAP_H and not is_water(world, x, y):
                    return (x, y)
    return (cx, cy)


def carve_road(gs: GameState, world: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    x, y = a
    tx, ty = b

    def try_set(px: int, py: int) -> None:
        if 0 <= px < MAP_W and 0 <= py < MAP_H and not is_water(world, px, py):
            if (px, py) not in gs.buildings:
                gs.buildings[(px, py)] = ROAD

    # horizontal first, then vertical (simple and readable)
    step = 1 if tx >= x else -1
    for px in range(x, tx + step, step):
        try_set(px, y)

    step = 1 if ty >= y else -1
    for py in range(y, ty + step, step):
        try_set(tx, py)


def spawn_peasants(gs: GameState, world: np.ndarray, n: int | None = None) -> None:
    """
    Spawn peasants near the center, avoiding water.
    If n is None, choose a deterministic random count based on seed.
    """
    rng = np.random.default_rng(gs.seed + 999)
    if n is None:
        n = int(rng.integers(8, 19))
    cx, cy = MAP_W // 2, MAP_H // 2

    for _ in range(n):
        for _attempt in range(80):
            x = int(cx + rng.integers(-6, 7))
            y = int(cy + rng.integers(-6, 7))
            if 0 <= x < MAP_W and 0 <= y < MAP_H and int(world[y, x]) != 2:     # not water
                gs.actors.append(Actor(x=x, y=y, glyph="@", fg=(230, 230, 230)))
                break


def reset_run(gs: GameState, world: np.ndarray) -> None:
    gs.tick = 0
    gs.paused = False
    gs.log.clear()
    gs.actors.clear()
    gs.buildings.clear()

    # reset economy
    gs.food = 120.0
    gs.morale = 0.75

    # Place Forum
    fx, fy = place_near_center_nonwater(gs, world)
    gs.buildings[(fx, fy)] = FORUM
    add_log(gs, "Forum established.")

    # Place some houses around forum
    rng = np.random.default_rng(gs.seed + 1234)
    house_count = int(rng.integers(3, 6))
    houses: list[tuple[int, int]] = []
    for _ in range(house_count):
        for _attempt in range(200):
            x = int(rng.integers(-6, 7))
            y = int(rng.integers(-6, 7))
            if 0 <= x < MAP_W and 0 <= y < MAP_H and not is_water(world, x, y):
                gs.buildings[(x, y)] = HOUSE
                houses.append((x, y))
                carve_road(gs, world, (fx, fy), (x, y))     # roads to houses
                break

    spawn_peasants(gs, world)

    # Assign homes (round-robin)
    if houses:
        for i, a in enumerate(gs.actors):
            a.home = houses[i % len(houses)]
    add_log(gs, f"Houses: {len(houses)} | Peasants: {len(gs.actors)}")


def update(gs: GameState, world: np.ndarray) -> list[str]:
    if gs.paused:
        return []

    gs.tick += 1
    events: list[str] = []

    rng = np.random.default_rng(gs.seed + gs.tick)
    forum = get_forum_pos(gs)
    for a in gs.actors:
        # Decide target: mostly forum by day, home sometimes (simple rhythm)
        target = None
        if forum is not None:
            if a.home is not None and (gs.tick % 20) >= 14:
                target = a.home     # go home phase
            else:
                target = forum     # gather at forum phase.

        if target is not None:
            dx = int(rng.integers(-1, 2))
            dy = int(rng.integers(-1, 2))
        else:
            fx, fy = target  # type: ignore
            step_x = 0 if a.x == fx else (1 if a.x < fx else -1)
            step_y = 0 if a.y == fy else (1 if a.y < fy else -1)

            # 75%: move towards forum, 25%: random wander
            if rng.random() < 0.75:
                dx = step_x
                dy = step_y
                # adding a little "imperfect" drift to avoid straight lines forever
                if rng.random() < 0.25:
                    dx = int(rng.integers(-1, 2))
                if rng.random() < 0.25:
                    dy = int(rng.integers(-1, 2))
            else:
                dx = int(rng.integers(-1, 2))
                dy = int(rng.integers(-1, 2))

        nx, ny = a.x + dx, a.y + dy
        if 0 <= nx < MAP_W and 0 <= ny < MAP_H and int(world[ny, nx]) != 2:
            a.x, a.y = nx, ny

    # Economy
    # -----------------------------
    pop = len(gs.actors)
    houses = sum(1 for b in gs.buildings.values() if b == HOUSE)

    # very simple: houses slightly improve stability; pop consumes food
    production = 0.10 * houses
    consumption = 0.25 * pop

    gs.food += production - consumption

    if gs.food < 0:
        gs.morale = max(0.0, gs.morale - 0.02)
        # occasionally lose someone if starving
        if pop > 0 and rng.random() < 0.03:
            gs.actors.pop()
            events.append("Starvation: a peasant was lost.")
            gs.food = max(gs.food, -10.0)
    else:
        gs.morale = min(1.0, gs.morale + 0.005)

    # Occasional
    capacity = houses * 4
    if pop < capacity and gs.food > 30 and gs.morale > 0.70 and rng.random() < 0.02:
        fx, fy = forum if forum is not None else (MAP_W // 2, MAP_H // 2)
        gs.actors.append(Actor(x=fx, y=fy, glyph="@", fg=(230, 230, 230), home=None))
        events.append("A newcomer arrived at the Forum.")

    if gs.tick % int(max(1, gs.tps)) == 0:
        events.append(f"Tick {gs.tick}: peasants wander...")

    return events


def new_game_state() -> GameState:
    return GameState(seed=int(time.time()) % 100_000)
