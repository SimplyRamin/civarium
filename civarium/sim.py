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
FARM = 4


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


def get_forum_pos(gs: GameState) -> tuple[int, int] | None:
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
    """
    Carve a road from a -> b without crossing water, using BFD shortest path.
    """
    (sx, sy) = a
    (tx, ty) = b
    if (sx, sy) == (tx, ty):
        return

    def passable(x: int, y: int) -> bool:
        # Roads can go on empty lands, do not go through water or overwrite forum/houses
        if not (0 <= x < MAP_W and 0 <= y < MAP_H):
            return False
        if is_water(world, x, y):
            return False
        blk = gs.buildings.get((x, y))
        return blk not in (FORUM, HOUSE)

    prev: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    q = deque([(sx, sy)])

    # 4-neighborhood for nicer roads
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        x, y = q.popleft()
        if (x, y) == (tx, ty):
            break
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if (nx, ny) in prev:
                continue
            if (nx, ny) == (tx, ty) or passable(nx, ny):
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))

    # if no path is found, do nothing (rare; means houses is isolated by water)
    if (tx, ty) not in prev:
        return

    # Reconstrcut path (exclude endpoints so we don't overwrite buildings)
    cur = (tx, ty)
    path: list[tuple[int, int]] = []
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    for (x, y) in path:
        if (x, y) in (a, b):
            continue
        if (x, y) not in gs.buildings:   # do not overwrite existing buildings
            gs.buildings[(x, y)] = ROAD


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


def can_place_farm(gs: GameState, world: np.ndarray, x: int, y: int) -> bool:
    if not (0 <= x < MAP_W and 0 <= y < MAP_H):
        return False
    if int(world[y, x]) != 0:  # only plains
        return False
    if (x, y) in gs.buildings:
        return False
    return True


def try_build_farm(gs: GameState, world: np.ndarray, around: tuple[int, int], rng: np.random.Generator) -> bool:
    ax, ay = around
    for _ in range(200):
        x = ax + int(rng.integers(-10, 11))
        y = ay + int(rng.integers(-10, 11))
        if can_place_farm(gs, world, x, y):
            gs.buildings[(x, y)] = FARM
            forum = get_forum_pos(gs)
            if forum is not None:
                carve_road(gs, world, forum, (x, y))
            return True
    return False


def reset_run(gs: GameState, world: np.ndarray) -> None:
    gs.tick = 0
    gs.paused = False
    gs.log.clear()
    gs.actors.clear()
    gs.buildings.clear()

    # reset economy
    gs.food = 80.0
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
            x = fx + int(rng.integers(-6, 7))
            y = fy + int(rng.integers(-6, 7))
            if 0 <= x < MAP_W and 0 <= y < MAP_H and not is_water(world, x, y):
                if (x, y) not in gs.buildings:
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

        if target is None:
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
    farms = sum(1 for b in gs.buildings.values() if b == FARM)

    # very simple: houses slightly improve stability; pop consumes food
    production = 1.1 * houses + 2.5 + 2.0 * farms
    consumption = 0.18 * pop
    gs.food += production - consumption

    capacity = houses * 4

    if gs.food < 0:
        gs.morale = max(0.0, gs.morale - 0.02)
        # occasionally lose someone if starving
        if pop > 0 and rng.random() < 0.005:
            gs.actors.pop()
            events.append("Starvation: a peasant was lost.")
            gs.food = max(gs.food, -10.0)
    else:
        gs.morale = min(1.0, gs.morale + 0.005)

    # Occasional
    if pop < capacity and gs.food > 30 and gs.morale > 0.70 and rng.random() < 0.02:
        fx, fy = forum if forum is not None else (MAP_W // 2, MAP_H // 2)
        gs.actors.append(Actor(x=fx, y=fy, glyph="@", fg=(230, 230, 230), home=None))
        events.append("A newcomer arrived at the Forum.")

    if gs.tick % int(max(1, gs.tps)) == 0:
        events.append(f"Tick {gs.tick}: peasants wander...")

    if gs.tick % 25 == 0:
        forum = get_forum_pos(gs)
        # If food is trending down or farms are too few, build a farm
        if gs.food < 60 or farms < max(1, houses):
            if try_build_farm(gs, world, forum, rng):   # type: ignore
                events.append("A new farm was built.")
        # If near capacity, build a house
        elif pop >= capacity - 1:
            events.append("Housing is tight.")

    return events


def new_game_state() -> GameState:
    return GameState(seed=int(time.time()) % 100_000)
