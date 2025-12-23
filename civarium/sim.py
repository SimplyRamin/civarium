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


# -------------------
# State
# -------------------
@dataclass
class Actor:
    x: int
    y: int
    glyph: str
    fg: tuple[int, int, int]


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


def add_log(gs: GameState, msg: str) -> None:
    gs.log.appendleft(msg)


def get_forum_pos(gs: GameState) -> Tuple[int, int] | None:
    for (x, y), b in gs.buildings.items():
        if b == FORUM:
            return (x, y)
    return None


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

    spawn_peasants(gs, world)
    add_log(gs, "Forum constructed.")

    cx, cy = MAP_W // 2, MAP_H // 2
    for r in range(6):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < MAP_W and 0 <= y < MAP_H and int(world[y, x]) != 2:
                    gs.buildings[(x, y)] = FORUM
                    return


def update(gs: GameState, world: np.ndarray) -> list[str]:
    if gs.paused:
        return []

    gs.tick += 1
    events: list[str] = []

    rng = np.random.default_rng(gs.seed + gs.tick)
    forum = get_forum_pos(gs)
    for a in gs.actors:
        if forum is not None:
            dx = int(rng.integers(-1, 2))
            dy = int(rng.integers(-1, 2))
        else:
            fx, fy = forum
            step_x = 0 if a.x == fx else (1 if a.x < fx else -1)
            step_y = 0 if a.y == fy else (1 if a.y < fy else -1)

            # 70%: move towards forum, 30%: random wander
            if rng.random() < 0.7:
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

    if gs.tick % int(max(1, gs.tps)) == 0:
        events.append(f"Tick {gs.tick}: peasants wander...")

    return events


def new_game_state() -> GameState:
    return GameState(seed=int(time.time()) % 100_000)
