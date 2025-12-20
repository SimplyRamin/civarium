# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import tcod


# -------------------
# layout config
# -------------------
MAP_W, MAP_H = 80, 40
PANEL_W = 30
LOG_H = 8

SCREEN_W = MAP_W + PANEL_W
SCREEN_H = MAP_H + LOG_H

FPS_CAP = 60

# -------------------
# Paths (portable)
# -------------------
ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "assets" / "fonts" / "DejaVuSansMono.ttf"


# ------------------------------------------
# Theme (Glyph-first, with future fallbacks)
# ------------------------------------------
THEME = {
    "plains": "·",
    "forest": "♣",
    "water": "≈",
    "hill": "▲"
}


# -------------------
# State
# -------------------
@dataclass
class GameState:
    seed: int
    tick: int = 0
    paused: bool = False
    tps: float = 10.0       # ticks per second
    log: Deque[str] = field(default_factory=lambda: deque(maxlen=LOG_H))
    cursor: Tuple[int, int] = (MAP_W // 2, MAP_H // 2)


def add_log(gs: GameState, msg: str) -> None:
    gs.log.appendleft(msg)


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def make_world(seed: int) -> np.ndarray:
    """
    MAP_H x MAP_W array of terrain codes:
    0=plains, 1=forest, 2=water, 3=hill
    """
    rng = np.random.default_rng(seed)
    world = np.zeros((MAP_H, MAP_W), dtype=np.uint8)

    # River-ish band
    river_y = int(rng.integers(low=MAP_H // 3, high=2 * MAP_H // 3))
    for x in range(MAP_W):
        wobble = int(rng.integers(-1, 2))
        y = max(0, min(MAP_H - 1, river_y + wobble))
        world[y, x] = 2
        if y + 1 < MAP_H and rng.random() < 0.30:
            world[y + 1, x] = 2
        if y - 1 >= 0 and rng.random() < 0.30:
            world[y - 1, x] = 2
        if rng.random() < 0.15:
            river_y = max(MAP_H // 6, min(5 * MAP_H // 6, river_y + int(rng.integers(-1, 2))))

    # Forest / hills
    forest_mask = rng.random((MAP_H, MAP_W)) < 0.08
    hill_mask = rng.random((MAP_H, MAP_W)) < 0.05
    world[forest_mask & (world == 0)] = 1
    world[hill_mask & (world == 0)] = 3

    return world


def terrain_glyph(code: int) -> str:
    return {
        0: THEME["plains"],
        1: THEME["forest"],
        2: THEME["water"],
        3: THEME["hill"],
    }.get(int(code), "?")


def update(gs: GameState) -> None:
    if gs.paused:
        return
    gs.tick += 1
    # tiny hearbeat log
    if gs.tick % int(max(1, gs.tps)) == 0:
        add_log(gs, f"Tick {gs.tick}: the world continues...")


def render(console: tcod.console.Console, world: np.ndarray, gs: GameState) -> None:
    console.clear()

    # Map
    for y in range(MAP_H):
        for x in range(MAP_W):
            console.print(x, y, terrain_glyph(world[y, x]))

    # Cursor highlight
    cx, cy = gs.cursor
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        ch = terrain_glyph(world[cy, cx])
        console.print(cx, cy, ch, fg=(0, 0, 0), bg=(200, 200, 200))

    # Right panel
    panel_x = MAP_W
    console.draw_frame(panel_x, 0, PANEL_W, MAP_H, title=" Civarium ", clear=False)

    px = panel_x + 2
    console.print(px, 2, f"Seed: {gs.seed}")
    console.print(px, 3, f"Tick: {gs.tick}")
    console.print(px, 4, f"Paused: {gs.paused}")
    console.print(px, 5, f"Speed: {gs.tps:.1f} tps")

    console.print(px, 7, "Controls:")
    console.print(px, 8, "Space: pause")
    console.print(px, 9, "+/- : speed")
    console.print(px, 10, "Arrows: cursor")
    console.print(px, 11, "R: restart")
    console.print(px, 12, "Q: quit")

    console.print(px, 14, "Inspect:")
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        code = int(world[cy, cx])
        tname = {0: "Plains", 1: "Forest", 2: "Water", 3: "Hill"}.get(code, "Unknown")
        console.print(px, 15, f"({cx}, {cy}) {tname}")

    # Bottom log
    log_y = MAP_H
    console.draw_frame(0, log_y, SCREEN_W, LOG_H, title=" Log ", clear=False)
    for i, line in enumerate(list(gs.log)[: LOG_H - 2]):
        console.print(2, log_y + 1 + i, line)


