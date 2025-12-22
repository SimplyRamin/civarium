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
TILESET_PATH = ROOT / "assets" / "tilesets" / "Alloy_curses_12x12.png"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Aesomatica_16x16.png"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Redjack17.png"


# ---------------------------------------------------
# Theme (Glyph-first + Colors, with future fallbacks)
# ---------------------------------------------------
THEME = {
    "plains": ("·", (120, 170, 120)),
    "forest": ("♣", (60, 140, 60)),
    "water": ("≈", (80, 130, 200)),
    "hill": ("▲", (170, 150, 110)),
}

UI = {
    "frame_fg": (200, 200, 200),
    "title_fg": (230, 230, 230),
    "text_fg": (210, 210, 210),
    "muted_fg": (160, 160, 160),
    "cursor_fg": (0, 0, 0),
    "cursor_bg": (230, 230, 230),
    "log_fg": (240, 240, 200)
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
    actors: list["Actor"] = field(default_factory=list)


@dataclass
class Actor:
    x: int
    y: int
    glyph: str
    fg: tuple[int, int, int]


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

    def sprinkle_forests_and_hills() -> None:
        forest_mask = rng.random((MAP_H, MAP_W)) < 0.08
        hill_mask = rng.random((MAP_H, MAP_W)) < 0.05
        world[forest_mask & (world == 0)] = 1
        world[hill_mask & (world == 0)] = 3

    def add_river_horizontal() -> None:
        y = int(rng.integers(low=MAP_H // 6, high=5 * MAP_H // 6))
        for x in range(MAP_W):
            y = int(np.clip(y + rng.integers(-1, 2), 0, MAP_H - 1))
            world[y, x] = 2
            # thickness
            if y + 1 < MAP_H and rng.random() < 0.25:
                world[y + 1, x] = 2
            if y - 1 >= 0 and rng.random() < 0.25:
                world[y - 1, x] = 2

    def add_river_vertical() -> None:
        x = int(rng.integers(low=MAP_W // 6, high=5 * MAP_W // 6))
        for y in range(MAP_H):
            x = int(np.clip(x + rng.integers(-1, 2), 0, MAP_W - 1))
            world[y, x] = 2
            if x + 1 < MAP_W and rng.random() < 0.25:
                world[y, x + 1] = 2
            if x - 1 >= 0 and rng.random() < 0.25:
                world[y, x - 1] = 2

    def add_river_diagonal() -> None:
        # choose one of two diagonal directions
        if rng.random() < 0.5:
            x, y = 0, int(rng.integers(0, MAP_H))
            dx = 1
        else:
            x, y = MAP_W - 1, int(rng.integers(0, MAP_H))
            dx = -1

        for _ in range(MAP_W, MAP_H):
            if 0 <= x < MAP_W and 0 <= y < MAP_H:
                world[y, x] = 2
                # ocassional thickness
                if y + 1 < MAP_H and rng.random() < 0.18:
                    world[y + 1, x] = 2
                if y - 1 >= 0 and rng.random() < 0.18:
                    world[y - 1, x] = 2

            # diagonal-ish step with wobble
            x += dx
            y += int(rng.integers(-1, 2))

            # bounce off top/bottom to keep it inside map
            if y < 0:
                y = 0
            elif y >= MAP_H:
                y = MAP_H - 1

            # stop if we left the map horizontally
            if x < 0 or x >= MAP_W:
                break

    def add_lake() -> None:
        # simple blob lake using random growth
        cx = int(rng.integers(MAP_W // 5, 4 * MAP_W // 5))
        cy = int(rng.integers(MAP_H // 5, 4 * MAP_H // 5))
        target = int(rng.integers(80, 220))

        frontier = [(cx, cy)]
        world[cy, cx] = 2
        placed = 1

        while frontier and placed < target:
            x, y = frontier.pop(int(rng.integers(0, len(frontier))))
            # try to expand to neighbors
            for _ in range(3):
                nx = x + int(rng.integers(-1, 2))
                ny = y + int(rng.integers(-1, 2))
                if 0 <= nx < MAP_W and 0 <= ny < MAP_H and world[ny, nx] != 2:
                    if rng.random() < 0.65:
                        world[ny, nx] = 2
                        placed += 1
                        frontier.append((nx, ny))
                if placed >= target:
                    break

    # choose water pattern(s)
    pattern = int(rng.integers(0, 4))       # 0..3
    if pattern == 0:
        add_river_horizontal()
    elif pattern == 1:
        add_river_vertical()
    elif pattern == 2:
        add_river_diagonal()
    else:
        add_lake()

    # sometimes add a small extra lake/rivulet for variety
    if rng.random() < 0.20:
        add_lake()

    sprinkle_forests_and_hills()
    return world


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


def terrain_style(code: int) -> tuple[str, tuple[int, int, int]]:
    if int(code) == 0:
        return THEME["plains"]
    if int(code) == 1:
        return THEME["forest"]
    if int(code) == 2:
        return THEME["water"]
    if int(code) == 3:
        return THEME["hill"]
    return "?", (255, 50, 50)


def update(gs: GameState, world: np.ndarray) -> None:
    if gs.paused:
        return
    gs.tick += 1

    rng = np.random.default_rng(gs.seed + gs.tick)
    for a in gs.actors:
        dx, dy = rng.integers(-1, 2), rng.integers(-1, 2)
        nx, ny = a.x + int(dx), a.y + int(dy)
        if 0 <= nx < MAP_W and 0 <= ny < MAP_H and int(world[ny, nx]) != 2:
            a.x, a.y = nx, ny

    if gs.tick % int(max(1, gs.tps)) == 0:
        add_log(gs, f"Tick {gs.tick}: peasants wander...")


def draw_frame(console: tcod.console.Console, x: int, y: int, w: int, h: int, title: str) -> None:
    """
    Small helper to avoid deprecated title usage in draw_frame.
    Uses box-drawingcharacters and prints a centered title.
    """
    fg = UI["frame_fg"]
    # corners + edges
    console.print(x, y, "┌" + "─" * (w - 2) + "┐", fg=fg)
    for row in range(1, h - 1):
        console.print(x, y + row, "│", fg=fg)
        console.print(x + w - 1, y + row, "│", fg=fg)
    console.print(x, y + h - 1,  "└" + "─" * (w - 2) + "┘", fg=fg)

    # title
    if title:
        t = f" {title} "
        start = x + max(1, (w - len(t)) // 2)
        if start + len(t) < x < w - 1:
            console.print(start, y, t, fg=UI["title_fg"], bg=None)


def render(console: tcod.console.Console, world: np.ndarray, gs: GameState) -> None:
    console.clear()

    # Map
    for y in range(MAP_H):
        for x in range(MAP_W):
            ch, fg = terrain_style(world[y, x])
            console.print(x, y, ch, fg=fg)

    # Actor rendering
    for a in gs.actors:
        console.print(a.x, a.y, a.glyph, fg=a.fg)

    # Cursor highlight
    cx, cy = gs.cursor
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        cg, _fg = terrain_style(world[cy, cx])
        console.print(cx, cy, ch, fg=UI["cursor_fg"], bg=UI["cursor_bg"])

    # Right panel
    panel_x = MAP_W
    draw_frame(console, panel_x, 0, PANEL_W, MAP_H, "Civarium")

    px = panel_x + 2
    console.print(px, 2, f"Seed: {gs.seed}", fg=UI["text_fg"])
    console.print(px, 3, f"Tick: {gs.tick}", fg=UI["text_fg"])
    console.print(px, 4, f"Paused: {gs.paused}", fg=UI["text_fg"])
    console.print(px, 5, f"Speed: {gs.tps:.1f} tps", fg=UI["text_fg"])

    console.print(px, 7, "Controls:", fg=UI["title_fg"])
    console.print(px, 8, "Space: pause", fg=UI["muted_fg"])
    console.print(px, 9, "+/- : speed", fg=UI["muted_fg"])
    console.print(px, 10, "Arrows: cursor", fg=UI["muted_fg"])
    console.print(px, 11, "R: restart", fg=UI["muted_fg"])
    console.print(px, 12, "Q: quit", fg=UI["muted_fg"])

    console.print(px, 14, "Inspect:", fg=UI["title_fg"])
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        code = int(world[cy, cx])
        tname = {0: "Plains", 1: "Forest", 2: "Water", 3: "Hill"}.get(code, "Unknown")
        console.print(px, 15, f"({cx}, {cy}) {tname}")

    # Bottom log
    log_y = MAP_H
    draw_frame(console, 0, log_y, SCREEN_W, LOG_H, "Log")
    for i, line in enumerate(list(gs.log)[: LOG_H - 2]):
        console.print(2, log_y + 1 + i, line, fg=UI["log_fg"])


def main() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Font file not found at {FONT_PATH}. Put a monospace .ttf there (e.g., DejaVuSansMono.ttf)."
        )

    gs = GameState(seed=int(time.time()) % 100_000)
    world = make_world(gs.seed)
    spawn_peasants(gs, world)
    add_log(gs, f"Spawned {len(gs.actors)} peasants.")
    add_log(gs, "Civarium booted.")
    add_log(gs, "Space pauses. R restarts.")

    tileset = tcod.tileset.load_tilesheet(
        str(TILESET_PATH),
        columns=16,
        rows=16,
        charmap=tcod.tileset.CHARMAP_CP437,
    )

    with tcod.context.new(
        columns=SCREEN_W,
        rows=SCREEN_H,
        tileset=tileset,
        title="Civarium",
        vsync=True
    ) as context:
        console = tcod.console.Console(SCREEN_W, SCREEN_H, order="F")

        last_time = time.perf_counter()
        acc = 0.0

        while True:
            # Input
            for event in tcod.event.wait(timeout=0.0):
                if event.type == "QUIT":
                    raise SystemExit()

                if event.type == "KEYDOWN":
                    key = event.sym

                    if key in (tcod.event.K_q, tcod.event.K_ESCAPE):
                        raise SystemExit()

                    if key == tcod.event.K_SPACE:
                        gs.paused = not gs.paused
                        add_log(gs, "Paused." if gs.paused else "Resumed.")

                    elif key in (tcod.event.K_PLUS, tcod.event.K_KP_PLUS, tcod.event.K_EQUALS):
                        gs.tps = min(60.0, gs.tps + 2.0)
                        add_log(gs, f"Speed: {gs.tps:.1f} tps")

                    elif key in (tcod.event.K_MINUS, tcod.event.K_KP_MINUS):
                        gs.tps = max(1.0, gs.tps - 2.0)
                        add_log(gs, f"Speed: {gs.tps:.1f} tps")

                    elif key == tcod.event.K_r:
                        gs.seed = int(time.time()) % 100_000
                        gs.tick = 0
                        gs.paused = False
                        gs.log.clear()
                        gs.actors.clear()
                        add_log(gs, f"Restarted (seed {gs.seed}).")
                        world = make_world(gs.seed)
                        spawn_peasants(gs, world)

                    elif key == tcod.event.K_LEFT:
                        x, y = gs.cursor
                        gs.cursor = (clamp(x - 1, 0, MAP_W - 1), y)
                    elif key == tcod.event.K_RIGHT:
                        x, y = gs.cursor
                        gs.cursor = (clamp(x + 1, 0, MAP_W - 1), y)
                    elif key == tcod.event.K_UP:
                        x, y = gs.cursor
                        gs.cursor = (x, clamp(y - 1, 0, MAP_H - 1))
                    elif key == tcod.event.K_DOWN:
                        x, y = gs.cursor
                        gs.cursor = (x, clamp(y + 1, 0, MAP_H - 1))

            # Timing
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            acc += dt

            step = 1.0 / max(1.0, gs.tps)
            while acc >= step:
                update(gs, world)
                acc -= step

            render(console, world, gs)
            context.present(console, integer_scaling=True)

            time.sleep(1.0 / FPS_CAP)


if __name__ == "__main__":
    main()
