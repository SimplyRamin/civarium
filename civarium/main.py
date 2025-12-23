# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================

from __future__ import annotations

import time
from pathlib import Path
import tcod
import numpy as np

from .render import SCREEN_H, SCREEN_W, render
from .sim import GameState, add_log, new_game_state, reset_run, update
from .worldgen import make_world, MAP_H, MAP_W

# -------------------
# Paths (portable)
# -------------------
ROOT = Path(__file__).resolve().parents[1]
# FONT_PATH = ROOT / "assets" / "fonts" / "DejaVuSansMono.ttf"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Alloy_curses_12x12.png"
TILESET_PATH = ROOT / "assets" / "tilesets" / "Bisasam_16x16.png"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Aesomatica_16x16.png"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Redjack17.png"


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def boot_world() -> tuple[GameState, np.ndarray]:
    """
    Create a new seed, generate world, reset state.
    """
    gs: GameState = new_game_state()
    world: np.ndarray = make_world(gs.seed)
    reset_run(gs, world)
    add_log(gs, "Civarium booted.")
    add_log(gs, "Space pauses. R restarts.")
    add_log(gs, f"Spawned {len(gs.actors)} peasants.")
    return gs, world


def restart_world(gs: GameState) -> np.ndarray:
    "Reset seed + world and respawn peasants."
    gs.seed = int(time.time()) % 100_000
    world: np.ndarray = make_world(gs.seed)
    reset_run(gs, world)
    add_log(gs, f"Restarted (seed {gs.seed}).")
    add_log(gs, f"Spawned {len(gs.actors)} peasants.")
    return world


def main() -> None:
    if not TILESET_PATH.exists():
        raise FileNotFoundError(
            f"Tileset not found: {TILESET_PATH}."
        )

    tileset = tcod.tileset.load_tilesheet(
        str(TILESET_PATH),
        columns=16,
        rows=16,
        charmap=tcod.tileset.CHARMAP_CP437,
    )

    gs, world = boot_world()

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

                    if key in (tcod.event.KeySym.Q, tcod.event.KeySym.ESCAPE):
                        raise SystemExit()

                    if key == tcod.event.KeySym.SPACE:
                        gs.paused = not gs.paused
                        add_log(gs, "Paused." if gs.paused else "Resumed.")

                    elif key in (tcod.event.KeySym.PLUS, tcod.event.KeySym.KP_PLUS, tcod.event.KeySym.EQUALS):
                        gs.tps = min(60.0, gs.tps + 2.0)
                        add_log(gs, f"Speed: {gs.tps:.1f} tps")

                    elif key in (tcod.event.KeySym.MINUS, tcod.event.KeySym.KP_MINUS):
                        gs.tps = max(1.0, gs.tps - 2.0)
                        add_log(gs, f"Speed: {gs.tps:.1f} tps")

                    elif key == tcod.event.KeySym.R:
                        world = restart_world(gs)

                    elif key == tcod.event.KeySym.LEFT:
                        x, y = gs.cursor
                        gs.cursor = (clamp(x - 1, 0, MAP_W - 1), y)
                    elif key == tcod.event.KeySym.RIGHT:
                        x, y = gs.cursor
                        gs.cursor = (clamp(x + 1, 0, MAP_W - 1), y)
                    elif key == tcod.event.KeySym.UP:
                        x, y = gs.cursor
                        gs.cursor = (x, clamp(y - 1, 0, MAP_H - 1))
                    elif key == tcod.event.KeySym.DOWN:
                        x, y = gs.cursor
                        gs.cursor = (x, clamp(y + 1, 0, MAP_H - 1))

            # Timing
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            acc += dt

            step = 1.0 / max(1.0, gs.tps)
            while acc >= step:
                for msg in update(gs, world):
                    add_log(gs, msg)
                acc -= step

            render(console, world, gs)
            context.present(console)


if __name__ == "__main__":
    main()
