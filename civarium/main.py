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
import argparse
import csv

from .render import SCREEN_H, SCREEN_W, render
from .sim import GameState, add_log, new_game_state, reset_run, update
from .worldgen import make_world, MAP_H, MAP_W

# -------------------
# Paths (portable)
# -------------------
ROOT = Path(__file__).resolve().parents[1]
# FONT_PATH = ROOT / "assets" / "fonts" / "DejaVuSansMono.ttf"
TYR_PATH = ROOT / "assets" / "tilesets" / "Tyr.png"
BISASM_PATH = ROOT / "assets" / "tilesets" / "Bisasam_16x16.png"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Aesomatica_16x16.png"
# TILESET_PATH = ROOT / "assets" / "tilesets" / "Redjack17.png"
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)


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


def run_headless(seed: int | None, years: int, max_ticks: int | None, print_all: bool) -> None:
    """
    Run the simulation without rendering (fast), printing yearly summaries and a final snapshot.
    """
    # Create state / world
    gs: GameState = new_game_state()
    year_rows: list[dict] = []
    if seed is not None:
        gs.seed = seed
    world: np.ndarray = make_world(gs.seed)
    reset_run(gs, world)

    # Optional: initial info
    print(f"Civarium headless | seed={gs.seed} | target_years={years}")

    ticks = 0
    last_year_printed = 0

    while gs.year <= years:
        prev_year = gs.year
        prev_stats = gs.stats

        msgs = update(gs, world)
        ticks += 1

        # log + print selectively
        for m in msgs:
            add_log(gs, m)

            # If you wrapped year summaries into multiple lines, printing all lines is OK.
            is_year_line = m.startswith("Year ") or m.startswith("=== Year")
            if print_all or is_year_line:
                print(m)

        # stop conditions
        if max_ticks is not None and ticks >= max_ticks:
            print(f"Stopped at max_ticks={max_ticks}.")
            break

        # small protection: avoid infinite loops if year counts breaks
        if gs.year != last_year_printed:
            ys = prev_stats   # stats were reset for the *new* year, so we want previous

            # snapshot building
            houses = sum(1 for b in gs.buildings.values() if b == 2)        # HOUSE
            farms = sum(1 for b in gs.buildings.values() if b == 4)         # FARM
            roads = sum(1 for b in gs.buildings.values() if b == 3)         # ROAD
            lumbers = sum(1 for b in gs.buildings.values() if b == 5)       # LUMBER
            garnaries = sum(1 for b in gs.buildings.values() if b == 6)     # GARNARY
            bridges = sum(1 for b in gs.buildings.values() if b == 7)       # BRIDGE (if you kept 7)

            year_rows.append({
                "seed": gs.seed,
                "year": prev_year,
                "pop_peak": ys.pop_peak,
                "deaths": ys.deaths,
                "immigrants": ys.immigrants,
                "events": ys.events_started,
                "food_avg": round(ys.food_avg(), 2),
                "food_min": round(ys.food_min, 2),
                "food_max": round(ys.food_max, 2),
                "wood_avg": round(ys.wood_avg(), 2),
                "wood_min": round(ys.wood_min, 2),
                "wood_max": round(ys.wood_max, 2),
                "houses": houses,
                "farms": farms,
                "lumbers": lumbers,
                "garnaries": garnaries,
                "roads": roads,
                "bridges": bridges,
            })

            last_year_printed = gs.year

    print("\n--- Final Snapshot ---")
    print(f"Year={gs.year} Tick={gs.tick}")
    print(f"Pop={len(gs.actors)} Food={gs.food:.1f} Wood={gs.wood:.1f} Morale={gs.morale:.2f}")
    print(f"Buildings: H={houses} F={farms} L={lumbers} G={garnaries} Roads={roads} Bridges={bridges}")
    print("----------------------")

    csv_path = OUT_DIR / f"yearly_seed{gs.seed}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=year_rows[0].keys())
        writer.writeheader()
        writer.writerows(year_rows)

    print(f"\nSaved yearly stats to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="civarium")
    parser.add_argument("--headless", action="store_true", help="Run without rendering (fast).")
    parser.add_argument("--seed", type=int, default=None, help="Seed for world generation.")
    parser.add_argument("--years", type=int, default=20, help="How many years to simulate in headless mode.")
    parser.add_argument("--max-ticks", type=int, default=None, help="Hard cap on ticks (safety).")
    parser.add_argument("--print-all", action="store_true", help="Print all event messages, not just year lines.")
    args = parser.parse_args()

    if args.headless:
        run_headless(seed=args.seed, years=args.years, max_ticks=args.max_ticks, print_all=args.print_all)
        return

    bisasm = tcod.tileset.load_tilesheet(
        str(BISASM_PATH), columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
    )
    tyr = tcod.tileset.load_tilesheet(
        str(TYR_PATH), columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
    )

    # Hybrid tileset: start with bisasm, overlay alloy for certain chars
    tileset = tcod.tileset.Tileset(16, 16)

    for codepoint in tcod.tileset.CHARMAP_CP437:
        tileset.set_tile(codepoint, bisasm.get_tile(codepoint))

    KEEP_FROM_WORLD = {
        ord("."),  # plains
        ord("~"),  # water
        ord("^"),  # hill
        ord("♣"),  # forest (if you're using it; see note below)
        ord("@"),  # peasants
        ord("#"),  # forum
        ord("="),  # roads
        ord('░'),  # farms if you used "
        }
    for codepoint in range(32, 127):
        if codepoint in KEEP_FROM_WORLD:
            continue
        tileset.set_tile(codepoint, tyr.get_tile(codepoint))

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
