# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================
from __future__ import annotations

import numpy as np
import tcod

from .sim import GameState
from .worldgen import MAP_W, MAP_H

PANEL_W = 30
LOG_H = 8

SCREEN_W = MAP_W + PANEL_W
SCREEN_H = MAP_H + LOG_H

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

BUILDING_GLYPHS = {
    1: ("#", (200, 180, 120)),      # Forum
}


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

    # Building rendering
    for (x, y), b in gs.buildings.items():
        ch, fg = BUILDING_GLYPHS[b]
        console.print(x, y, ch, fg=fg)

    # Actor rendering
    for a in gs.actors:
        console.print(a.x, a.y, a.glyph, fg=a.fg)

    # Cursor highlight
    cx, cy = gs.cursor
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        # If an actor is on the cursor tile, highlight that actor glyph
        actor_glyph = None
        actor_fg = None
        for a in gs.actors:
            if a.x == cx and a.y == cy:
                actor_glyph = a.glyph
                actor_fg = a.fg
                break

        if actor_glyph is not None:
            # keep the actor glyph, just add cursor background
            console.print(cx, cy, actor_glyph, fg=actor_fg, bg=UI["cursor_bg"])
        else:
            # otherwise highligt the underlying terrain glyph/
            ch, fg = terrain_style(world[cy, cx])
            console.print(cx, cy, ch, fg=fg, bg=UI["cursor_bg"])

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
