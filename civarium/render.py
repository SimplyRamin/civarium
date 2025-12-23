# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================
from __future__ import annotations

import numpy as np
import tcod

from .sim import GameState, FORUM, HOUSE, ROAD
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
    2: ("⌂", (180, 160, 120)),      # House
    3: ("=", (140, 140, 140)),      # Road
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
        for a in gs.actors:
            if a.x == cx and a.y == cy:
                console.print(cx, cy, a.glyph, fg=UI["cursor_fg"], bg=UI["cursor_bg"])
                break
        else:
            # building on tile?
            b = gs.buildings.get((cx, cy))
            if b is not None:
                ch, fg = BUILDING_GLYPHS[b]
                console.print(cx, cy, ch, fg=fg, bg=UI["cursor_bg"])
            else:
                # terrain
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
    console.print(px, 6, f"Food: {gs.food:.1f}", fg=UI["text_fg"])
    console.print(px, 7, f"Morale: {gs.morale:.2f}", fg=UI["text_fg"])

    console.print(px, 8, "Controls:", fg=UI["title_fg"])
    console.print(px, 9, "Space: pause", fg=UI["muted_fg"])
    console.print(px, 10, "+/- : speed", fg=UI["muted_fg"])
    console.print(px, 11, "Arrows: cursor", fg=UI["muted_fg"])
    console.print(px, 12, "R: restart", fg=UI["muted_fg"])
    console.print(px, 13, "Q: quit", fg=UI["muted_fg"])

    console.print(px, 14, "Inspect:", fg=UI["title_fg"])
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        b = gs.buildings.get((cx, cy))
        if b is not None:
            bname = {1: "Forum", 2: "House", 3: "Road"}.get(b, "Building")
            console.print(px, 15, f"({cx},{cy}) {bname}", fg=UI["text_fg"])
        else:
            code = int(world[cy, cx])
            tname = {0: "Plains", 1: "Forest", 2: "Water", 3: "Hill"}.get(code, "Unknown")
            console.print(px, 15, f"({cx},{cy}) {tname}", fg=UI["text_fg"])

    # Bottom log
    log_y = MAP_H
    draw_frame(console, 0, log_y, SCREEN_W, LOG_H, "Log")
    for i, line in enumerate(list(gs.log)[: LOG_H - 2]):
        console.print(2, log_y + 1 + i, line, fg=UI["log_fg"])
