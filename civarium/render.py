# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================
from __future__ import annotations

import numpy as np
import tcod

from .sim import GameState, is_roadlike
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
    3: ("─", (140, 140, 140)),      # Road
    4: ("░", (120, 200, 120)),      # Farm
    5: ("£", (160, 120, 80)),       # Lumber Camp
    6: ("Θ", (200, 170, 90)),       # Garnary
    7: ("=", (170, 170, 210)),      # Bridge

}

ROAD_GLYPHS = {
    "h": "─",
    "v": "│",
    "ur": "└",  # coming from up, turning right
    "ul": "┘",  # coming from up, turning left
    "dr": "┌",  # coming from down, turning right
    "dl": "┐",  # coming from down, turning left
    "x": "┼",   # 4-way
    "t_up": "┴",
    "t_down": "┬",
    "t_left": "┤",
    "t_right": "├",

}


def roadlike_glyph(gs, x: int, y: int) -> str:
    n = is_roadlike(gs, (x, y - 1))
    s = is_roadlike(gs, (x, y + 1))
    w = is_roadlike(gs, (x - 1, y))
    e = is_roadlike(gs, (x + 1, y))

    # All four
    if n and s and w and e:
        return "╬"

    # T junctions
    if n and s and w:
        return "╣"
    if n and s and e:
        return "╠"
    if n and w and e:
        return "╩"
    if s and w and e:
        return "╦"

    # Corners
    if s and e:
        return "╔"
    if s and w:
        return "╗"
    if n and e:
        return "╚"
    if n and w:
        return "╝"

    # Straights
    if n or s:
        return "║"
    if w or e:
        return "═"

    # Isolated (should be rare)
    return "═"


def tint(rgb: tuple[int, int, int], mul: float) -> tuple[int, int, int]:
    r, g, b = rgb
    return (min(255, int(r*mul)), min(255, int(g*mul)), min(255, int(b*mul)))


def blend(rgb: tuple[int, int, int], target: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    r, g, b = rgb
    tr, tg, tb = target
    return (
        int(r*(1-alpha) + tr*alpha),
        int(g*(1-alpha) + tg*alpha),
        int(b*(1-alpha) + tb*alpha),
    )


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


def road_glyph(gs, x: int, y: int) -> str:
    def is_road(px: int, py: int) -> bool:
        return gs.buildings.get((px, py)) == 3  # ROAD

    up = is_road(x, y - 1)
    dn = is_road(x, y + 1)
    lf = is_road(x - 1, y)
    rt = is_road(x + 1, y)

    # connections count
    c = up + dn + lf + rt

    if c <= 1:
        # dead-end: prefer horizontal if connected left / right, else vertical
        return "─" if (lf or rt) else "│"
    if up and dn and lf and rt:
        return "┼"
    if up and dn and lf:
        return "┤"
    if up and dn and rt:
        return "├"
    if lf and rt and up:
        return "┴"
    if lf and rt and dn:
        return "┬"
    if lf and rt:
        return "─"
    if up and dn:
        return "│"
    if up and rt:
        return "└"
    if up and lf:
        return "┘"
    if dn and rt:
        return "┌"
    if dn and lf:
        return "┐"

    return "─"


def render(console: tcod.console.Console, world: np.ndarray, gs: GameState) -> None:
    console.clear()
    season = gs.season.season.value

    # Map
    for y in range(MAP_H):
        for x in range(MAP_W):
            ch, fg = terrain_style(world[y, x])
            # applying seasonal filters
            if season == "Winter":
                fg = blend(fg, (235, 235, 245), 0.55)   # wash toward snow / grey
            elif season == "Autumn":
                fg = blend(fg, (210, 170, 90), 0.25)    # warm tint
            elif season == "Spring":
                fg = blend(fg, (160, 220, 160), 0.18)   # slight green pop
            console.print(x, y, ch, fg=fg)

    # Building rendering
    for (x, y), b in gs.buildings.items():
        if b == 3:
            ch = road_glyph(gs, x, y)
            console.print(x, y, ch, fg=BUILDING_GLYPHS[b][1])
        elif b == 7:
            ch = roadlike_glyph(gs, x, y)
            console.print(x, y, ch, fg=BUILDING_GLYPHS[b][1])
        else:
            ch, fg = BUILDING_GLYPHS[b]
            console.print(x, y, ch, fg=fg)

    # Actor rendering
    for a in gs.actors:
        fg = a.fg
        if getattr(a, "role", "laborer") == "farmer":
            fg = (230, 210, 120)
        if getattr(a, "role", "laborer") == "lumberjack":
            fg = (200, 170, 140)
        console.print(a.x, a.y, a.glyph, fg=fg)

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

    # Calculations - Right panel
    garnaries = sum(1 for b in gs.buildings.values() if b == 6)
    houses = sum(1 for b in gs.buildings.values() if b == 2)
    farms = sum(1 for b in gs.buildings.values() if b == 4)
    lumber_camps = sum(1 for b in gs.buildings.values() if b == 5)
    farmers = sum(1 for a in gs.actors if a.role == "farmer")
    laborer = sum(1 for a in gs.actors if a.role == "laborer")
    lumber_jack = sum(1 for a in gs.actors if a.role == "lumberjack")
    pop = len(gs.actors)
    food_cap = 120.0 + 80.0 * garnaries
    event_str = "-" if not gs.active_event else f"{gs.active_event} ({gs.event_ticks_left})"

    # Right panel
    panel_x = MAP_W
    draw_frame(console, panel_x, 0, PANEL_W, MAP_H, "Civarium")

    px = panel_x + 2
    console.print(px, 2, f"Seed: {gs.seed}", fg=UI["text_fg"])
    console.print(px, 3, f"Year: {gs.year}", fg=UI["text_fg"])
    console.print(px, 4, f"Season: {gs.season.season.value} ({gs.season.ticks_left})", fg=UI["text_fg"])
    console.print(px, 5, f"Event: {event_str}", fg=UI["text_fg"])
    console.print(px, 6, f"Paused: {gs.paused}", fg=UI["text_fg"])
    console.print(px, 7, f"Speed: {gs.tps:.1f} tps", fg=UI["text_fg"])
    console.print(px, 8, f"Food: {gs.food:6.1f} / {food_cap:5.0f}", fg=UI["text_fg"])
    console.print(px, 9, f"Wood: {gs.wood:6.1f}", fg=UI["text_fg"])
    console.print(px, 10, f"Morale: {gs.morale:.2f}", fg=UI["text_fg"])

    console.print(px, 12, f"Deaths YTD: {gs.stats.deaths}", fg=UI["text_fg"])
    console.print(px, 13, f"Immigr YTD: {gs.stats.immigrants}", fg=UI["text_fg"])

    console.print(px, 15, f"Pop: {pop} (F:{farmers} L:{lumber_jack} U:{laborer})", fg=UI["text_fg"])
    console.print(px, 16, f"Bld: H={houses} F={farms} L={lumber_camps} G={garnaries}", fg=UI["text_fg"])

    console.print(px, 17, "Controls:", fg=UI["title_fg"])
    console.print(px, 18, "Space: pause", fg=UI["muted_fg"])
    console.print(px, 19, "+/- : speed", fg=UI["muted_fg"])
    console.print(px, 20, "Arrows: cursor", fg=UI["muted_fg"])
    console.print(px, 21, "R: restart", fg=UI["muted_fg"])
    console.print(px, 22, "Q: quit", fg=UI["muted_fg"])

    console.print(px, 24, "Inspect:", fg=UI["title_fg"])
    if 0 <= cx < MAP_W and 0 <= cy < MAP_H:
        b = gs.buildings.get((cx, cy))
        if b is not None:
            bname = {1: "Forum", 2: "House", 3: "Road",
                     4: "Farm", 5: "Lumber Camp", 6: "Garnary",
                     7: "Bridge"}.get(b, "Building")
            console.print(px, 25, f"({cx},{cy}) {bname}", fg=UI["text_fg"])
        else:
            code = int(world[cy, cx])
            tname = {0: "Plains", 1: "Forest", 2: "Water", 3: "Hill"}.get(code, "Unknown")
            console.print(px, 25, f"({cx},{cy}) {tname}", fg=UI["text_fg"])
        here = [a for a in gs.actors if (a.x, a.y) == (cx, cy)]
        if here:
            roles: dict[str, int] = {}
            for a in here:
                roles[a.role] = roles.get(a.role, 0) + 1
            role_str = ", ".join(f"{k}:{v}" for k, v in roles.items())
            console.print(px, 26, f"Actors: {len(here)} ({role_str})", fg=UI["muted_fg"])

    # Bottom log
    log_y = MAP_H
    draw_frame(console, 0, log_y, SCREEN_W, LOG_H, "Log")
    for i, line in enumerate(list(gs.log)[: LOG_H - 2]):
        console.print(2, log_y + 1 + i, line, fg=UI["log_fg"])
