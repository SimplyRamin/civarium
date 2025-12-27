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
from .season import SeasonClock, Season

LOG_H = 8
PLAZA_R = 3
FORUM = 1
HOUSE = 2
ROAD = 3
FARM = 4
LUMBER = 5
GARNARY = 6

# -------------------
# Costs
# -------------------
HOUSE_WOOD_COST = 8.0
FARM_WOOD_COST = 4.0
LUMBER_WOOD_COST = 2.0
GARNARY_WOOD_COST = 10.0

BASE_FOOD_CAP = 120.0
FOOD_CAP_PER_GARNARY = 80.0
SPOILAGE_RATE = 0.04  # % of food that spoils per tick
WOOD_RESERVE = 6.0

FARM_MULT = {
    Season.SPRING: 0.5,
    Season.SUMMER: 1.0,
    Season.AUTUMN: 1.5,
    Season.WINTER: 0.0
}


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
    role: str = "laborer"                   # laborer or farmer or lumber
    work: tuple[int, int] | None = None     # farm tile if farmer
    work_timer: int = 0


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
    wood: float = 30.0
    season: SeasonClock = field(default_factory=SeasonClock)


def add_log(gs: GameState, msg: str) -> None:
    gs.log.appendleft(msg)


def get_forum_pos(gs: GameState) -> tuple[int, int] | None:
    for (x, y), b in gs.buildings.items():
        if b == FORUM:
            return (x, y)
    return None


def is_water(world: np.ndarray, x: int, y: int) -> bool:
    return int(world[y, x]) == 2


def is_forest(world: np.ndarray, x: int, y: int) -> bool:
    return int(world[y, x]) == 1


def is_blocked(gs: GameState, world: np.ndarray, x: int, y: int) -> bool:
    if is_water(world, x, y):
        return True
    b = gs.buildings.get((x, y))
    if b is None:
        return False
    return b == FORUM or b == HOUSE     # Roads are walkable


def assign_farmers(gs: GameState, rng: np.random.Generator) -> None:
    farms = [pos for pos, b in gs.buildings.items() if b == FARM]
    if not farms:
        return

    # target: ~1 farmer per farm (or 2 if pop is high)
    target = min(len(gs.actors), max(1, len(farms)))

    current = [a for a in gs.actors if a.role == "farmer" and a.work is not None]
    need = max(0, target - len(current))
    if need == 0:
        return

    candidates = [a for a in gs.actors if a.role != "farmer"]
    rng.shuffle(candidates)

    for a in candidates[:need]:
        a.role = "farmer"
        a.work = farms[int(rng.integers(0, len(farms)))]


def assing_lumberjacks(gs: GameState, rng: np.random.Generator) -> None:
    camps = [pos for pos, b in gs.buildings.items() if b == LUMBER]
    if not camps:
        return

    # target ~1 per camp, but don;t steal all farmers
    pop = len(gs.actors)
    farms = sum(1 for b in gs.buildings.values() if b == FARM)
    farmer_target = min(pop, max(1, farms))         # keep atleast this many farmers

    max_lumberjacks = max(0, pop - farmer_target)   # whatever actor is left can be lumberjack
    target = min(len(camps), max_lumberjacks)
    if target <= 0:
        return

    current = [a for a in gs.actors if a.role == "lumberjack" and a.work is not None]
    need = max(0, target - len(current))
    if need == 0:
        return

    candidates = [a for a in gs.actors if a.role == "laborer"]
    rng.shuffle(candidates)

    for a in candidates[:need]:
        a.role = "lumberjack"
        a.work = camps[int(rng.integers(0, len(camps)))]


def place_near_center_nonwater(gs: GameState, world: np.ndarray) -> tuple[int, int]:
    cx, cy = MAP_W // 2, MAP_H // 2
    for r in range(10):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < MAP_W and 0 <= y < MAP_H and not is_water(world, x, y):
                    return (x, y)
    return (cx, cy)


def near_forest(world: np.ndarray, x: int, y: int) -> bool:
    # 8-neighborhood
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAP_W and 0 <= ny < MAP_H and is_forest(world, nx, ny):
                return True
    return False


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


def nearest_road(gs: GameState, target: tuple[int, int]) -> tuple[int, int] | None:
    tx, ty = target
    best = None
    best_d = 10**9
    for (x, y), b in gs.buildings.items():
        if b == ROAD:
            d = abs(x - tx) + abs(y - ty)
            if d < best_d:
                best_d = d
                best = (x, y)
    return best


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
    def place(gs: GameState, world: np.ndarray, x: int, y: int):
        gs.buildings[(x, y)] = FARM
        gs.wood -= FARM_WOOD_COST
        start = nearest_road(gs, (x, y)) or around
        carve_road(gs, world, start, (x, y))
        assign_farmers(gs, rng)

    start_r = 10 + len([b for b in gs.buildings.values() if b == FARM]) // 2
    return _try_place_with_expanding_radius(
        gs, world, around, rng,
        can_place=can_place_farm,
        place=place,
        start_r=start_r,
    )


def can_place_house(gs: GameState, world: np.ndarray, x: int, y: int) -> bool:
    if not (0 <= x < MAP_W and 0 <= y < MAP_H):
        return False
    if is_water(world, x, y):
        return False
    if (x, y) in gs.buildings:
        return False
    return True


def try_build_house(gs: GameState, world: np.ndarray, center: tuple[int, int]) -> bool:
    cx, cy = center

    def place(x: int, y: int) -> None:
        gs.buildings[(x, y)] = HOUSE
        gs.wood -= HOUSE_WOOD_COST
        start = nearest_road(gs, (x, y)) or center
        carve_road(gs, world, start, (x, y))

    # start near the forum, expand outward
    max_r = max(MAP_W, MAP_H)

    for r in range(2, max_r):
        # walk the square ring at radius r (perimeter only)
        x0, x1 = cx - r, cx + r
        y0, y1 = cy - r, cy + r

        # top and bottom edges
        for x in range(x0, x1 + 1):
            y = y0
            if can_place_house(gs, world, x, y):
                place(x, y)
                return True
            y = y1
            if can_place_house(gs, world, x, y):
                place(x, y)
                return True

        # left and right edges (excluding corners already checked)
        for y in range(y0 + 1, y1):
            x = x0
            if can_place_house(gs, world, x, y):
                place(x, y)
                return True
            x = x1
            if can_place_house(gs, world, x, y):
                place(x, y)
                return True

    return False


def can_place_lumber(gs: GameState, world: np.ndarray, x: int, y: int) -> bool:
    if not (0 <= x < MAP_W and 0 <= y < MAP_H):
        return False
    if is_water(world, x, y):
        return False

    # don't overwrite anything, including roads
    if (x, y) in gs.buildings:
        return False

    # must be on forest or next to forest
    return is_forest(world, x, y) or near_forest(world, x, y)


def try_build_lumber(gs: GameState, world: np.ndarray, around: tuple[int, int], rng: np.random.Generator) -> bool:
    def place(gs: GameState, world: np.ndarray, x: int, y: int):
        gs.buildings[(x, y)] = LUMBER
        gs.wood -= LUMBER_WOOD_COST
        start = nearest_road(gs, (x, y)) or around
        carve_road(gs, world, start, (x, y))
        assing_lumberjacks(gs, rng)

    # prefer near forum but expand out if no forests nearby
    start_r = 10
    return _try_place_with_expanding_radius(
        gs, world, around, rng,
        can_place=can_place_lumber,
        place=place,
        start_r=start_r,
    )


def can_place_garnary(gs: GameState, world: np.ndarray, x: int, y: int) -> bool:
    if not (0 <= x < MAP_W and 0 <= y < MAP_H):
        return False
    if is_water(world, x, y):
        return False
    if (x, y) in gs.buildings:
        return False
    return True


def try_build_garnary(gs: GameState, world: np.ndarray, around: tuple[int, int], rng: np.random.Generator) -> bool:
    if gs.wood < GARNARY_WOOD_COST:
        return False

    def place(gs: GameState, world: np.ndarray, x: int, y: int) -> None:
        gs.buildings[(x, y)] = GARNARY
        gs.wood -= GARNARY_WOOD_COST
        start = nearest_road(gs, (x, y)) or around
        carve_road(gs, world, start, (x, y))

    # Prefer near the forum but, but expand outward
    return _try_place_with_expanding_radius(
        gs, world, around, rng,
        can_place=can_place_garnary,
        place=place,
        start_r=6,
        attempts_per_r=220,
    )


def _try_place_with_expanding_radius(
        gs: GameState,
        world: np.ndarray,
        center: tuple[int, int],
        rng: np.random.Generator,
        can_place,
        place,
        start_r: int = 8,
        max_r: int | None = None,
        attempts_per_r: int = 250) -> bool:
    cx, cy = center
    if max_r is None:
        max_r = max(MAP_W, MAP_H)

    r = start_r
    while r <= max_r:
        for _ in range(attempts_per_r):
            x = cx + int(rng.integers(-r, r + 1))   # type: ignore
            y = cy + int(rng.integers(-r, r + 1))   # type: ignore
            # using plaza radius to avoid placing too close to center
            if abs(x - cx) <= PLAZA_R and abs(y - cy) <= PLAZA_R:
                continue
            if 0 <= x < MAP_W and 0 <= y < MAP_H and can_place(gs, world, x, y):
                place(gs, world, x, y)
                return True
        r = int(r * 1.35) + 1
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
    gs.wood = 30.0

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

    assign_farmers(gs, rng)
    assing_lumberjacks(gs, rng)


def update(gs: GameState, world: np.ndarray) -> list[str]:
    if gs.paused:
        return []

    gs.tick += 1
    gs.season.advance()

    events: list[str] = []

    rng = np.random.default_rng(gs.seed + gs.tick)
    forum = get_forum_pos(gs)

    # simple day cycle of 20 ticks
    phase = gs.tick % 20
    work_hours = 2 <= phase <= 17
    for a in gs.actors:
        stick = False
        target = None
        if a.role in ("farmer", "lumberjack") and a.work is not None:
            # decie if actor should be working now or not
            want_work = work_hours or (gs.food < 15)
            if not want_work:
                a.work_timer = 0
                target = a.home or forum
                stick = False
            else:
                if a.work_timer > 0:
                    target = a.work
                    stick = True
                    a.work_timer -= 1
                else:
                    if work_hours:
                        target = a.work
                        # when they arrive, start a work session
                        if (a.x, a.y) == a.work:
                            a.work_timer = 12
                            stick = True
        else:
            if forum is None:
                target = a.home
            else:
                target = a.home if (a.home is not None and phase >= 14) else forum

        if stick:
            dx, dy = 0, 0
        else:
            if target is None:
                dx = int(rng.integers(-1, 2))
                dy = int(rng.integers(-1, 2))
            else:
                tx, ty = target
                dx = 0 if a.x == tx else (1 if a.x < tx else -1)
                dy = 0 if a.y == ty else (1 if a.y < ty else -1)

                # only random movement for non-farmers or outside thw work hours
                if not (a.role == "farmer" and work_hours):
                    if rng.random() < 0.35:
                        dx = int(rng.integers(-1, 2))
                    if rng.random() < 0.35:
                        dy = int(rng.integers(-1, 2))

        nx, ny = a.x + dx, a.y + dy
        if 0 <= nx < MAP_W and 0 <= ny < MAP_H and not is_blocked(gs, world, nx, ny):
            a.x, a.y = nx, ny

    # Economy
    # -----------------------------

    # Base foraging
    base_food = 0.8

    # Count staffed farms: farmers currently standing on their assigned work tile.
    staffed_farmer = 0
    for a in gs.actors:
        if a.role == "farmer" and a.work is not None and (a.x, a.y) == a.work:
            staffed_farmer += 1

    # Count staffed lumbers
    staffed_lumber = 0
    for a in gs.actors:
        if a.role == "lumberjack" and a.work is not None and (a.x, a.y) == a.work:
            staffed_lumber += 1

    # Overall Staffed
    # staffed = sum([staffed_farmer, staffed_lumber])

    # Seasons variables
    season = gs.season.season
    winter = season.name == "WINTER"
    farm_multiplier = FARM_MULT[season]

    food_from_staffed = 1.1 * staffed_farmer * farm_multiplier

    pop = len(gs.actors)
    houses = sum(1 for b in gs.buildings.values() if b == HOUSE)
    farms = sum(1 for b in gs.buildings.values() if b == FARM)
    lumbers = sum(1 for b in gs.buildings.values() if b == LUMBER)
    garnaries = sum(1 for b in gs.buildings.values() if b == GARNARY)
    food_cap = BASE_FOOD_CAP + garnaries * FOOD_CAP_PER_GARNARY
    desired_farms = max(1, pop // 4)

    # Spoilage
    if gs.food > food_cap:
        excess = gs.food - food_cap
        gs.food -= SPOILAGE_RATE * excess

    # houses slightly improve stability; pop consumes food
    production = base_food + food_from_staffed
    consumption = 0.14 * pop
    gs.food += production - consumption
    gs.food = max(0.0, gs.food)

    gs.wood += 0.20 * staffed_lumber

    capacity = houses * 4

    if gs.food <= 0.0:
        gs.morale = max(0.0, gs.morale - 0.02)
        # occasionally lose someone if starving
        if pop > 0 and rng.random() < 0.01:
            gs.actors.pop()
            assign_farmers(gs, rng)
            events.append("Starvation: a peasant was lost.")
    else:
        gs.morale = min(1.0, gs.morale + 0.005)

    # Immigration
    # Immigration in winter is more restricted due to the fact farming is not available
    food_req = 40 if not winter else 70
    if (pop < capacity
       and gs.food > food_req
       and gs.morale > 0.70
       and rng.random() < 0.02
       and staffed_farmer >= 1):
        fx, fy = forum if forum is not None else (MAP_W // 2, MAP_H // 2)
        gs.actors.append(Actor(x=fx, y=fy, glyph="@", fg=(230, 230, 230), home=None))
        assign_farmers(gs, rng)
        assing_lumberjacks(gs, rng)
        events.append("A newcomer arrived at the Forum.")

    if gs.tick % 25 == 0:
        forum = get_forum_pos(gs)
        if forum is not None:
            built = False
            # Priority 1: if housing is tight, built a house.
            if pop >= capacity - 1:
                if gs.wood >= HOUSE_WOOD_COST:
                    built = try_build_house(gs, world, forum)
                    if built:
                        events.append("Built a house.")
                    else:
                        events.append("House build failed (no space).")
                else:
                    if gs.wood >= LUMBER_WOOD_COST and try_build_lumber(gs, world, forum, rng):
                        events.append("Built a lumber camp.")
                    else:
                        events.append("Need wood for housing.")

            # Priority 2: food pressure -> try farm, else get wood.
            elif (not winter) and gs.food < 60 and farms < desired_farms:
                if gs.wood >= FARM_WOOD_COST:
                    built = try_build_farm(gs, world, forum, rng)
                    if built:
                        events.append("Built a farm.")
                    else:
                        events.append("Farm build failed.")
                else:
                    if gs.wood >= LUMBER_WOOD_COST and try_build_lumber(gs, world, forum, rng):
                        events.append("Built a lumber camp.")
                    else:
                        events.append("Need wood for farms.")

            # Priority 3: if food is near/at cap, build storage (stop waste).
            elif (gs.food >= 0.92 * food_cap
                  and gs.wood >= GARNARY_WOOD_COST + WOOD_RESERVE
                  and lumbers >= 1
                  and farms >= 2
                  and staffed_farmer >= 1
                  and garnaries < max(1, farms // 2)):
                built = try_build_garnary(gs, world, forum, rng)
                if built:
                    events.append("Built a garnary.")
                else:
                    events.append("Garnary build failed.")

    # if gs.tick % 50 == 0:
    #     assign_farmers(gs, rng)
    #     farmers = sum(1 for a in gs.actors if a.role == "farmer")
    #     add_log(gs, f"Pop={pop} Farms={farms} Farmers={farmers} Staffed={staffed} Food={gs.food:.1f}")

    return events


def new_game_state() -> GameState:
    return GameState(seed=int(time.time()) % 100_000)
