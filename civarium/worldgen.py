# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================
import numpy as np

MAP_W, MAP_H = 80, 45


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
