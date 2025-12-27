# =================================================================================================
#                                           Written by Ramin F.
#                                      AI Engineer & Data Scientist
#                            Ferdos.ramin@gmail.com | simplyramin.github.io
# =================================================================================================
from enum import Enum


class Season(Enum):
    SPRING = "Spring"
    SUMMER = "Summer"
    AUTUMN = "Autumn"
    WINTER = "Winter"


TICKS_PER_SEASON = 30
SEASONS = [
    Season.SPRING,
    Season.SUMMER,
    Season.AUTUMN,
    Season.WINTER,
]


class SeasonClock:
    def __init__(self):
        self.tick = 0

    def advance(self):
        self.tick += 1

    @property
    def year_tick(self) -> int:
        return self.tick % (len(SEASONS) * TICKS_PER_SEASON)

    @property
    def season_index(self) -> int:
        return self.year_tick // TICKS_PER_SEASON

    @property
    def season(self) -> Season:
        return SEASONS[self.season_index]

    @property
    def ticks_left(self) -> int:
        return TICKS_PER_SEASON - (self.year_tick % TICKS_PER_SEASON)
