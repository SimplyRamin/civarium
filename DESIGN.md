# Civarium v0.1 - Aquarium Mode

## Time
- 1 tick = 1 day (abstract). Seasons/events may modify yields later.

## Core stats (kept small on purpose)
- population: int
- food: float (sorted food)
- morale: float [0, 1]
- farms: int
- houses: int
- (optional later) storage_cap, health, security

## Derived state
- food_delta = farm_yield * far,s = consumption * population
- carrying_capacity ~ houses * people_per_house

## Dynamics per tick
1) Food updates: food += food_delta
2) Morale updates:
    - shortage => morale down
    - surplus/stability => morale slowly up
3) Population:
    - if food > 0 and morale high and capacity available => births/immigration
    - if food <0 => deaths/emigration
4) Autopilot (every Nth ticks): build farm/house based on deficits
5) Emit events for the log (shortage, frowth, build, disaster)

## Non-goals (for now)
- combat, diplomacy, tech tree, pathfindings, UI polish



## Self notes
- farms and houses are built across the rivers have no access and farmers cannot reach them.
- add information to the side, for instance how many peasants are there, how many are farmers and how many are peasants and classes.