# Civarium (v1.0) - Settlement Autopilot Terminal Sim  
Civarium is a lightweight terminal-based settlement simulation focused on **emergent behavior**, **system fragility**, and **observation without control**.

There is no player agency, no building placement, no priorities to set, and no way to intervene.
Once a world is created, it runs on its own.

You do not rule the settlement.
You **witness** it.



## What Civarium Is

Civarium is an autonomous system inspired by:

* the *emergence* and internal logic of Dwarf Fortress
* the *background persistence* of Travian-style growth
* agent-based models and ecological simulations

A settlement forms, adapts, expands, struggles, and sometimes collapses — entirely according to its internal rules and environment.

Your role is to:

* observe
* inspect
* replay
* understand

Nothing more.



## What Civarium Is Not

Civarium is **not**:

* a strategy game
* a city builder
* a god game
* a management sim
* an optimization puzzle

You cannot:

* place buildings
* assign priorities
* issue policies
* prevent disasters
* “fix” bad outcomes

All outcomes are consequences of the system itself.



## Requirements  
### Install  
```bash
git clone SimplyRamin/civarium.git
cd civarium
uv sync
```
### Run (interactive / rendered)
```bash
uv run -m civarium.main
```  
Run with a specific seed:
```bash
uv run -m civarium.main --seed 123
```  
### Run (headless / fast simulation)
```bash
uv run -m civarium.main --headless --seed 123 --years 50
```  
Optional:
* `--print-all` prints all event messages (not only year summaries)
* `--max-ticks` safety cap  

## Controls
* Arrow keys / mouse: move cursor / inspect
* `Space`: pause / resume
* `+` / `-`: speed up / slow down
* `R`: restart (new seed / new world)
* `Q` / `Esc`: quit  



## Design Philosophy

Civarium intentionally avoids player control to prevent the “god complex” common in simulation games.

The goal is not mastery, efficiency, or optimization.

The goal is:

* to see how fragile systems behave
* to observe unintended consequences
* to understand how structure, geography, and timing shape outcomes

Failure is not a mistake.
Collapse is not a bug.

They are valid system states.



## Current Status

**v1.0 — Settlement Autopilot**

The autonomous simulation engine is stable and complete.

Future versions will focus on:

* deeper observation tools
* replay and analysis
* multi-world comparisons

Not on player control.



## License

MIT License

Copyright (c) 2025 Ramin Ferdos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

