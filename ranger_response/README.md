# Ranger response planner

Given a fire detected at one or more cells, choose where k ranger crews should build firebreaks, and in what order, to keep as much value as possible from burning. The model is a simplified version of the Moving Firefighter Problem (Gutiérrez-De-La-Paz et al., *Mathematics* 11(1):179, 2023; doi 10.3390/math11010179): a firefighter that needs travel time between vertices while the fire spreads on a graph. This code does not implement that paper's formulation. It is purely classical (Dijkstra, greedy search, branch and bound).

## Model

- Fuel cells: cells of the 20x20 grid (500 m each) with `hazard >= 0.55`. That gives 253 of 400 cells (63%) for seed 42. The threshold was picked from the hazard distribution so that 55-70% of cells are fuel while keeping the fuel in separate pieces joined by narrow gaps. At 0.45 the fuel is 78% and almost one block; at 0.6 it is 56% but a clean ring. Override with `--fuel-threshold`.
- Spread graph: edges join 4-neighbours that both have fuel. Spread takes 1 step, or 2 steps when moving downhill by more than 50 m between adjacent cells (fire climbs faster than it descends).
- Fire: starts at the ignition cells at step 0 and reaches each cell at the earliest time through cells that are not defended (multi-source Dijkstra).
- Crews: each starts at a station at step 0 and walks in a straight line at `speed` cells per step. Travel time is `ceil(euclid / speed)`, at least 1 for a different cell, 0 for the same cell. Then `work_time` steps (default 1) finish the firebreak.
- A firebreak finished at step f holds if f is at or before the step the fire would reach that cell. A held cell never burns and blocks spread. A failed firebreak does nothing, but the crew still spent the time.
- Objective: summed `priority` value of fuel cells that never burn (within an optional horizon; default is until the fire stops). `simulate()` in `model.py` is the only scorer; all solvers are scored through it.

## Solvers

- `solve_greedy`: repeatedly appends the (crew, cell) defence that raises the simulated objective most, until none helps.
- `solve_exact`: depth-first branch and bound over joint plans. Defences are added in the order they finish across all crews (finish step, then crew index), and each is tried only if it holds against the defences already added, which are exactly the ones that finish earlier. A defence that only holds because another crew finished first is therefore found, and every joint plan whose defences all hold is visited once. Plans containing a defence that fails are not needed, since dropping one only frees crew time. It starts from the greedy plan, orders branches by the value downstream of the cell in the fire's spread tree, and prunes with an upper bound: a burning cell can only still be saved if some defence reachable in time sits on its spread path. It returns the best plan found and `proven_optimal`, true only if the search finished before the node or time limit.
- `solve_greedy` appends one defence at a time, so it cannot find a defence that only works together with a later one (for example the second crew's earlier firebreak); it is a heuristic only.

## Simplifications

- Straight-line travel ignores terrain, roads and the fire itself.
- Spread is deterministic: no wind, no fuel-dependent rates, no spotting.
- Synthetic terrain and values, not real Khao Yai data.
- A firebreak is a single cell; one or two crews are supported by the tests (more work in the solvers but are untested).
- Fuel cells are fixed by a hazard threshold; hazard is not used as spread probability.

## Running

```
MPLBACKEND=Agg python -m ranger_response.run [--seed 42] [--ignition r,c] [--station r,c] [--crews 1] [--speed 3]
```

Other options: `--work-time`, `--fuel-threshold`, `--time-limit`, `--node-limit`, `--out`. It prints the baseline/greedy/exact table and the plan, and writes `ranger_response/ranger_response.png`. Default ignition is the highest-hazard fuel cell, default station the middle of the grid edge nearest the ignition. Tests: `python -m pytest tests/test_ranger.py`.

## What was tested and what is proven

- Optimality is proven only when `proven_optimal` is true; for one crew on the default instance it is. For two crews the search is exhaustive in principle, but on the full grid it hits the time limit, so the two-crew default result is the best plan found, not proven optimal.
- Tests (`tests/test_ranger.py`) check the exact solver against brute force over all plans: 16 random one-crew instances, and 41 random two-crew instances (up to 9 fuel cells, work time 1-3, speed 1-3) enumerating every joint plan, with and without the greedy warm start. 11 of those are seeds on which an earlier, incomplete version of the solver returned a worse plan while claiming optimality, and one is a hand-sized regression case. All run on grids of at most 20 cells. Optimality is not tested on instances larger than that.
- `simulate` is also compared with an independent step-by-step fire simulation on random plans.

## Default run result

Ignition (2, 5), station (0, 10), one crew at 3 cells per step: with no response 230 of 253 fuel cells burn; greedy saves value 10.5 (210 burn) because one firebreak in a wide fuel band saves nothing until the cut is complete; exact saves 61.0 (42 burn) with six firebreaks and finishes in about 11 s with optimality proven under this model. With two crews (`--crews 2 --station 0,10 --station 10,0`) greedy saves 17.8 and exact finds 68.0 (24 burn) but stops at its 20 s limit without a proof. These are statements about the model on synthetic data, not about a real fire.
