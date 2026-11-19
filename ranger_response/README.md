# Ranger response planner

Given a fire detected at one or more cells, choose where k ranger crews should build firebreaks, and in what order, to keep as much value as possible from burning. The model is a simplified version of the Moving Firefighter Problem (Gutiérrez-De-La-Paz et al., *Mathematics* 11(1):179, 2023; doi 10.3390/math11010179): a firefighter that needs travel time between vertices while the fire spreads on a graph. This code does not implement that paper's formulation. It is purely classical (Dijkstra, greedy search, branch and bound).

## Model

- Fuel cells: cells of the 20x20 grid (500 m each). Synthetic terrain: `hazard >= 0.55`, 253 of 400 cells (63%) for seed 42. The threshold was picked from the hazard distribution so that 55-70% of cells are fuel while keeping the fuel in separate pieces joined by narrow gaps. At 0.45 the fuel is 78% and almost one block; at 0.6 it is 56% but a clean ring. Override with `--fuel-threshold`. Real terrain: see "Real terrain" below; the default is also `hazard >= 0.55` (256 of 400 cells), and a land-cover rule is available.
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
- On synthetic terrain, values and terrain are not real Khao Yai data. On real terrain the inputs (DEM, FIRMS counts, land cover, park boundary) are real, but hazard, priority and the fuel rule are modelling assumptions, and the spread model is not calibrated against any real fire.
- A firebreak is a single cell; one or two crews are supported by the tests (more work in the solvers but are untested).
- Fuel cells are fixed by a rule (hazard threshold or land cover); hazard is not used as spread probability.

## Real terrain

`--terrain real` (the default) runs on the same 20 x 20 window (rows 15-34, cols 3-22 of the 40 x 40 extract `data/khao_yai_500m.npz`) that `main.py` and the website show. The surface comes from `build_priority_surface(20, 20, 42, source="real")`, exactly as in `main.py`, so DEM, hazard, priority and FIRMS counts match the site cell for cell (a test compares them with `dashboards/public/results.json`). Value is the project `priority` layer. `--terrain synthetic` keeps the old seeded random surface.

**Fuel rule.** Two rules exist (`--fuel-rule hazard|landcover`):

- `landcover`: fuel if tree + shrub + grass + crop cover at least 50% of the cell (the burnable test in `detection/fire_sim.py`, without its "bare" class, which cannot carry fire). Water, built-up, bare and wetland cells would be non-fuel. This is the more defensible rule physically, because it asks whether there is vegetation to burn. On this window it marks all 400 cells as fuel: every cell is at least 65% tree, shrub, grass or crop. The land is one unbroken block.
- `hazard` (default): fuel if `hazard >= 0.55`, 256 of 400 cells. Hazard is a weighted index of fire history (weight 0.40), fuel load, dryness and slope, not a statement about whether a cell can burn, so this rule treats low-hazard forest as non-burnable. It is an assumption, and the results depend on it strongly (see below). It is the default because with land cover alone one-cell firebreaks have nothing to cut, so the planner is trivial.

**Ignition.** Default: the fuel cell with the most NASA FIRMS detections in 2012-2025 (ties by hazard): (1, 13), 11 detections. Alternatives used in the tables: (12, 4) with 9, (18, 6) with 7, (16, 12) with 7 (each at least 3 cells from the earlier ones). **Station:** unchanged, the middle of the window edge nearest the ignition, here (0, 10). There is no road or ranger-station data in the project, so none was invented. A second crew defaults to the same station.

## Running

```
MPLBACKEND=Agg python -m ranger_response.run [--terrain real|synthetic] [--fuel-rule hazard|landcover]
         [--seed 42] [--ignition r,c] [--station r,c] [--crews 1] [--speed 3]
python -m ranger_response.export_dashboard [--time-limit 60]      # writes dashboards/public/ranger.json
```

Other options: `--work-time`, `--fuel-threshold`, `--time-limit`, `--node-limit`, `--out`. It prints the baseline/greedy/exact table and the plan, and writes `ranger_response/ranger_response.png` (real) or `ranger_response_synthetic.png` (synthetic). Default station is the middle of the grid edge nearest the ignition. `export_dashboard` runs six scenarios (default ignition with 1 and 2 crews, three alternative ignitions, and the land-cover rule) and writes the arrays the website draws, with a `caveats` list. Tests: `python -m pytest tests/test_ranger.py`.

## What was tested and what is proven

- Optimality is proven only when `proven_optimal` is true; for one crew on the default instance it is. For two crews the search is exhaustive in principle, but on the full grid it hits the time limit, so the two-crew default result is the best plan found, not proven optimal.
- Tests (`tests/test_ranger.py`) check the exact solver against brute force over all plans: 16 random one-crew instances, and 41 random two-crew instances (up to 9 fuel cells, work time 1-3, speed 1-3) enumerating every joint plan, with and without the greedy warm start. 11 of those are seeds on which an earlier, incomplete version of the solver returned a worse plan while claiming optimality, and one is a hand-sized regression case. All run on grids of at most 20 cells. Optimality is not tested on instances larger than that.
- `simulate` is also compared with an independent step-by-step fire simulation on random plans.

## Default run result (real terrain)

Window (15, 3), fuel = `hazard >= 0.55` (256 cells), ignition (1, 13), station (0, 10), speed 3 cells per step, work 1 step, exact limit 60 s, run on a 2-core machine. No response burns 237 of the 256 fuel cells (total fuel value 108.9, 9.2 of it survives).

Every setting that was tried is listed (values are "saved / cells burned"; exact limit 60 s for the first eight rows, 40 s for the rest; speed in cells per step; the first row's no-response baseline is 237 burned of 256, 301 of 304 at 0.5, 338 of 339 at 0.45, 205 of 212 at 0.6, 23 of 170 at 0.65, 400 of 400 for land cover):

| setting | greedy | exact | exact proven |
|---|---|---|---|
| hazard >= 0.55, fuel 256, ignition (1, 13), 1 crew, speed 3, work 1 | 18.7 / 209 | 73.4 / 39 | no |
| hazard >= 0.55, fuel 256, ignition (1, 13), 2 crews, speed 3, work 1 | 31.4 / 178 | 73.9 / 39 | no |
| hazard >= 0.55, fuel 256, ignition (12, 4), 1 crew, speed 3, work 1 | 16.4 / 211 | 44.8 / 168 | no |
| hazard >= 0.55, fuel 256, ignition (12, 4), 2 crews, speed 3, work 1 | 25.7 / 199 | 48.8 / 160 | no |
| hazard >= 0.55, fuel 256, ignition (18, 6), 1 crew, speed 3, work 1 | 17.2 / 210 | 44.0 / 168 | no |
| hazard >= 0.55, fuel 256, ignition (18, 6), 2 crews, speed 3, work 1 | 29.1 / 193 | 53.7 / 151 | no |
| hazard >= 0.55, fuel 256, ignition (16, 12), 1 crew, speed 3, work 1 | 17.1 / 211 | 50.8 / 160 | no |
| hazard >= 0.55, fuel 256, ignition (16, 12), 2 crews, speed 3, work 1 | 26.6 / 197 | 71.8 / 92 | no |
| hazard >= 0.55, fuel 256, ignition (1, 13), 1 crew, speed 1, work 1 | 11.4 / 218 | 66.5 / 50 | yes |
| hazard >= 0.55, fuel 256, ignition (1, 13), 1 crew, speed 2, work 1 | 15.5 / 213 | 77.9 / 31 | yes |
| hazard >= 0.55, fuel 256, ignition (1, 13), 1 crew, speed 5, work 1 | 20.8 / 205 | 26.3 / 194 | no |
| hazard >= 0.55, fuel 256, ignition (1, 13), 2 crews, speed 1, work 1 | 15.8 / 212 | 79.5 / 29 | no |
| hazard >= 0.55, fuel 256, ignition (1, 13), 2 crews, speed 5, work 1 | 33.2 / 175 | 87.8 / 16 | no |
| hazard >= 0.55, fuel 256, ignition (1, 13), 1 crew, speed 3, work 2 | 13.9 / 215 | 13.9 / 215 | no |
| hazard >= 0.55, fuel 256, ignition (1, 13), 2 crews, speed 3, work 2 | 23.7 / 200 | 79.5 / 29 | no |
| hazard >= 0.45, fuel 339, ignition (1, 13), 1 crew, speed 3, work 1 | 7.5 / 321 | 7.5 / 321 | no |
| hazard >= 0.5, fuel 304, ignition (1, 13), 1 crew, speed 3, work 1 | 9.0 / 287 | 9.0 / 287 | no |
| hazard >= 0.6, fuel 212, ignition (1, 13), 1 crew, speed 3, work 1 | 65.0 / 27 | 69.0 / 21 | yes |
| hazard >= 0.65, fuel 170, ignition (1, 13), 1 crew, speed 3, work 1 | 1.6 / 21 | 4.4 / 17 | yes |
| hazard >= 0.5, fuel 304, ignition (1, 13), 2 crews, speed 3, work 1 | 18.3 / 268 | 18.3 / 268 | no |
| hazard >= 0.6, fuel 212, ignition (1, 13), 2 crews, speed 3, work 1 | 67.6 / 23 | 74.1 / 14 | no |
| land cover, fuel 400, ignition (1, 13), 1 crew, speed 3, work 1 | 8.8 / 388 | 8.8 / 388 | no |
| land cover, fuel 400, ignition (1, 13), 2 crews, speed 3, work 1 | 16.7 / 375 | 16.7 / 375 | no |
| land cover, fuel 400, ignition (12, 4), 1 crew, speed 3, work 1 | 7.7 / 390 | 10.9 / 384 | no |
| land cover, fuel 400, ignition (12, 4), 2 crews, speed 3, work 1 | 15.6 / 378 | 16.7 / 370 | no |
| land cover, fuel 400, ignition (1, 13), 1 crew, speed 10, work 1 | 11.6 / 382 | 11.6 / 382 | no |

Reading the numbers:

- Exact is not proven optimal at speed 3. A faster crew can do anything a slower one can, so the proven speed-2 optimum (77.9) is a lower bound for speed 3, which means the 73.4 shown for speed 3 is below the true optimum. The best plan found changes with the time limit and the machine (73.4 and 72.0 in two runs of the same case; with the 300,000-node limit reached after about 75 s it found 75.4, burning 35 cells, still unproven). Treat exact numbers marked unproven as "at least this much". The search also struggles at higher speed or longer work time: at speed 5 one crew's exact result (26.3) is far below the speed-3 and speed-2 results, and at work time 2 one crew's exact result equals greedy, so "exact equals greedy" in an unproven row is not evidence that nothing better exists.
- The "greedy fails, exact much better" effect appears on real terrain, but only under the hazard rule near 0.55. Greedy saves 16-18 of 109 on every ignition; exact saves 45-73. The cause is the same as on synthetic terrain: the first firebreak across a neck of fuel saves nothing until the cut is complete. The effect is fragile: at `--fuel-threshold 0.45` or 0.5 the fuel is one block and greedy equals exact (7.6 and 9.0 saved of 128-138); at 0.6 greedy already saves 65.0 against 69.0; at 0.65 the fire burns only 23 cells and nothing can be saved.
- With the land-cover rule (all 400 cells fuel) nothing worthwhile can be saved: one-cell firebreaks cannot stop a fire that goes around them, greedy saves 8.8 and exact 8.8 to 10.9, about 6% of the value, and both burn at least 385 of 400 cells.
- The saving under the default rule comes from cutting a 2-3 cell neck that joins the burning north-east cluster to a large southern block. That neck exists because cells just below the hazard cut-off are declared non-fuel, although land cover says they are forest. If fire can cross them, the saving disappears.
- Two crews help little here: the best two-crew plan found saves 73.9 against 73.4 for one crew, both unproven, and the two-crew plan contains a firebreak the fire would never reach.

## Synthetic terrain result (secondary)

`--terrain synthetic`: ignition (2, 5), station (0, 10), one crew at 3 cells per step: with no response 230 of 253 fuel cells burn; greedy saves value 10.5 (210 burn) because one firebreak in a wide fuel band saves nothing until the cut is complete; exact saves 61.0 (42 burn) with six firebreaks and finished in about 11 s with optimality proven under this model when the README was first written (re-run on a slower 2-core machine it reaches the same 61.0 but stops at the 20 s limit, unproven). With two crews (`--crews 2 --station 0,10 --station 10,0`) greedy saves 17.8 and exact finds 68.0 (24 burn) but stops at its 20 s limit without a proof. These are statements about the model on synthetic data, not about a real fire.
