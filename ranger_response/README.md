# Ranger response planner

Given a fire detected at a cell, choose where one or two ranger crews should cut firebreaks, and in what order, to keep as much value as possible from burning. It is a simplified version of the Moving Firefighter Problem (Gutiérrez-De-La-Paz et al., *Mathematics* 11(1):179, 2023; doi 10.3390/math11010179): crews that need time to travel and to cut a break, while the fire spreads on a graph. This code does not implement that paper's formulation. It is purely classical (Dijkstra, greedy search, branch and bound).

There are two models. **Model v2 is the default on the real Khao Yai window** and is what `dashboards/public/ranger.json` holds. Model v1 (binary fuel, unit-step spread) is kept, with its tests, and is described in the appendix at the end together with why it was replaced.

## Why v1 was replaced

v1 showed a striking gap on real terrain: greedy saved about 17 value units and the exact search 45 to 73 (of 109 in fuel). That result existed only for one arbitrary definition of fuel, `hazard >= 0.55`:

- at `hazard >= 0.5` or `0.45` the fuel was one block and greedy equalled exact (about 8 to 9 saved of 128 to 138);
- at `hazard >= 0.6` greedy saved 65 against 69; at 0.65 nothing could be saved;
- with the physically defensible land-cover rule every one of the 400 cells in this window is fuel (each is at least 65% tree, shrub, grass or crop), and single-cell firebreaks saved about 9 of 153 for both methods.

The gap came from a 2 to 3 cell neck of "fuel" that existed only because cells just under the hazard cut-off were declared non-burnable although land cover says they are forest. A result that depends on that threshold must not be shown on the website. v2 removes the threshold and the unit-step spread, then measures whether anything differs.

## Model v2

Code: `model2.py` (instance and simulator), `solvers2.py` (solvers), `experiment.py`, `export_dashboard.py`. All times are in hours after ignition.

- **Terrain.** The same 20 x 20, 500 m window as the site (`rows 15-34, cols 3-22` of the 40 x 40 extract; `main.py` and `results.json`). Cells are land-cover burnable by `detection/fire_sim.py`'s rule (at least 50% tree, shrub, grass, crop or bare); on this window that is all 400 cells. Value is the project `priority` layer (total 152.7).
- **Spread.** Reused, not edited, from `detection/fire_sim.py`: its 8-neighbour graph, land-cover fuel factors (grass 1.0, shrub 0.9, tree 0.6, crop 0.5, bare 0.1), slope factor (`exp(3.93 s)`, clipped to 0.5-4, McArthur's doubling per 10 degrees uphill) and wind factor (`exp(0.12 min(U, 15 m/s) cos(angle))`), with base rate `r0 = 2 m/min` at fuel factor 1. The graph and wind-free edge times come from `fire_sim._Graph`; the wind factor is repeated in `model2.wind_edge_hours`, and a test checks that no-response arrival times equal `fire_sim.simulate_scenarios` on the real window. Fire arrives at each cell at the minimum travel time (Dijkstra). r0 = 2 m/min is an assumption inside the 0.44 to 2.55 m/min of published Thai deciduous-forest burns cited in `fire_sim.py`; no rate for Khao Yai's evergreen forest was found, and all spread times scale with it (sensitivity below).
- **Wind.** One constant (speed, direction) per scenario, drawn from the daytime (10:00 to 18:00, UTC+7) fire-season hourly record `data/wind_fire_season.npz` via `fire_sim.daytime_wind_pool`.
- **Horizon.** 24 hours (`fire_sim`'s default). A cell counts as burned if the fire reaches it within 24 h; value "saved" means not reached within 24 h. A line that merely delays the fire past 24 h counts as saved (see the 48 h check below).
- **Firebreaks block properly, including diagonally.** Spread is 8-neighbour, as in `fire_sim` (4-neighbour spread would drop its direction-dependent wind and slope physics and make fire spread unrealistically slow along the grid diagonals). A diagonal move is therefore forbidden at arrival time t if both cells that share its corner are blocked at t, where a cell is blocked if it is not burnable or holds a finished firebreak, and at least one of the two is a firebreak. Natural gaps between two non-burnable cells are left as `fire_sim` has them. Consequence: two firebreaks touching at a corner stop a fire at the diagonal, and a firebreak on a cell the fire never reaches can still matter as a corner. A failed firebreak (finished after the fire arrived) is inert and does not count as a corner.
- **Crews.** Start at their station at t = 0 (no dispatch delay, an assumption), walk in a straight line between cell centres at 2 km/h (an assumption; slow, to allow for a loaded walk through forest; walking takes 0.25 h per cell, small next to cutting), and then cut a line across a 500 m cell. A crew executes an ordered list of cells, so multi-cell lines arise naturally. A cell holds iff its break finishes at or before the time the fire would reach it given the other firebreaks that hold (the v1 semantics). A failed firebreak still costs its crew the time.
- **Cutting time.** Source: *Fireline production rate of handcrews in wildfires*, International Journal of Wildland Fire 32(11):1503 (field data from wildfires in southern Spain, 2014 to 2019; I did not retrieve the author list). Mean handline rates per firefighter: 0.16 m/min for indirect attack in timber fuels, 0.44 for direct attack in timber understory, 1.06 for direct attack in timber litter (range over all conditions 0.10 to 2.85). A firebreak cut ahead of the fire is indirect line, so the default is 0.16. With an assumed crew of 10 people: 10 x 0.16 x 60 = 96 m/h, so **5.2 h per 500 m cell**. The crew size is an assumption and the rate varies by a factor of about 6 between fuel types and attack modes, so the sensitivity below runs the rate at x0.5, x2 and x2.75 (the last is the direct-attack timber-understory rate, 0.44/0.16). The Spanish rates are not Thai rates, and Thai forest crews may differ.
- **Stations.** The v1 rule, unchanged: the middle of the window edge nearest the ignition (no road or ranger-station data exists in the project, none was invented). Two crews start from the same station.
- **Objective.** Total value of cells the fire does not reach within the horizon. `model2.simulate2` is the single scorer for every solver.

## Solvers (v2)

- `solve_greedy2`: repeatedly appends the (crew, cell) defence that raises the simulated objective most.
- `solve_lookahead2` (the stronger middle baseline): greedy with a 2-step lookahead. Each round it scores every single defence and every pair of defences whose second cell is within 2 cells (Chebyshev) of the first, by the objective after both; it commits the pair if that beats the best single, else the best single if it improves the objective. It finds two-cell lines whose first cell saves nothing alone; three-cell lines whose partial versions save nothing are still invisible to it. It is a heuristic of my own design, not a published algorithm.
- `solve_exact2`: depth-first branch and bound, as in v1: defences are added in order of finish time across crews, each only if it holds against the earlier-finishing ones. It is warm-started from the better of greedy and lookahead, so it can never score below them. Candidate defences are cells the fire reaches plus the corner cells of diagonal steps the fire attempts (a firebreak on an unreached corner cell can close a squeeze; a brute-force test found this case, and the first version of the solver, which ignored such cells, was wrong). `proven_optimal` is true only if the search finished before the time limit. Ties between exactly equal finish times are not enumerated in both orders (probability zero with continuous times).
- **Upper bound, and why it still holds.** A burning cell x can only still be saved if the fire's current shortest-path-tree path to x is cut. Take the first place where the new world differs from the current one along that path; up to there arrival times are unchanged. Either (a) a new firebreak sits on a path cell q, which needs `earliest_finish(q) <= current arrival at q` (the arrival at q is unchanged because the path before it is), exactly as in v1; or (b) a diagonal step p to x is squeezed shut, which needs a new firebreak on one of its two corner cells c finished by the unchanged arrival time at x. c's own arrival time may change in the new world, so no condition on it is used (a relaxation, so the bound stays valid but is looser). `earliest_finish` is the best over crews of travel plus cutting from the current crew state, and at least the finish time of the last defence added. The bound is tested against brute force (below).

## Protocol

Fixed before the full run, in the defaults of `experiment.py`; this section was written after the run, so the text is a description of that fixed setup, and anything added later is marked **post-hoc**.

1. **Scenarios.** 20, seed 42, from `fire_sim.simulate_scenarios` on the site window: ignition cell sampled in proportion to the smoothed FIRMS 2012-2025 fire history (`fire_sim.ignition_weights_from_counts`, smoothed on the large grid then sliced to the window), one wind per scenario drawn from the daytime wind pool. The same 20 ignitions and winds are used in every variant.
2. **Crews.** One and two, from the station rule above.
3. **Methods.** No response; greedy; greedy with 2-step lookahead; exact branch and bound with a 30 s limit per instance (sensitivity runs: 20 s). Everything runs on a 2-core machine, 2 instances in parallel; the exact search is time limited, so unproven results can differ slightly between machines.
4. **Metrics.** Per method: mean and median value saved as a percentage of the value that burns within 24 h with no response, cells burned, proven-optimal count, runtime. Distribution over scenarios of (exact - greedy) and (exact - lookahead) in percentage points of that value (pp). Absolute value units are given too (window total 152.7).
5. **Sensitivity.** Crew cutting rate x0.5 and x2 (cut time 10.4 h and 2.6 h), x2.75 (the direct-attack rate, 1.9 h), and r0 = 1 m/min, all on the same 20 scenarios. (The sensitivity was first planned on the first 10 scenarios with a 10 s limit; after seeing from the run log that the main experiment took only 2.6 minutes, and before any result was looked at, it was widened to all 20.)
6. **Reporting.** Everything is reported, including scenarios where methods tie or the heuristic beats the other heuristic. No scenario was dropped.
7. **Post-hoc, not in the protocol:** a 48 h horizon rerun, and the plan-slack diagnostic, both added after reading the main results.

Reproduce: `python -m ranger_response.experiment` (about 5.5 minutes on 2 cores; writes `experiment_results.json`), then `python -m ranger_response.experiment --posthoc-horizon 48` (about 6.5 minutes), `python -m ranger_response.experiment --tables` (these tables), `python -m ranger_response.export_dashboard`.

## Results

Under the default model each fire burns a mean of 30.4 cells (median 27) within 24 h, but only about 10.8 of the window's total value of 152.7 (7%), because most burning is in low-priority cells. All percentages below are of that 10.8.

**Main experiment: 20 scenarios, value saved as % of the value that burns within 24 h with no response**

| crews | method | mean % saved | median % saved | mean value saved | mean cells burned | proven optimal | mean / max runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | no response | 0 | 0 | 0 | 30.4 | n/a | n/a |
| 1 | greedy | 37.7 | 37.8 | 3.19 | 23.8 | n/a | 0.00 / 0.0 |
| 1 | lookahead | 42.9 | 43.0 | 3.67 | 22.2 | n/a | 0.02 / 0.1 |
| 1 | exact | 43.7 | 43.9 | 3.74 | 22.1 | 20 / 20 | 0.60 / 4.1 |
| 2 | no response | 0 | 0 | 0 | 30.4 | n/a | n/a |
| 2 | greedy | 56.8 | 55.0 | 5.25 | 17.9 | n/a | 0.01 / 0.0 |
| 2 | lookahead | 62.5 | 59.0 | 5.80 | 16.1 | n/a | 0.08 / 0.3 |
| 2 | exact | 67.8 | 70.6 | 6.34 | 14.4 | 12 / 20 | 13.61 / 30.0 |

**Main experiment: differences in % points of the no-response burned value (pp)**

| crews | gap | scenarios better / equal / worse | mean (pp) | median (pp) | 25th-75th pct (pp) | max (pp) | mean (value units) |
|---|---|---|---|---|---|---|---|
| 1 | exact - greedy | 16 / 4 / 0 | 5.92 | 5.83 | 3.29 to 8.47 | 16.76 | 0.544 |
| 1 | exact - lookahead | 5 / 15 / 0 | 0.80 | 0.00 | 0.00 to 0.06 | 5.87 | 0.068 |
| 1 | lookahead - greedy | 12 / 8 / 0 | 5.12 | 5.02 | 0.00 to 8.47 | 16.76 | 0.476 |
| 2 | exact - greedy | 20 / 0 / 0 | 11.05 | 10.27 | 6.03 to 15.98 | 22.81 | 1.091 |
| 2 | exact - lookahead | 11 / 9 / 0 | 5.35 | 1.53 | 0.00 to 11.09 | 16.38 | 0.545 |
| 2 | lookahead - greedy | 13 / 3 / 4 | 5.70 | 4.18 | 0.00 to 10.07 | 22.81 | 0.545 |

**Plan diagnostics (main experiment, 24 h)**

| crews | method | mean firebreaks | median of per-plan minimum slack (h) | plans with a firebreak finished < 0.25 h before the fire | median slack over all firebreaks (h) |
|---|---|---|---|---|---|
| 1 | greedy | 3.25 | 1.17 | 3 / 20 | 3.54 |
| 1 | lookahead | 3.25 | 1.26 | 3 / 20 | 2.31 |
| 1 | exact | 3.30 | 1.26 | 3 / 20 | 2.46 |
| 2 | greedy | 5.95 | 0.74 | 3 / 20 | 3.80 |
| 2 | lookahead | 5.65 | 0.93 | 5 / 20 | 3.01 |
| 2 | exact | 6.00 | 0.39 | 9 / 20 | 2.23 |

**Sensitivity (same 20 scenarios, exact limit 20 s; 'default' is the main experiment, exact limit 30 s)**

| variant | crews | mean % saved: greedy / lookahead / exact | exact - greedy: mean / median pp (better, equal, worse) | exact - lookahead: mean / median pp (better, equal, worse) | exact proven | no-response cells burned |
|---|---|---|---|---|---|---|
| default | 1 | 37.7 / 42.9 / 43.7 | 5.9 / 5.8 (16, 4, 0) | 0.8 / 0.0 (5, 15, 0) | 20 / 20 | 30.4 |
| default | 2 | 56.8 / 62.5 / 67.8 | 11.0 / 10.3 (20, 0, 0) | 5.3 / 1.5 (11, 9, 0) | 12 / 20 | 30.4 |
| crew rate x0.5 | 1 | 16.8 / 17.7 / 17.8 | 1.0 / 0.0 (6, 14, 0) | 0.1 / 0.0 (3, 17, 0) | 20 / 20 | 30.4 |
| crew rate x0.5 | 2 | 30.7 / 32.0 / 33.0 | 2.3 / 0.5 (12, 8, 0) | 1.1 / 0.3 (11, 9, 0) | 20 / 20 | 30.4 |
| crew rate x2 | 1 | 61.0 / 71.5 / 75.9 | 14.9 / 11.8 (20, 0, 0) | 4.4 / 2.7 (15, 5, 0) | 15 / 20 | 30.4 |
| crew rate x2 | 2 | 79.1 / 84.1 / 90.3 | 11.2 / 10.5 (19, 1, 0) | 6.2 / 3.4 (13, 7, 0) | 16 / 20 | 30.4 |
| crew rate x2.75 (direct-attack timber-understory rate, 0.44 m/min/person) | 1 | 69.1 / 80.4 / 87.5 | 18.4 / 16.4 (20, 0, 0) | 7.1 / 6.5 (15, 5, 0) | 17 / 20 | 30.4 |
| crew rate x2.75 (direct-attack timber-understory rate, 0.44 m/min/person) | 2 | 85.1 / 90.5 / 93.6 | 8.5 / 5.7 (18, 2, 0) | 3.1 / 3.5 (12, 8, 0) | 19 / 20 | 30.4 |
| r0 = 1 m/min | 1 | 59.1 / 62.4 / 62.6 | 3.5 / 0.0 (6, 14, 0) | 0.2 / 0.0 (1, 19, 0) | 20 / 20 | 10.1 |
| r0 = 1 m/min | 2 | 74.4 / 77.2 / 80.3 | 5.8 / 0.3 (10, 10, 0) | 3.1 / 0.0 (8, 12, 0) | 20 / 20 | 10.1 |

**Post-hoc (added after reading the main results): horizon 48 h, exact limit 20 s**

| crews | method | mean % saved | median % saved | mean value saved | mean cells burned | proven optimal | mean / max runtime (s) |
|---|---|---|---|---|---|---|---|
| 1 | no response | 0 | 0 | 0 | 92.8 | n/a | n/a |
| 1 | greedy | 42.9 | 44.0 | 12.57 | 70.2 | n/a | 0.05 / 0.1 |
| 1 | lookahead | 48.8 | 53.9 | 14.08 | 66.8 | n/a | 0.41 / 1.2 |
| 1 | exact | 54.0 | 57.1 | 15.90 | 63.1 | 1 / 20 | 19.30 / 20.0 |
| 2 | no response | 0 | 0 | 0 | 92.8 | n/a | n/a |
| 2 | greedy | 71.3 | 73.9 | 23.54 | 42.9 | n/a | 0.15 / 0.5 |
| 2 | lookahead | 75.6 | 78.8 | 23.95 | 38.6 | n/a | 2.21 / 6.6 |
| 2 | exact | 82.4 | 85.8 | 27.07 | 30.4 | 3 / 20 | 17.74 / 20.0 |

**Post-hoc horizon 48 h: differences (pp)**

| crews | gap | scenarios better / equal / worse | mean (pp) | median (pp) | 25th-75th pct (pp) | max (pp) | mean (value units) |
|---|---|---|---|---|---|---|---|
| 1 | exact - greedy | 16 / 4 / 0 | 11.09 | 5.90 | 2.00 to 19.22 | 38.75 | 3.324 |
| 1 | exact - lookahead | 12 / 8 / 0 | 5.14 | 1.08 | 0.00 to 7.04 | 39.14 | 1.814 |
| 1 | lookahead - greedy | 14 / 0 / 6 | 5.95 | 3.45 | -0.55 to 8.20 | 30.00 | 1.510 |
| 2 | exact - greedy | 15 / 5 / 0 | 11.17 | 10.45 | 0.17 to 16.81 | 38.98 | 3.530 |
| 2 | exact - lookahead | 14 / 6 / 0 | 6.83 | 3.69 | 0.00 to 13.46 | 29.86 | 3.123 |
| 2 | lookahead - greedy | 12 / 1 / 7 | 4.34 | 1.23 | -2.56 to 12.02 | 38.98 | 0.407 |

**Per scenario (% of no-response burned value saved; * = exact not proven optimal)**

| id | ignition | FIRMS count | wind m/s from deg | no-response cells / value burned | 1 crew: greedy / lookahead / exact | 2 crews: greedy / lookahead / exact |
|---|---|---|---|---|---|---|
| 0 | (18, 8) | 1 | 1.8 / 193 | 28 / 3.80 | 43.7 / 43.7 / 46.4 | 61.0 / 56.9 / 72.0 |
| 1 | (18, 7) | 1 | 2.3 / 230 | 36 / 4.15 | 33.7 / 33.7 / 39.6 | 46.2 / 46.2 / 54.1* |
| 2 | (17, 13) | 1 | 2.5 / 217 | 51 / 21.30 | 25.3 / 36.1 / 36.1 | 54.2 / 54.4 / 70.1* |
| 3 | (8, 8) | 1 | 1.0 / 37 | 43 / 25.19 | 21.4 / 21.4 / 21.4 | 36.9 / 47.6 / 47.6* |
| 4 | (19, 3) | 2 | 3.9 / 193 | 18 / 1.47 | 45.7 / 45.7 / 45.7 | 62.7 / 56.4 / 68.4 |
| 5 | (3, 17) | 0 | 1.7 / 225 | 22 / 6.43 | 27.7 / 33.5 / 33.5 | 42.1 / 59.8 / 59.8 |
| 6 | (19, 17) | 2 | 2.5 / 243 | 24 / 7.02 | 43.0 / 43.0 / 46.6 | 58.5 / 58.5 / 72.4 |
| 7 | (15, 13) | 1 | 1.6 / 38 | 59 / 26.75 | 18.0 / 18.0 / 18.0 | 30.2 / 40.1 / 40.1* |
| 8 | (3, 7) | 1 | 2.5 / 217 | 30 / 17.43 | 30.7 / 34.2 / 34.4 | 55.0 / 54.6 / 71.0* |
| 9 | (9, 18) | 4 | 2.5 / 222 | 16 / 5.48 | 38.6 / 44.6 / 44.6 | 52.1 / 72.7 / 72.7 |
| 10 | (13, 1) | 1 | 1.0 / 61 | 23 / 1.54 | 40.8 / 55.2 / 55.2 | 56.5 / 62.4 / 73.1 |
| 11 | (4, 0) | 0 | 2.1 / 73 | 18 / 2.53 | 60.3 / 60.3 / 60.3 | 92.6 / 96.5 / 96.5 |
| 12 | (6, 14) | 1 | 2.8 / 57 | 57 / 28.09 | 13.6 / 20.8 / 20.8 | 27.5 / 33.3 / 33.3* |
| 13 | (18, 11) | 2 | 1.1 / 170 | 23 / 4.39 | 62.2 / 72.9 / 72.9 | 89.6 / 98.3 / 98.3 |
| 14 | (11, 3) | 0 | 0.9 / 45 | 30 / 3.09 | 47.0 / 55.5 / 55.5 | 62.3 / 65.7 / 65.8* |
| 15 | (1, 13) | 11 | 1.6 / 263 | 32 / 17.30 | 16.0 / 24.5 / 24.5 | 37.5 / 36.3 / 43.6* |
| 16 | (18, 11) | 2 | 1.7 / 237 | 23 / 5.08 | 60.5 / 77.3 / 77.3 | 83.5 / 95.6 / 95.6 |
| 17 | (0, 9) | 3 | 3.0 / 201 | 30 / 20.50 | 55.4 / 55.4 / 58.9 | 79.9 / 79.9 / 82.8 |
| 18 | (15, 11) | 1 | 1.2 / 222 | 26 / 13.06 | 34.1 / 38.3 / 38.3 | 55.1 / 59.5 / 64.2 |
| 19 | (12, 1) | 1 | 2.2 / 210 | 18 / 1.26 | 37.1 / 43.1 / 43.1 | 52.0 / 74.8 / 74.8 |

## What the results say

- **The gap between greedy and exact is real under this model, and it is not an artefact of a fuel threshold.** All methods are scored by the same simulator and every plan is replayable. With one crew exact saves 43.7% of the burning value on average against greedy's 37.7%: better in 16 of 20 scenarios, equal in 4, never worse; median gap 5.8 pp, largest 16.8 pp. With two crews it is 67.8% against 56.8%: better in all 20, median 10.3 pp, largest 22.8 pp. Exact was never below greedy in any setting (the main experiment and the four sensitivity variants, one and two crews), as it must be, since it is warm-started from the heuristics.
- **Most of the one-crew gap, and part of the two-crew gap, is closed by a cheap 2-step lookahead.** Lookahead reaches 42.9% (one crew), within 0.8 pp of exact on average (median 0, exact better in 5 of 20, up to 5.9 pp). With two crews lookahead reaches 62.5% against exact's 67.8% (median gap 1.5 pp, exact better in 11 of 20, up to 16.4 pp); there exact is proven optimal in only 12 of 20 scenarios, so its numbers are lower bounds and the true gap is at least this. Lookahead is not uniformly better than greedy: with two crews it was worse in 4 of 20 scenarios (by up to 6.3 pp).
- **The stakes are small in absolute terms.** One crew's exact plan saves a mean of 3.7 value units, and exact over greedy adds 0.54, which is 0.36% of the window's total value. Per scenario, exact over greedy is worth up to 2.3 value units with one crew (scenario 2) and up to 3.4 with two (scenario 2; 9 of 20 scenarios gain at least 1 unit), and nothing in 4 of 20 one-crew scenarios.
- **The size of the gap depends on the crew's line-cutting rate and on r0.** If crews are half as fast (10.4 h per cell) all methods save little (17 to 18% with one crew) and exact beats greedy by about 1 pp on average. If crews are faster (the direct-attack rate), all methods save much more (69 to 88% with one crew) and the gap with one crew grows to 18 pp. At r0 = 1 m/min fires are smaller (10 cells) and the one-crew gap is 3.5 pp mean, 0 median (exact better in 6 of 20). The cutting rate is the least certain input.
- **Exact plans are brittle.** The best plans work by finishing breaks shortly before the fire arrives. With two crews, 9 of 20 exact plans contain a break finished less than 15 minutes before the modelled fire, against 3 of 20 for greedy. The model is deterministic; real fire arrival times are uncertain by far more than that, so the extra value exact finds may not survive. (The default scenario shown by `python -m ranger_response.run --scenario 15 --crews 2` has breaks finished 0.01 h and 0.02 h before the fire.)
- **A longer horizon (post-hoc).** At 48 h the fires are larger (93 cells) and the same pattern holds: one crew greedy 42.9%, lookahead 48.8%, exact 54.0% (exact proven in only 1 of 20, so a lower bound); two crews 71.3%, 75.6%, 82.4%. The percentages saved are not much lower than at 24 h, so the savings are not simply fire being delayed past 24 h, but exact is mostly unproven at 48 h.
- **Nothing here is validated against a real fire.** The fire model is uncalibrated, wind is constant, winds are mostly 1 to 4 m/s, crew speeds and cutting rate come from elsewhere, and 20 scenarios is a small sample (the gap's mean has a wide uncertainty). The results describe the model.

## Tests

`python -m pytest tests/test_ranger_v2.py` (about 15 s), in addition to the v1 tests in `tests/test_ranger.py`:

- blocking: a held break blocks spread and a break finished after the fire arrives fails; exactly at arrival it holds; two held breaks touching at a corner stop a diagonal move, one does not, a failed break does not close the corner, a non-burnable corner plus a break closes it while a natural gap between two non-burnable cells stays open;
- crew timing (walking, cutting, delay) and hand-computed arrival times with fuel factors and wind; no-response arrivals equal `fire_sim.simulate_scenarios` on six real-window scenarios;
- exact branch and bound against brute force over all plans: 30 random one-crew instances (3 x 3 and 3 x 4 grids with non-burnable holes, random wind, slope, fuel, cut time, walking speed, horizon) and 8 two-crew instances, with a regression for the unreached-corner case;
- the upper bound against brute force at the root and at the children of the root for 10 instances (a wider ad-hoc check, not in the test suite, of 300 further instances at the root, the children and the grandchildren of the root found no violation; neither did 400 further one-crew and 60 two-crew exact-versus-brute-force comparisons after the corner-cell fix, apart from a few instances whose optimum needs more firebreaks than the enumeration allowed, which a longer enumeration confirmed);
- exporter schema, per-cell arrays consistent with the reported value saved and with a replay of the stored plan, size below 120 KiB, featured-scenario rule on hand-made results.

## Running

```
MPLBACKEND=Agg python -m ranger_response.run                   # v2, real terrain, default fire (no wind, ignition (1, 13), 1 crew)
MPLBACKEND=Agg python -m ranger_response.run --scenario 15 --crews 2
python -m ranger_response.run --ignition 12,4 --wind 3,225 --r0 1 --rate-mult 2
MPLBACKEND=Agg python -m ranger_response.run --model v1 [--terrain real|synthetic] [--fuel-rule hazard|landcover] ...
python -m ranger_response.experiment ; python -m ranger_response.export_dashboard
```

`ranger_response.png` is the v2 plan for scenario 15 with two crews (the highest fire-history ignition); `ranger_response_v1.png` and `ranger_response_synthetic.png` are the v1 figures.

## Export

`dashboards/public/ranger.json` (about 100 KiB, `model: "v2"`) holds: parameters with units and sources; the experiment protocol, the summary statistics and per-scenario results for all 20 scenarios; the sensitivity and post-hoc tables; three featured scenarios, each for one and two crews, with per-cell arrival times in hours (no response and each method), the defences (crew, order, cell, `arrive_h`, `finish_h`, `fire_arrival_h`, `held`), `proven_optimal` and `stop_reason`; a caveats list; the generation date. The featured scenarios are chosen by rules fixed up front: the scenario whose ignition cell has the highest smoothed fire-history weight (**most-fire-prone**), the scenario whose one-crew exact-minus-greedy gap is the median of the 20 (**median-gap**, lower median), and the scenario with the largest such gap (**largest-gap**, shown so the top of the distribution is visible; it is not typical). The exporter runs no solver: it replays the plans stored in `experiment_results.json` and fails if a stored value does not reproduce.

---

# Appendix: model v1

The text below is the original v1 documentation, kept unchanged apart from heading levels. `python -m ranger_response.run --model v1` still runs it.

### Model

- Fuel cells: cells of the 20x20 grid (500 m each). Synthetic terrain: `hazard >= 0.55`, 253 of 400 cells (63%) for seed 42. The threshold was picked from the hazard distribution so that 55-70% of cells are fuel while keeping the fuel in separate pieces joined by narrow gaps. At 0.45 the fuel is 78% and almost one block; at 0.6 it is 56% but a clean ring. Override with `--fuel-threshold`. Real terrain: see "Real terrain" below; the default is also `hazard >= 0.55` (256 of 400 cells), and a land-cover rule is available.
- Spread graph: edges join 4-neighbours that both have fuel. Spread takes 1 step, or 2 steps when moving downhill by more than 50 m between adjacent cells (fire climbs faster than it descends).
- Fire: starts at the ignition cells at step 0 and reaches each cell at the earliest time through cells that are not defended (multi-source Dijkstra).
- Crews: each starts at a station at step 0 and walks in a straight line at `speed` cells per step. Travel time is `ceil(euclid / speed)`, at least 1 for a different cell, 0 for the same cell. Then `work_time` steps (default 1) finish the firebreak.
- A firebreak finished at step f holds if f is at or before the step the fire would reach that cell. A held cell never burns and blocks spread. A failed firebreak does nothing, but the crew still spent the time.
- Objective: summed `priority` value of fuel cells that never burn (within an optional horizon; default is until the fire stops). `simulate()` in `model.py` is the only scorer; all solvers are scored through it.

### Solvers

- `solve_greedy`: repeatedly appends the (crew, cell) defence that raises the simulated objective most, until none helps.
- `solve_exact`: depth-first branch and bound over joint plans. Defences are added in the order they finish across all crews (finish step, then crew index), and each is tried only if it holds against the defences already added, which are exactly the ones that finish earlier. A defence that only holds because another crew finished first is therefore found, and every joint plan whose defences all hold is visited once. Plans containing a defence that fails are not needed, since dropping one only frees crew time. It starts from the greedy plan, orders branches by the value downstream of the cell in the fire's spread tree, and prunes with an upper bound: a burning cell can only still be saved if some defence reachable in time sits on its spread path. It returns the best plan found and `proven_optimal`, true only if the search finished before the node or time limit.
- `solve_greedy` appends one defence at a time, so it cannot find a defence that only works together with a later one (for example the second crew's earlier firebreak); it is a heuristic only.

### Simplifications

- Straight-line travel ignores terrain, roads and the fire itself.
- Spread is deterministic: no wind, no fuel-dependent rates, no spotting.
- On synthetic terrain, values and terrain are not real Khao Yai data. On real terrain the inputs (DEM, FIRMS counts, land cover, park boundary) are real, but hazard, priority and the fuel rule are modelling assumptions, and the spread model is not calibrated against any real fire.
- A firebreak is a single cell; one or two crews are supported by the tests (more work in the solvers but are untested).
- Fuel cells are fixed by a rule (hazard threshold or land cover); hazard is not used as spread probability.

### Real terrain

`--terrain real` (the default) runs on the same 20 x 20 window (rows 15-34, cols 3-22 of the 40 x 40 extract `data/khao_yai_500m.npz`) that `main.py` and the website show. The surface comes from `build_priority_surface(20, 20, 42, source="real")`, exactly as in `main.py`, so DEM, hazard, priority and FIRMS counts match the site cell for cell (a test compares them with `dashboards/public/results.json`). Value is the project `priority` layer. `--terrain synthetic` keeps the old seeded random surface.

**Fuel rule.** Two rules exist (`--fuel-rule hazard|landcover`):

- `landcover`: fuel if tree + shrub + grass + crop cover at least 50% of the cell (the burnable test in `detection/fire_sim.py`, without its "bare" class, which cannot carry fire). Water, built-up, bare and wetland cells would be non-fuel. This is the more defensible rule physically, because it asks whether there is vegetation to burn. On this window it marks all 400 cells as fuel: every cell is at least 65% tree, shrub, grass or crop. The land is one unbroken block.
- `hazard` (default): fuel if `hazard >= 0.55`, 256 of 400 cells. Hazard is a weighted index of fire history (weight 0.40), fuel load, dryness and slope, not a statement about whether a cell can burn, so this rule treats low-hazard forest as non-burnable. It is an assumption, and the results depend on it strongly (see below). It is the default because with land cover alone one-cell firebreaks have nothing to cut, so the planner is trivial.

**Ignition.** Default: the fuel cell with the most NASA FIRMS detections in 2012-2025 (ties by hazard): (1, 13), 11 detections. Alternatives used in the tables: (12, 4) with 9, (18, 6) with 7, (16, 12) with 7 (each at least 3 cells from the earlier ones). **Station:** unchanged, the middle of the window edge nearest the ignition, here (0, 10). There is no road or ranger-station data in the project, so none was invented. A second crew defaults to the same station.

### Running

```
MPLBACKEND=Agg python -m ranger_response.run [--terrain real|synthetic] [--fuel-rule hazard|landcover]
         [--seed 42] [--ignition r,c] [--station r,c] [--crews 1] [--speed 3]
python -m ranger_response.export_dashboard [--time-limit 60]      # writes dashboards/public/ranger.json
```

Other options: `--work-time`, `--fuel-threshold`, `--time-limit`, `--node-limit`, `--out`. It prints the baseline/greedy/exact table and the plan, and writes `ranger_response/ranger_response.png` (real) or `ranger_response_synthetic.png` (synthetic). Default station is the middle of the grid edge nearest the ignition. `export_dashboard` runs six scenarios (default ignition with 1 and 2 crews, three alternative ignitions, and the land-cover rule) and writes the arrays the website draws, with a `caveats` list. Tests: `python -m pytest tests/test_ranger.py`.

### What was tested and what is proven

- Optimality is proven only when `proven_optimal` is true; for one crew on the default instance it is. For two crews the search is exhaustive in principle, but on the full grid it hits the time limit, so the two-crew default result is the best plan found, not proven optimal.
- Tests (`tests/test_ranger.py`) check the exact solver against brute force over all plans: 16 random one-crew instances, and 41 random two-crew instances (up to 9 fuel cells, work time 1-3, speed 1-3) enumerating every joint plan, with and without the greedy warm start. 11 of those are seeds on which an earlier, incomplete version of the solver returned a worse plan while claiming optimality, and one is a hand-sized regression case. All run on grids of at most 20 cells. Optimality is not tested on instances larger than that.
- `simulate` is also compared with an independent step-by-step fire simulation on random plans.

### Default run result (real terrain)

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

### Synthetic terrain result (secondary)

`--terrain synthetic`: ignition (2, 5), station (0, 10), one crew at 3 cells per step: with no response 230 of 253 fuel cells burn; greedy saves value 10.5 (210 burn) because one firebreak in a wide fuel band saves nothing until the cut is complete; exact saves 61.0 (42 burn) with six firebreaks and finished in about 11 s with optimality proven under this model when the README was first written (re-run on a slower 2-core machine it reaches the same 61.0 but stops at the 20 s limit, unproven). With two crews (`--crews 2 --station 0,10 --station 10,0`) greedy saves 17.8 and exact finds 68.0 (24 burn) but stops at its 20 s limit without a proof. These are statements about the model on synthetic data, not about a real fire.
