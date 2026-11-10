# Wildfire-response-system

Q-Pithak: a wildfire-detection sensor placement optimiser for Khao Yai National Park, Thailand. Built by a team of five at the SEA Quantum Leader Summit 2026 hackathon (problem track: wildfire and emergency response).

Given a fixed number of sensor masts, where should they go so they watch the most fire-prone, highest-value ground? The pipeline:

1. **Priority surface** (`priority_surface.py`): a 10 km × 10 km grid of 500 m cells scoring hazard × value, on synthetic terrain matched to Khao Yai's elevation range (300–1,350 m).
2. **Viewsheds** (`viewshed.py`): what each candidate site can see from a 30 m mast within 4 km, with line of sight blocked by ridges.
3. **Placement**: choose N sites that maximise covered priority (weighted maximum coverage), solved several ways in `solvers/`: exact ILP, greedy (CELF), simulated annealing on a QUBO formulation, and three QAOA variants on a statevector simulator. Plain QAOA uses an X mixer with a penalty for placing the wrong number of sensors. XY-QAOA starts from an equal superposition of all N-sensor placements and uses an XY mixer, so it never leaves them. WSXY-QAOA (warm start with XY mixer) starts each qubit from the LP relaxation of the problem and uses the same mixer.
4. **Benchmark** (`benchmark.py`): scores every method against the ILP optimum on three small instances and writes `approx_ratio.csv` and `benchmark.png`.
5. **Map** (`benchmarks/map_viz.py`): `placement_map.png` and an interactive `placement_map.html`.

## Results

Covered priority as a fraction of the ILP optimum. Greedy and simulated annealing return one placement. The QAOA columns are the expected value of a single measurement shot with no classical repair, where a shot with the wrong number of sensors scores 0; "Random" is the same expected value for an N-sensor placement drawn uniformly at random.

| Instance | Greedy | SA | Random | QAOA | XY-QAOA | WSXY start | WSXY-QAOA |
|---|---|---|---|---|---|---|---|
| 8 sites, 2 sensors | 1.000 | 1.000 | 0.666 | 0.315 | 0.841 | 0.209 | 0.221 |
| 10 sites, 3 sensors | 0.971 | 1.000 | 0.719 | 0.315 | 0.870 | 0.197 | 0.211 |
| 12 sites, 3 sensors | 0.957 | 0.894 | 0.680 | 0.247 | 0.839 | 0.141 | 0.152 |

"WSXY start" is the warm-start state measured before any QAOA layer is applied.

There is no quantum advantage here. On every instance the placement from greedy or simulated annealing beats the expected QAOA shot, and the ILP solves every instance exactly. Plain QAOA does worse than random placement because more than half its shots place the wrong number of sensors. XY-QAOA does better than random (it measures an optimal placement 6 to 16 times as often) but its expected shot stays below both classical methods. WSXY-QAOA scores lowest: its warm-start state puts only 17 to 25% of shots on exactly N sensors, the XY mixer cannot change that share, and the circuit adds about one percentage point over the starting state. The instances are also small enough, at 28 to 220 possible placements, that 2,000 shots from any of these distributions found the optimum, so "best shot" figures say nothing about the circuit.

Limits of the QAOA results: they come from a noiseless simulator that applies the exact coverage objective as a phase, which is not a shallow circuit on real hardware. `solvers/qaoa_hardware.py` uses a two-qubit approximation of the objective instead and classically repairs each shot; no hardware results are included in this repo.

Also in the repo: `dashboards/` (a React front end; its source files are incomplete, so it does not build yet), `ranger_reponse/` (a Moving Firefighter Problem implementation for ranger response, which needs two external modules that are not included), and `solvers/qaoa_hardware.py` (a script for running QAOA on IBM quantum hardware, or IQM hardware through qBraid).

## Run

```bash
pip install -r requirements.txt
python main.py           # full pipeline on a 20 × 20 grid
python main.py --quick   # 15 × 15 grid (7.5 km), skips QAOA
python benchmark.py      # the three benchmark instances
python -m pytest tests   # checks the QUBO and QAOA against brute force (pip install pytest)
```
