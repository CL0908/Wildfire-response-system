# Wildfire-response-system

Q-Pithak: a wildfire-detection sensor placement optimiser for Khao Yai National Park, Thailand. Built by a team of five at the SEA Quantum Leader Summit 2026 hackathon (problem track: wildfire and emergency response).

Given a fixed number of sensor masts, where should they go so they watch the most fire-prone, highest-value ground? The pipeline:

1. **Priority surface** (`priority_surface.py`): a 10 km × 10 km grid of 500 m cells scoring hazard × value, on synthetic terrain matched to Khao Yai's elevation range (300–1,350 m).
2. **Viewsheds** (`viewshed.py`): what each candidate site can see from a 30 m mast within 4 km, with line of sight blocked by ridges.
3. **Placement**: choose N sites that maximise covered priority (weighted maximum coverage), solved several ways in `solvers/`: exact ILP, greedy (CELF), simulated annealing on a QUBO formulation, and QAOA / warm-start QAOA on a statevector simulator.
4. **Benchmark** (`benchmark.py`): scores every method the same way against the ILP optimum on three small instances. Results are in `approx_ratio.csv` and `benchmark.png`. On these instances the classical greedy method already matches the optimum, so the quantum methods show no advantage here.
5. **Map** (`benchmarks/map_viz.py`): `placement_map.png` and an interactive `placement_map.html`.

Also in the repo: `dashboards/` (a React front end), `ranger_reponse/` (a Moving Firefighter Problem implementation for ranger response), and `solvers/qaoa_hardware.py` (a script for running QAOA on IBM quantum hardware).

## Run

```bash
pip install -r requirements.txt
python main.py           # full pipeline
python main.py --quick   # smaller grid, skips QAOA
```
