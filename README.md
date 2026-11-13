# Wildfire-response-system

Q-Pithak: a wildfire-detection sensor placement optimiser for Khao Yai National Park, Thailand. Built by a team of five at the SEA Quantum Leader Summit 2026 hackathon (problem track: wildfire and emergency response).

Given a fixed number of sensor masts, where should they go so they watch the most fire-prone, highest-value ground? The pipeline:

1. **Priority surface** (`priority_surface.py`): a grid of 500 m cells scoring hazard × value. By default `main.py` uses real data for a 10 km × 10 km window on the park's north-west edge; a synthetic terrain is still available (`--terrain synthetic`).
2. **Viewsheds** (`viewshed.py`): what each candidate site can see from a 30 m mast within 4 km, with line of sight blocked by ridges.
3. **Placement**: choose N sites that maximise covered priority (weighted maximum coverage), solved several ways in `solvers/`: exact ILP, greedy (CELF), simulated annealing on a QUBO formulation, and three QAOA variants on a statevector simulator. Plain QAOA uses an X mixer with a penalty for placing the wrong number of sensors. XY-QAOA starts from an equal superposition of all N-sensor placements and uses an XY mixer, so it never leaves them. WSXY-QAOA (warm start with XY mixer) starts each qubit from the LP relaxation of the problem and uses the same mixer.
4. **Benchmark** (`benchmark.py`): scores every method against the ILP optimum over many instances and writes `benchmark_runs.csv`, `approx_ratio.csv` and `benchmark.png`.
5. **Map** (`benchmarks/map_viz.py`): `placement_map.png` and an interactive `placement_map.html`.

## Data

The real terrain is a 20 km × 20 km region (40 × 40 cells) at the park's north-west edge, chosen by rule: mostly inside the park, with the most satellite fire detections. `data/khao_yai_500m.npz` holds per-cell summaries only; `data/DATA_SOURCES.md` gives each source's licence and attribution.

| Layer | Source | Real or modelled |
|---|---|---|
| Elevation | Copernicus DEM GLO-30 | Real |
| Fire detections, 2012 to 2025 | NASA FIRMS, VIIRS S-NPP | Real |
| Land cover, 2021 | ESA WorldCover | Real |
| Park boundary | OpenStreetMap | Real |
| Hazard | 0.40 fire density + 0.25 fuel + 0.20 dryness + 0.15 terrain spread (slope and aspect) | Modelled: the weights, the fuel score per land-cover class and the dryness rule are assumptions |
| Value | Higher inside the park and under tree cover | Modelled |

Fire detections are uneven from year to year (2020 alone is 41% of the region's total), and about half fall just outside the park boundary. `python data/fetch_khao_yai.py` regenerates the extract (needs `requirements-data.txt`; downloads about 180 MB to a cache outside the repo).

### Does the hazard layer predict later fires?

`data/HAZARD_VALIDATION.md` reports a test on a wider 100 km × 100 km area (`data/hazard_training_500m.npz`, about 33,000 detections): every method saw detections from 2012 to 2019 and was scored on 2020 to 2024. The 20% of cells ranked highest captured 60% of later detections with the hand formula above, 62% with smoothed past fire counts alone, and 61% with a logistic model that also uses distance to roads, settlements and the park boundary; inside the park the figures are 74%, 84% and 74%, a gap that is within the bootstrap interval on this measure. A fitted model did not beat past fire counts, which carry nearly all the signal. The pipeline still uses the hand formula; `hazard_model.py` holds the fitted model for reference and is not used by `main.py`.

## Results

Covered priority as a fraction of the ILP optimum, on real terrain: mean ± standard deviation over 20 instances per size. Each instance is a 10 km window of the region with its own candidate sites. The 20 windows overlap (by half on average) and the same 20 are used at every size, so the instances are not independent and the spread understates the uncertainty; between sizes only the candidate sites and the number of sensors change. Greedy and simulated annealing return one placement. The QAOA columns are the expected value of a single measurement shot with no classical repair, where a shot with the wrong number of sensors scores 0; "Random" is the same expected value for an N-sensor placement drawn uniformly at random.

| Instance | Greedy | SA | Random | QAOA | XY-QAOA | WSXY start | WSXY-QAOA |
|---|---|---|---|---|---|---|---|
| 8 sites, 2 sensors | 0.997 ± 0.016 | 0.955 ± 0.071 | 0.633 ± 0.087 | 0.295 ± 0.040 | 0.850 ± 0.044 | 0.209 ± 0.009 | 0.226 ± 0.008 |
| 12 sites, 3 sensors | 1.000 ± 0.000 | 0.966 ± 0.045 | 0.627 ± 0.076 | 0.232 ± 0.033 | 0.788 ± 0.054 | 0.138 ± 0.005 | 0.148 ± 0.006 |
| 16 sites, 4 sensors | 0.997 ± 0.008 | 0.954 ± 0.048 | 0.640 ± 0.069 | 0.191 ± 0.026 | 0.784 ± 0.055 | 0.101 ± 0.005 | 0.108 ± 0.006 |

"WSXY start" is the warm-start state measured before any QAOA layer is applied.

There is no quantum advantage here. In all 60 instances greedy scores at least as high as every QAOA variant, and the ILP solves every instance exactly. Greedy is the best method: it finds the optimum in 95%, 100% and 70% of instances at the three sizes. Simulated annealing is weaker (optimum in 50%, 35% and 15%) and more variable. Plain QAOA does worse than random placement because more than half its shots place the wrong number of sensors. XY-QAOA does better than random at every size but stays about 0.15 to 0.21 below greedy, and its absolute lead over random in finding the optimum shrinks with size: at 16 sites one shot is optimal 0.6% of the time, against 0.05% for a random placement. WSXY-QAOA scores lowest: its warm-start state puts only 13 to 25% of shots on exactly N sensors, the XY mixer cannot change that share, and the circuit adds between 0.6 and 1.7 percentage points to the starting state. Much of the low score is the warm-start clipping (`warm_eps`, 0.25 here): in a separate check on three synthetic instances, not part of the committed benchmark, a value of 0.05 raised the expected shot to 0.58 to 0.68, still at or below random placement.

Limits of the QAOA results: they come from a noiseless simulator that applies the exact coverage objective as a phase, which is not a shallow circuit on real hardware.

## Hardware runs

`hardware_runs/iqm_garnet_2026-06-01/` holds the raw data of the first of two jobs on real quantum hardware: plain QAOA with two layers on IQM Garnet (20 qubits), 2,000 shots, on a small synthetic instance with 8 sites and 2 sensors. Scored per shot with no repair, 22.5% of shots placed exactly 2 sensors and 0.5% were the optimal placement; a random 2-sensor placement is optimal 3.6% of the time. The circuit used the original cost function, which double-counted overlapping coverage, and the device's output was closer to uniform noise than to the noiseless circuit. See that folder's README for how the circuit and the counts were decoded.

`hardware_runs/iqm_garnet_2026-11-12/` holds the follow-up job of 12 November 2026 on the same device and instance: one layer, the cost function fixed, 1,000 shots. 38.8% of shots placed exactly 2 sensors and 2.8% were the optimal placement, and the output was closer to the noiseless circuit than to uniform noise. The average shot still scored 0.284 of the optimum, against 0.666 for a random 2-sensor placement, so this run shows no quantum advantage either. `hardware_runs/proposed_garnet_run/` holds the scripts that prepared, submitted and scored it.

## Dashboard

`dashboards/` is a single-page React view of one pipeline run, in 2D or draped over the real terrain in 3D: the terrain layers (priority, hazard, ecological value, elevation, recorded fires, park mask), the candidate sites, and each method's placement and coverage, with the benchmark table and the hardware runs underneath. `python main.py` writes `dashboards/public/results.json` and `python hardware_runs/export_dashboard.py` writes `dashboards/public/hardware.json`; nothing in it is mocked and nothing is live. `.github/workflows/pages.yml` publishes it to GitHub Pages once a repo admin sets Settings > Pages > Source to "GitHub Actions". See `dashboards/README.md`.

## Ranger response

`ranger_response/` plans where a ranger crew should cut firebreaks, and in what order, once a fire is detected. It runs on the synthetic terrain. It is a classical planner modelled on the Moving Firefighter Problem: fire spreads cell to cell through fuel, the crew takes time to travel, and a firebreak only holds if it is finished before the fire arrives. On the default case one crew's six firebreaks cut the burned area from 230 cells to 42. See `ranger_response/README.md` for the model and its simplifications.

Also in the repo: `solvers/qaoa_hardware.py`, a script for running QAOA on IBM quantum hardware, or IQM hardware through qBraid.

## Run

```bash
pip install -r requirements.txt
python main.py                      # full pipeline on the real 10 km window
python main.py --terrain synthetic  # the same on synthetic terrain
python main.py --quick              # 15 × 15 grid (7.5 km), skips QAOA
python benchmark.py                 # 20 instances at each of three sizes, about 12 minutes
python -m ranger_response.run       # ranger response plan for the default fire
python -m pytest tests              # checks the solvers against brute force (pip install pytest)
```

`requirements-hardware.txt` adds what the hardware scripts need, and `requirements-data.txt` what `data/fetch_khao_yai.py` needs. `.github/workflows/ci.yml` runs the tests and the dashboard build on every push.
