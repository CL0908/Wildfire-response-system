# Q-Pithak dashboard

Single-page view of the sensor-placement results for Khao Yai National Park: a 20 × 20 grid of 500 m cells, coloured by priority, hazard, ecological value, elevation, inside-park share or recorded fires. Pick a method (ILP, Greedy, SA, QAOA variants run on a simulator) to see which cells its sensors cover, and compare methods in the benchmark table.

With real terrain the map shows elevation, fire history and land cover for the north-west edge of the park, with modelled hazard and value; the Data block lists sources and required attribution. If `results.json` has `terrain: "synthetic"`, the page says so and omits the real-data layers and attribution. QAOA results come from a simulator. There is no live sensor data. The benchmark shows no quantum advantage: the classical methods match or beat every QAOA variant, and the page checks this against the data before saying so.

## Data

The page reads two files from `public/`, both written by Python. From the repo root:

| File | Written by | Used for |
|------|-----------|----------|
| `public/results.json` | `python main.py` | map layers, methods, benchmark table |
| `public/hardware.json` | `python hardware_runs/export_dashboard.py` | Hardware run section |

Do not edit either by hand. If `results.json` is missing, the page says so. `hardware.json` is optional: if it is missing or malformed, the Hardware run section is left out. The page makes no network requests beyond these two files.

## Commands

```
npm install
npm run dev      # development server
npm run build    # type-check and build to dist/
npm run lint
```
