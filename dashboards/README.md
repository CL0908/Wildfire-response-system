# Q-Pithak dashboard

Single-page view of the sensor-placement results for Khao Yai National Park, in 3D (the default) or 2D: a 20 × 20 grid of 500 m cells, coloured by priority, hazard, ecological value, elevation, inside-park share or recorded fires. Pick a method (ILP, Greedy, SA, QAOA variants run on a simulator) to see which cells its sensors cover, and compare methods in the benchmark table.

With real terrain the map shows elevation, fire history and land cover for the north-west edge of the park, with modelled hazard and value; the Data block lists sources and required attribution. If `results.json` has `terrain: "synthetic"`, the page says so and omits the real-data layers and attribution. QAOA results come from a simulator. There is no live sensor data. The benchmark shows no quantum advantage: the classical methods match or beat every QAOA variant, and the page checks this against the data before saying so.

## Layout

The first screen is the map at full window size, with a floating card for the view (2D or 3D), data layer and method. Scrolling moves down the page; hold Ctrl (⌘ on Mac) and scroll, pinch, or use the buttons to zoom. Below it, one column of four sections: the selected placement, the benchmark, the hardware runs and the data sources. Each section's heading is a sentence computed from the data, and the full tables sit behind "Show the numbers". In the method charts a QAOA bar is the average shot; the best of 2,000 shots is a separate open marker.

## Data

The page reads these files from `public/`, all written by Python. From the repo root:

| File | Written by | Used for |
|------|-----------|----------|
| `public/results.json` | `python main.py` | map layers, methods, benchmark table |
| `public/hardware.json` | `python hardware_runs/export_dashboard.py` | Hardware runs section |
| `public/terrain.bin`, `public/terrain.json` | `python data/export_terrain_mesh.py` (needs `requirements-data.txt`) | 3D terrain surface |

Do not edit any of them by hand. If `results.json` is missing, the page says so. `hardware.json` is optional: if it is missing or malformed, the Hardware runs section is left out. The page makes no network requests beyond these files.

## 3D view

The default layer is Landscape: the ground is painted from ESA WorldCover 2021 land cover and Copernicus elevation (`public/landcover.bin`, written by `python data/export_landcover_mesh.py`), with tree clumps placed where the data shows tree cover. The clumps, the shading, the sky and the sun are illustrative, and the page says so. The data layers below are drawn flat-lit on a dark background so their colours match the legend.

The 3D view drapes the same 500 m cells over the real elevation at 30 m (Copernicus DEM GLO-30), with heights exaggerated 2×. Sensor masts are not to scale. Coverage is computed on the 500 m grid, so on the 30 m surface a covered cell can look hidden behind a ridge the coarse model does not see. Shading is kept light so colours stay close to the legend; hover or tap a cell for its exact values. The 2D map shows the same information and is the keyboard-accessible view. If the browser cannot run WebGL, the page falls back to 2D.

The 3D view uses [three.js](https://threejs.org) 0.170 (MIT licence), loaded as a separate chunk. Re-run `data/export_terrain_mesh.py` if the map window in `results.json` changes.

## Commands

```
npm install
npm run dev      # development server
npm run build    # type-check and build to dist/
npm run lint
```
