# Q-Pithak dashboard

Single-page view of the sensor-placement results for Khao Yai National Park: a 20 × 20 grid of 500 m cells on synthetic terrain, coloured by priority, hazard, ecological value or elevation. Pick a method (ILP, Greedy, SA, QAOA variants run on a simulator) to see which cells its sensors cover, and compare methods in the benchmark table.

The terrain and all results are simulated. There is no live sensor data. The benchmark shows no quantum advantage: the classical methods match or beat every QAOA variant.

## Data

The page reads `public/results.json`, which the Python pipeline writes. From the repo root:

```
python main.py
```

Do not edit `results.json` by hand. If it is missing, the page says so. The page needs no network access.

## Commands

```
npm install
npm run dev      # development server
npm run build    # type-check and build to dist/
npm run lint
```
