# QPreFire Dashboard

Real-time wildfire detection dashboard for Khao Yai National Park.
Detection + sensor-placement layer only (no routing).

## Run

```bash
cd qprefire/dashboard
npm install
npm run dev
# → http://localhost:5173
```

## Plug in real data

### 1 · NASA FIRMS fire feed

**Where:** `src/services/mockFeed.ts` → `MockFeed`

Create a class that implements `DataFeed` (see `src/services/feedInterface.ts`):

```ts
// src/services/nasaFirmsFeed.ts
import type { DataFeed } from './feedInterface';
import type { FeedMessage } from '../types';

export class NasaFirmsFeed implements DataFeed {
  private timer: ReturnType<typeof setInterval> | null = null;
  constructor(private apiKey: string) {}

  start(cb: (msg: FeedMessage) => void) {
    this.timer = setInterval(async () => {
      const rows = await fetchFirmsHotspots(this.apiKey);
      for (const row of rows) {
        cb({
          type: 'fire',
          event: {
            id: row.acq_date + row.acq_time,
            timestamp: Date.now(),
            position: { lat: row.latitude, lng: row.longitude },
            confidence: Number(row.confidence),
            severity: row.frp > 50 ? 'high' : 'medium',
            detectedBy: 'FIRMS',
            acknowledged: false,
          },
        });
      }
    }, 5 * 60_000);
  }

  stop() { if (this.timer) clearInterval(this.timer); }
}
```

Then in `src/App.tsx` (line marked `REAL FEED PLUG-IN POINT`), swap:

```ts
feedRef.current = new NasaFirmsFeed(import.meta.env.VITE_FIRMS_KEY);
```

### 2 · Optimizer placement output

**Where:** `src/data/mockData.ts` → `SENSOR_PLACEMENTS` and `buildGrid()`

Replace the hardcoded placements with a fetch from your optimizer's JSON output
(`python main.py` can be extended to write a `placement_output.json`).
The `StaticDataset` interface in `src/types/index.ts` documents the full shape.

### 3 · Real sensor heartbeats

Implement `DataFeed` against your sensor network's WebSocket endpoint:

```ts
export class SensorNetworkFeed implements DataFeed {
  private ws: WebSocket | null = null;
  start(cb: (msg: FeedMessage) => void) {
    this.ws = new WebSocket('wss://your-sensor-hub/live');
    this.ws.onmessage = e => cb(JSON.parse(e.data) as FeedMessage);
  }
  stop() { this.ws?.close(); }
}
```

## Architecture

```
src/
  types/         TypeScript interfaces (Sensor, DetectionEvent, FeedMessage…)
  data/          mockData.ts — static grid + sensor placements
  services/      feedInterface.ts (DataFeed contract), mockFeed.ts
  store/         useAppStore.ts — Zustand: sensors, events, KPIs, trend
  components/
    SensorList   Left sidebar: clock, status pill, sensor rows
    MapPanel     Leaflet map: heatmap, viewshed polygons, fire markers
    KpiCards     4 animated metric cards
    AlertFeed    Live alert list with ACK button
    TrendChart   Recharts 24h detections / risk-index line chart
  App.tsx        Layout + feed bootstrap
```
