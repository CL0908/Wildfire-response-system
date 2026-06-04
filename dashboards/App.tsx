import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './index.css';

import SensorList    from './components/SensorList';
import MapPanel      from './components/MapPanel';
import KpiCards      from './components/KpiCards';
import AlertFeed     from './components/AlertFeed';
import TrendChart    from './components/TrendChart';
import WindCompass   from './components/WindCompass';
import ResponseModal from './components/ResponseModal';

import { MockFeed }      from './services/mockFeed';
import { WeatherService } from './services/weatherService';
import { DataService }   from './services/dataService';
import { useAppStore }   from './store/useAppStore';
import type { WeatherSnapshot } from './types';

// ─── Trend updater (60s cadence, uses live weather risk) ─────────────────────
function useTrendUpdater() {
  const updateTrendPoint = useAppStore(s => s.updateTrendPoint);
  const events           = useAppStore(s => s.events);

  useEffect(() => {
    const id = setInterval(() => {
      const now = new Date();
      const hh  = now.getHours().toString().padStart(2, '0');
      const mm  = now.getMinutes().toString().padStart(2, '0');
      const { weather } = useAppStore.getState();
      const recentFires = events.filter(e => Date.now() - e.timestamp < 3_600_000).length;
      updateTrendPoint({
        time:       `${hh}:${mm}`,
        detections: recentFires,
        // Use real weather risk when available, else keep a plausible sim value
        riskIndex:  weather ? weather.riskIndex : Math.round(30 + Math.random() * 30),
      });
    }, 60_000);
    return () => clearInterval(id);
  }, [events, updateTrendPoint]);
}

// ─── DataService bootstrap ────────────────────────────────────────────────────
function useDataService() {
  const applyMessage = useAppStore(s => s.applyMessage);
  const applyWeather = useAppStore(s => s.applyWeather);
  const dsRef        = useRef<DataService | null>(null);

  useEffect(() => {
    const detection = new MockFeed(
      // Getter reads current weather risk from store — no prop-drilling needed
      () => useAppStore.getState().weather?.riskIndex ?? 50
    );
    const weather   = new WeatherService();
    const ds        = new DataService(detection, weather);

    dsRef.current = ds;
    ds.start(applyMessage, (snap: WeatherSnapshot | null) => applyWeather(snap));

    return () => ds.stop();
  }, [applyMessage, applyWeather]);

  return dsRef;
}

// ─── Status bar tint ─────────────────────────────────────────────────────────
function StatusBar() {
  const systemStatus = useAppStore(s => s.systemStatus);
  const isOk         = systemStatus === 'Operational';

  return (
    <motion.div
      className="h-0.5 w-full flex-shrink-0"
      animate={{ background: isOk ? 'var(--ok)' : 'var(--danger)' }}
      transition={{ duration: 0.4 }}
    />
  );
}

// ─── Weather status badge (top-right of map) ──────────────────────────────────
function WeatherBadge() {
  const weatherStatus = useAppStore(s => s.weatherStatus);
  const weather       = useAppStore(s => s.weather);

  return (
    <AnimatePresence>
      {weatherStatus !== 'loading' && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute top-3 right-3 z-[400] flex items-center gap-2 text-[9px] font-semibold px-2 py-1 rounded"
          style={{
            background: 'rgba(12,15,18,0.88)',
            border: '1px solid var(--border)',
            backdropFilter: 'blur(4px)',
            color: weatherStatus === 'live' ? 'var(--ok)' : weatherStatus === 'stale' ? 'var(--warn)' : 'var(--danger)',
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{
              background: weatherStatus === 'live' ? 'var(--ok)' : weatherStatus === 'stale' ? 'var(--warn)' : 'var(--danger)',
              animation: weatherStatus === 'live' ? 'breathe-ok 3s ease-in-out infinite' : 'none',
            }}
          />
          {weatherStatus === 'live' && weather
            ? `${weather.temperature.toFixed(1)}°C · ${weather.humidity}% RH`
            : weatherStatus === 'stale' ? 'Weather stale'
            : 'Weather error'}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  useDataService();
  useTrendUpdater();

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: 'var(--base)' }}>
      {/* Response routing modal — portals above all layout */}
      <ResponseModal />
      {/* Thin top status strip */}
      <StatusBar />

      {/* Main layout */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left sidebar */}
        <SensorList />

        {/* Centre: map — height:100% propagates into MapPanel's h-full chain */}
        <main className="flex-1 min-w-0 p-2 relative" style={{ height: '100%', minHeight: 0 }}>
          <MapPanel />
          <WeatherBadge />
        </main>

        {/* Right panel */}
        <aside
          className="flex flex-col gap-2 p-2 overflow-y-auto flex-shrink-0"
          style={{
            width: 304,
            borderLeft: '1px solid var(--border)',
            background: 'var(--base)',
          }}
        >
          <KpiCards />
          <WindCompass />
          <AlertFeed />
          <TrendChart />
        </aside>
      </div>
    </div>
  );
}
