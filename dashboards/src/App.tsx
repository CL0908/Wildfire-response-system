import { Suspense, lazy, useCallback, useEffect, useState, type ComponentType } from 'react'
import { useResults } from './lib/data'
import { formatDate, formatRange } from './lib/format'
import type { LayerKey, Results } from './types'
import type Terrain3DView from './components/Terrain3D'
import MapView from './components/MapView'
import LayerSwitch from './components/LayerSwitch'
import ViewSwitch, { type ViewMode } from './components/ViewSwitch'
import MethodPicker from './components/MethodPicker'
import MethodDetail from './components/MethodDetail'
import Benchmark from './components/Benchmark'
import DataSources from './components/DataSources'
import Hardware from './components/Hardware'
import { useHardware } from './lib/hardware'
import { availableLayers } from './lib/layers'

type Terrain3DProps = Parameters<typeof Terrain3DView>[0]

function ChunkFailed({ onUnavailable }: Pick<Terrain3DProps, 'onUnavailable'>) {
  useEffect(() => onUnavailable?.(), [onUnavailable])
  return (
    <p role="alert" className="error">
      The 3D view could not be loaded. The 2D map is still available.
    </p>
  )
}

// A type-only import above keeps three out of the main chunk; the view loads on demand.
const loadTerrain3D = (): Promise<{ default: ComponentType<Terrain3DProps> }> =>
  import('./components/Terrain3D').catch(() => ({ default: ChunkFailed }))
const Terrain3D = lazy(loadTerrain3D)
// Start fetching the 3D code beside results.json, since 3D is the default view.
void loadTerrain3D()

function Dashboard({ results }: { results: Results }) {
  const [layer, setLayer] = useState<LayerKey>('priority')
  const [view, setView] = useState<ViewMode>('3d')
  const [fellBack, setFellBack] = useState(false)
  // Only the default 3D view falls back by itself; after a manual pick the failure shows in place.
  const [picked, setPicked] = useState(false)
  const fallBackTo2D = useCallback(() => {
    setView('2d')
    setFellBack(true)
  }, [])
  const pickView = (mode: ViewMode) => {
    setPicked(true)
    setFellBack(false)
    setView(mode)
  }
  const defaultMethod = results.methods.findIndex((m) => m.name === 'ILP')
  const [methodIndex, setMethodIndex] = useState(Math.max(0, defaultMethod))
  const method = results.methods[methodIndex]
  const { grid } = results
  const hardware = useHardware()
  const layerInfos = availableLayers(results.layers)
  const terrainFacts =
    results.terrain === 'real'
      ? 'real elevation, fire history and land cover; modelled hazard'
      : 'synthetic terrain'

  return (
    <div className="page">
      <header className="header">
        <h1>Q-Pithak</h1>
        <p>
          Sensor placement, Khao Yai National Park, north-west edge · {terrainFacts} ·{' '}
          {grid.rows} × {grid.cols} grid of {grid.cell_size_m} m cells · {results.mast_height_m} m
          masts, {formatRange(results.range_m)} range · generated{' '}
          {formatDate(results.generated)}
        </p>
      </header>
      <main className="layout">
        <section className="map-col" aria-label="Map">
          <ViewSwitch value={view} onChange={pickView} />
          <LayerSwitch layers={layerInfos} value={layer} onChange={setLayer} />
          {fellBack && (
            <p role="status" className="muted">
              The 3D view could not be loaded, so the 2D map is shown.
            </p>
          )}
          {view === '2d' ? (
            <MapView results={results} layer={layer} method={method} />
          ) : (
            <Suspense
              fallback={
                <div className="terrain3d">
                  <div className="terrain-box">
                    <p role="status" className="terrain-msg muted">
                      Loading 3D view…
                    </p>
                  </div>
                </div>
              }
            >
              <Terrain3D
                results={results}
                layer={layer}
                method={method}
                onUnavailable={!picked ? fallBackTo2D : undefined}
              />
            </Suspense>
          )}
          <DataSources results={results} />
        </section>
        <div className="side-col">
          <section aria-label="Methods">
            <MethodPicker
              results={results}
              selected={methodIndex}
              onSelect={setMethodIndex}
            />
            <MethodDetail results={results} method={method} />
          </section>
          <Benchmark rows={results.benchmark} />
          {hardware && <Hardware data={hardware} />}
        </div>
      </main>
    </div>
  )
}

export default function App() {
  const state = useResults()
  if (state.status === 'loading') {
    return (
      <div className="page">
        <p role="status">Loading results.json…</p>
      </div>
    )
  }
  if (state.status === 'error') {
    return (
      <div className="page">
        <h1>Q-Pithak</h1>
        <p role="alert" className="error">
          {state.message}
        </p>
      </div>
    )
  }
  return <Dashboard results={state.results} />
}
