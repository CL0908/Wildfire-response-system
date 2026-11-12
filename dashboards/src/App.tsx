import { Suspense, lazy, useCallback, useEffect, useLayoutEffect, useRef, useState, type ComponentType } from 'react'
import { useResults } from './lib/data'
import { formatDate, formatRange } from './lib/format'
import type { Results } from './types'
import type Terrain3DView from './components/Terrain3D'
import MapView from './components/MapView'
import LayerSwitch from './components/LayerSwitch'
import ViewSwitch, { type ViewMode } from './components/ViewSwitch'
import ControlCard from './components/ControlCard'
import MethodPicker from './components/MethodPicker'
import MethodDetail from './components/MethodDetail'
import Benchmark from './components/Benchmark'
import Placement from './components/Placement'
import DataSources from './components/DataSources'
import Hardware from './components/Hardware'
import Footer from './components/Footer'
import { useHardware } from './lib/hardware'
import { LANDSCAPE, availableLayers, type StageLayer } from './lib/layers'

type Terrain3DProps = Parameters<typeof Terrain3DView>[0]

function ChunkFailed({ onUnavailable }: Pick<Terrain3DProps, 'onUnavailable'>) {
  useEffect(() => onUnavailable?.(), [onUnavailable])
  return (
    <div className="stage-msg">
      <p role="alert" className="error">
        The 3D view could not be loaded. The 2D map is still available.
      </p>
    </div>
  )
}

// A type-only import above keeps three out of the main chunk; the view loads on demand.
const loadTerrain3D = (): Promise<{ default: ComponentType<Terrain3DProps> }> =>
  import('./components/Terrain3D').catch(() => ({ default: ChunkFailed }))
const Terrain3D = lazy(loadTerrain3D)
// Start fetching the 3D code beside results.json, since 3D is the default view.
void loadTerrain3D()

function Dashboard({ results }: { results: Results }) {
  const [layer, setLayer] = useState<StageLayer>('landscape')
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
  // Landscape is a 3D-only look; the 2D map shows Priority in its place.
  const layer2d = layer === 'landscape' ? 'priority' : layer
  // The 2D panel starts below the title block, whatever height the title wraps to.
  const stageRef = useRef<HTMLElement>(null)
  const titleRef = useRef<HTMLElement>(null)
  useLayoutEffect(() => {
    const stage = stageRef.current
    const title = titleRef.current
    if (!stage || !title) return
    const measure = () => stage.style.setProperty('--title-bottom', `${Math.ceil(title.offsetTop + title.offsetHeight)}px`)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(title)
    return () => observer.disconnect()
  }, [])
  const terrainFacts =
    results.terrain === 'real'
      ? 'real elevation, fire history and land cover; modelled hazard'
      : 'synthetic terrain'

  return (
    <>
      <a className="skip-link" href="#results">
        Skip to results
      </a>
      <section ref={stageRef} className={`stage${view === '2d' ? ' view-2d' : ''}${fellBack ? ' has-notice' : ''}${view === '3d' && layer === 'landscape' ? ' look-day' : ''}`} aria-label="Map">
        <div className="stage-scrim" aria-hidden="true" />
        <header ref={titleRef} className="stage-title">
          <h1>Q-Pithak</h1>
          <p>Wildfire sensor placement · Khao Yai National Park</p>
          <p className="stage-meta">
            North-west edge · {terrainFacts} · {grid.rows} × {grid.cols} grid of {grid.cell_size_m} m
            cells · {results.mast_height_m} m masts, {formatRange(results.range_m)} range · generated{' '}
            {formatDate(results.generated)}
          </p>
        </header>
        <ControlCard>
          <div className="card-section inline">
            <span className="card-label" aria-hidden="true">
              View
            </span>
            <ViewSwitch value={view} onChange={pickView} />
          </div>
          <div className="card-section">
            <span className="card-label" aria-hidden="true">
              Layer
            </span>
            {view === '2d' ? (
              <LayerSwitch layers={layerInfos} value={layer2d} onChange={setLayer} />
            ) : (
              <LayerSwitch layers={[LANDSCAPE, ...layerInfos]} value={layer} onChange={setLayer} />
            )}
          </div>
          <div className="card-section">
            <MethodPicker results={results} selected={methodIndex} onSelect={setMethodIndex} />
            <MethodDetail results={results} method={method} />
          </div>
        </ControlCard>
        {view === '2d' ? (
          <div className="stage-2d">
            <div className="panel-2d">
              <MapView results={results} layer={layer2d} method={method} />
            </div>
          </div>
        ) : (
          <Suspense
            fallback={
              <div className="stage-msg">
                <p role="status">Loading 3D view…</p>
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
        {fellBack && (
          <p role="status" className="stage-notice glass">
            The 3D view could not be loaded, so the 2D map is shown.
          </p>
        )}
        <a className="results-link glass" href="#results">
          Results <span aria-hidden="true">↓</span>
        </a>
      </section>
      <main id="results" className="results" tabIndex={-1}>
        <Placement results={results} method={method} />
        <Benchmark rows={results.benchmark} />
        {hardware && <Hardware data={hardware} />}
        <DataSources results={results} />
      </main>
      <Footer generated={results.generated} />
    </>
  )
}

export default function App() {
  const state = useResults()
  if (state.status === 'loading') {
    return (
      <section className="stage" aria-label="Map">
        <div className="stage-msg">
          <p role="status">Loading results.json…</p>
        </div>
      </section>
    )
  }
  if (state.status === 'error') {
    return (
      <section className="stage">
        <header className="stage-title">
          <h1>Q-Pithak</h1>
        </header>
        <div className="stage-msg">
          <p role="alert" className="error">
            {state.message}
          </p>
        </div>
      </section>
    )
  }
  return <Dashboard results={state.results} />
}
