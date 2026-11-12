import { Suspense, lazy, memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentType } from 'react'
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
import FireHistory from './components/FireHistory'
import DataSources from './components/DataSources'
import Hardware from './components/Hardware'
import Footer from './components/Footer'
import FireTimeline, { ReplaySwitch } from './components/FireTimeline'
import { useHardware } from './lib/hardware'
import { LANDSCAPE, availableLayers, type StageLayer } from './lib/layers'
import { loadFires, type FiresState } from './lib/fires'
import { createReplay } from './lib/fireReplay'

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

type HardwareData = NonNullable<ReturnType<typeof useHardware>>

// Memoised so the replay clock, which re-renders the dashboard every frame, leaves these alone.
const ResultSections = memo(function ResultSections({
  results,
  method,
  hardware,
}: {
  results: Results
  method: Results['methods'][number]
  hardware: HardwareData | null
}) {
  return (
    <main id="results" className="results" tabIndex={-1}>
      <Placement results={results} method={method} />
      <FireHistory results={results} method={method} />
      <Benchmark rows={results.benchmark} />
      {hardware && <Hardware data={hardware} />}
      <DataSources results={results} />
    </main>
  )
})

function Dashboard({ results }: { results: Results }) {
  const [layer, setLayer] = useState<StageLayer>('landscape')
  const [view, setView] = useState<ViewMode>('3d')
  const [fellBack, setFellBack] = useState(false)
  // Only the default 3D view falls back by itself; after a manual pick the failure shows in place.
  const [picked, setPicked] = useState(false)
  // Fire replay: on or off, playback time in seconds (0 to replay.duration) and whether it is running.
  // The replay is 3D only, so leaving 3D switches it off.
  const [fireState, setFireState] = useState<FiresState | null>(null)
  const [replayOn, setReplayOn] = useState(false)
  const [replayT, setReplayT] = useState(0)
  const [replayPlaying, setReplayPlaying] = useState(false)
  useEffect(() => {
    let live = true
    void loadFires().then((s) => {
      if (live) setFireState(s)
    })
    return () => {
      live = false
    }
  }, [])
  const replay = useMemo(
    () => (fireState?.status === 'ready' ? createReplay(fireState.fires.detections) : null),
    [fireState],
  )
  const setReplay = useCallback((on: boolean) => {
    setReplayOn(on)
    setReplayPlaying(false)
    if (on) setReplayT(0)
  }, [])
  const fallBackTo2D = useCallback(() => {
    setView('2d')
    setFellBack(true)
    setReplay(false)
  }, [setReplay])
  const pickView = (mode: ViewMode) => {
    setPicked(true)
    setFellBack(false)
    setView(mode)
    if (mode === '2d') setReplay(false)
  }
  const defaultMethod = results.methods.findIndex((m) => m.name === 'ILP')
  const [methodIndex, setMethodIndex] = useState(Math.max(0, defaultMethod))
  const method = results.methods[methodIndex]
  const { grid } = results
  const hardware = useHardware()
  const layerInfos = availableLayers(results.layers)
  const replayShown = replayOn && view === '3d' && replay !== null
  const sameGrid =
    fireState?.status === 'ready' &&
    fireState.fires.grid.rows === grid.rows &&
    fireState.fires.grid.cols === grid.cols
  const coveredCells = useMemo(() => (sameGrid ? new Set(method.covered_cells) : null), [sameGrid, method])
  const replayReason =
    view === '2d'
      ? 'Needs the 3D view.'
      : fireState?.status === 'error'
        ? 'The fire detections could not be loaded.'
        : undefined
  const closeReplayFromBar = () => {
    setReplay(false)
    // The bar disappears with the focus on it; hand focus back to the switch, or to the Controls button.
    window.setTimeout(() => {
      stageRef.current?.querySelector<HTMLElement>('input[name="fire-replay"]:checked, .controls-open')?.focus()
    }, 0)
  }
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
      <section ref={stageRef} className={`stage${view === '2d' ? ' view-2d' : ''}${fellBack ? ' has-notice' : ''}${view === '3d' && layer === 'landscape' ? ' look-day' : ''}${replayShown ? ' has-replay' : ''}`} aria-label="Map">
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
          <ReplaySwitch
            on={replayShown}
            disabled={view === '2d' || replay === null}
            reason={replayReason}
            onChange={setReplay}
          />
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
              replayShown={replayShown}
              replay={replay}
              replayT={replayT}
              replayPlaying={replayPlaying}
              detections={fireState?.status === 'ready' ? fireState.fires.detections : null}
              onUnavailable={!picked ? fallBackTo2D : undefined}
            />
          </Suspense>
        )}
        {fellBack && (
          <p role="status" className="stage-notice glass">
            The 3D view could not be loaded, so the 2D map is shown.
          </p>
        )}
        {replayShown && replay && (
          <FireTimeline
            replay={replay}
            t={replayT}
            playing={replayPlaying}
            onTime={setReplayT}
            onPlaying={setReplayPlaying}
            onClose={closeReplayFromBar}
            coveredCells={coveredCells}
            methodName={method.name}
          />
        )}
        <a className="results-link glass" href="#results">
          Results <span aria-hidden="true">↓</span>
        </a>
      </section>
      <ResultSections results={results} method={method} hardware={hardware} />
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
