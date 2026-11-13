import { Suspense, lazy, memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentType } from 'react'
import { useResults } from './lib/data'
import { formatDate, formatRange } from './lib/format'
import type { Results } from './types'
import type Terrain3DView from './components/Terrain3D'
import MapView from './components/MapView'
import ParkMapView, { type ParkMapLayer } from './components/ParkMapView'
import LayerSwitch from './components/LayerSwitch'
import ViewSwitch, { type ViewMode } from './components/ViewSwitch'
import ScaleSwitch from './components/ScaleSwitch'
import { ParkControls, ParkStats } from './components/ParkControls'
import type { Scale } from './lib/scale'
import { PARK_LAYERS, loadPark, parkPlacement, type ParkK, type ParkLayer, type ParkRule, type ParkState } from './lib/park'
import ControlCard from './components/ControlCard'
import MethodPicker from './components/MethodPicker'
import MethodDetail from './components/MethodDetail'
import Benchmark from './components/Benchmark'
import Placement from './components/Placement'
import FireHistory from './components/FireHistory'
import DataSources from './components/DataSources'
import Hardware from './components/Hardware'
import ParkSections from './components/ParkSections'
import { parkTitleMeta } from './lib/parkText'
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
  // The 3D view's extent: the 10 km window the data covers, or the whole park (terrain only).
  const [scale, setScale] = useState<Scale>('window')
  // Whole-park state: its data is fetched when the park scale is first chosen.
  const [parkState, setParkState] = useState<ParkState>({ status: 'loading' })
  const [parkLayer, setParkLayer] = useState<ParkLayer>('landscape')
  const [parkK, setParkK] = useState<ParkK>(20)
  const [parkRule, setParkRule] = useState<ParkRule>('fire_history')
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
  const pickScale = (next: Scale) => {
    // A park that fails to load shows its message in place instead of dropping to 2D.
    setPicked(true)
    setFellBack(false)
    setScale(next)
    if (next === 'park') setReplay(false)
  }
  const park = scale === 'park'
  useEffect(() => {
    if (scale !== 'park') return
    let live = true
    void loadPark().then((s) => {
      if (live) setParkState(s)
    })
    return () => {
      live = false
    }
  }, [scale])
  const parkData = parkState.status === 'ready' ? parkState.park : null
  const parkMetrics = parkData ? parkPlacement(parkData, parkK, parkRule).metrics : null
  // Until the park data arrives the ground stays Landscape.
  const parkGround: ParkLayer = parkData ? parkLayer : 'landscape'
  // Landscape is a 3D-only look; the flat park map shows Priority in its place, as the window's 2D map does.
  const parkLayer2d: ParkMapLayer = parkLayer === 'landscape' ? 'priority' : parkLayer
  const defaultMethod = results.methods.findIndex((m) => m.name === 'ILP')
  const [methodIndex, setMethodIndex] = useState(Math.max(0, defaultMethod))
  const method = results.methods[methodIndex]
  const { grid } = results
  const hardware = useHardware()
  const layerInfos = availableLayers(results.layers)
  const replayShown = replayOn && view === '3d' && replay !== null && !park
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
      <section ref={stageRef} className={`stage${view === '2d' ? ' view-2d' : ''}${fellBack ? ' has-notice' : ''}${view === '3d' && (park ? parkGround === 'landscape' : layer === 'landscape') ? ' look-day' : ''}${replayShown ? ' has-replay' : ''}`} aria-label="Map">
        <div className="stage-scrim" aria-hidden="true" />
        <header ref={titleRef} className="stage-title">
          <h1>Q-Pithak</h1>
          <p>Wildfire sensor placement · Khao Yai National Park</p>
          <p className="stage-meta">
            {park ? (
              parkTitleMeta(parkData)
            ) : (
              <>
                North-west edge · {terrainFacts} · {grid.rows} × {grid.cols} grid of {grid.cell_size_m} m
            cells · {results.mast_height_m} m masts, {formatRange(results.range_m)} range · generated{' '}
            {formatDate(results.generated)}
              </>
            )}
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
            <ScaleSwitch value={scale} onChange={pickScale} />
          </div>
          <div className="card-section">
            <span className="card-label" aria-hidden="true">
              Layer
            </span>
            {view === '2d' ? (
              park ? (
                <LayerSwitch
                  layers={PARK_LAYERS.filter((l) => l.key !== 'landscape')}
                  value={parkLayer2d}
                  onChange={setParkLayer}
                />
              ) : (
                <LayerSwitch layers={layerInfos} value={layer2d} onChange={setLayer} />
              )
            ) : park ? (
              <LayerSwitch layers={PARK_LAYERS} value={parkLayer} onChange={setParkLayer} />
            ) : (
              <LayerSwitch layers={[LANDSCAPE, ...layerInfos]} value={layer} onChange={setLayer} />
            )}
          </div>
          {!park && (
            <ReplaySwitch
              on={replayShown}
              disabled={view === '2d' || replay === null}
              reason={replayReason}
              onChange={setReplay}
            />
          )}
          {park ? (
            <>
              <ParkControls k={parkK} rule={parkRule} onK={setParkK} onRule={setParkRule} />
              {parkMetrics ? (
                <ParkStats metrics={parkMetrics} />
              ) : (
                <p role={parkState.status === 'error' ? 'alert' : 'status'} className="muted">
                  {parkState.status === 'error' ? parkState.message : 'Loading placements…'}
                </p>
              )}
            </>
          ) : (
            <div className="card-section">
              <MethodPicker results={results} selected={methodIndex} onSelect={setMethodIndex} />
              <MethodDetail results={results} method={method} />
            </div>
          )}
        </ControlCard>
        {view === '2d' ? (
          <div className="stage-2d">
            {park ? (
              parkData ? (
                <div className="panel-2d panel-park">
                  <ParkMapView park={parkData} layer={parkLayer2d} k={parkK} rule={parkRule} />
                </div>
              ) : (
                <div className="stage-msg">
                  <p className={parkState.status === 'error' ? 'error' : undefined}>
                    {parkState.status === 'error' ? parkState.message : 'Loading placements…'}
                  </p>
                </div>
              )
            ) : (
              <div className="panel-2d">
                <MapView results={results} layer={layer2d} method={method} />
              </div>
            )}
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
              key={scale}
              scale={scale}
              results={results}
              layer={park ? 'landscape' : layer}
              parkData={parkData}
              parkLayer={parkGround}
              parkK={parkK}
              parkRule={parkRule}
              method={method}
              replayShown={replayShown}
              replay={replay}
              replayT={replayT}
              replayPlaying={replayPlaying}
              detections={!park && fireState?.status === 'ready' ? fireState.fires.detections : null}
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
      {park ? (
        <ParkSections
          state={parkState}
          k={parkK}
          rule={parkRule}
          onSelect={(k, rule) => {
            setParkK(k)
            setParkRule(rule)
          }}
        />
      ) : (
        <ResultSections results={results} method={method} hardware={hardware} />
      )}
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
