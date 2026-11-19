import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useMediaQuery } from '../lib/useMediaQuery'
import { formatHM } from '../lib/ranger'
import {
  CREW_COUNTS,
  RESPONSE_MODES,
  aggregateLine,
  burnedCount,
  crewWord,
  responseSentence,
  summarise,
  type CrewCount,
  type ResponseMode,
  type ResponseScene,
  type ResponseState,
} from '../lib/response'

/** Playback speed at 1x: simulated hours per real second. */
const HOURS_PER_SECOND = 1
const SPEEDS = [1, 2] as const
/** With reduced motion, playback jumps this many hours at a time, this often (real seconds at 1x). */
const STEP_HOURS = 1
const STEP_SECONDS = 0.7
/** Arrow keys move the scrubber this many hours; Page Up and Page Down move more. */
const KEY_STEP = 0.5
const KEY_STEP_LARGE = 3

const MODE_LABEL: Record<ResponseMode, string> = { sensors: 'Sensors', satellite: 'Satellite only' }

interface SwitchProps {
  on: boolean
  disabled: boolean
  /** Why the switch is unavailable; shown under the row. */
  reason?: string
  onChange: (on: boolean) => void
}

/** The "Fire response" row of the control card, same look as the Fire replay row. */
export function ResponseSwitch({ on, disabled, reason, onChange }: SwitchProps) {
  const hintId = useId()
  return (
    <div className="card-section">
      <div className="card-row">
        <span className="card-label" aria-hidden="true">
          Fire response
        </span>
        <div
          className="segmented view-switch"
          role="radiogroup"
          aria-label="Fire response"
          aria-describedby={reason ? hintId : undefined}
          style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
        >
          <label>
            <input type="radio" name="fire-response" className="sr-only" disabled={disabled} checked={!on} onChange={() => onChange(false)} />
            <span>Off</span>
          </label>
          <label>
            <input type="radio" name="fire-response" className="sr-only" disabled={disabled} checked={on} onChange={() => onChange(true)} />
            <span>On</span>
          </label>
        </div>
      </div>
      {reason && (
        <p id={hintId} className="card-hint">
          {reason}
        </p>
      )}
    </div>
  )
}

function Segmented<K extends string | number>({
  name,
  label,
  options,
  value,
  onChange,
}: {
  name: string
  label: string
  options: { key: K; label: string }[]
  value: K
  onChange: (key: K) => void
}) {
  return (
    <div className="card-section">
      <span className="card-label" aria-hidden="true">
        {label}
      </span>
      <div
        className="segmented"
        role="radiogroup"
        aria-label={label}
        style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
      >
        {options.map((o) => (
          <label key={o.key}>
            <input type="radio" name={name} className="sr-only" checked={value === o.key} onChange={() => onChange(o.key)} />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

interface ControlsProps {
  state: ResponseState
  mode: ResponseMode
  crews: CrewCount
  ignition: number
  onMode: (mode: ResponseMode) => void
  onCrews: (crews: CrewCount) => void
  onIgnition: (cell: number) => void
  scene: ResponseScene | null
}

/** Card section while the mode is on: detection, crews, a row and column picker, the text and the assumptions. */
export function ResponseControls({ state, mode, crews, ignition, onMode, onCrews, onIgnition, scene }: ControlsProps) {
  if (state.status === 'loading') {
    return (
      <p role="status" className="card-hint">
        Loading the fire-response data…
      </p>
    )
  }
  if (state.status === 'error') {
    return (
      <p role="alert" className="card-hint rf-error">
        {state.message}
      </p>
    )
  }
  if (state.status !== 'ready') return null
  const { data } = state
  const { rows, cols } = data
  const row = Math.floor(ignition / cols)
  const col = ignition % cols
  const aggregate = aggregateLine(data, crews)
  return (
    <div className="card-section rf-controls">
      <p className="card-hint">Click or tap any cell on the map to start a fire there. With the map focused, the arrow keys move it.</p>
      <Segmented
        name="response-mode"
        label="Detection"
        options={RESPONSE_MODES.map((m) => ({ key: m, label: MODE_LABEL[m] }))}
        value={mode}
        onChange={onMode}
      />
      <Segmented
        name="response-crews"
        label="Crews"
        options={CREW_COUNTS.map((c) => ({ key: c, label: String(c) }))}
        value={crews}
        onChange={onCrews}
      />
      <div className="card-section">
        <span className="card-label" aria-hidden="true">
          Ignition cell (from the north-west corner)
        </span>
        <div className="rf-pick">
          <label>
            Row
            <select value={row + 1} onChange={(e) => onIgnition((Number(e.target.value) - 1) * cols + col)}>
              {Array.from({ length: rows }, (_, i) => (
                <option key={i} value={i + 1}>
                  {i + 1}
                </option>
              ))}
            </select>
          </label>
          <label>
            Column
            <select value={col + 1} onChange={(e) => onIgnition(row * cols + Number(e.target.value) - 1)}>
              {Array.from({ length: cols }, (_, i) => (
                <option key={i} value={i + 1}>
                  {i + 1}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      {scene && <p className="card-hint rf-text">{responseSentence(data, ignition, crews)}</p>}
      {aggregate && <p className="card-hint rf-aggregate">{aggregate}</p>}
      <details className="rf-assumptions">
        <summary>Assumptions</summary>
        <ul>
          {data.assumptions.map((a) => (
            <li key={a}>{a}</li>
          ))}
          <li>{data.note}</li>
        </ul>
      </details>
    </div>
  )
}

interface BarProps {
  scene: ResponseScene
  /** Hours since ignition, 0 to the horizon. */
  t: number
  playing: boolean
  onTime: (t: number) => void
  onPlaying: (playing: boolean) => void
  onClose: () => void
  /** The bar's height in px, so the 3D view can frame the terrain above it; 0 when it goes. */
  onHeight: (px: number) => void
  /** Where the Ranger section's caveats are, when that section is on the page. */
  caveatsHref: string | null
}

const signed = (x: number, digits: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(digits)}`

/** Play bar for the fire response: play, scrubber over the hours the fire spreads, comparison of the two ways to detect. */
export function ResponseBar({ scene, t, playing, onTime, onPlaying, onClose, onHeight, caveatsHref }: BarProps) {
  const { data, ignition } = scene
  const horizon = data.horizon
  const reduced = useMediaQuery('(prefers-reduced-motion: reduce)')
  const [speed, setSpeed] = useState<number>(1)
  const bar = useRef<HTMLElement>(null)
  const atEnd = t >= horizon - 1e-6

  const latest = useRef({ t, onTime, onPlaying })
  useLayoutEffect(() => {
    latest.current = { t, onTime, onPlaying }
  })

  // requestAnimationFrame runs only while playing.
  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    let time = latest.current.t
    let emitted = time
    let acc = 0
    const tick = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.1)
      last = now
      // A scrub since the last frame moves playback to where it was dragged.
      if (latest.current.t !== emitted) time = latest.current.t
      if (reduced) {
        acc += dt * speed
        if (acc >= STEP_SECONDS) {
          acc = 0
          time = Math.floor(time / STEP_HOURS + 1e-6) * STEP_HOURS + STEP_HOURS
        }
      } else {
        time += dt * speed * HOURS_PER_SECOND
      }
      time = Math.min(horizon, time)
      if (time !== emitted) {
        emitted = time
        latest.current.onTime(time)
      }
      if (time >= horizon) {
        latest.current.onPlaying(false)
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, reduced, horizon])

  // Tell the stage how tall the bar is, so the Results link, the legend and the 2D map clear it.
  useLayoutEffect(() => {
    const el = bar.current
    const stage = el?.closest<HTMLElement>('.stage')
    if (!el || !stage) return
    const observer = new ResizeObserver(() => {
      const h = Math.ceil(el.offsetHeight)
      stage.style.setProperty('--replay-h', `${h}px`)
      onHeight(h)
    })
    observer.observe(el)
    return () => {
      observer.disconnect()
      stage.style.removeProperty('--replay-h')
      onHeight(0)
    }
  }, [onHeight])

  // Cells newly reached in each hour, for the strip behind the scrubber.
  const perHour = useMemo(() => {
    const bins = new Array<number>(Math.ceil(horizon)).fill(0)
    for (const h of scene.arrival) if (Number.isFinite(h)) bins[Math.min(bins.length - 1, Math.floor(h))]++
    return bins
  }, [scene.arrival, horizon])
  const maxBin = Math.max(1, ...perHour)
  const bars = useMemo(
    () => (
      <svg viewBox={`0 0 ${horizon} 100`} preserveAspectRatio="none" aria-hidden="true">
        {perHour.map((n, i) => {
          const h = n === 0 ? 0 : Math.max(8, (n / maxBin) * 100)
          return <rect key={i} x={i} y={100 - h} width={0.9} height={h} />
        })}
      </svg>
    ),
    [perHour, horizon, maxBin],
  )

  const sens = summarise(data, ignition.id, 'sensors', scene.crews)
  const sat = summarise(data, ignition.id, 'satellite', scene.crews)
  const base = ignition.baseline
  const lead = sat.detectionH - sens.detectionH
  const better = (a: number, b: number, lowerIsBetter: boolean) => (a === b ? 'tie' : (a < b) === lowerIsBetter ? 'win' : 'lose')
  const cellRows = better(sens.cellsBurned, sat.cellsBurned, true)
  const valueRows = better(sens.valueSaved, sat.valueSaved, false)
  const detectRows = better(sens.detectionH, sat.detectionH, true)

  const togglePlay = () => {
    if (playing) {
      onPlaying(false)
      return
    }
    if (atEnd) onTime(0)
    onPlaying(true)
  }
  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    const step = e.key === 'PageUp' || e.key === 'PageDown' ? KEY_STEP_LARGE : KEY_STEP
    let dir = 0
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'PageUp') dir = 1
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown' || e.key === 'PageDown') dir = -1
    if (dir === 0) return
    e.preventDefault()
    onTime(Math.min(horizon, Math.max(0, t + dir * step)))
  }
  const nextSpeed = SPEEDS[(SPEEDS.indexOf(speed as (typeof SPEEDS)[number]) + 1) % SPEEDS.length]
  const progress = (t / horizon) * 100
  const ticks = [0, 6, 12, 18, 24].filter((h) => h <= horizon)
  const marker = (h: number) => `${(h / horizon) * 100}%`
  const burned = burnedCount(scene.arrival, t)

  return (
    <section ref={bar} className="fire-timeline response-bar glass" aria-label="Fire response">
      <div className="ft-body">
        <button
          type="button"
          className="ft-btn ft-play"
          aria-label={playing ? 'Pause' : atEnd ? 'Play again from the start' : 'Play'}
          onClick={togglePlay}
        >
          {playing ? (
            <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
              <rect x="3.5" y="2.5" width="4" height="13" rx="1" fill="currentColor" />
              <rect x="10.5" y="2.5" width="4" height="13" rx="1" fill="currentColor" />
            </svg>
          ) : atEnd ? (
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <path d="M3.5 9a5.5 5.5 0 1 0 1.8-4.1M3.5 3v3h3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
              <path d="M5 2.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4L6.2 2.1A.8.8 0 0 0 5 2.8Z" fill="currentColor" />
            </svg>
          )}
        </button>

        <div className="ft-track">
          <div className="ft-bars ft-bars-dim">{bars}</div>
          <div className="ft-bars ft-bars-lit" style={{ clipPath: `inset(0 ${100 - progress}% 0 0)` }}>
            {bars}
          </div>
          <span className="rf-mark rf-mark-detect" style={{ left: marker(scene.info.detection_h) }} aria-hidden="true" />
          <span className="rf-mark rf-mark-leave" style={{ left: marker(scene.info.start_h) }} aria-hidden="true" />
          <input
            type="range"
            className="ft-range"
            aria-label="Hours since ignition"
            aria-valuetext={`${formatHM(t)} after ignition`}
            min={0}
            max={horizon}
            step={0.05}
            value={t}
            onChange={(e) => onTime(Number(e.target.value))}
            onKeyDown={onKeyDown}
          />
          <div className="ft-years rf-ticks" aria-hidden="true">
            {ticks.map((h) => (
              <span key={h} style={{ left: marker(h) }}>
                {h} h
              </span>
            ))}
          </div>
        </div>

        <button
          type="button"
          className="ft-btn ft-speed"
          aria-label={`Playback speed ${speed} times. Switch to ${nextSpeed} times.`}
          onClick={() => setSpeed(nextSpeed)}
        >
          {speed}×
        </button>
        <button type="button" className="ft-btn ft-close" aria-label="Close fire response" onClick={onClose}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>

        <div className="ft-info rf-info">
          <strong className="ft-date">
            {formatHM(t)} <span className="rf-of">of {formatHM(horizon)} after ignition</span>
          </strong>
          <div className="ft-counts">
            <span>
              {burned} {burned === 1 ? 'cell' : 'cells'} reached by the fire
            </span>
          </div>
        </div>

        <div className="rf-compare-wrap">
          <table className="rf-compare">
            <caption className="sr-only">
              Sensors against satellite only detection, with {crewWord(scene.crews)}, ignition at row {ignition.cell[0] + 1}, column {ignition.cell[1] + 1}
            </caption>
            <thead>
              <tr>
                <td />
                <th scope="col" className={scene.mode === 'sensors' ? 'is-shown' : undefined}>
                  Sensors
                </th>
                <th scope="col" className={scene.mode === 'satellite' ? 'is-shown' : undefined}>
                  Satellite only
                </th>
                <th scope="col" className="rf-edge">
                  Sensors gain
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Detected</th>
                <td className={detectRows === 'win' ? 'rf-win' : undefined}>
                  {formatHM(sens.detectionH)} <small>{sens.detectedBy}</small>
                </td>
                <td>
                  {formatHM(sat.detectionH)} <small>{sat.detectedBy}</small>
                </td>
                <td className="rf-edge">{lead < 0.01 ? 'none' : `${formatHM(lead)} h earlier`}</td>
              </tr>
              <tr className="rf-row-leave">
                <th scope="row">Crews leave</th>
                <td>{formatHM(sens.startH)}</td>
                <td>{formatHM(sat.startH)}</td>
                <td className="rf-edge">{lead < 0.01 ? 'none' : `${formatHM(lead)} h earlier`}</td>
              </tr>
              <tr>
                <th scope="row">
                  Cells burned <small>{base.cells_burned} with no crews</small>
                </th>
                <td className={cellRows === 'win' ? 'rf-win' : undefined}>{sens.cellsBurned}</td>
                <td>{sat.cellsBurned}</td>
                <td className="rf-edge">
                  {sat.cellsBurned === sens.cellsBurned ? 'none' : `${Math.abs(sat.cellsBurned - sens.cellsBurned)} ${sens.cellsBurned < sat.cellsBurned ? 'fewer' : 'more'}`}
                </td>
              </tr>
              <tr>
                <th scope="row">
                  Value saved <small>of {base.value_burned.toFixed(2)} that burns</small>
                </th>
                <td className={valueRows === 'win' ? 'rf-win' : undefined}>
                  {sens.valueSaved.toFixed(2)} <small>{sens.pctSaved.toFixed(0)}%</small>
                </td>
                <td>
                  {sat.valueSaved.toFixed(2)} <small>{sat.pctSaved.toFixed(0)}%</small>
                </td>
                <td className="rf-edge">
                  {sens.valueSaved === sat.valueSaved ? 'none' : `${signed(sens.valueSaved - sat.valueSaved, 2)} (${signed(sens.pctSaved - sat.pctSaved, 0)} pts)`}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="ft-note rf-caveat">
          Simulated: one fixed wind, heuristic plans not proven optimal. Crews leave{' '}
          {data.dispatchH === null ? 'after' : `${formatHM(data.dispatchH)} h after`} detection; the satellite detects at {formatHM(data.satelliteH)}.{' '}
          {caveatsHref ? <a href={caveatsHref}>Model limits</a> : null}
        </p>
      </div>
    </section>
  )
}
