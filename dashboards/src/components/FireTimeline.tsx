import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { formatDateLong, formatInt } from '../lib/format'
import type { Replay } from '../lib/fireReplay'
import { useMediaQuery } from '../lib/useMediaQuery'
import type { ParkRule } from '../lib/park'

/** Seconds between monthly steps when the person prefers reduced motion. */
const STEP_SECONDS = 0.7
/** Arrow keys move the scrubber this many playback seconds; Page Up and Page Down move more. */
const KEY_STEP = 1
const KEY_STEP_LARGE = 5
/** Year labels closer than this many pixels to the previous one are left out. */
const YEAR_LABEL_GAP_PX = 36
const YEAR_LABEL_WIDTH_PX = 30
const SPEEDS = [1, 2] as const

const isoOf = (d: Date) => d.toISOString().slice(0, 10)

interface SwitchProps {
  on: boolean
  disabled: boolean
  /** Why the switch is unavailable; shown under the row. */
  reason?: string
  onChange: (on: boolean) => void
}

/** The "Fire replay" row of the control card, same look as the View row. */
export function ReplaySwitch({ on, disabled, reason, onChange }: SwitchProps) {
  const hintId = useId()
  return (
    <div className="card-section">
      <div className="card-row">
        <span className="card-label" aria-hidden="true">
          Fire replay
        </span>
        <div
          className="segmented view-switch"
          role="radiogroup"
          aria-label="Fire replay"
          aria-describedby={reason ? hintId : undefined}
          style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
        >
          <label>
            <input
              type="radio"
              name="fire-replay"
              className="sr-only"
              disabled={disabled}
              checked={!on}
              onChange={() => onChange(false)}
            />
            <span>Off</span>
          </label>
          <label>
            <input
              type="radio"
              name="fire-replay"
              className="sr-only"
              disabled={disabled}
              checked={on}
              onChange={() => onChange(true)}
            />
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

interface Props {
  replay: Replay
  /** Playback time in seconds, 0 to replay.duration. */
  t: number
  playing: boolean
  onTime: (t: number) => void
  onPlaying: (playing: boolean) => void
  onClose: () => void
  /** Flat cell indices covered by the selected method; null when the cells do not match the fire grid. */
  coveredCells: ReadonlySet<number> | null
  methodName: string
  /** Set in the whole-park view: the counts then show the detections inside the park and the sensors' view. */
  parkRule?: ParkRule
}

/** Play bar for the fire replay: play and pause, scrubber over a monthly histogram, date and counts. */
export default function FireTimeline({ replay, t, playing, onTime, onPlaying, onClose, coveredCells, methodName, parkRule }: Props) {
  const { duration } = replay
  const reduced = useMediaQuery('(prefers-reduced-motion: reduce)')
  const [speed, setSpeed] = useState<number>(1)
  const bar = useRef<HTMLElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)
  const [trackWidth, setTrackWidth] = useState(0)
  const atEnd = t >= duration - 1e-6

  // The latest time and the callbacks, readable from the animation loop without restarting it.
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
          time = replay.nextStep(time)
        }
      } else {
        time += dt * speed
      }
      time = Math.min(duration, time)
      if (time !== emitted) {
        emitted = time
        latest.current.onTime(time)
      }
      if (time >= duration) {
        latest.current.onPlaying(false)
        return
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, reduced, replay, duration])

  // Tell the stage how tall the bar is, so the Results link and the legend can clear it.
  useLayoutEffect(() => {
    const el = bar.current
    const stage = el?.closest<HTMLElement>('.stage')
    if (!el || !stage) return
    const measure = () => stage.style.setProperty('--replay-h', `${Math.ceil(el.offsetHeight)}px`)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => {
      observer.disconnect()
      stage.style.removeProperty('--replay-h')
    }
  }, [])

  useLayoutEffect(() => {
    const el = trackRef.current
    if (!el) return
    const measure = () => setTrackWidth(el.clientWidth)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const { count, covered, inside } = replay.countsAt(t, coveredCells)
  const date = replay.dateAt(t)
  const longDate = formatDateLong(isoOf(date))
  const monthYear = longDate.split(' ').slice(1).join(' ')

  const maxCount = useMemo(() => Math.max(1, ...replay.monthBins.map((b) => b.count)), [replay])
  const bars = useMemo(
    () => (
      <svg viewBox={`0 0 ${duration} 100`} preserveAspectRatio="none" aria-hidden="true">
        {replay.monthBins.map((b) => {
          const h = Math.max(8, (b.count / maxCount) * 100)
          return <rect key={`${b.year}-${b.month}`} x={b.t0} y={100 - h} width={Math.max(0.12, b.t1 - b.t0)} height={h} />
        })}
      </svg>
    ),
    [replay, duration, maxCount],
  )

  // Every year gets a tick; the label is left out where it would touch the previous one or the end.
  const yearLabels = useMemo(() => {
    const out: { year: number; pct: number; label: boolean }[] = []
    let lastPx = -Infinity
    for (const m of replay.yearMarks) {
      const px = (m.t / duration) * trackWidth
      const label = px - lastPx >= YEAR_LABEL_GAP_PX && px + YEAR_LABEL_WIDTH_PX <= trackWidth
      if (label) lastPx = px
      out.push({ year: m.year, pct: (m.t / duration) * 100, label })
    }
    return out
  }, [replay, duration, trackWidth])

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
    onTime(Math.min(duration, Math.max(0, t + dir * step)))
  }

  const nextSpeed = SPEEDS[(SPEEDS.indexOf(speed as (typeof SPEEDS)[number]) + 1) % SPEEDS.length]
  const progress = (t / duration) * 100

  return (
    <section ref={bar} className="fire-timeline glass" aria-label="Fire replay">
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
            <path
              d="M3.5 9a5.5 5.5 0 1 0 1.8-4.1M3.5 3v3h3"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <path d="M5 2.8v12.4a.8.8 0 0 0 1.2.7l10-6.2a.8.8 0 0 0 0-1.4L6.2 2.1A.8.8 0 0 0 5 2.8Z" fill="currentColor" />
          </svg>
        )}
      </button>

      <div className="ft-track" ref={trackRef}>
        <div className="ft-bars ft-bars-dim">{bars}</div>
        <div className="ft-bars ft-bars-lit" style={{ clipPath: `inset(0 ${100 - progress}% 0 0)` }}>
          {bars}
        </div>
        <input
          type="range"
          className="ft-range"
          aria-label="Replay date"
          aria-valuetext={longDate}
          min={0}
          max={duration}
          step={0.01}
          value={t}
          onChange={(e) => onTime(Number(e.target.value))}
          onKeyDown={onKeyDown}
        />
        <div className="ft-years" aria-hidden="true">
          {yearLabels
            .filter((y) => y.label)
            .map((y) => (
              <span key={y.year} style={{ left: `${y.pct}%` }}>
                {y.year}
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
      <button type="button" className="ft-btn ft-close" aria-label="Close fire replay" onClick={onClose}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </button>

      <div className="ft-info">
        <strong className="ft-date">{monthYear}</strong>
        <div className="ft-counts">
          <span>
            {formatInt(count)} of {formatInt(replay.total)} detections
          </span>
          {parkRule && <span>{formatInt(inside)} inside the park</span>}
          {covered !== null && (
            <span>
              {parkRule
                ? `${formatInt(covered)} in view of the sensors, margin included`
                : `${formatInt(covered)} inside cells covered by ${methodName}`}
              {parkRule && parkRule !== 'coverage' && (
                <span className="ft-caveat"> (placed from these detections, so not an independent test)</span>
              )}
            </span>
          )}
        </div>
      </div>
      <p className="ft-note">Real satellite hot-spot detections (NASA FIRMS), located to about 375 m. Not fire outlines.</p>
      </div>
    </section>
  )
}
