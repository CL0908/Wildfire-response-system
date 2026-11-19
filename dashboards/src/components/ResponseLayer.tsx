import { useId } from 'react'
import { rampColor } from '../lib/color'
import { FAILED, INK, RAMP_LOW, UNREACHED, arrivalColor, crewLetter } from '../lib/ranger'
import { breakProgress, breakStatus, calloutText, crewPosition, type ResponseScene } from '../lib/response'

/** Cell size in the 2D map's SVG units (MapView draws in the same ones). */
const CELL = 20
/** The detecting sensor's ring and the ignition mark. */
const RING = '#facc15'

/** What MapView needs to draw the fire-response mode over its cells. */
export interface ResponseOverlay {
  scene: ResponseScene
  /** Hours since ignition. */
  t: number
  onPick: (cell: number) => void
  /** Arrow keys on the focused map: rows down and columns right. */
  onMove: (dRow: number, dCol: number) => void
}

/** Cells the fire has reached by hour t, coloured by the hour it arrives (the ranger section's scale). */
export function ResponseCells2D({ scene, t }: Pick<ResponseOverlay, 'scene' | 't'>) {
  const { cols, horizon } = scene.data
  const out = []
  for (let i = 0; i < scene.arrival.length; i++) {
    const h = scene.arrival[i]
    if (!(h <= t)) continue
    out.push(
      <rect
        key={i}
        className="rf-burned"
        x={(i % cols) * CELL}
        y={Math.floor(i / cols) * CELL}
        width={CELL}
        height={CELL}
        fill={arrivalColor(h, horizon)}
        stroke="#ffffff"
        strokeOpacity="0.5"
        strokeWidth="0.5"
      />,
    )
  }
  return <g pointerEvents="none">{out}</g>
}

const px = (v: number) => v * CELL

/** Ignition, station, crew paths and markers, firebreaks and the detection callout, over the sensors. */
export function ResponseMarks2D({ scene, t }: Pick<ResponseOverlay, 'scene' | 't'>) {
  const hatch = `rf-hatch-${useId().replace(/:/g, '')}`
  const { cols } = scene.data
  const { ignition, info } = scene
  const ig = { x: ignition.cell[1] + 0.5, y: ignition.cell[0] + 0.5 }
  const st = { x: ignition.station[1] + 0.5, y: ignition.station[0] + 0.5 }
  const detected = t >= info.detection_h - 1e-9
  const anchor = scene.detector ? { x: scene.detector[1] + 0.5, y: scene.detector[0] + 0.5 } : ig
  const text = calloutText(scene)
  const w = text.length * 4.5 + 12
  const x0 = Math.min(cols * CELL - w - 2, Math.max(2, px(anchor.x) - w / 2))
  const below = anchor.y < 2.2
  const y0 = below ? px(anchor.y) + 15 : px(anchor.y) - 31

  return (
    <g pointerEvents="none" className="rf-marks">
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="5" height="5" fill="#ffffff" fillOpacity="0.6" />
          <line x1="0" y1="0" x2="0" y2="5" stroke={FAILED} strokeWidth="1.6" />
        </pattern>
      </defs>
      {scene.tracks.map((track) => {
        if (t < info.start_h) return null
        const pos = crewPosition(track, t)
        const done = track.stops.filter((s, i) => i > 0 && s.t <= t)
        const travelled = [track.points[0], ...done, pos].map((p) => `${px(p.x)},${px(p.y)}`).join(' ')
        const planned = track.points.map((p) => `${px(p.x)},${px(p.y)}`).join(' ')
        return (
          <g key={`path-${track.crew}`} className="rf-path">
            <polyline points={planned} fill="none" stroke="#ffffff" strokeWidth="3.4" strokeOpacity="0.8" strokeLinejoin="round" />
            <polyline points={planned} fill="none" stroke={INK} strokeWidth="1.1" strokeDasharray="2.5 2.5" strokeLinejoin="round" />
            <polyline points={travelled} fill="none" stroke={INK} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" />
          </g>
        )
      })}
      {scene.plan.defences.map((d) => {
        const s = breakStatus(d, t)
        if (s === 'pending') return null
        const x = d.cell[1] * CELL
        const y = d.cell[0] * CELL
        const label = `${crewLetter(d.crew)}${d.order + 1}${s === 'failed' ? '×' : ''}`
        const chipFill = s === 'held' ? INK : '#ffffff'
        const chipInk = s === 'held' ? '#ffffff' : s === 'failed' ? FAILED : INK
        return (
          <g key={`${d.crew}-${d.order}`} className={`rf-break rf-break-${s}`}>
            {s === 'cutting' && (
              <>
                <rect x={x + 2} y={y + CELL - 2 - (CELL - 4) * breakProgress(d, t)} width={CELL - 4} height={(CELL - 4) * breakProgress(d, t)} fill={INK} fillOpacity="0.35" />
                <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill="none" stroke={INK} strokeWidth="1.6" strokeDasharray="1.5 2" />
              </>
            )}
            {s === 'held' && <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill="none" stroke={INK} strokeWidth="2.6" />}
            {s === 'failed' && (
              <>
                <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill={`url(#${hatch})`} />
                <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill="none" stroke={FAILED} strokeWidth="2.4" strokeDasharray="4 2.5" />
              </>
            )}
            <rect x={x + 3} y={y + CELL - 9} width={CELL - 6} height="7.5" rx="1.5" fill={chipFill} stroke={s === 'cutting' ? INK : 'none'} strokeWidth="0.6" />
            <text className="rf-chip" x={x + CELL / 2} y={y + CELL - 3.4} textAnchor="middle" fill={chipInk}>
              {label}
            </text>
          </g>
        )
      })}
      <polygon
        className="rf-station"
        points={`${px(st.x)},${px(st.y) - 6.5} ${px(st.x) + 6.5},${px(st.y) + 5.5} ${px(st.x) - 6.5},${px(st.y) + 5.5}`}
        fill="#ffffff"
        stroke={INK}
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <polygon
        className="rf-ignition"
        points={`${px(ig.x)},${px(ig.y) - 8} ${px(ig.x) + 8},${px(ig.y)} ${px(ig.x)},${px(ig.y) + 8} ${px(ig.x) - 8},${px(ig.y)}`}
        fill="var(--s-quantum)"
        stroke="#ffffff"
        strokeWidth="1.8"
      />
      {scene.tracks.map((track) => {
        const p = crewPosition(track, t)
        return (
          <g key={`crew-${track.crew}`} className="rf-crew">
            <circle cx={px(p.x)} cy={px(p.y)} r="5.6" fill="#ffffff" stroke={INK} strokeWidth="1.8" />
            <text className="rf-crew-letter" x={px(p.x)} y={px(p.y) + 2.8} textAnchor="middle" fill={INK}>
              {crewLetter(track.crew)}
            </text>
          </g>
        )
      })}
      {detected && (
        <g className="rf-detect">
          {scene.detector && (
            <>
              <circle cx={px(anchor.x)} cy={px(anchor.y)} r="10" fill="none" stroke={INK} strokeWidth="3.6" />
              <circle cx={px(anchor.x)} cy={px(anchor.y)} r="10" fill="none" stroke={RING} strokeWidth="1.8" strokeDasharray="3 2" />
            </>
          )}
          <rect x={x0} y={y0} width={w} height="15" rx="7.5" fill={INK} stroke="#ffffff" strokeWidth="1" />
          <text className="rf-callout" x={x0 + w / 2} y={y0 + 10.4} textAnchor="middle" fill="#ffffff">
            {text}
          </text>
        </g>
      )}
    </g>
  )
}

function KeyMark({ children }: { children?: React.ReactNode }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <rect x="0.5" y="0.5" width="15" height="15" fill={UNREACHED} stroke="var(--s-ref)" strokeWidth="0.6" />
      {children}
    </svg>
  )
}

/** Colour scale and mark key for the mode, in the 2D panel and the 3D chip. */
export function ResponseKey({ horizon }: { horizon: number }) {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((f) => rampColor(RAMP_LOW + (1 - RAMP_LOW) * (1 - f))).join(', ')
  const hatch = `rf-key-hatch-${useId().replace(/:/g, '')}`
  return (
    <div className="legends rf-key">
      <div className="legend">
        <span className="legend-title">Fire arrives</span>
        <span className="muted">0 h</span>
        <span className="ramp" aria-hidden="true">
          <i style={{ background: `linear-gradient(to right, ${stops})` }} />
        </span>
        <span className="muted">{horizon} h</span>
      </div>
      <ul className="key" aria-label="Fire response key">
        <li>
          <KeyMark />
          Unburned
        </li>
        <li>
          <KeyMark>
            <polygon points="8,2.5 13.5,8 8,13.5 2.5,8" fill="var(--s-quantum)" stroke="#ffffff" strokeWidth="1.2" />
          </KeyMark>
          Ignition
        </li>
        <li>
          <KeyMark>
            <polygon points="8,3 13,12.5 3,12.5" fill="#ffffff" stroke={INK} strokeWidth="1.4" strokeLinejoin="round" />
          </KeyMark>
          Crew station
        </li>
        <li>
          <KeyMark>
            <circle cx="8" cy="8" r="5" fill="#ffffff" stroke={INK} strokeWidth="1.6" />
          </KeyMark>
          Crew (A, B)
        </li>
        <li>
          <KeyMark>
            <rect x="2.5" y="2.5" width="11" height="11" fill="none" stroke={INK} strokeWidth="1.5" strokeDasharray="1.5 2" />
          </KeyMark>
          Firebreak being cut
        </li>
        <li>
          <KeyMark>
            <rect x="2" y="2" width="12" height="12" fill="none" stroke={INK} strokeWidth="2.6" />
          </KeyMark>
          Holds
        </li>
        <li>
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <defs>
              <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <rect width="5" height="5" fill="#ffffff" fillOpacity="0.6" />
                <line x1="0" y1="0" x2="0" y2="5" stroke={FAILED} strokeWidth="1.6" />
              </pattern>
            </defs>
            <rect x="0.5" y="0.5" width="15" height="15" fill={UNREACHED} stroke="var(--s-ref)" strokeWidth="0.6" />
            <rect x="2" y="2" width="12" height="12" fill={`url(#${hatch})`} stroke={FAILED} strokeWidth="2.2" strokeDasharray="3.5 2.5" />
          </svg>
          Fails (marked ×)
        </li>
        <li>
          <KeyMark>
            <circle cx="8" cy="8" r="5.5" fill="none" stroke={INK} strokeWidth="3.4" />
            <circle cx="8" cy="8" r="5.5" fill="none" stroke={RING} strokeWidth="1.6" strokeDasharray="2.8 1.8" />
          </KeyMark>
          Detecting sensor
        </li>
      </ul>
    </div>
  )
}
