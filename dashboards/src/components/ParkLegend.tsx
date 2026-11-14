import { rampColor } from '../lib/color'
import { LAYERS, NO_FIRE_FILL, formatLayerValue, legendT } from '../lib/layers'
import type { ParkLayer } from '../lib/park'
import { PARK_LINE_COLORS } from '../lib/parkLines'

const LEGEND_STEPS = 8

interface LegendProps {
  layer: 'priority' | 'coverage' | 'fire_count'
  lo: number
  hi: number
}

/** Colour key for the whole-park data layers. */
export function ParkLegend({ layer, lo, hi }: LegendProps) {
  if (layer === 'coverage') {
    return (
      <div className="legends">
        <ul className="key" aria-label="Coverage">
          <li>
            <i className="swatch" style={{ background: rampColor(0.8) }} aria-hidden="true" />
            In view of a sensor
          </li>
          <li>
            <i className="swatch" style={{ background: rampColor(0.12) }} aria-hidden="true" />
            Not in view
          </li>
        </ul>
      </div>
    )
  }
  const info = LAYERS.find((l) => l.key === layer)
  if (!info) return null
  const fires = info.scale === 'fires'
  return (
    <div className="legends">
      <div className="legend">
        <span className="legend-title">{layer === 'priority' ? 'Priority (modelled)' : 'Fire history'}</span>
        {fires && (
          <>
            <span className="ramp" aria-hidden="true">
              <i style={{ background: NO_FIRE_FILL }} />
            </span>
            <span className="muted">0</span>
          </>
        )}
        <span className="muted">{formatLayerValue(info, fires ? Math.min(1, hi) : lo)}</span>
        <span className="ramp" aria-hidden="true">
          {Array.from({ length: LEGEND_STEPS }, (_, i) => (
            <i key={i} style={{ background: rampColor(legendT(info, i, LEGEND_STEPS)) }} />
          ))}
        </span>
        <span className="muted">
          {formatLayerValue(info, hi)}
          {fires ? ' detections' : ''}
        </span>
      </div>
    </div>
  )
}

/** Sensor, park boundary and 10 km window, in the colours the 3D lines use for this layer. */
export function ParkKey({ layer }: { layer: ParkLayer }) {
  const data = layer !== 'landscape'
  const c = PARK_LINE_COLORS[data ? 'data' : 'landscape']
  // The dark data-layer lines sit on a pale chip, as they sit on the pale ground.
  const back = data ? <rect width="16" height="16" rx="2" fill="#e5e7eb" /> : null
  return (
    <ul className="key" aria-label="Overlay key">
      <li>
        <svg width="16" height="16" aria-hidden="true">
          <circle cx="8" cy="8" r="6" fill="var(--accent)" stroke="#fff" strokeWidth="1.5" />
        </svg>
        Sensor
      </li>
      <li>
        <svg width="16" height="16" aria-hidden="true">
          {back}
          <path d="M2 11 L6 5 L10 9 L14 3" fill="none" stroke={c.boundary} strokeWidth="1.5" />
        </svg>
        Park boundary
      </li>
      <li>
        <svg width="16" height="16" aria-hidden="true">
          {back}
          <rect x="2.5" y="3.5" width="11" height="9" fill="none" stroke={c.window} strokeWidth="1.5" />
        </svg>
        10 km window
      </li>
    </ul>
  )
}

/** The fire replay's keys in the whole-park view: the window's two marks, and the dimmer ones outside the park. */
export function ParkFireKey() {
  return (
    <>
      <ul className="key" aria-label="Fire replay key">
        <li>
          <svg width="16" height="16" aria-hidden="true">
            <defs>
              <radialGradient id="park-fire-key-ember">
                <stop offset="0" stopColor="#fff5b8" />
                <stop offset="0.45" stopColor="#ffa01a" />
                <stop offset="0.8" stopColor="#ff4a0e" />
                <stop offset="1" stopColor="#ff4a0e" stopOpacity="0" />
              </radialGradient>
            </defs>
            <circle cx="8" cy="8" r="7.5" fill="url(#park-fire-key-ember)" />
          </svg>
          Active detection
        </li>
        <li>
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="4" fill="#211812" stroke="#94491f" strokeWidth="1.2" />
          </svg>
          Earlier detection
        </li>
        <li>
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="4" fill="#211812" stroke="#94491f" strokeWidth="1.2" opacity="0.4" />
          </svg>
          Outside the park, dimmer
        </li>
      </ul>
      <p className="land-note">Detections are real; their glow and fade are illustrative.</p>
    </>
  )
}
