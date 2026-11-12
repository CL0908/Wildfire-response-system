import { rampColor } from '../lib/color'
import { NO_FIRE_FILL, formatLayerValue, legendT, type LayerInfo } from '../lib/layers'

const LEGEND_STEPS = 8

interface Props {
  info: LayerInfo
  lo: number
  hi: number
}

/** Colour ramp and overlay key, shared by the 2D map and the 3D view. */
export default function Legend({ info, lo, hi }: Props) {
  return (
    <div className="legends">
      <div className="legend">
        <span className="legend-title">{info.label}</span>
        {info.scale === 'fires' && (
          <>
            <span className="ramp" aria-hidden="true">
              <i style={{ background: NO_FIRE_FILL }} />
            </span>
            <span className="muted">0</span>
          </>
        )}
        <span className="muted">
          {formatLayerValue(info, info.scale === 'fires' ? Math.min(1, hi) : lo)}
        </span>
        <span className="ramp" aria-hidden="true">
          {Array.from({ length: LEGEND_STEPS }, (_, i) => (
            <i key={i} style={{ background: rampColor(legendT(info, i, LEGEND_STEPS)) }} />
          ))}
        </span>
        <span className="muted">
          {formatLayerValue(info, hi)}
          {info.scale === 'fires' ? ' detections' : ''}
        </span>
      </div>
      <ul className="key" aria-label="Overlay key">
        <li>
          <svg width="16" height="16" aria-hidden="true">
            <defs>
              <pattern
                id="hatch-key"
                width="6"
                height="6"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(45)"
              >
                <rect width="6" height="6" fill="#fff" fillOpacity="0.5" />
                <line x1="0" y1="0" x2="0" y2="6" stroke="#4b5563" strokeWidth="1.4" />
              </pattern>
            </defs>
            <rect width="16" height="16" fill="#c9dbd6" />
            <rect width="16" height="16" fill="url(#hatch-key)" />
          </svg>
          Not covered
        </li>
        <li>
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="3" fill="#f9fafb" stroke="#1f2937" strokeWidth="1.2" />
          </svg>
          Candidate site
        </li>
        <li>
          <svg width="16" height="16" aria-hidden="true">
            <circle cx="8" cy="8" r="6" fill="var(--accent)" stroke="#fff" strokeWidth="1.5" />
          </svg>
          Chosen sensor
        </li>
      </ul>
    </div>
  )
}
