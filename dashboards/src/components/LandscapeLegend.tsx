import { type LandClass } from '../lib/landcover'

interface Props {
  /** Classes present in the window, or 'none' when land cover could not be loaded. */
  classes: LandClass[] | 'none'
  /** Add the fire replay's keys while the replay is shown. */
  fires?: boolean
  /** The whole-park view has no sensors or trees. */
  park?: boolean
}

/** Legend for the Landscape layer: land-cover swatches, then the same site keys as the data layers. */
export default function LandscapeLegend({ classes, fires = false, park = false }: Props) {
  return (
    <div className="legends">
      <ul className="land-key" aria-label="Land cover">
        {classes === 'none' ? (
          <li className="muted">Land cover unavailable; ground colour follows elevation.</li>
        ) : (
          classes.map((c) => (
            <li key={c.code}>
              <i style={{ background: c.color }} aria-hidden="true" />
              {c.name}
            </li>
          ))
        )}
      </ul>
      {!park && (
      <ul className="key" aria-label="Overlay key">
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
        {fires && (
          <>
            <li>
              <svg width="16" height="16" aria-hidden="true">
                <defs>
                  <radialGradient id="fire-key-ember">
                    <stop offset="0" stopColor="#fff5b8" />
                    <stop offset="0.45" stopColor="#ffa01a" />
                    <stop offset="0.8" stopColor="#ff4a0e" />
                    <stop offset="1" stopColor="#ff4a0e" stopOpacity="0" />
                  </radialGradient>
                </defs>
                <circle cx="8" cy="8" r="7.5" fill="url(#fire-key-ember)" />
              </svg>
              Active detection
            </li>
            <li>
              <svg width="16" height="16" aria-hidden="true">
                <circle cx="8" cy="8" r="4" fill="#211812" stroke="#94491f" strokeWidth="1.2" />
              </svg>
              Earlier detection
            </li>
          </>
        )}
      </ul>
      )}
      <p className="land-note">
        {park
          ? 'Land cover and elevation are real. Sun and sky are illustrative.'
          : 'Land cover and elevation are real. Trees, sun and sky are illustrative.'}
      </p>
      {fires && <p className="land-note">Detections are real; their glow and fade are illustrative.</p>}
    </div>
  )
}
