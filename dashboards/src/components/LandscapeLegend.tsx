import { type LandClass } from '../lib/landcover'

interface Props {
  /** Classes present in the window, or 'none' when land cover could not be loaded. */
  classes: LandClass[] | 'none'
}

/** Legend for the Landscape layer: land-cover swatches, then the same site keys as the data layers. */
export default function LandscapeLegend({ classes }: Props) {
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
      </ul>
      <p className="land-note">Land cover and elevation are real. Trees, clouds, sun and sky are illustrative.</p>
    </div>
  )
}
