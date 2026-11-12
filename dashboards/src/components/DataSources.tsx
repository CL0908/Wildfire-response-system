import { formatBounds } from '../lib/format'
import { VERTICAL_EXAGGERATION } from '../lib/terrain'
import type { Results } from '../types'

const ATTRIBUTION = [
  '© OpenStreetMap contributors',
  '© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium',
  'Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA',
  "We acknowledge the use of data from NASA's Fire Information for Resource Management System (FIRMS)",
]

const YEARS = /\s*\b\d{4}(?: to \d{4})?\b/

interface Source {
  /** Who provides it. */
  name: string
  /** What it provides, without the years. */
  what: string
  years: string
}

/** "Fire detections 2012 to 2025: NASA FIRMS VIIRS S-NPP" -> provider, layer and years. */
function parseSource(s: string): Source {
  const i = s.indexOf(':')
  if (i < 0) return { name: s, what: '', years: '' }
  const label = s.slice(0, i).trim()
  const years = YEARS.exec(label)?.[0].trim() ?? ''
  return { name: s.slice(i + 1).trim(), what: label.replace(YEARS, '').trim(), years }
}

const lowerFirst = (s: string) => s.charAt(0).toLowerCase() + s.slice(1)

function joinList(items: string[]): string {
  if (items.length <= 1) return items.join('')
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

/** "Hazard and value are weighted combinations ..." -> ["hazard", "value"]; empty if the sentence has another shape. */
function modelledLayers(modelled: string | undefined): string[] {
  const m = modelled ? /^(.+?) (?:are|is) /.exec(modelled) : null
  if (!m) return []
  return m[1].split(/,\s*|\s+and\s+/).map((l) => l.trim().toLowerCase()).filter(Boolean)
}

/** The window and grid, what is real and what is modelled, and the attribution real data requires. */
export default function DataSources({ results }: { results: Results }) {
  const { grid, bounds, modelled, sources = [] } = results
  const gridText = `${grid.rows} × ${grid.cols} grid of ${grid.cell_size_m} m cells`

  if (results.terrain !== 'real') {
    return (
      <section className="data" aria-labelledby="data-title">
        <div className="sec-head">
          <h2 id="data-title">This map uses synthetic terrain; no layer is real data.</h2>
          <p className="lede">The map is a {gridText}.</p>
        </div>
      </section>
    )
  }

  const parsed = sources.map(parseSource)
  const realNames = parsed.map((p) => lowerFirst(p.what || p.name))
  const layers = modelledLayers(modelled)
  const headline =
    realNames.length === 0
      ? 'This map uses real terrain data.'
      : `${joinList(realNames.map((l, i) => (i === 0 ? l.charAt(0).toUpperCase() + l.slice(1) : l)))} ${
          realNames.length === 1 ? 'is' : 'are'
        } real data${layers.length > 0 ? `; ${joinList(layers)} ${layers.length === 1 ? 'is' : 'are'} modelled from ${realNames.length === 1 ? 'it' : 'them'}` : ''}.`

  return (
    <section className="data" aria-labelledby="data-title">
      <div className="sec-head">
        <h2 id="data-title">{headline}</h2>
        <p className="lede">
          {bounds ? `The window covers ${formatBounds(bounds)}, as a ${gridText}.` : `The map is a ${gridText}.`}
        </p>
      </div>
      <div className="data-cols">
        {parsed.length > 0 && (
          <div>
            <h3>Real</h3>
            <ul className="data-list">
              {parsed.map((p) => (
                <li key={p.name + p.what}>
                  <span className="src-name">{p.name}</span>
                  {(p.what || p.years) && (
                    <span className="src-what">{[p.what, p.years].filter(Boolean).join(' · ')}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}
        {(layers.length > 0 || modelled) && (
          <div>
            <h3>Modelled</h3>
            {layers.length > 0 && (
              <ul className="data-list">
                {layers.map((l) => (
                  <li key={l}>
                    <span className="src-name">{l.charAt(0).toUpperCase() + l.slice(1)}</span>
                  </li>
                ))}
              </ul>
            )}
            {modelled && <p className="muted">{modelled}</p>}
          </div>
        )}
      </div>
      <p className="method-note">
        The 3D surface uses the same Copernicus elevation at 30 m, with heights exaggerated {VERTICAL_EXAGGERATION}×.
      </p>
      <ul className="attribution">
        {ATTRIBUTION.map((a) => (
          <li key={a}>{a}</li>
        ))}
      </ul>
    </section>
  )
}
