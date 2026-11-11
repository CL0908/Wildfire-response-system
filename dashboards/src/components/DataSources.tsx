import { formatBounds } from '../lib/format'
import type { Results } from '../types'

const ATTRIBUTION = [
  '© OpenStreetMap contributors',
  '© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium',
  'Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA',
  "We acknowledge the use of data from NASA's Fire Information for Resource Management System (FIRMS)",
]

/** Sources, modelling note and required attribution; only for real terrain. */
export default function DataSources({ results }: { results: Results }) {
  if (results.terrain !== 'real') return null
  const { sources = [], modelled, bounds } = results
  return (
    <section className="data" aria-labelledby="data-title">
      <h2 id="data-title">Data</h2>
      {sources.length > 0 && (
        <ul className="data-list">
          {sources.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      )}
      {bounds && <p>Window: {formatBounds(bounds)}</p>}
      {modelled && <p>{modelled}</p>}
      <ul className="attribution">
        {ATTRIBUTION.map((a) => (
          <li key={a}>{a}</li>
        ))}
      </ul>
    </section>
  )
}
