import { useState } from 'react'
import { useResults } from './lib/data'
import { formatDate, formatRange } from './lib/format'
import type { LayerKey, Results } from './types'
import MapView from './components/MapView'
import LayerSwitch from './components/LayerSwitch'
import MethodPicker from './components/MethodPicker'
import MethodDetail from './components/MethodDetail'
import Benchmark from './components/Benchmark'

function Dashboard({ results }: { results: Results }) {
  const [layer, setLayer] = useState<LayerKey>('priority')
  const defaultMethod = results.methods.findIndex((m) => m.name === 'ILP')
  const [methodIndex, setMethodIndex] = useState(Math.max(0, defaultMethod))
  const method = results.methods[methodIndex]
  const { grid } = results

  return (
    <div className="page">
      <header className="header">
        <h1>Q-Pithak</h1>
        <p>
          Sensor placement, Khao Yai ({results.terrain} terrain) · {grid.rows} ×{' '}
          {grid.cols} grid of {grid.cell_size_m} m cells · {results.mast_height_m} m
          masts, {formatRange(results.range_m)} range · generated{' '}
          {formatDate(results.generated)}
        </p>
      </header>
      <main className="layout">
        <section className="map-col" aria-label="Map">
          <LayerSwitch value={layer} onChange={setLayer} />
          <MapView results={results} layer={layer} method={method} />
        </section>
        <div className="side-col">
          <section aria-label="Methods">
            <MethodPicker
              results={results}
              selected={methodIndex}
              onSelect={setMethodIndex}
            />
            <MethodDetail results={results} method={method} />
          </section>
          <Benchmark rows={results.benchmark} />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  const state = useResults()
  if (state.status === 'loading') {
    return (
      <div className="page">
        <p role="status">Loading results.json…</p>
      </div>
    )
  }
  if (state.status === 'error') {
    return (
      <div className="page">
        <h1>Q-Pithak</h1>
        <p role="alert" className="error">
          {state.message}
        </p>
      </div>
    )
  }
  return <Dashboard results={state.results} />
}
