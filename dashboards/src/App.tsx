import { useState } from 'react'
import { useResults } from './lib/data'
import { formatDate, formatRange } from './lib/format'
import type { LayerKey, Results } from './types'
import MapView from './components/MapView'
import LayerSwitch from './components/LayerSwitch'
import MethodPicker from './components/MethodPicker'
import MethodDetail from './components/MethodDetail'
import Benchmark from './components/Benchmark'
import DataSources from './components/DataSources'
import Hardware from './components/Hardware'
import { useHardware } from './lib/hardware'
import { availableLayers } from './lib/layers'

function Dashboard({ results }: { results: Results }) {
  const [layer, setLayer] = useState<LayerKey>('priority')
  const defaultMethod = results.methods.findIndex((m) => m.name === 'ILP')
  const [methodIndex, setMethodIndex] = useState(Math.max(0, defaultMethod))
  const method = results.methods[methodIndex]
  const { grid } = results
  const hardware = useHardware()
  const layerInfos = availableLayers(results.layers)
  const terrainFacts =
    results.terrain === 'real'
      ? 'real elevation, fire history and land cover; modelled hazard'
      : 'synthetic terrain'

  return (
    <div className="page">
      <header className="header">
        <h1>Q-Pithak</h1>
        <p>
          Sensor placement, Khao Yai National Park, north-west edge · {terrainFacts} ·{' '}
          {grid.rows} × {grid.cols} grid of {grid.cell_size_m} m cells · {results.mast_height_m} m
          masts, {formatRange(results.range_m)} range · generated{' '}
          {formatDate(results.generated)}
        </p>
      </header>
      <main className="layout">
        <section className="map-col" aria-label="Map">
          <LayerSwitch layers={layerInfos} value={layer} onChange={setLayer} />
          <MapView results={results} layer={layer} method={method} />
          <DataSources results={results} />
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
          {hardware && <Hardware data={hardware} />}
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
