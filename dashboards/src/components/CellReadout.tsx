import { availableLayers, formatLayerValue } from '../lib/layers'
import type { Method, Results } from '../types'

interface Props {
  results: Results
  method: Method
  /** Cells the method covers, as a set of flat cell indices (row * cols + col). */
  covered: Set<number>
  /** Flat index of the cell to describe, or null for the idle text. */
  cell: number | null
  /** Site whose viewshed summary is added (2D map only). */
  siteId?: number | null
  idle: string
}

/** Details line under the 2D map and the 3D view: the cell, its coverage and every layer value. */
export default function CellReadout({ results, method, covered, cell, siteId = null, idle }: Props) {
  const { layers, viewsheds } = results
  const { cols } = results.grid

  const siteSummary = (id: number) => {
    const cellsOfSite = viewsheds[String(id)] ?? []
    const pr = cellsOfSite.reduce(
      (sum, i) => sum + layers.priority[Math.floor(i / cols)][i % cols],
      0,
    )
    return `Site ${id}: viewshed ${cellsOfSite.length} cells, priority sum ${pr.toFixed(2)}`
  }

  const sr = cell === null ? 0 : Math.floor(cell / cols)
  const sc = cell === null ? 0 : cell % cols

  return (
    <div className="readout" aria-live="polite">
      {cell === null ? (
        <span className="muted">{idle}</span>
      ) : (
        <>
          <div>
            <strong>
              Row {sr}, column {sc}
            </strong>
            {' · '}
            {covered.has(cell) ? 'covered' : 'not covered'} by {method.name}
          </div>
          <div className="muted">
            {availableLayers(layers)
              .map((l) => `${l.label} ${formatLayerValue(l, layers[l.key]![sr][sc])}`)
              .join(' · ')}
          </div>
          {siteId !== null && <div>{siteSummary(siteId)}</div>}
        </>
      )}
    </div>
  )
}
