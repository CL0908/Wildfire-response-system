import type { ParkData, ParkLayer, ParkPlacement } from '../lib/park'

export interface ParkHover {
  /** Flat index in the park grid (row * cols + col), or null off the grid. */
  cell: number | null
  elevation: number
  landClass: string
  treePct: number | null
}

interface Props {
  hover: ParkHover | null
  park: ParkData | null
  layer: ParkLayer
  placement: ParkPlacement | null
  covered: Set<number> | null
}

/** Details line for the whole-park view: the cell, its elevation, park membership and the layer's value. */
export default function ParkReadout({ hover, park, layer, placement, covered }: Props) {
  if (hover === null) {
    return (
      <div className="readout" aria-live="polite">
        <span className="muted">Hover or tap the ground for details.</span>
      </div>
    )
  }
  const cell = hover.cell
  const cols = park?.meta.cols ?? 0
  let place = ''
  let value = ''
  if (park && cell !== null) {
    place = park.layers.inside_park[cell] > 0.5 ? 'Inside park' : 'Outside park'
    if (layer === 'landscape') {
      value = `${hover.landClass}${hover.treePct !== null ? ` · ${hover.treePct}% tree cover` : ''}`
    } else if (layer === 'priority') {
      value = `Priority (modelled) ${park.layers.priority[cell].toFixed(2)}`
    } else if (layer === 'fire_count') {
      const n = park.layers.fire_count[cell]
      value = `Fire history ${n} ${n === 1 ? 'detection' : 'detections'}`
    } else if (placement && covered) {
      value = covered.has(cell) ? 'In view of a sensor' : 'Not in view'
    }
  }
  return (
    <div className="readout" aria-live="polite">
      <div>
        {cell === null || cols === 0 ? (
          <strong>Outside the grid</strong>
        ) : (
          <strong>
            Row {Math.floor(cell / cols)}, column {cell % cols}
          </strong>
        )}
        {' · '}
        {Math.round(hover.elevation).toLocaleString('en-US')} m
      </div>
      <div className="muted">{[place, value].filter(Boolean).join(' · ')}</div>
    </div>
  )
}
