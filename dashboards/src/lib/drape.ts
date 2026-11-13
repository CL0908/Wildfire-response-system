import { rampColor } from './color'
import { LAYERS, cellFill, layerRange } from './layers'
import type { LayerKey, Method, Results } from '../types'
import type { ParkData, ParkLayer, ParkPlacement } from './park'

/** Canvas pixels per grid cell. */
export const PX_PER_CELL = 64

// MapView draws in a 20-unit cell; the hatch below is that same pattern scaled to the canvas.
const SVG_CELL = 20
const HATCH_PERIOD = 6
const HATCH_WIDTH = 0.9 // thinner than the 2D map's 1.4: slopes stretch the lines

/**
 * Draws the selected layer as one flat-coloured square per grid cell, with the 2D map's
 * thin cell boundaries and its hatch over cells the method does not cover. Canvas top = north.
 */
export function drawDrape(results: Results, layer: LayerKey, method: Method): HTMLCanvasElement {
  const { rows, cols } = results.grid
  const values = results.layers[layer]
  const info = LAYERS.find((l) => l.key === layer)
  const canvas = document.createElement('canvas')
  canvas.width = cols * PX_PER_CELL
  canvas.height = rows * PX_PER_CELL
  const ctx = canvas.getContext('2d')
  if (!ctx || !values || !info) return canvas

  const [lo, hi] = layerRange(results.layers, layer)
  const covered = new Set(method.covered_cells)
  const S = PX_PER_CELL

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      ctx.fillStyle = cellFill(info, values[r][c], lo, hi)
      ctx.fillRect(c * S, r * S, S, S)
    }
  }

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
  ctx.lineWidth = 0.5 * (S / SVG_CELL)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) ctx.strokeRect(c * S, r * S, S, S)
  }

  ctx.save()
  ctx.beginPath()
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!covered.has(r * cols + c)) ctx.rect(c * S, r * S, S, S)
    }
  }
  ctx.clip()
  ctx.fillStyle = 'rgba(255, 255, 255, 0.5)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  // The SVG pattern is rotated 45 degrees with a stroke on x' = 0 of each 6-unit tile, which
  // gives the diagonal lines x + y = k * 6 * sqrt(2), anchored at the map's top-left corner.
  const scale = S / SVG_CELL
  const step = HATCH_PERIOD * Math.SQRT2 * scale
  ctx.strokeStyle = '#4b5563'
  ctx.lineWidth = HATCH_WIDTH * scale
  ctx.beginPath()
  for (let k = 0; k * step <= canvas.width + canvas.height; k++) {
    const d = k * step
    ctx.moveTo(d, 0)
    ctx.lineTo(d - canvas.height, canvas.height)
  }
  ctx.stroke()
  ctx.restore()

  return canvas
}

/** Pixels per cell in the whole-park drape: 187 columns at 8 px is 1,496 px wide. */
export const PARK_PX_PER_CELL = 8
const PARK_DIM = 'rgba(91, 100, 112, 0.6)' // over cells outside the park, in every data layer
const COVERAGE_BASE = rampColor(0.12)
const COVERAGE_HIT = rampColor(0.8)

function flatRange(values: number[]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  return [lo, hi]
}

/** Priority and Fire history ramps for the park, as the colour ramp and range the legend shows. */
export function parkLayerRange(park: ParkData, layer: 'priority' | 'fire_count'): [number, number] {
  return flatRange(park.layers[layer])
}

/**
 * The whole-park data layers as one flat-coloured square per 500 m cell, with the same colour
 * ramps as the window and cells outside the park dimmed. Coverage shows the cells in view of the
 * chosen placement over a pale base. Canvas top = north.
 */
export function drawParkDrape(
  park: ParkData,
  layer: Exclude<ParkLayer, 'landscape'>,
  placement: ParkPlacement,
): HTMLCanvasElement {
  const { rows, cols } = park.meta
  const S = PARK_PX_PER_CELL
  const canvas = document.createElement('canvas')
  canvas.width = cols * S
  canvas.height = rows * S
  const ctx = canvas.getContext('2d')
  if (!ctx) return canvas

  const inside = park.layers.inside_park
  const covered = new Set(placement.covered)
  const info = layer === 'coverage' ? null : LAYERS.find((l) => l.key === layer)
  const values = layer === 'coverage' ? null : park.layers[layer]
  const [lo, hi] = layer === 'coverage' || !values ? [0, 1] : flatRange(values)

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c
      ctx.fillStyle =
        info && values ? cellFill(info, values[i], lo, hi) : covered.has(i) ? COVERAGE_HIT : COVERAGE_BASE
      ctx.fillRect(c * S, r * S, S, S)
    }
  }
  ctx.fillStyle = PARK_DIM
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (inside[r * cols + c] <= 0.5) ctx.fillRect(c * S, r * S, S, S)
    }
  }
  return canvas
}
