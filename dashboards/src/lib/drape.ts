import { LAYERS, cellFill, layerRange } from './layers'
import type { LayerKey, Method, Results } from '../types'

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
