import { rampColor } from './color'
import type { LayerKey, Layers } from '../types'

export type LayerScale = 'linear' | 'share' | 'fires'

export interface LayerInfo {
  key: LayerKey
  label: string
  unit: string
  digits: number
  scale: LayerScale
}

/**
 * What the 3D stage can show: a data layer or the Landscape look. Landscape is not data, so
 * everything keyed by `LayerKey` (2D map, drape, legend ramp) never sees it.
 */
export type StageLayer = LayerKey | 'landscape'

export const LANDSCAPE: { key: 'landscape'; label: string } = { key: 'landscape', label: 'Landscape' }

/** Every layer the page knows about, in display order. */
export const LAYERS: LayerInfo[] = [
  { key: 'priority', label: 'Priority', unit: '', digits: 2, scale: 'linear' },
  { key: 'hazard', label: 'Hazard', unit: '', digits: 2, scale: 'linear' },
  { key: 'value', label: 'Ecological value', unit: '', digits: 2, scale: 'linear' },
  { key: 'elevation_m', label: 'Elevation', unit: ' m', digits: 0, scale: 'linear' },
  { key: 'inside_park', label: 'Inside park', unit: '', digits: 0, scale: 'share' },
  { key: 'fire_count', label: 'Recorded fires', unit: '', digits: 0, scale: 'fires' },
]

/** The layers present in this results file (synthetic files have only four). */
export function availableLayers(layers: Layers): LayerInfo[] {
  return LAYERS.filter((l) => Array.isArray(layers[l.key]))
}

export function formatLayerValue(info: LayerInfo, v: number): string {
  if (info.scale === 'share') return `${Math.round(v * 100)}%`
  return `${v.toLocaleString('en-US', {
    minimumFractionDigits: info.digits,
    maximumFractionDigits: info.digits,
  })}${info.unit}`
}

export function layerRange(layers: Layers, key: LayerKey): [number, number] {
  if (LAYERS.find((l) => l.key === key)?.scale === 'share') return [0, 1]
  let lo = Infinity
  let hi = -Infinity
  for (const row of layers[key] ?? []) {
    for (const v of row) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  return [lo, hi]
}

/** Cells with no recorded fire get a flat neutral fill, not the low end of the ramp. */
export const NO_FIRE_FILL = rampColor(0)
const FIRE_T_MIN = 0.3

/** Fire counts are skewed and mostly zero: log scale, starting well above the zero fill. */
function fireT(count: number, max: number): number {
  if (max <= 1) return 0.65
  return FIRE_T_MIN + (1 - FIRE_T_MIN) * (Math.log(count) / Math.log(max))
}

/** Ramp position (0 to 1) for the legend's i-th of n swatches. */
export function legendT(info: LayerInfo, i: number, n: number): number {
  const f = i / (n - 1)
  return info.scale === 'fires' ? FIRE_T_MIN + (1 - FIRE_T_MIN) * f : f
}

export function cellFill(info: LayerInfo, v: number, lo: number, hi: number): string {
  if (info.scale === 'fires') {
    return v <= 0 ? NO_FIRE_FILL : rampColor(fireT(v, hi))
  }
  return rampColor((v - lo) / (hi - lo || 1))
}
