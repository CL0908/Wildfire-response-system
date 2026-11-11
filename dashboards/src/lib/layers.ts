import type { LayerKey, Layers } from '../types'

export interface LayerInfo {
  key: LayerKey
  label: string
  unit: string
  digits: number
}

export const LAYERS: LayerInfo[] = [
  { key: 'priority', label: 'Priority', unit: '', digits: 2 },
  { key: 'hazard', label: 'Hazard', unit: '', digits: 2 },
  { key: 'value', label: 'Ecological value', unit: '', digits: 2 },
  { key: 'elevation_m', label: 'Elevation', unit: ' m', digits: 0 },
]

export function formatLayerValue(info: LayerInfo, v: number): string {
  return `${v.toLocaleString('en-US', {
    minimumFractionDigits: info.digits,
    maximumFractionDigits: info.digits,
  })}${info.unit}`
}

export function layerRange(layers: Layers, key: LayerKey): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const row of layers[key]) {
    for (const v of row) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  return [lo, hi]
}
