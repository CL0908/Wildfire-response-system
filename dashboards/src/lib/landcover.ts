/**
 * ESA WorldCover 2021 on the terrain's own pixel grid (public/landcover.bin + landcover.json).
 * Layout: rows * cols class codes, then rows * cols tree-cover fractions (0..255), uint8,
 * row-major, row 0 = north, column 0 = west. Pixel (r, c) is terrain vertex (r, c).
 */

import { SCALE_FILES, type Scale } from './scale'

export interface LandcoverData {
  rows: number
  cols: number
  /** WorldCover class code per pixel (10, 20, 30, 40, 50, 60, 80). */
  classes: Uint8Array
  /** Tree-cover fraction per pixel, 0..255. */
  tree: Uint8Array
  /** Share of pixels per class code, 0..1, counted from the pixels. */
  shares: Map<number, number>
}

export type LandcoverState =
  | { status: 'error' }
  | { status: 'ready'; data: LandcoverData }

export interface LandClass {
  code: number
  name: string
  /** Base colour in the Landscape layer (sRGB), also the legend swatch. */
  color: string
}

/** Classes in legend order. Tree cover's colour is its mid-elevation green. */
export const LAND_CLASSES: LandClass[] = [
  { code: 10, name: 'Tree cover', color: '#2c6638' },
  { code: 20, name: 'Shrubland', color: '#7c8a4a' },
  { code: 30, name: 'Grassland', color: '#a9b85a' },
  { code: 40, name: 'Cropland', color: '#c3b27c' },
  { code: 50, name: 'Built-up', color: '#9b928a' },
  { code: 60, name: 'Bare ground', color: '#cdb78c' },
  { code: 80, name: 'Water', color: '#1f5163' },
]

export const WATER_CODE = 80

export function landClassName(code: number): string {
  return LAND_CLASSES.find((c) => c.code === code)?.name ?? `Class ${code}`
}

/** Classes covering more than `minShare` of the window, largest first. */
export function presentClasses(data: LandcoverData, minShare = 0.001): LandClass[] {
  return LAND_CLASSES.filter((c) => (data.shares.get(c.code) ?? 0) > minShare).sort(
    (a, b) => (data.shares.get(b.code) ?? 0) - (data.shares.get(a.code) ?? 0),
  )
}

interface GridMeta {
  rows: number
  cols: number
  width_m: number
  height_m: number
}

/** Nearest terrain pixel to scene x (east), z (south), clamped to the mesh. */
export function pixelAt(meta: GridMeta, x: number, z: number): { row: number; col: number } {
  const col = Math.round((x / meta.width_m + 0.5) * (meta.cols - 1))
  const row = Math.round((z / meta.height_m + 0.5) * (meta.rows - 1))
  return {
    row: Math.min(meta.rows - 1, Math.max(0, row)),
    col: Math.min(meta.cols - 1, Math.max(0, col)),
  }
}

/** Hook for the tree stage: WorldCover class code at scene x, z. */
export function landClassAt(data: LandcoverData, meta: GridMeta, x: number, z: number): number {
  const { row, col } = pixelAt(meta, x, z)
  return data.classes[row * data.cols + col]
}

/** Hook for the tree stage: tree-cover fraction (0..1) at scene x, z. */
export function treeFractionAt(data: LandcoverData, meta: GridMeta, x: number, z: number): number {
  const { row, col } = pixelAt(meta, x, z)
  return data.tree[row * data.cols + col] / 255
}

/** True when the land-cover grid is the same size as the terrain mesh. */
export function matchesTerrain(data: LandcoverData, meta: { rows: number; cols: number }): boolean {
  return data.rows === meta.rows && data.cols === meta.cols
}

async function fetchLandcover(scale: Scale): Promise<LandcoverState> {
  const base = import.meta.env.BASE_URL
  const name = SCALE_FILES[scale].landcover
  try {
    const [metaRes, binRes] = await Promise.all([
      fetch(`${base}${name}.json`),
      fetch(`${base}${name}.bin`),
    ])
    if (!metaRes.ok || !binRes.ok) return { status: 'error' }
    const meta = (await metaRes.json()) as { rows?: unknown; cols?: unknown }
    const { rows, cols } = meta
    if (!Number.isInteger(rows) || !Number.isInteger(cols)) return { status: 'error' }
    const n = (rows as number) * (cols as number)
    const buf = await binRes.arrayBuffer()
    if (n <= 0 || buf.byteLength !== n * 2) return { status: 'error' }
    const all = new Uint8Array(buf)
    const classes = all.slice(0, n)
    const tree = all.slice(n, n * 2)
    const counts = new Map<number, number>()
    for (let i = 0; i < n; i++) counts.set(classes[i], (counts.get(classes[i]) ?? 0) + 1)
    const shares = new Map<number, number>()
    for (const [code, count] of counts) shares.set(code, count / n)
    return {
      status: 'ready',
      data: { rows: rows as number, cols: cols as number, classes, tree, shares },
    }
  } catch {
    return { status: 'error' }
  }
}

const cached = new Map<Scale, Promise<LandcoverState>>()

/** Fetches once per scale per page load; never rejects. Errors are retried on the next call. */
export function loadLandcover(scale: Scale = 'window'): Promise<LandcoverState> {
  let p = cached.get(scale)
  if (!p) {
    p = fetchLandcover(scale).then((s) => {
      if (s.status === 'error') cached.delete(scale)
      return s
    })
    cached.set(scale, p)
  }
  return p
}
