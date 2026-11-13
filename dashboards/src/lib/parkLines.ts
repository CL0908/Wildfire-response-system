import { sampleHeight, type SceneRect, type TerrainMeta } from './terrain'
import type { ParkData } from './park'

/** Line colours for the park boundary and the 10 km window: light over Landscape, dark over the pale data layers. */
export const PARK_LINE_COLORS = {
  landscape: { boundary: '#ffffff', window: '#fbbf24' },
  data: { boundary: '#111827', window: '#92400e' },
} as const

const STEP_M = 100 // sample spacing along a line, a bit under the terrain's 150 m pixels

/** Pairs of scene points (x, y, z) following the ground, for a LineSegments geometry. */
class Segments {
  readonly out: number[] = []
  private heights: Float32Array
  private meta: TerrainMeta
  private exaggeration: number
  private lift: number

  constructor(heights: Float32Array, meta: TerrainMeta, exaggeration: number, lift: number) {
    this.heights = heights
    this.meta = meta
    this.exaggeration = exaggeration
    this.lift = lift
  }

  private point(x: number, z: number): [number, number, number] {
    return [x, sampleHeight(this.heights, this.meta, x, z) * this.exaggeration + this.lift, z]
  }

  edge(x0: number, z0: number, x1: number, z1: number) {
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, z1 - z0) / STEP_M))
    let prev = this.point(x0, z0)
    for (let i = 1; i <= n; i++) {
      const t = i / n
      const next = this.point(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)
      this.out.push(...prev, ...next)
      prev = next
    }
  }

  array(): Float32Array {
    return new Float32Array(this.out)
  }
}

/** The edges between cells inside the park (inside_park above 0.5) and cells outside it or off the grid. */
export function parkBoundaryPositions(
  park: ParkData,
  heights: Float32Array,
  meta: TerrainMeta,
  rect: SceneRect,
  exaggeration: number,
  lift: number,
): Float32Array {
  const { rows, cols } = park.meta
  const flag = park.layers.inside_park
  const inside = (r: number, c: number) => r >= 0 && r < rows && c >= 0 && c < cols && flag[r * cols + c] > 0.5
  const x = (c: number) => rect.x0 + (c / cols) * (rect.x1 - rect.x0)
  const z = (r: number) => rect.z0 + (r / rows) * (rect.z1 - rect.z0)
  const seg = new Segments(heights, meta, exaggeration, lift)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c <= cols; c++) {
      if (inside(r, c - 1) !== inside(r, c)) seg.edge(x(c), z(r), x(c), z(r + 1))
    }
  }
  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (inside(r - 1, c) !== inside(r, c)) seg.edge(x(c), z(r), x(c + 1), z(r))
    }
  }
  return seg.array()
}

/**
 * The 10 km analysis window as a flat ribbon of triangles that follows the ground. A WebGL line is
 * always 1 px wide, so the window's outline is a strip `widthM` wide, centred on the window's edge.
 * Positions are non-indexed triangles (x, y, z).
 */
export function parkWindowBand(
  park: ParkData,
  heights: Float32Array,
  meta: TerrainMeta,
  rect: SceneRect,
  exaggeration: number,
  lift: number,
  widthM: number,
): Float32Array {
  const { rows, cols, window: w } = park.meta
  const x0 = rect.x0 + (w.col0 / cols) * (rect.x1 - rect.x0)
  const x1 = rect.x0 + ((w.col0 + w.n_cols) / cols) * (rect.x1 - rect.x0)
  const z0 = rect.z0 + (w.row0 / rows) * (rect.z1 - rect.z0)
  const z1 = rect.z0 + ((w.row0 + w.n_rows) / rows) * (rect.z1 - rect.z0)
  const half = widthM / 2
  const out: number[] = []
  const point = (x: number, z: number) => [x, sampleHeight(heights, meta, x, z) * exaggeration + lift, z]
  // One side: from (ax, az) to (bx, bz), widened by half on both ends so the corners are filled.
  const side = (ax: number, az: number, bx: number, bz: number) => {
    const len = Math.hypot(bx - ax, bz - az)
    const dx = (bx - ax) / len
    const dz = (bz - az) / len
    const nx = -dz * half
    const nz = dx * half
    const n = Math.max(1, Math.ceil(len / STEP_M))
    const at = (i: number) => {
      const t = -half + ((len + widthM) * i) / n
      return [ax + dx * t, az + dz * t] as const
    }
    for (let i = 0; i < n; i++) {
      const [px, pz] = at(i)
      const [qx, qz] = at(i + 1)
      const a = point(px + nx, pz + nz)
      const b = point(px - nx, pz - nz)
      const c = point(qx + nx, qz + nz)
      const d = point(qx - nx, qz - nz)
      out.push(...a, ...b, ...c, ...b, ...d, ...c)
    }
  }
  side(x0, z0, x1, z0)
  side(x1, z0, x1, z1)
  side(x1, z1, x0, z1)
  side(x0, z1, x0, z0)
  return new Float32Array(out)
}

/** The park boundary as [col0, row0, col1, row1] unit edges in grid coordinates (for flat maps). */
export function parkBoundaryEdges(park: ParkData): [number, number, number, number][] {
  const { rows, cols } = park.meta
  const flag = park.layers.inside_park
  const inside = (r: number, c: number) => r >= 0 && r < rows && c >= 0 && c < cols && flag[r * cols + c] > 0.5
  const out: [number, number, number, number][] = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c <= cols; c++) if (inside(r, c - 1) !== inside(r, c)) out.push([c, r, c, r + 1])
  }
  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c < cols; c++) if (inside(r - 1, c) !== inside(r, c)) out.push([c, r, c + 1, r])
  }
  return out
}
