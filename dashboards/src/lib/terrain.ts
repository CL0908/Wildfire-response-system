import type { Bounds } from '../types'

export interface TerrainMeta {
  rows: number
  cols: number
  width_m: number
  height_m: number
  min_m: number
  max_m: number
  /** Lat/lon of the first and last vertex rows and columns (the mesh outline). */
  bounds?: Bounds
  source?: string
}

export interface Terrain {
  meta: TerrainMeta
  /** Elevation in metres, row-major, row 0 = north edge, column 0 = west edge. */
  heights: Float32Array
}

export type TerrainState =
  | { status: 'error'; message: string }
  | { status: 'ready'; terrain: Terrain }

/** Applied to Y only; the scene note under the 3D view must state the same number. */
export const VERTICAL_EXAGGERATION = 2

const HINT = 'The 2D map is still available.'
const DECIMETRES_PER_METRE = 10

function looksLikeMeta(x: unknown): x is TerrainMeta {
  if (typeof x !== 'object' || x === null) return false
  const m = x as Partial<TerrainMeta> & { unit?: unknown; dtype?: unknown }
  const finite = (v: unknown) => typeof v === 'number' && Number.isFinite(v)
  return (
    Number.isInteger(m.rows) &&
    Number.isInteger(m.cols) &&
    (m.rows as number) >= 2 &&
    (m.cols as number) >= 2 &&
    finite(m.width_m) &&
    finite(m.height_m) &&
    finite(m.min_m) &&
    finite(m.max_m) &&
    m.unit === 'decimetres' &&
    m.dtype === 'uint16-le'
  )
}

/** Decodes uint16 little-endian decimetres into metres. Returns null if the size is wrong. */
export function decodeHeights(buf: ArrayBuffer, meta: TerrainMeta): Float32Array | null {
  const n = meta.rows * meta.cols
  if (buf.byteLength !== n * 2) return null
  const view = new DataView(buf)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = view.getUint16(i * 2, true) / DECIMETRES_PER_METRE
  }
  return out
}

/**
 * Vertex positions, x = east, y = up (scaled by `exaggeration`), z = south, in metres,
 * centred on the origin. Vertex i = r * cols + c, so row 0 (north) is at -Z and column 0
 * (west) is at -X.
 */
export function buildPositions(
  heights: Float32Array,
  meta: TerrainMeta,
  exaggeration: number,
): Float32Array {
  const { rows, cols, width_m, height_m } = meta
  const out = new Float32Array(rows * cols * 3)
  for (let r = 0; r < rows; r++) {
    const z = (r / (rows - 1) - 0.5) * height_m
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c
      out[i * 3] = (c / (cols - 1) - 0.5) * width_m
      out[i * 3 + 1] = heights[i] * exaggeration
      out[i * 3 + 2] = z
    }
  }
  return out
}

/** Two triangles per cell, wound counter-clockwise seen from above (normals point +Y). */
export function buildIndices(rows: number, cols: number): Uint32Array {
  const out = new Uint32Array((rows - 1) * (cols - 1) * 6)
  let k = 0
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const a = r * cols + c
      const b = a + cols
      const d = a + 1
      const e = b + 1
      out[k++] = a
      out[k++] = b
      out[k++] = d
      out[k++] = d
      out[k++] = b
      out[k++] = e
    }
  }
  return out
}

/** Scene-space outline of the analysis grid: x0 = west, x1 = east, z0 = north, z1 = south. */
export interface SceneRect {
  x0: number
  x1: number
  z0: number
  z1: number
}

type MeshSize = Pick<TerrainMeta, 'width_m' | 'height_m'>
type GridSize = { rows: number; cols: number }

/** Lat/lon to scene x (east) and z (south). `mesh` is the lat/lon outline of the vertices. */
export function latLonToScene(
  lat: number,
  lon: number,
  mesh: Bounds,
  size: MeshSize,
): { x: number; z: number } {
  return {
    x: ((lon - mesh.west) / (mesh.east - mesh.west) - 0.5) * size.width_m,
    z: ((mesh.north - lat) / (mesh.north - mesh.south) - 0.5) * size.height_m,
  }
}

export function sceneToLatLon(
  x: number,
  z: number,
  mesh: Bounds,
  size: MeshSize,
): { lat: number; lon: number } {
  return {
    lat: mesh.north - (z / size.height_m + 0.5) * (mesh.north - mesh.south),
    lon: mesh.west + (x / size.width_m + 0.5) * (mesh.east - mesh.west),
  }
}

/**
 * Where the analysis grid (outer cell edges, `gridBounds`) sits in the scene. Falls back to
 * the whole mesh if either set of bounds is missing.
 */
export function gridSceneRect(meta: TerrainMeta, gridBounds: Bounds | undefined): SceneRect {
  if (!meta.bounds || !gridBounds) {
    return { x0: -meta.width_m / 2, x1: meta.width_m / 2, z0: -meta.height_m / 2, z1: meta.height_m / 2 }
  }
  const nw = latLonToScene(gridBounds.north, gridBounds.west, meta.bounds, meta)
  const se = latLonToScene(gridBounds.south, gridBounds.east, meta.bounds, meta)
  return { x0: nw.x, x1: se.x, z0: nw.z, z1: se.z }
}

/** Scene x, z of the centre of grid cell (row, col); row 0 = north, column 0 = west. */
export function cellCentreScene(
  row: number,
  col: number,
  grid: GridSize,
  rect: SceneRect,
): { x: number; z: number } {
  return {
    x: rect.x0 + ((col + 0.5) / grid.cols) * (rect.x1 - rect.x0),
    z: rect.z0 + ((row + 0.5) / grid.rows) * (rect.z1 - rect.z0),
  }
}

/** The grid cell under scene point (x, z), or null outside the grid. */
export function sceneToCell(
  x: number,
  z: number,
  grid: GridSize,
  rect: SceneRect,
): { row: number; col: number } | null {
  const u = (x - rect.x0) / (rect.x1 - rect.x0)
  const t = (z - rect.z0) / (rect.z1 - rect.z0)
  if (u < 0 || u >= 1 || t < 0 || t >= 1) return null
  return { row: Math.floor(t * grid.rows), col: Math.floor(u * grid.cols) }
}

/**
 * Per-vertex texture coordinates, clamped to 0..1, for a texture whose canvas covers the
 * grid rect (canvas top = north). `flipY` is three's default for canvas textures, so the
 * canvas top is v = 1: v = 1 - (distance from the north edge as a share of the grid).
 */
export function buildUVs(meta: TerrainMeta, rect: SceneRect): Float32Array {
  const { rows, cols, width_m, height_m } = meta
  const out = new Float32Array(rows * cols * 2)
  const clamp = (v: number) => Math.min(1, Math.max(0, v))
  for (let r = 0; r < rows; r++) {
    const z = (r / (rows - 1) - 0.5) * height_m
    const v = clamp(1 - (z - rect.z0) / (rect.z1 - rect.z0))
    for (let c = 0; c < cols; c++) {
      const x = (c / (cols - 1) - 0.5) * width_m
      const i = r * cols + c
      out[i * 2] = clamp((x - rect.x0) / (rect.x1 - rect.x0))
      out[i * 2 + 1] = v
    }
  }
  return out
}

/** Bilinear terrain height in metres (not exaggerated) at scene x, z; clamped to the mesh. */
export function sampleHeight(heights: Float32Array, meta: TerrainMeta, x: number, z: number): number {
  const { rows, cols, width_m, height_m } = meta
  const fc = Math.min(cols - 1, Math.max(0, (x / width_m + 0.5) * (cols - 1)))
  const fr = Math.min(rows - 1, Math.max(0, (z / height_m + 0.5) * (rows - 1)))
  const c0 = Math.min(cols - 2, Math.floor(fc))
  const r0 = Math.min(rows - 2, Math.floor(fr))
  const tc = fc - c0
  const tr = fr - r0
  const h = (r: number, c: number) => heights[r * cols + c]
  const top = h(r0, c0) * (1 - tc) + h(r0, c0 + 1) * tc
  const bottom = h(r0 + 1, c0) * (1 - tc) + h(r0 + 1, c0 + 1) * tc
  return top * (1 - tr) + bottom * tr
}

/**
 * First point where a ray meets the terrain surface (heights scaled by `exaggeration`),
 * found by stepping along the ray and bisecting the step that crosses the ground. Only the
 * stretch inside the mesh's bounding box is searched. Returns null if the ray misses or
 * starts below the ground. `step` is in metres, about half the 30 m vertex spacing.
 */
export function rayTerrainHit(
  origin: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
  heights: Float32Array,
  meta: TerrainMeta,
  exaggeration: number,
  step = 15,
): { x: number; z: number } | null {
  let t0 = 0
  let t1 = Infinity
  const slab = (o: number, d: number, lo: number, hi: number) => {
    if (Math.abs(d) < 1e-12) return o >= lo && o <= hi
    let a = (lo - o) / d
    let b = (hi - o) / d
    if (a > b) [a, b] = [b, a]
    t0 = Math.max(t0, a)
    t1 = Math.min(t1, b)
    return t0 <= t1
  }
  if (
    !slab(origin.x, dir.x, -meta.width_m / 2, meta.width_m / 2) ||
    !slab(origin.y, dir.y, meta.min_m * exaggeration, meta.max_m * exaggeration) ||
    !slab(origin.z, dir.z, -meta.height_m / 2, meta.height_m / 2)
  ) {
    return null
  }
  // Positive while the ray is above the ground.
  const gap = (t: number) => {
    const x = origin.x + dir.x * t
    const z = origin.z + dir.z * t
    return origin.y + dir.y * t - sampleHeight(heights, meta, x, z) * exaggeration
  }
  if (gap(t0) <= 0) return null
  let prev = t0
  for (let t = t0 + step; ; t += step) {
    const last = t >= t1
    const here = last ? t1 : t
    if (gap(here) <= 0) {
      let lo = prev
      let hi = here
      for (let i = 0; i < 14; i++) {
        const mid = (lo + hi) / 2
        if (gap(mid) > 0) lo = mid
        else hi = mid
      }
      return { x: origin.x + dir.x * hi, z: origin.z + dir.z * hi }
    }
    if (last) return null
    prev = here
  }
}

async function fetchTerrain(): Promise<TerrainState> {
  const base = import.meta.env.BASE_URL
  let metaRes: Response
  let binRes: Response
  try {
    ;[metaRes, binRes] = await Promise.all([
      fetch(`${base}terrain.json`),
      fetch(`${base}terrain.bin`),
    ])
  } catch {
    return { status: 'error', message: `Could not fetch the terrain files. ${HINT}` }
  }
  if (metaRes.status === 404 || binRes.status === 404) {
    return { status: 'error', message: `Terrain files not found. ${HINT}` }
  }
  if (!metaRes.ok || !binRes.ok) {
    return {
      status: 'error',
      message: `Could not load the terrain files (HTTP ${!metaRes.ok ? metaRes.status : binRes.status}). ${HINT}`,
    }
  }
  try {
    const meta: unknown = await metaRes.json()
    if (looksLikeMeta(meta)) {
      const heights = decodeHeights(await binRes.arrayBuffer(), meta)
      if (heights) return { status: 'ready', terrain: { meta, heights } }
    }
  } catch {
    // fall through to the shared message
  }
  return {
    status: 'error',
    message: `The terrain files are missing or not in the expected format. ${HINT}`,
  }
}

let cached: Promise<TerrainState> | null = null

/** Fetches once per page load, so toggling 2D / 3D does not re-download; errors are retried. */
export function loadTerrain(): Promise<TerrainState> {
  cached ??= fetchTerrain().then((s) => {
    if (s.status === 'error') cached = null
    return s
  })
  return cached
}
