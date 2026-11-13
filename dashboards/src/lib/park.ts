import type { Bounds } from '../types'

/** The whole-park data (public/park.json): 500 m cells, row 0 = north, column 0 = west. */
export type ParkRule = 'coverage' | 'fire_history' | 'timed_greedy'
export type ParkK = 10 | 20 | 40

export const PARK_K_VALUES: ParkK[] = [10, 20, 40]
export const PARK_RULES: { key: ParkRule; label: string }[] = [
  { key: 'coverage', label: 'Priority coverage' },
  { key: 'fire_history', label: 'Fire history' },
  { key: 'timed_greedy', label: 'Time to detection' },
]

/** What the 3D park can show on the ground: the Landscape look or one of three data layers. */
export type ParkLayer = 'landscape' | 'priority' | 'coverage' | 'fire_count'
export const PARK_LAYERS: { key: ParkLayer; label: string }[] = [
  { key: 'landscape', label: 'Landscape' },
  { key: 'priority', label: 'Priority (modelled)' },
  { key: 'coverage', label: 'Coverage' },
  { key: 'fire_count', label: 'Fire history' },
]

export interface ParkMetrics {
  n_sensors: number
  /** Modelled share of the priority surface in view of at least one sensor. */
  priority_share: number
  area_share: number
  /** Share of real VIIRS detections (2012 to 2025) that fall in view. */
  fire_share: number
  /** Simulated; not a real detection time. */
  mean_detection_h: number
  saved_h: number
  share_at_ignition: number
  share_after_spread: number
  share_satellite: number
}

export interface ParkPlacement {
  /** [row, col] of each chosen mast. */
  sites: [number, number][]
  /** Sorted flat cell indices seen by at least one chosen sensor. */
  covered: number[]
  metrics: ParkMetrics
}

export interface ParkData {
  meta: {
    rows: number
    cols: number
    cell_m: number
    bounds: Bounds
    window: { row0: number; col0: number; n_rows: number; n_cols: number }
    mast_height_m: number
    range_m: number
    k_values: number[]
    note: string
    /** ISO date of the run. */
    generated: string
    /** What is real, modelled or simulated: a key (snake_case) and a source sentence each. */
    labels: Record<'real' | 'modelled' | 'simulated', Record<string, string>>
  }
  layers: {
    inside_park: number[]
    priority: number[]
    fire_count: number[]
    elevation_m: number[]
  }
  placements: Record<string, Record<ParkRule, ParkPlacement>>
  /** Greedy priority-coverage placement, one point per sensor count. Shares are of the park (in-park cells). */
  coverage_curve: ParkCurvePoint[]
}

export interface ParkCurvePoint {
  k: number
  /** Modelled priority in view. */
  priority_share: number
  area_share: number
  /** Real VIIRS detections in view. */
  fire_share: number
}

export type ParkState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; park: ParkData }

const HINT = 'The whole-park placements could not be loaded.'

const isObject = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)

function looksLikePark(x: unknown): x is ParkData {
  if (!isObject(x) || !isObject(x.meta) || !isObject(x.layers) || !isObject(x.placements)) return false
  const m = x.meta
  const { rows, cols } = m
  if (!isNum(rows) || !isNum(cols) || rows < 1 || cols < 1 || !isNum(m.cell_m)) return false
  if (!isObject(m.bounds) || !['north', 'south', 'west', 'east'].every((k) => isNum((m.bounds as Record<string, unknown>)[k]))) {
    return false
  }
  const w = m.window
  if (!isObject(w) || !['row0', 'col0', 'n_rows', 'n_cols'].every((k) => isNum(w[k]))) return false
  if (
    (w.row0 as number) < 0 ||
    (w.col0 as number) < 0 ||
    (w.row0 as number) + (w.n_rows as number) > rows ||
    (w.col0 as number) + (w.n_cols as number) > cols
  ) {
    return false
  }
  if (typeof m.generated !== 'string' || !isObject(m.labels)) return false
  for (const group of ['real', 'modelled', 'simulated']) if (!isObject((m.labels as Record<string, unknown>)[group])) return false
  const curve = x.coverage_curve
  if (
    !Array.isArray(curve) ||
    curve.length < 2 ||
    !curve.every((c) => isObject(c) && ['k', 'priority_share', 'area_share', 'fire_share'].every((f) => isNum(c[f])))
  ) {
    return false
  }
  const cells = rows * cols
  for (const key of ['inside_park', 'priority', 'fire_count', 'elevation_m']) {
    const v = x.layers[key]
    if (!Array.isArray(v) || v.length !== cells) return false
  }
  for (const k of PARK_K_VALUES) {
    const byRule = x.placements[String(k)]
    if (!isObject(byRule)) return false
    for (const { key } of PARK_RULES) {
      const p = byRule[key]
      if (!isObject(p) || !Array.isArray(p.sites) || !Array.isArray(p.covered) || !isObject(p.metrics)) return false
      if (p.sites.length !== k) return false
      for (const s of p.sites) {
        if (!Array.isArray(s) || s.length !== 2 || !isNum(s[0]) || !isNum(s[1])) return false
        if (s[0] < 0 || s[0] >= rows || s[1] < 0 || s[1] >= cols) return false
      }
      if (p.covered.some((i) => !isNum(i) || i < 0 || i >= cells)) return false
      const mt = p.metrics
      if (!['priority_share', 'area_share', 'fire_share', 'mean_detection_h', 'share_at_ignition', 'share_after_spread', 'share_satellite'].every((f) => isNum(mt[f]))) return false
    }
  }
  return true
}

async function fetchPark(): Promise<ParkState> {
  let res: Response
  try {
    res = await fetch(`${import.meta.env.BASE_URL}park.json`)
  } catch {
    return { status: 'error', message: HINT }
  }
  if (!res.ok) return { status: 'error', message: HINT }
  try {
    const json: unknown = await res.json()
    if (looksLikePark(json)) return { status: 'ready', park: json }
  } catch {
    // fall through to the shared message
  }
  return { status: 'error', message: `park.json is missing or not in the expected format. ${HINT}` }
}

let cached: Promise<ParkState> | null = null

/** Fetches once per page load, on the first call; a failure is retried on the next call. */
export function loadPark(): Promise<ParkState> {
  if (!cached) {
    cached = fetchPark().then((s) => {
      if (s.status === 'error') cached = null
      return s
    })
  }
  return cached
}

export function parkPlacement(park: ParkData, k: ParkK, rule: ParkRule): ParkPlacement {
  return park.placements[String(k)][rule]
}

export const RULE_LABEL: Record<ParkRule, string> = {
  coverage: 'priority coverage',
  fire_history: 'fire history',
  timed_greedy: 'time to detection',
}
