import type { Bounds, Grid } from '../types'
import type { Scale } from './scale'

/** One satellite hot-spot detection. `cell` is the flat grid index, row * cols + col. */
export interface Detection {
  /** ISO date, "2020-03-02". */
  date: string
  year: number
  /** 1 to 12. */
  month: number
  lon: number
  lat: number
  /** Fire radiative power in MW. */
  frp: number
  row: number
  col: number
  cell: number
  night: boolean
  /** False for a detection in the park's 5 km margin (whole-park file only); true everywhere else. */
  inside: boolean
}

export interface Fires {
  source: string
  note: string
  bounds: Bounds
  grid: Pick<Grid, 'rows' | 'cols'>
  firstDate: string
  lastDate: string
  /** Sorted by date. */
  detections: Detection[]
}

export type FiresState =
  | { status: 'error'; message: string }
  | { status: 'ready'; fires: Fires }

const FIELDS = ['date', 'lon', 'lat', 'frp', 'row', 'col', 'night']
/** The whole-park file adds one trailing field. */
const FIELDS_WITH_INSIDE = [...FIELDS, 'inside']
/** Files under public/ for each scale. */
const FIRE_FILES: Record<Scale, string> = { window: 'fires.json', park: 'fires_park.json' }
const UNAVAILABLE = 'The fire history could not be loaded.'

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
const isStr = (x: unknown): x is string => typeof x === 'string'

function parseBounds(x: unknown): Bounds | null {
  if (!isObj(x) || !isNum(x.north) || !isNum(x.south) || !isNum(x.west) || !isNum(x.east)) return null
  return { north: x.north, south: x.south, west: x.west, east: x.east }
}

/** Null unless the file has the expected field order and every row is well formed. */
function parseFires(json: unknown): Fires | null {
  if (!isObj(json)) return null
  const { source, note, grid, fields, detections, first_date, last_date } = json
  const bounds = parseBounds(json.bounds)
  if (
    !isStr(source) ||
    !isStr(note) ||
    !bounds ||
    !isObj(grid) ||
    !Number.isInteger(grid.rows) ||
    !Number.isInteger(grid.cols) ||
    !isStr(first_date) ||
    !isStr(last_date) ||
    !Array.isArray(fields) ||
    !Array.isArray(detections)
  ) {
    return null
  }
  const withInside = fields.join() === FIELDS_WITH_INSIDE.join()
  if (!withInside && fields.join() !== FIELDS.join()) return null
  const width = withInside ? FIELDS_WITH_INSIDE.length : FIELDS.length
  const rows = grid.rows as number
  const cols = grid.cols as number
  const parsed: Detection[] = []
  for (const d of detections) {
    if (!Array.isArray(d) || d.length !== width) return null
    const [date, lon, lat, frp, row, col, night, inside] = d as unknown[]
    const m = isStr(date) ? /^(\d{4})-(\d{2})-\d{2}$/.exec(date) : null
    if (
      !m ||
      !isNum(lon) ||
      !isNum(lat) ||
      !isNum(frp) ||
      !Number.isInteger(row) ||
      !Number.isInteger(col) ||
      (row as number) < 0 ||
      (row as number) >= rows ||
      (col as number) < 0 ||
      (col as number) >= cols ||
      (withInside && inside !== 0 && inside !== 1)
    ) {
      return null
    }
    parsed.push({
      date: date as string,
      year: Number(m[1]),
      month: Number(m[2]),
      lon,
      lat,
      frp,
      row: row as number,
      col: col as number,
      cell: (row as number) * cols + (col as number),
      night: night === 1 || night === true,
      inside: !withInside || inside === 1,
    })
  }
  return { source, note, bounds, grid: { rows, cols }, firstDate: first_date, lastDate: last_date, detections: parsed }
}

async function fetchFires(scale: Scale): Promise<FiresState> {
  const file = FIRE_FILES[scale]
  let res: Response
  try {
    res = await fetch(`${import.meta.env.BASE_URL}${file}`)
  } catch {
    return { status: 'error', message: `${UNAVAILABLE} The file could not be fetched.` }
  }
  if (res.status === 404) return { status: 'error', message: `${UNAVAILABLE} ${file} was not found.` }
  if (!res.ok) return { status: 'error', message: `${UNAVAILABLE} The server answered HTTP ${res.status}.` }
  try {
    const fires = parseFires(await res.json())
    if (fires) return { status: 'ready', fires }
  } catch {
    // fall through to the shared message
  }
  return { status: 'error', message: `${UNAVAILABLE} ${file} is not in the expected format.` }
}

const cached: Partial<Record<Scale, Promise<FiresState>>> = {}

/** Fetches each scale's file once per page load, on its first call; errors are retried on the next call. */
export function loadFires(scale: Scale = 'window'): Promise<FiresState> {
  const hit = cached[scale]
  if (hit) return hit
  const p = fetchFires(scale).then((s) => {
    if (s.status === 'error') delete cached[scale]
    return s
  })
  cached[scale] = p
  return p
}

export interface YearMonthCounts {
  /** Every year from the first to the last detection, including years with none. */
  years: number[]
  /** counts[i][m] is the number of detections in years[i], month m + 1. */
  counts: number[][]
}

export function countsByYearMonth(detections: readonly Detection[]): YearMonthCounts {
  if (detections.length === 0) return { years: [], counts: [] }
  let first = Infinity
  let last = -Infinity
  for (const d of detections) {
    first = Math.min(first, d.year)
    last = Math.max(last, d.year)
  }
  const years = Array.from({ length: last - first + 1 }, (_, i) => first + i)
  const counts = years.map(() => new Array<number>(12).fill(0))
  for (const d of detections) counts[d.year - first][d.month - 1] += 1
  return { years, counts }
}

/** Detections per year, same order as `countsByYearMonth(...).years`. */
export function countsByYear(detections: readonly Detection[]): { years: number[]; totals: number[] } {
  const { years, counts } = countsByYearMonth(detections)
  return { years, totals: counts.map((row) => row.reduce((a, b) => a + b, 0)) }
}

/** The detections whose grid cell is in `coveredCells` (flat indices, row * cols + col). */
export function detectionsInCells(detections: readonly Detection[], coveredCells: ReadonlySet<number>): Detection[] {
  return detections.filter((d) => coveredCells.has(d.cell))
}
