import { useEffect, useState } from 'react'

export const RANGER_METHODS = ['greedy', 'lookahead', 'exact'] as const
export type RangerMethod = (typeof RANGER_METHODS)[number]

export interface Defence {
  crew: number
  order: number
  cell: [number, number]
  arrive_h: number
  finish_h: number
  fire_arrival_h: number | null
  held: boolean
}

export interface MethodRun {
  value_saved: number
  pct_saved: number
  cells_burned: number
  proven_optimal: boolean | null
  stop_reason: string | null
  defences: Defence[]
  arrival_h: (number | null)[]
}

export interface CrewRun {
  crews: number
  stations: [number, number][]
  baseline: { cells_burned: number; value_burned: number; total_value: number; arrival_h: (number | null)[] }
  methods: Partial<Record<RangerMethod, MethodRun>>
}

export interface FeaturedScenario {
  scenario: number
  roles: string[]
  ignition: [number, number]
  wind_speed_ms: number
  wind_from_deg: number
  runs: CrewRun[]
}

export interface MethodSummary {
  mean_pct_saved: number
  median_pct_saved: number
  proven_optimal: number | null
}

export interface GapSummary {
  n_positive: number
  n_zero: number
  n_negative: number
  mean_pct_points: number
  median_pct_points: number
}

export interface CrewSummary {
  n: number
  methods: Partial<Record<RangerMethod, MethodSummary>>
  gaps: Record<string, GapSummary>
  baseline: { mean_cells_burned: number; mean_value_burned: number; mean_total_value: number }
}

export interface Parameter {
  value: number | string | Record<string, number>
  unit: string
  source: string
}

export interface RangerData {
  generated: string
  grid: { rows: number; cols: number; cell_size_m: number }
  parameters: Record<string, Parameter>
  experiment: {
    n_scenarios: number
    summary: Record<string, CrewSummary>
    sensitivity: Record<string, Record<string, CrewSummary>>
  }
  featured: FeaturedScenario[]
  caveats: string[]
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
const isStr = (x: unknown): x is string => typeof x === 'string'
const isCell = (x: unknown): x is [number, number] => Array.isArray(x) && x.length === 2 && x.every(isNum)
const isHours = (x: unknown, n: number): x is (number | null)[] =>
  Array.isArray(x) && x.length === n && x.every((v) => v === null || isNum(v))

function isDefence(x: unknown): x is Defence {
  return (
    isObj(x) &&
    isNum(x.crew) &&
    isNum(x.order) &&
    isCell(x.cell) &&
    isNum(x.arrive_h) &&
    isNum(x.finish_h) &&
    (x.fire_arrival_h === null || isNum(x.fire_arrival_h)) &&
    typeof x.held === 'boolean'
  )
}

function isMethodRun(x: unknown, n: number): x is MethodRun {
  return (
    isObj(x) &&
    isNum(x.value_saved) &&
    isNum(x.pct_saved) &&
    isNum(x.cells_burned) &&
    (x.proven_optimal === null || typeof x.proven_optimal === 'boolean') &&
    (x.stop_reason === null || isStr(x.stop_reason)) &&
    Array.isArray(x.defences) &&
    x.defences.every(isDefence) &&
    isHours(x.arrival_h, n)
  )
}

function parseCrewRun(x: unknown, n: number): CrewRun | null {
  if (!isObj(x) || !isNum(x.crews) || !Array.isArray(x.stations) || !x.stations.every(isCell)) return null
  const b = x.baseline
  if (!isObj(b) || !isNum(b.cells_burned) || !isNum(b.value_burned) || !isNum(b.total_value) || !isHours(b.arrival_h, n)) {
    return null
  }
  if (!isObj(x.methods)) return null
  const methods: Partial<Record<RangerMethod, MethodRun>> = {}
  for (const m of RANGER_METHODS) {
    const run = x.methods[m]
    if (isMethodRun(run, n)) methods[m] = run
  }
  if (Object.keys(methods).length === 0) return null
  return { crews: x.crews, stations: x.stations, baseline: b as CrewRun['baseline'], methods }
}

function parseScenario(x: unknown, n: number): FeaturedScenario | null {
  if (
    !isObj(x) ||
    !isNum(x.scenario) ||
    !Array.isArray(x.roles) ||
    !isCell(x.ignition) ||
    !isNum(x.wind_speed_ms) ||
    !isNum(x.wind_from_deg) ||
    !Array.isArray(x.runs)
  ) {
    return null
  }
  const runs = x.runs.map((r) => parseCrewRun(r, n)).filter((r): r is CrewRun => r !== null)
  if (runs.length === 0) return null
  return {
    scenario: x.scenario,
    roles: x.roles.filter(isStr),
    ignition: x.ignition,
    wind_speed_ms: x.wind_speed_ms,
    wind_from_deg: x.wind_from_deg,
    runs,
  }
}

function isCrewSummary(x: unknown): x is CrewSummary {
  if (!isObj(x) || !isNum(x.n) || !isObj(x.methods) || !isObj(x.gaps) || !isObj(x.baseline)) return false
  const b = x.baseline
  if (!isNum(b.mean_cells_burned) || !isNum(b.mean_value_burned) || !isNum(b.mean_total_value)) return false
  const methodsOk = Object.values(x.methods).every(
    (m) => isObj(m) && isNum(m.mean_pct_saved) && isNum(m.median_pct_saved) && (m.proven_optimal === null || isNum(m.proven_optimal)),
  )
  const gapsOk = Object.values(x.gaps).every(
    (g) =>
      isObj(g) &&
      isNum(g.n_positive) &&
      isNum(g.n_zero) &&
      isNum(g.n_negative) &&
      isNum(g.mean_pct_points) &&
      isNum(g.median_pct_points),
  )
  return methodsOk && gapsOk
}

function isSummaryByCrews(x: unknown): x is Record<string, CrewSummary> {
  return isObj(x) && Object.values(x).every(isCrewSummary)
}

/** Returns null unless the file has the pieces the section reads. */
function parse(json: unknown): RangerData | null {
  if (!isObj(json) || json.terrain !== 'real') return null
  const { grid, value, parameters, experiment, featured, caveats } = json
  if (!isObj(grid) || !isNum(grid.rows) || !isNum(grid.cols) || !isNum(grid.cell_size_m)) return null
  const n = grid.rows * grid.cols
  if (!Array.isArray(value) || value.length !== n) return null
  if (!isObj(parameters) || !isObj(experiment) || !isNum(experiment.n_scenarios)) return null
  if (!isSummaryByCrews(experiment.summary) || !Object.keys(experiment.summary).length) return null
  const sensitivity = isObj(experiment.sensitivity)
    ? Object.fromEntries(Object.entries(experiment.sensitivity).filter(([, v]) => isSummaryByCrews(v)))
    : {}
  if (!Array.isArray(featured) || !Array.isArray(caveats)) return null
  const scenarios = featured.map((f) => parseScenario(f, n)).filter((f): f is FeaturedScenario => f !== null)
  if (scenarios.length === 0) return null
  return {
    generated: isStr(json.generated) ? json.generated : '',
    grid: { rows: grid.rows, cols: grid.cols, cell_size_m: grid.cell_size_m },
    parameters: parameters as Record<string, Parameter>,
    experiment: {
      n_scenarios: experiment.n_scenarios,
      summary: experiment.summary,
      sensitivity: sensitivity as RangerData['experiment']['sensitivity'],
    },
    featured: scenarios,
    caveats: caveats.filter(isStr),
  }
}

async function load(): Promise<RangerData | null> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}ranger.json`)
    if (!res.ok) return null
    return parse(await res.json())
  } catch {
    return null
  }
}

/** The ranger file is optional: null while loading and whenever it is missing or malformed. */
export function useRanger(): RangerData | null {
  const [data, setData] = useState<RangerData | null>(null)
  useEffect(() => {
    let live = true
    void load().then((d) => {
      if (live) setData(d)
    })
    return () => {
      live = false
    }
  }, [])
  return data
}

/** 5.208 -> "5:12". Hours after ignition as h:mm. */
export function formatHM(hours: number): string {
  const total = Math.round(hours * 60)
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}
