import { useEffect, useState } from 'react'
import { formatShare } from './format'
import { crewLetter, formatHM, type Defence } from './ranger'

/**
 * The click-to-respond model: for every ignition cell, when the sensors (or the satellite) detect the
 * fire, which firebreaks the crews cut, and when the fire reaches each cell. Everything is precomputed
 * (ranger_response/response_explorer.py); the browser only decodes and draws.
 */

export type ResponseMode = 'sensors' | 'satellite'
export type CrewCount = 1 | 2

export const RESPONSE_MODES: ResponseMode[] = ['sensors', 'satellite']
export const CREW_COUNTS: CrewCount[] = [1, 2]

export interface ModeInfo {
  detection_h: number
  /** Site id of the sensor that detects first, or 'satellite'. */
  detecting_site: number | 'satellite'
  start_h: number
}

export interface PlanResult {
  value_saved: number
  cells_burned: number
  pct_saved: number
  defences: Defence[]
}

export interface Ignition {
  id: number
  cell: [number, number]
  station: [number, number]
  sensor_first_h: number | null
  baseline: { cells_burned: number; value_burned: number }
  modes: Record<ResponseMode, ModeInfo>
  plans: Record<string, PlanResult>
}

/** Figures over all ignitions for one crew count. */
export interface CrewAggregate {
  sensorsSaved: number
  satelliteSaved: number
  sensorsBetter: number
  sensorsWorse: number
}

export interface ResponseAggregate {
  n: number
  sensorsDetectShare: number
  detectedAtIgnitionShare: number
  baselineValueMean: number
  byCrews: Record<number, CrewAggregate>
}

export interface ResponseData {
  rows: number
  cols: number
  horizon: number
  stepH: number
  notReached: number
  planNames: string[]
  sensors: { id: number; cell: [number, number] }[]
  /** Cells any placed sensor sees (flat indices). */
  seenCells: number[]
  assumptions: string[]
  note: string
  dispatchH: number | null
  satelliteH: number
  ignitions: Ignition[]
  aggregate: ResponseAggregate | null
  /** uint8[ignition][plan][cell], see response.json arrival_bin. */
  arrival: Uint8Array
}

const isObj = (x: unknown): x is Record<string, unknown> => typeof x === 'object' && x !== null
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
const isStr = (x: unknown): x is string => typeof x === 'string'
const isCell = (x: unknown): x is [number, number] => Array.isArray(x) && x.length === 2 && x.every(isNum)

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

function isModeInfo(x: unknown): x is ModeInfo {
  return isObj(x) && isNum(x.detection_h) && isNum(x.start_h) && (isNum(x.detecting_site) || x.detecting_site === 'satellite')
}

function isPlan(x: unknown): x is PlanResult {
  return (
    isObj(x) &&
    isNum(x.value_saved) &&
    isNum(x.cells_burned) &&
    isNum(x.pct_saved) &&
    Array.isArray(x.defences) &&
    x.defences.every(isDefence)
  )
}

function isIgnition(x: unknown, planNames: string[]): x is Ignition {
  if (
    !isObj(x) ||
    !isNum(x.id) ||
    !isCell(x.cell) ||
    !isCell(x.station) ||
    !(x.sensor_first_h === null || isNum(x.sensor_first_h)) ||
    !isObj(x.baseline) ||
    !isNum(x.baseline.cells_burned) ||
    !isNum(x.baseline.value_burned) ||
    !isObj(x.modes) ||
    !isModeInfo(x.modes.sensors) ||
    !isModeInfo(x.modes.satellite) ||
    !isObj(x.plans)
  ) {
    return false
  }
  const plans = x.plans
  return planNames.filter((p) => p !== 'no_response').every((p) => isPlan(plans[p]))
}

function parseAggregate(x: unknown): ResponseAggregate | null {
  if (!isObj(x) || !isNum(x.n_ignitions) || !isObj(x.detection) || !isObj(x.baseline)) return null
  const d = x.detection
  const baseValue = isObj(x.baseline.value_burned) ? x.baseline.value_burned.mean : null
  if (!isNum(d.sensors_detect_share) || !isNum(d.detected_at_ignition_share) || !isNum(baseValue)) return null
  const byCrews: Record<number, CrewAggregate> = {}
  for (const k of CREW_COUNTS) {
    const c = x[`crews_${k}`]
    if (!isObj(c) || !isObj(c.sensors) || !isObj(c.satellite) || !isObj(c.sensors_minus_satellite)) continue
    const s = isObj(c.sensors.value_saved) ? c.sensors.value_saved.mean : null
    const t = isObj(c.satellite.value_saved) ? c.satellite.value_saved.mean : null
    const diff = c.sensors_minus_satellite
    if (!isNum(s) || !isNum(t) || !isNum(diff.ignitions_sensors_better) || !isNum(diff.ignitions_sensors_worse)) continue
    byCrews[k] = { sensorsSaved: s, satelliteSaved: t, sensorsBetter: diff.ignitions_sensors_better, sensorsWorse: diff.ignitions_sensors_worse }
  }
  return {
    n: x.n_ignitions,
    sensorsDetectShare: d.sensors_detect_share,
    detectedAtIgnitionShare: d.detected_at_ignition_share,
    baselineValueMean: baseValue,
    byCrews,
  }
}

/** Null unless both files agree with each other. */
function parse(json: unknown, bin: ArrayBuffer): ResponseData | null {
  if (!isObj(json) || !isObj(json.grid) || !isObj(json.arrival_bin) || !isObj(json.sensors)) return null
  const { rows, cols, horizon_h: horizon } = json.grid
  if (!isNum(rows) || !isNum(cols) || !isNum(horizon)) return null
  const n = rows * cols
  const ab = json.arrival_bin
  const planNames = Array.isArray(ab.plans) ? ab.plans.filter(isStr) : []
  if (!isNum(ab.step_h) || !isNum(ab.not_reached) || planNames.length === 0 || planNames[0] !== 'no_response') return null
  if (bin.byteLength !== n * planNames.length * n) return null
  if (!Array.isArray(json.ignitions) || json.ignitions.length !== n) return null
  const ignitions = json.ignitions
  if (!ignitions.every((g, i) => isIgnition(g, planNames) && g.id === i)) return null
  const { site_ids: ids, cells } = json.sensors
  if (!Array.isArray(ids) || !isObj(cells)) return null
  const sensors = ids.filter(isNum).map((id) => ({ id, cell: cells[String(id)] }))
  if (sensors.length === 0 || !sensors.every((s): s is { id: number; cell: [number, number] } => isCell(s.cell))) return null
  const p = isObj(json.parameters) ? json.parameters : {}
  const param = (key: string): number | null => {
    const v = p[key]
    return isObj(v) && isNum(v.value) ? v.value : null
  }
  const seen = isObj(json.sensors.seen_cells) ? Object.values(json.sensors.seen_cells) : []
  const seenCells = [...new Set(seen.flatMap((v) => (Array.isArray(v) ? v.filter(isNum) : [])))]
  return {
    rows,
    cols,
    horizon,
    stepH: ab.step_h,
    notReached: ab.not_reached,
    planNames,
    sensors,
    seenCells,
    assumptions: Array.isArray(json.assumptions) ? json.assumptions.filter(isStr) : [],
    note: isStr(json.note) ? json.note : '',
    dispatchH: param('dispatch_delay_h'),
    satelliteH: param('satellite_detection_h') ?? 6,
    ignitions: ignitions as Ignition[],
    aggregate: parseAggregate(json.aggregate),
    arrival: new Uint8Array(bin),
  }
}

export type ResponseState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: ResponseData }

const FAIL = 'The fire-response data could not be loaded, so this mode is unavailable. The rest of the page still works; reload to try again.'

async function load(): Promise<ResponseState> {
  const base = import.meta.env.BASE_URL
  try {
    const [j, b] = await Promise.all([fetch(`${base}response.json`), fetch(`${base}response_arrival.bin`)])
    if (!j.ok || !b.ok) return { status: 'error', message: FAIL }
    const data = parse(await j.json(), await b.arrayBuffer())
    return data ? { status: 'ready', data } : { status: 'error', message: 'The fire-response files are not in the expected format, so this mode is unavailable.' }
  } catch {
    return { status: 'error', message: FAIL }
  }
}

/** Both files (about 1.6 MB) are fetched the first time `enabled` is true, then kept. */
export function useResponse(enabled: boolean): ResponseState {
  const [result, setResult] = useState<ResponseState | null>(null)
  useEffect(() => {
    if (!enabled || result) return
    let live = true
    void load().then((s) => {
      if (live) setResult(s)
    })
    return () => {
      live = false
    }
  }, [enabled, result])
  return result ?? (enabled ? { status: 'loading' } : { status: 'idle' })
}

/** The most fire-prone cell: highest fire count, first in reading order on a tie. Falls back to the grid centre. */
export function defaultIgnition(fireCount: number[][] | undefined, rows: number, cols: number): number {
  let best = Math.floor(rows / 2) * cols + Math.floor(cols / 2)
  if (!fireCount) return best
  let top = -Infinity
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = fireCount[r]?.[c]
      if (typeof v === 'number' && v > top) {
        top = v
        best = r * cols + c
      }
    }
  }
  return best
}

export interface CrewTrack {
  crew: number
  /** Station, then each firebreak cell in order; cell-unit coordinates of the cell centres (x = column + 0.5). */
  points: { x: number; y: number }[]
  /** The same points with the hour the crew is there (a break's point carries its arrival hour). */
  stops: { x: number; y: number; t: number }[]
  /** Hours the crew stays at each break after arriving: stops[i + 1] is reached at arrive, left at finish. */
  leaves: number[]
  defences: Defence[]
}

export interface ResponseScene {
  data: ResponseData
  ignition: Ignition
  mode: ResponseMode
  crews: CrewCount
  info: ModeInfo
  plan: PlanResult
  /** Hours at which the fire reaches each cell with this plan; Infinity where it does not within the horizon. */
  arrival: Float32Array
  /** The same with no response. */
  baseline: Float32Array
  tracks: CrewTrack[]
  /** Cell of the sensor that detects first, or null when the satellite does. */
  detector: [number, number] | null
}

function decode(data: ResponseData, ignition: number, plan: number): Float32Array {
  const n = data.rows * data.cols
  const out = new Float32Array(n)
  const at = (ignition * data.planNames.length + plan) * n
  for (let i = 0; i < n; i++) {
    const v = data.arrival[at + i]
    out[i] = v === data.notReached ? Infinity : v * data.stepH
  }
  return out
}

export function buildScene(data: ResponseData, ignitionId: number, mode: ResponseMode, crews: CrewCount): ResponseScene {
  const ignition = data.ignitions[ignitionId]
  const key = `${mode}_${crews}`
  const plan = ignition.plans[key]
  const info = ignition.modes[mode]
  const centre = (c: [number, number]) => ({ x: c[1] + 0.5, y: c[0] + 0.5 })
  const crewIds = [...new Set(plan.defences.map((d) => d.crew))].sort((a, b) => a - b)
  const tracks: CrewTrack[] = crewIds.map((crew) => {
    const defences = plan.defences.filter((d) => d.crew === crew).sort((a, b) => a.order - b.order)
    const s = centre(ignition.station)
    const stops = [{ ...s, t: info.start_h }]
    const leaves: number[] = []
    for (const d of defences) {
      stops.push({ ...centre(d.cell), t: d.arrive_h })
      leaves.push(d.finish_h)
    }
    return { crew, points: stops.map(({ x, y }) => ({ x, y })), stops, leaves, defences }
  })
  const site = typeof info.detecting_site === 'number' ? data.sensors.find((s) => s.id === info.detecting_site) : undefined
  return {
    data,
    ignition,
    mode,
    crews,
    info,
    plan,
    arrival: decode(data, ignitionId, data.planNames.indexOf(key)),
    baseline: decode(data, ignitionId, data.planNames.indexOf('no_response')),
    tracks,
    detector: site ? site.cell : null,
  }
}

/** Where a crew is at hour t, in cell units: waiting at the station, walking straight to each break, cutting, on. */
export function crewPosition(track: CrewTrack, t: number): { x: number; y: number } {
  const { stops, leaves } = track
  if (t <= stops[0].t) return stops[0]
  for (let i = 1; i < stops.length; i++) {
    const a = stops[i - 1]
    const from = i === 1 ? a.t : leaves[i - 2]
    const b = stops[i]
    if (t < from) return a
    if (t < b.t) {
      const f = (t - from) / (b.t - from || 1)
      return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f }
    }
    if (t < leaves[i - 1]) return b
  }
  return stops[stops.length - 1]
}

export type BreakStatus = 'pending' | 'cutting' | 'held' | 'failed'

export function breakStatus(d: Defence, t: number): BreakStatus {
  if (t < d.arrive_h) return 'pending'
  if (t < d.finish_h) return 'cutting'
  return d.held ? 'held' : 'failed'
}

/** Share of the cut done at hour t, 0 to 1. */
export function breakProgress(d: Defence, t: number): number {
  return Math.min(1, Math.max(0, (t - d.arrive_h) / (d.finish_h - d.arrive_h || 1)))
}

export function burnedCount(arrival: Float32Array, t: number): number {
  let n = 0
  for (let i = 0; i < arrival.length; i++) if (arrival[i] <= t) n++
  return n
}

/** One line for the cell readout while the mode is on: when the fire gets there and what stands in the way. */
export function cellNote(scene: ResponseScene, cell: number): string {
  const { cols } = scene.data
  const r = Math.floor(cell / cols)
  const c = cell % cols
  const parts: string[] = []
  if (scene.ignition.cell[0] === r && scene.ignition.cell[1] === c) parts.push('Ignition')
  const station = scene.ignition.station
  if (station[0] === r && station[1] === c) parts.push('Crew station')
  const d = scene.plan.defences.find((x) => x.cell[0] === r && x.cell[1] === c)
  if (d) {
    parts.push(
      `Firebreak ${crewLetter(d.crew)}${d.order + 1}: crew arrives ${formatHM(d.arrive_h)}, done ${formatHM(d.finish_h)}, ${d.held ? 'holds' : 'fails (finished after the fire)'}`,
    )
  }
  const h = scene.arrival[cell]
  parts.push(Number.isFinite(h) ? `fire arrives ${formatHM(h)}` : `not reached in ${scene.data.horizon} h`)
  return parts.join(' · ')
}

export interface ModeSummary {
  mode: ResponseMode
  detectionH: number
  startH: number
  detectedBy: string
  cellsBurned: number
  valueSaved: number
  pctSaved: number
}

export function summarise(data: ResponseData, ignitionId: number, mode: ResponseMode, crews: CrewCount): ModeSummary {
  const g = data.ignitions[ignitionId]
  const info = g.modes[mode]
  const plan = g.plans[`${mode}_${crews}`]
  return {
    mode,
    detectionH: info.detection_h,
    startH: info.start_h,
    detectedBy: typeof info.detecting_site === 'number' ? `sensor ${info.detecting_site}` : 'satellite',
    cellsBurned: plan.cells_burned,
    valueSaved: plan.value_saved,
    pctSaved: plan.pct_saved,
  }
}

export const crewWord = (n: number) => (n === 1 ? 'one crew' : n === 2 ? 'two crews' : `${n} crews`)
const hm = formatHM

/** "In the model, sensors detected this fire at 0:12, 5:48 h before the satellite; ..." */
export function responseSentence(data: ResponseData, ignitionId: number, crews: CrewCount): string {
  const g = data.ignitions[ignitionId]
  const s = summarise(data, ignitionId, 'sensors', crews)
  const t = summarise(data, ignitionId, 'satellite', crews)
  const burning = g.baseline.value_burned.toFixed(2)
  const lead = t.detectionH - s.detectionH
  const dispatch = data.dispatchH === null ? '' : `, which leave ${hm(data.dispatchH)} h after detection`
  if (lead < 0.01) {
    return (
      `In the model, no sensor sees this fire before the satellite pass at ${hm(t.detectionH)}, so sensors give no head start here. ` +
      `${crewWord(crews)}${dispatch}, save ${s.valueSaved.toFixed(2)} of the ${burning} that would burn (${s.pctSaved.toFixed(0)}%), and ${s.cellsBurned} cells burn instead of ${g.baseline.cells_burned}.`
    )
  }
  return (
    `In the model, ${s.detectedBy} detected this fire at ${hm(s.detectionH)}, ${hm(lead)} h before the satellite (${hm(t.detectionH)}). ` +
    `With ${crewWord(crews)}${dispatch}, the crews saved ${s.valueSaved.toFixed(2)} of the ${burning} that would burn (${s.pctSaved.toFixed(0)}%) and ${s.cellsBurned} cells burn, ` +
    `against ${t.valueSaved.toFixed(2)} (${t.pctSaved.toFixed(0)}%) and ${t.cellsBurned} cells when the satellite detects it.`
  )
}

/** One line over all ignitions, from response.json's aggregate block. Null when the block is missing. */
export function aggregateLine(data: ResponseData, crews: CrewCount): string | null {
  const a = data.aggregate
  const c = a?.byCrews[crews]
  if (!a || !c) return null
  const worse = c.sensorsWorse === 0 ? 'and worse in none' : `and worse in ${c.sensorsWorse}`
  return (
    `Over all ${a.n} ignition cells, sensors detect first in ${formatShare(a.sensorsDetectShare)} (${formatShare(a.detectedAtIgnitionShare)} at the moment of ignition). ` +
    `With ${crewWord(crews)} the mean value saved is ${c.sensorsSaved.toFixed(2)} with sensors against ${c.satelliteSaved.toFixed(2)} with the satellite alone, ` +
    `of ${a.baselineValueMean.toFixed(1)} that burns on average with no response; sensors do better in ${c.sensorsBetter} cells ${worse}.`
  )
}

/** The detection callout on the map: "Sensor 6 detects at 0:12". */
export function calloutText(scene: ResponseScene): string {
  const { info } = scene
  return typeof info.detecting_site === 'number'
    ? `Sensor ${info.detecting_site} detects at ${formatHM(info.detection_h)}`
    : `Satellite detects at ${formatHM(info.detection_h)}`
}

