import type { Detection } from './fires'

/**
 * Playback model for the fire replay. Pure functions, no framework.
 *
 * Playback time `t` is in seconds. Calendar time runs at one steady rate through every stretch
 * that has detections; a stretch of more than GAP_DAYS between detections is crossed in GAP_SECONDS.
 */

/** A detection counts as burning for this many calendar days after its date. */
export const GLOW_DAYS = 10
/** Detections further apart than this many days start a new active stretch. */
const GAP_DAYS = 30
/** Playback seconds spent crossing one idle gap. */
const GAP_SECONDS = 0.6
/** Calendar days shown before the first detection, so the replay opens on an empty map. */
const LEAD_DAYS = 3
/** The whole replay at 1x speed, in seconds. */
const TARGET_SECONDS = 52

const DAY_MS = 86_400_000

export const PHASE_PENDING = 0
export const PHASE_BURNING = 1
export const PHASE_SCORCH = 2

export type ReplayDetection = Pick<Detection, 'date' | 'cell'>

export interface ReplayState {
  /** Calendar day (days since 1970-01-01, fractional) the state belongs to. */
  day: number
  /** Per detection, in the order given to createReplay: PHASE_PENDING, PHASE_BURNING or PHASE_SCORCH. */
  phase: Uint8Array
  /** 0 at the detection date to 1 at the end of the glow window while burning; 1 once it is scorch, 0 while pending. */
  age: Float32Array
  /** Detections that have happened (burning or scorch). */
  count: number
  /** Detections currently burning. */
  burning: number
  /** Detections that have happened inside the covered cells, or null when no cells were given. */
  countCovered: number | null
}

export interface MonthBin {
  year: number
  /** 1 to 12. */
  month: number
  /** Playback-time extent of the calendar month. */
  t0: number
  t1: number
  count: number
}

export interface Replay {
  /** Playback seconds at 1x speed. */
  duration: number
  total: number
  /** Calendar date (UTC midnight, fractional days included) at playback time t. */
  dateAt(t: number): Date
  /** Playback time of a calendar date ("2020-03-14" or a Date). Dates inside a gap map inside the gap. */
  tAt(date: Date | string): number
  stateAt(t: number, coveredCells?: ReadonlySet<number> | null): ReplayState
  /** The two running counts only, without building the per-detection arrays. */
  countsAt(t: number, coveredCells?: ReadonlySet<number> | null): { count: number; covered: number | null }
  /** Months with at least one detection, positioned in playback time, in order. */
  monthBins: readonly MonthBin[]
  /** First day of each year that has detections, in playback time (the first year starts at 0). */
  yearMarks: readonly { year: number; t: number }[]
  /** Playback times of month starts inside active stretches, for stepping a month at a time. */
  stepTimes: readonly number[]
  /** The next monthly step after t, or the end. */
  nextStep(t: number): number
}

interface Segment {
  active: boolean
  day0: number
  day1: number
  t0: number
  t1: number
}

function dayOf(iso: string): number {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!m) return NaN
  return Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])) / DAY_MS
}

const monthStartDay = (year: number, month: number) => Date.UTC(year, month - 1, 1) / DAY_MS

/** Null when there are no usable detections. */
export function createReplay(detections: readonly ReplayDetection[]): Replay | null {
  const days = detections.map((d) => dayOf(d.date))
  if (days.length === 0 || days.some((d) => !Number.isFinite(d))) return null
  const sorted = [...days].sort((a, b) => a - b)

  // Stretches with detections, each extended by the glow window of its last detection.
  const stretches: { day0: number; day1: number }[] = []
  let start = sorted[0]
  let prev = sorted[0]
  for (const d of sorted) {
    if (d - prev > GAP_DAYS) {
      stretches.push({ day0: start, day1: prev + GLOW_DAYS })
      start = d
    }
    prev = d
  }
  stretches.push({ day0: start, day1: prev + GLOW_DAYS })
  stretches[0].day0 -= LEAD_DAYS

  const gaps = stretches.length - 1
  const activeDays = stretches.reduce((sum, s) => sum + (s.day1 - s.day0), 0)
  const secondsPerDay = (TARGET_SECONDS - gaps * GAP_SECONDS) / activeDays

  const segments: Segment[] = []
  let clock = 0
  stretches.forEach((s, i) => {
    if (i > 0) {
      const before = stretches[i - 1]
      segments.push({ active: false, day0: before.day1, day1: s.day0, t0: clock, t1: clock + GAP_SECONDS })
      clock += GAP_SECONDS
    }
    const length = (s.day1 - s.day0) * secondsPerDay
    segments.push({ active: true, day0: s.day0, day1: s.day1, t0: clock, t1: clock + length })
    clock += length
  })
  const duration = clock
  const firstDay = segments[0].day0
  const lastDay = segments[segments.length - 1].day1

  const clamp = (t: number) => Math.min(duration, Math.max(0, t))

  function dayAt(t: number): number {
    const c = clamp(t)
    const seg = segments.find((s) => c <= s.t1) ?? segments[segments.length - 1]
    const f = seg.t1 > seg.t0 ? (c - seg.t0) / (seg.t1 - seg.t0) : 0
    return seg.day0 + f * (seg.day1 - seg.day0)
  }

  function tAtDay(day: number): number {
    if (day <= firstDay) return 0
    if (day >= lastDay) return duration
    const seg = segments.find((s) => day <= s.day1) ?? segments[segments.length - 1]
    const f = seg.day1 > seg.day0 ? (day - seg.day0) / (seg.day1 - seg.day0) : 0
    return seg.t0 + f * (seg.t1 - seg.t0)
  }

  const total = days.length

  function countsAt(t: number, covered?: ReadonlySet<number> | null) {
    const day = dayAt(t)
    let count = 0
    let inside = 0
    for (let i = 0; i < total; i++) {
      if (days[i] > day) continue
      count++
      if (covered?.has(detections[i].cell)) inside++
    }
    return { count, covered: covered ? inside : null }
  }

  function stateAt(t: number, covered?: ReadonlySet<number> | null): ReplayState {
    const day = dayAt(t)
    const phase = new Uint8Array(total)
    const age = new Float32Array(total)
    let count = 0
    let burning = 0
    let inside = 0
    for (let i = 0; i < total; i++) {
      const since = day - days[i]
      if (since < 0) continue
      count++
      if (covered?.has(detections[i].cell)) inside++
      if (since < GLOW_DAYS) {
        phase[i] = PHASE_BURNING
        age[i] = since / GLOW_DAYS
        burning++
      } else {
        phase[i] = PHASE_SCORCH
        age[i] = 1
      }
    }
    return { day, phase, age, count, burning, countCovered: covered ? inside : null }
  }

  // Monthly histogram, from the first month with data to the last.
  const counts = new Map<string, number>()
  for (const d of sorted) {
    const date = new Date(d * DAY_MS)
    const key = `${date.getUTCFullYear()}-${date.getUTCMonth() + 1}`
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  const monthBins: MonthBin[] = []
  const years = new Set<number>()
  for (const [key, count] of counts) {
    const [year, month] = key.split('-').map(Number)
    years.add(year)
    const next = month === 12 ? monthStartDay(year + 1, 1) : monthStartDay(year, month + 1)
    monthBins.push({ year, month, t0: tAtDay(monthStartDay(year, month)), t1: tAtDay(next), count })
  }
  monthBins.sort((a, b) => a.t0 - b.t0)
  const yearMarks = [...years].sort((a, b) => a - b).map((year) => ({ year, t: tAtDay(monthStartDay(year, 1)) }))

  const stepSet = new Set<number>([0])
  for (const seg of segments) if (seg.active) stepSet.add(seg.t0)
  for (let year = new Date(firstDay * DAY_MS).getUTCFullYear(); monthStartDay(year, 1) <= lastDay; year++) {
    for (let month = 1; month <= 12; month++) {
      const day = monthStartDay(year, month)
      if (day <= firstDay || day >= lastDay) continue
      if (segments.some((s) => s.active && day >= s.day0 && day <= s.day1)) stepSet.add(tAtDay(day))
    }
  }
  const stepTimes = [...stepSet].sort((a, b) => a - b)

  return {
    duration,
    total,
    dateAt: (t) => new Date(dayAt(t) * DAY_MS),
    tAt: (date) => tAtDay(typeof date === 'string' ? dayOf(date) : date.getTime() / DAY_MS),
    stateAt,
    countsAt,
    monthBins,
    yearMarks,
    stepTimes,
    nextStep: (t) => stepTimes.find((s) => s > t + 1e-6) ?? duration,
  }
}
