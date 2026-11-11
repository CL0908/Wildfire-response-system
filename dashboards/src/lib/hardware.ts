import { useEffect, useState } from 'react'

export interface RunStats {
  p_feasible: number
  p_optimal: number
  mean_ratio: number
}

export interface HardwareRun {
  id: string
  device: string
  access: string
  date: string
  circuit: string
  note: string
  shots: number
  optimal_shots: number
  measured: RunStats
  noiseless: RunStats
}

export interface HardwareData {
  instance: {
    n_sites: number
    n_sensors: number
    seed: number
    terrain: string
    n_placements: number
  }
  baselines: { uniform_bitstrings: RunStats; random_placement: RunStats }
  runs: HardwareRun[]
}

const isObj = (x: unknown): x is Record<string, unknown> =>
  typeof x === 'object' && x !== null
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x)
const isStr = (x: unknown): x is string => typeof x === 'string'

function isStats(x: unknown): x is RunStats {
  return (
    isObj(x) && isNum(x.p_feasible) && isNum(x.p_optimal) && isNum(x.mean_ratio)
  )
}

function isRun(x: unknown): x is HardwareRun {
  return (
    isObj(x) &&
    isStr(x.id) &&
    isStr(x.device) &&
    isStr(x.access) &&
    isStr(x.date) &&
    isStr(x.circuit) &&
    isStr(x.note) &&
    isNum(x.shots) &&
    isNum(x.optimal_shots) &&
    isStats(x.measured) &&
    isStats(x.noiseless)
  )
}

/** Returns null unless the file is well formed and has at least one valid run. */
function parse(json: unknown): HardwareData | null {
  if (!isObj(json)) return null
  const { instance, baselines, runs } = json
  if (
    !isObj(instance) ||
    !isNum(instance.n_sites) ||
    !isNum(instance.n_sensors) ||
    !isStr(instance.terrain) ||
    !isObj(baselines) ||
    !isStats(baselines.uniform_bitstrings) ||
    !isStats(baselines.random_placement) ||
    !Array.isArray(runs)
  ) {
    return null
  }
  const valid = runs.filter(isRun)
  if (valid.length === 0) return null
  return { instance, baselines, runs: valid } as unknown as HardwareData
}

async function load(): Promise<HardwareData | null> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}hardware.json`)
    if (!res.ok) return null
    return parse(await res.json())
  } catch {
    return null
  }
}

/** The hardware file is optional: null while loading and whenever it is missing or malformed. */
export function useHardware(): HardwareData | null {
  const [data, setData] = useState<HardwareData | null>(null)
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
