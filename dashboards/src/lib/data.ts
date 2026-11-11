import { useEffect, useState } from 'react'
import type { Results } from '../types'

export type LoadState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; results: Results }

const HINT = 'Run `python main.py` from the repo root to generate it.'

function looksLikeResults(x: unknown): x is Results {
  if (typeof x !== 'object' || x === null) return false
  const r = x as Partial<Results>
  return (
    !!r.grid &&
    !!r.layers &&
    Array.isArray(r.sites) &&
    Array.isArray(r.methods) &&
    r.methods.length > 0 &&
    !!r.viewsheds &&
    Array.isArray(r.benchmark)
  )
}

async function load(): Promise<LoadState> {
  let res: Response
  try {
    res = await fetch(`${import.meta.env.BASE_URL}results.json`)
  } catch {
    return { status: 'error', message: `Could not fetch results.json. ${HINT}` }
  }
  if (res.status === 404) {
    return { status: 'error', message: `results.json not found. ${HINT}` }
  }
  if (!res.ok) {
    return {
      status: 'error',
      message: `Could not load results.json (HTTP ${res.status}). ${HINT}`,
    }
  }
  try {
    const json: unknown = await res.json()
    if (looksLikeResults(json)) return { status: 'ready', results: json }
  } catch {
    // fall through to the shared message
  }
  return {
    status: 'error',
    message: `results.json is missing or not in the expected format. ${HINT}`,
  }
}

export function useResults(): LoadState {
  const [state, setState] = useState<LoadState>({ status: 'loading' })
  useEffect(() => {
    let live = true
    void load().then((s) => {
      if (live) setState(s)
    })
    return () => {
      live = false
    }
  }, [])
  return state
}
