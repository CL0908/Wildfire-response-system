import { useEffect, useState } from 'react'
import { formatInt } from '../lib/format'
import { countsByYear, countsByYearMonth, detectionsInCells, loadFires, type Fires, type FiresState } from '../lib/fires'
import type { Method, Results } from '../types'
import FireCalendar from './charts/FireCalendar'
import { useWidth } from './charts/useWidth'

const MONTHS_LONG = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]
const NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five']
/** How many of the busiest years the headline sums. */
const TOP_YEARS = 3
/** Width of the busiest run of consecutive months in the lede. */
const SEASON_MONTHS = 3

const pct = (part: number, whole: number) => `${Math.round((part / whole) * 100)}%`

function listOf(items: string[]): string {
  return items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

/** The fire history, or null while loading; a failure is a state of its own. */
function useFires(): FiresState | null {
  const [state, setState] = useState<FiresState | null>(null)
  useEffect(() => {
    let live = true
    void loadFires().then((s) => {
      if (live) setState(s)
    })
    return () => {
      live = false
    }
  }, [])
  return state
}

function Body({ fires, results, method }: { fires: Fires; results: Results; method: Method }) {
  const { detections } = fires
  const [ref, width] = useWidth<HTMLDivElement>()
  const total = detections.length
  const { years, counts } = countsByYearMonth(detections)
  const yearTotals = countsByYear(detections).totals
  const monthTotals = Array.from({ length: 12 }, (_, m) => counts.reduce((sum, row) => sum + row[m], 0))

  const busiest = years
    .map((year, i) => ({ year, n: yearTotals[i] }))
    .sort((a, b) => b.n - a.n)
    .slice(0, TOP_YEARS)
  const busiestN = busiest.reduce((s, y) => s + y.n, 0)
  const byYear = [...busiest].sort((a, b) => a.year - b.year)

  let seasonStart = 0
  let seasonN = -1
  for (let m = 0; m + SEASON_MONTHS <= 12; m++) {
    const n = monthTotals.slice(m, m + SEASON_MONTHS).reduce((a, b) => a + b, 0)
    if (n > seasonN) {
      seasonN = n
      seasonStart = m
    }
  }
  const seasonText = `${MONTHS_LONG[seasonStart]} to ${MONTHS_LONG[seasonStart + SEASON_MONTHS - 1]}`

  const firstYear = years[0]
  const lastYear = years[years.length - 1]
  const headline = `Satellites recorded ${formatInt(total)} fire detections here from ${firstYear} to ${lastYear}, and ${pct(busiestN, total)} of them fell in ${NUMBER_WORDS[TOP_YEARS] ?? TOP_YEARS} years.`
  const lede = `${pct(seasonN, total)} of the detections fell from ${seasonText}. The busiest years were ${listOf(
    byYear.map((y) => `${y.year} (${formatInt(y.n)})`),
  )}.`

  const plural = (n: number) => `${formatInt(n)} detection${n === 1 ? '' : 's'}`
  const chartLabel =
    `Fire detections by year and month, ${firstYear} to ${lastYear}. ` +
    years.map((y, i) => `${y}: ${yearTotals[i]}.`).join(' ') +
    ` Most fall from ${seasonText}.`

  // Cell indices only mean the same thing if both files describe the same grid.
  const sameGrid = fires.grid.rows === results.grid.rows && fires.grid.cols === results.grid.cols
  const inCovered = sameGrid ? detectionsInCells(detections, new Set(method.covered_cells)).length : null
  const subject = method.sampling ? `the sensors in ${method.name}’s best simulated shot` : `${method.name}’s sensors`

  return (
    <>
      <div className="sec-head">
        <h2 id="fires-title">{headline}</h2>
        <p className="lede">{lede}</p>
      </div>
      <div className="chart-block">
        <h3>Detections per month, by year</h3>
        <div ref={ref} className="chart-box">
          {width > 0 && <FireCalendar years={years} counts={counts} width={width} label={chartLabel} />}
        </div>
      </div>
      {inCovered !== null && (
        <div className="fire-check">
          <p>
            <strong>
              {formatInt(inCovered)} of {formatInt(total)}
            </strong>{' '}
            detections ({pct(inCovered, total)}) fell in cells covered by {subject}. Those cells are{' '}
            {pct(method.covered_cells.length, results.grid.rows * results.grid.cols)} of the map.
          </p>
          <p className="muted">
            Fire history is one of the inputs to the hazard layer that the placement optimises, so this is a consistency
            check, not an independent test of the placement.
          </p>
        </div>
      )}
      <p className="method-note">
        Darker cells have more detections; empty cells had none. Each detection is a satellite hot-spot reading located
        to about 375 m, not a fire outline, and detections with low confidence are excluded.
      </p>
      <details>
        <summary>Show the numbers</summary>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Fire detections by year and month">
          <table>
            <caption>
              Detections per month, {firstYear} to {lastYear}: {plural(total)} in all.
            </caption>
            <thead>
              <tr>
                <th scope="col">Year</th>
                {MONTHS_LONG.map((m) => (
                  <th key={m} scope="col" className="num">
                    {m.slice(0, 3)}
                  </th>
                ))}
                <th scope="col" className="num">
                  Total
                </th>
              </tr>
            </thead>
            <tbody>
              {years.map((y, i) => (
                <tr key={y}>
                  <th scope="row">{y}</th>
                  {counts[i].map((n, m) => (
                    <td key={m} className="num">
                      {n}
                    </td>
                  ))}
                  <td className="num">{yearTotals[i]}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr>
                <th scope="row">All years</th>
                {monthTotals.map((n, m) => (
                  <td key={m} className="num">
                    {n}
                  </td>
                ))}
                <td className="num">{total}</td>
              </tr>
            </tfoot>
          </table>
        </div>
      </details>
    </>
  )
}

export default function FireHistory({ results, method }: { results: Results; method: Method }) {
  const state = useFires()
  return (
    <section aria-labelledby={state?.status === 'ready' ? 'fires-title' : undefined} aria-label={state?.status === 'ready' ? undefined : 'Fire history'}>
      {state === null && <p role="status" className="muted">Loading the fire history…</p>}
      {state?.status === 'error' && (
        <p role="alert" className="error">
          {state.message}
        </p>
      )}
      {state?.status === 'ready' && <Body fires={state.fires} results={results} method={method} />}
    </section>
  )
}
