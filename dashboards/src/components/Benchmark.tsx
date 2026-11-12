import type { BenchmarkRow } from '../types'
import ChartLegend from './charts/ChartLegend'
import DotPanel, { type DotRow } from './charts/DotPanel'
import { PANEL_PAD } from './charts/layout'
import type { Series } from './charts/series'
import { useWidth } from './charts/useWidth'

const COLUMNS: { key: string; label: string; series: Series; hollow?: boolean }[] = [
  { key: 'greedy', label: 'Greedy', series: 'classical' },
  { key: 'sa', label: 'SA', series: 'classical' },
  { key: 'random_mean', label: 'Random', series: 'baseline' },
  { key: 'qaoa_mean', label: 'QAOA', series: 'quantum' },
  { key: 'xy_qaoa_mean', label: 'XY-QAOA', series: 'quantum' },
  { key: 'wsxy_start_mean', label: 'WSXY start', series: 'quantum', hollow: true },
  { key: 'wsxy_qaoa_mean', label: 'WSXY-QAOA', series: 'quantum' },
]

/** The QAOA variants proper; the WSXY start is a warm-start state, not a QAOA result. */
const QAOA_KEYS = ['qaoa_mean', 'xy_qaoa_mean', 'wsxy_qaoa_mean']

/** "|S|=8, N=2" -> "8 sites, 2 sensors"; anything else is shown as is. */
function instanceLabel(instance: string): string {
  const m = /^\|S\|=(\d+), N=(\d+)$/.exec(instance)
  return m ? `${m[1]} sites, ${m[2]} sensors` : instance
}

function num(v: string | undefined): number | null {
  if (v === undefined || v.trim() === '') return null
  const n = Number(v)
  return Number.isNaN(n) ? null : n
}

function fixed3(n: number | null): string {
  return n === null ? '–' : n.toFixed(3)
}

/** True only if, in every row, the better classical method is at least every QAOA mean. */
function classicalMatchesOrBeats(rows: BenchmarkRow[]): boolean {
  if (rows.length === 0) return false
  return rows.every((r) => {
    const classical = [num(r.greedy), num(r.sa)].filter((v) => v !== null)
    const qaoa = QAOA_KEYS.map((k) => num(r[k]))
    if (classical.length === 0 || qaoa.some((v) => v === null)) return false
    return Math.max(...classical) >= Math.max(...(qaoa as number[]))
  })
}

function unique(values: (string | undefined)[]): string[] {
  return [...new Set(values.filter((v): v is string => !!v && v.trim() !== ''))]
}

function Cell({ row, column }: { row: BenchmarkRow; column: string }) {
  const mean = num(row[column])
  const std = num(row[`${column}_std`])
  return (
    <td className={mean !== null && std !== null ? 'num has-std' : 'num'}>
      {fixed3(mean)}
      {mean !== null && std !== null && (
        <>
          {' '}
          <span className="pm">±&nbsp;{fixed3(std)}</span>
        </>
      )}
    </td>
  )
}

/** Below this container width the three panels stack. */
const WIDE_MIN = 760
const PANEL_GAP = 24
const MIN_PLOT = 120

function panelLabel(row: BenchmarkRow, dots: DotRow[]): string {
  const known = dots.filter((d): d is DotRow & { mean: number } => d.mean !== null)
  if (known.length === 0) return `${instanceLabel(row.instance)}: no results.`
  const best = known.reduce((a, b) => (b.mean > a.mean ? b : a))
  const qaoa = known.filter((d) => QAOA_KEYS.includes(d.key))
  const bestQaoa = qaoa.length > 0 ? qaoa.reduce((a, b) => (b.mean > a.mean ? b : a)) : null
  return (
    `${instanceLabel(row.instance)}, mean fraction of the exact optimum: ` +
    `${best.label} is highest at ${fixed3(best.mean)}` +
    (bestQaoa && bestQaoa !== best ? `; the best QAOA variant, ${bestQaoa.label}, reaches ${fixed3(bestQaoa.mean)}.` : '.') +
    ` ${known.map((d) => `${d.label} ${fixed3(d.mean)}`).join(', ')}.`
  )
}

export default function Benchmark({ rows }: { rows: BenchmarkRow[] }) {
  const counts = unique(rows.map((r) => r.n_instances))
  const terrains = unique(rows.map((r) => r.terrain))
  const showCounts = counts.length > 1
  const averaged = counts.length > 0
  const [ref, width] = useWidth<HTMLDivElement>()
  const wide = width >= WIDE_MIN
  const n = rows.length
  const plotW = wide
    ? Math.max(
        MIN_PLOT,
        Math.floor(
          (width - PANEL_GAP * (n - 1) - PANEL_PAD.label - PANEL_PAD.left * (n - 1) - PANEL_PAD.right * n) / n,
        ),
      )
    : Math.max(MIN_PLOT, width - PANEL_PAD.label - PANEL_PAD.right)

  const explanation =
    'Covered priority as a fraction of the exact optimum. Greedy and SA: the placement returned. Others: expected value of one shot, no classical repair. WSXY start: the warm-start state before any QAOA layer.'

  const lede = averaged
    ? `Each panel runs ${showCounts ? 'several' : counts[0]} instances of one size${
        terrains.length > 0 ? ` on ${terrains.join(' and ')} terrain` : ''
      }, scoring every method against the exact optimum.`
    : 'Each panel is one instance, smaller than the map above, scored against the exact optimum.'

  return (
    <section aria-labelledby="benchmark-title">
      <div className="sec-head">
        <h2 id="benchmark-title">
          {classicalMatchesOrBeats(rows)
            ? 'No quantum advantage on these instances: the classical methods match or beat every QAOA variant.'
            : 'Results are mixed on these instances.'}
        </h2>
        <p className="lede">{lede}</p>
      </div>
      <div className="chart-block">
        <ChartLegend
          items={[
            { label: 'Classical', series: 'classical', mark: 'dot' },
            { label: 'Random baseline', series: 'baseline', mark: 'dot' },
            { label: 'Quantum (simulated)', series: 'quantum', mark: 'dot' },
            { label: 'WSXY start, not a QAOA result', series: 'quantum', mark: 'hollow' },
          ]}
        />
        <div ref={ref} className={wide ? 'panels wide' : 'panels'}>
          {width > 0 &&
            rows.map((r, i) => {
              const dots: DotRow[] = COLUMNS.map((c) => ({
                key: c.key,
                label: c.label,
                series: c.series,
                hollow: c.hollow,
                mean: num(r[c.key]),
                std: num(r[`${c.key}_std`]),
              }))
              const labels = wide ? i === 0 : true
              return (
                <div key={r.instance} className="panel">
                  <h3 style={{ paddingLeft: labels ? PANEL_PAD.label : PANEL_PAD.left }}>{instanceLabel(r.instance)}</h3>
                  <DotPanel
                    rows={dots}
                    plotW={plotW}
                    labels={labels}
                    label={panelLabel(r, dots)}
                    formatValue={fixed3}
                  />
                </div>
              )
            })}
        </div>
      </div>
      <p className="method-note">
        {explanation} Dots are means; whiskers are one standard deviation across instances.
      </p>
      <details>
        <summary>Show the numbers</summary>
        <div className="table-wrap" tabIndex={0} role="region" aria-label="Benchmark table">
          <table>
            <caption>
              {averaged
                ? `Mean ± standard deviation over ${
                    showCounts ? 'several' : counts[0]
                  } instances per size${
                    terrains.length > 0 ? ` on ${terrains.join(' and ')} terrain` : ''
                  }${showCounts ? ' (see the Instances column)' : ''}. `
                : 'Three smaller instances than the map above. '}
              {explanation}
            </caption>
            <thead>
              <tr>
                <th scope="col">Instance</th>
                {showCounts && (
                  <th scope="col" className="num">
                    Instances
                  </th>
                )}
                {COLUMNS.map((c) => (
                  <th key={c.key} scope="col" className="num">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.instance}>
                  <th scope="row">{instanceLabel(r.instance)}</th>
                  {showCounts && <td className="num">{r.n_instances ?? '–'}</td>}
                  {COLUMNS.map((c) => (
                    <Cell key={c.key} row={r} column={c.key} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  )
}
