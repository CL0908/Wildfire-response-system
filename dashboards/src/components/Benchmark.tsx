import type { BenchmarkRow } from '../types'

const COLUMNS: { key: string; label: string }[] = [
  { key: 'greedy', label: 'Greedy' },
  { key: 'sa', label: 'SA' },
  { key: 'random_mean', label: 'Random' },
  { key: 'qaoa_mean', label: 'QAOA' },
  { key: 'xy_qaoa_mean', label: 'XY-QAOA' },
  { key: 'wsxy_start_mean', label: 'WSXY start' },
  { key: 'wsxy_qaoa_mean', label: 'WSXY-QAOA' },
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

export default function Benchmark({ rows }: { rows: BenchmarkRow[] }) {
  const counts = unique(rows.map((r) => r.n_instances))
  const terrains = unique(rows.map((r) => r.terrain))
  const showCounts = counts.length > 1
  const averaged = counts.length > 0

  const explanation =
    'Covered priority as a fraction of the exact optimum. Greedy and SA: the placement returned. Others: expected value of one shot, no classical repair. WSXY start: the warm-start state before any QAOA layer.'

  return (
    <section className="benchmark" aria-labelledby="benchmark-title">
      <h2 id="benchmark-title">Benchmark</h2>
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
      <p>
        {classicalMatchesOrBeats(rows)
          ? 'No quantum advantage on these instances: the classical methods match or beat every QAOA variant.'
          : 'Results are mixed on these instances; see the table.'}
      </p>
    </section>
  )
}
