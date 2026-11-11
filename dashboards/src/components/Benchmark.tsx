import type { BenchmarkRow } from '../types'

const COLUMNS: { key: keyof BenchmarkRow; label: string }[] = [
  { key: 'greedy', label: 'Greedy' },
  { key: 'sa', label: 'SA' },
  { key: 'random_mean', label: 'Random' },
  { key: 'qaoa_mean', label: 'QAOA' },
  { key: 'xy_qaoa_mean', label: 'XY-QAOA' },
  { key: 'wsxy_start_mean', label: 'WSXY start' },
  { key: 'wsxy_qaoa_mean', label: 'WSXY-QAOA' },
]

/** "|S|=8, N=2" -> "8 sites, 2 sensors"; anything else is shown as is. */
function instanceLabel(instance: string): string {
  const m = /^\|S\|=(\d+), N=(\d+)$/.exec(instance)
  return m ? `${m[1]} sites, ${m[2]} sensors` : instance
}

function fixed3(v: string | undefined): string {
  const n = Number(v)
  return v === undefined || v === '' || Number.isNaN(n) ? '–' : n.toFixed(3)
}

export default function Benchmark({ rows }: { rows: BenchmarkRow[] }) {
  return (
    <section className="benchmark" aria-labelledby="benchmark-title">
      <h2 id="benchmark-title">Benchmark</h2>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Benchmark table">
        <table>
          <caption>
            Three smaller instances than the map above. Covered priority as a
            fraction of the exact optimum. Greedy and SA: the placement returned.
            Others: expected value of one shot, no classical repair. WSXY start: the
            warm-start state before any QAOA layer.
          </caption>
          <thead>
            <tr>
              <th scope="col">Instance</th>
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
                {COLUMNS.map((c) => (
                  <td key={c.key} className="num">
                    {fixed3(r[c.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        No quantum advantage on these instances: the classical methods match or beat
        every QAOA variant.
      </p>
    </section>
  )
}
