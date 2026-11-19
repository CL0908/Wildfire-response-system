import { formatInt, formatPercent } from '../lib/format'
import { randomBestHitProbability } from '../lib/sampling'
import type { Method, Results } from '../types'
import ChartLegend from './charts/ChartLegend'
import MethodBars, { type BarRow } from './charts/MethodBars'
import type { Series } from './charts/series'

interface Props {
  results: Results
  method: Method
}

const seriesOf = (m: Method): Series => (m.kind === 'classical' ? 'classical' : 'quantum')

/** The number a method hands back: the placement for classical methods, the average shot for QAOA. */
const barValue = (m: Method): number => (m.sampling ? m.sampling.mean_ratio : m.ratio)

export default function Placement({ results, method }: Props) {
  const { grid, total_priority, optimum } = results
  const totalCells = grid.rows * grid.cols
  const sensors = method.chosen.length
  const covered = method.covered_cells.length
  const shareOfMap = method.objective / total_priority
  const shareOfOptimum = method.objective / optimum
  const shots = method.sampling?.shots ?? results.methods.find((m) => m.sampling)?.sampling?.shots ?? 0
  const shotsText = formatInt(shots)
  const ref = results.methods.find((m) => m.sampling)?.sampling ?? null

  const subject = method.sampling
    ? `${method.name}’s best simulated shot (of ${shotsText})`
    : method.name
  const headline = `${subject} places ${sensors} sensor${sensors === 1 ? '' : 's'} that cover ${formatInt(covered)} of ${formatInt(totalCells)} cells.`
  const lede = `${method.sampling ? 'The best shot' : 'That'} is ${Math.round(shareOfMap * 100)}% of the priority on the map and ${formatPercent(shareOfOptimum)} of what the best possible placement covers.`

  const ordered = [...results.methods].sort(
    (a, b) => Number(a.kind !== 'classical') - Number(b.kind !== 'classical'),
  )
  const rows: BarRow[] = ordered.map((m) => ({
    name: m.name,
    series: seriesOf(m),
    value: barValue(m),
    best: m.sampling ? m.ratio : undefined,
    selected: m.name === method.name,
  }))

  const classical = ordered.filter((m) => !m.sampling)
  const quantum = ordered.filter((m) => m.sampling)
  const range = (values: number[]) => {
    const lo = Math.min(...values)
    const hi = Math.max(...values)
    return lo === hi ? formatPercent(lo) : `${formatPercent(lo)} to ${formatPercent(hi)}`
  }
  const chartLabel =
    quantum.length > 0 && classical.length > 0
      ? `Share of the exact optimum for each method. Classical placements reach ${range(classical.map((m) => m.ratio))}. QAOA average shots reach ${range(quantum.map(barValue))}, and the single best of ${shotsText} shots reaches ${range(quantum.map((m) => m.ratio))}. Random draws can reach it too (see the note below the table).`
      : 'Share of the exact optimum for each method.'

  return (
    <section aria-labelledby="placement-title">
      <div className="sec-head">
        <h2 id="placement-title">{headline}</h2>
        <p className="lede">{lede}</p>
      </div>
      <dl className="figures">
        <div>
          <dt>Sensors placed</dt>
          <dd>{sensors}</dd>
        </div>
        <div>
          <dt>Cells covered</dt>
          <dd>
            {formatInt(covered)} <span className="of">of {formatInt(totalCells)}</span>
          </dd>
        </div>
        <div>
          <dt>Covered priority</dt>
          <dd>
            {method.objective.toFixed(2)} <span className="of">of {total_priority.toFixed(2)}</span>
          </dd>
        </div>
      </dl>
      <div className="chart-block">
        <h3>All six methods on this map</h3>
        <ChartLegend
          items={[
            { label: 'Classical, placement returned', series: 'classical', mark: 'bar' },
            { label: 'Quantum (simulated), mean per shot', series: 'quantum', mark: 'bar' },
            ...(shots > 0 ? [{ label: `best of ${shotsText} shots (control, not evidence)`, series: 'quantum' as const, mark: 'hollow' as const }] : []),
          ]}
        />
        <MethodBars
          rows={rows}
          bestTitle={`best of ${shotsText} shots (not evidence, see note)`}
          label={chartLabel}
          formatValue={formatPercent}
        />
        <ul className="sr-only">
          {ordered.map((m) => (
            <li key={m.name}>
              {m.sampling
                ? `${m.name}: mean per shot ${formatPercent(m.sampling.mean_ratio)} of the optimum, random placement ${formatPercent(m.sampling.mean_ratio_random)}; best of ${formatInt(m.sampling.shots)} shots ${formatPercent(m.ratio)} (secondary, see caveat).`
                : `${m.name}: placement returned, ${formatPercent(m.ratio)} of the optimum.`}
            </li>
          ))}
        </ul>
      </div>
      {ref && (
        <>
          <p className="method-note">
            Simulated QAOA, {ref.p_layers} layers, {formatInt(ref.shots)} shots each. Scores are shares of the optimum. QAOA shots with the wrong number of
            sensors score 0; the random-placement column draws only valid {results.n_sensors}-sensor placements, which is a hard bar for QAOA and
            WSXY-QAOA when many of their shots are invalid. WSXY-QAOA&rsquo;s like-for-like baseline is its warm-start state.
          </p>
          <div className="table-wrap" tabIndex={0} role="region" aria-label="QAOA scores per shot compared with random placement">
            <table className="hw-table">
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  <th scope="col" className="num">Mean per shot</th>
                  <th scope="col" className="num">Random placement</th>
                  <th scope="col" className="num">Warm-start state</th>
                  <th scope="col" className="num">Shots with {results.n_sensors} sensors</th>
                  <th scope="col" className="num">Best of {shotsText} (secondary)</th>
                  <th scope="col" className="num">Random best-of hit chance</th>
                </tr>
              </thead>
              <tbody>
                {quantum.map((m) => {
                  const s = m.sampling!
                  return (
                    <tr key={m.name} className={m.name === method.name ? 'is-selected' : undefined}>
                      <th scope="row">{m.name}</th>
                      <td className="num">{formatPercent(s.mean_ratio)}</td>
                      <td className="num">{formatPercent(s.mean_ratio_random)}</td>
                      <td className="num">{s.start_mean_ratio !== undefined ? formatPercent(s.start_mean_ratio) : 'n/a'}</td>
                      <td className="num">{formatPercent(s.p_feasible)}</td>
                      <td className="num">{formatPercent(m.ratio)}</td>
                      <td className="num">{formatPercent(randomBestHitProbability(s))}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="method-note">
            Best of {shotsText} shots is not evidence of the circuit working: there are only {formatInt(ref.n_placements)} possible placements, so{' '}
            {formatInt(ref.shots)} uniformly random draws reach the optimum with probability {formatPercent(randomBestHitProbability(ref))}.
          </p>
        </>
      )}
      <p className="method-note">
        Bars show covered priority as a share of the exact optimum. Greedy, ILP and SA return a placement, and the bar is that placement. QAOA returns one random
        sample per shot, so its bar is the mean score per shot (infeasible shots count 0); the open circle is the best of {shotsText} shots, which is the
        placement drawn on the map. QAOA is run on a classical simulator, not on quantum hardware.
      </p>
    </section>
  )
}
