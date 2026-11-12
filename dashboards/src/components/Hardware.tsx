import { formatDateLong, formatInt, formatScore, formatShare } from '../lib/format'
import type { HardwareData, HardwareRun, RunStats } from '../lib/hardware'
import ChartLegend from './charts/ChartLegend'
import HardwareDots, { HARDWARE_PAD, type HardwareDotRow } from './charts/HardwareDots'
import { useWidth } from './charts/useWidth'

const MIN_PLOT = 120

function StatsCells({ s, optimalNote }: { s: RunStats; optimalNote?: string }) {
  return (
    <>
      <td className="num">{formatShare(s.p_feasible)}</td>
      <td className="num">
        {formatShare(s.p_optimal)}
        {optimalNote ? ` (${optimalNote})` : ''}
      </td>
      <td className="num">{formatScore(s.mean_ratio)}</td>
    </>
  )
}

function RunTable({ run, data }: { run: HardwareRun; data: HardwareData }) {
  const { instance, baselines } = data
  const n = instance.n_sensors
  return (
    <div
      className="table-wrap"
      tabIndex={0}
      role="region"
      aria-label={`Hardware results, ${run.device}, ${formatDateLong(run.date)}`}
    >
      <table className="hw-table">
        <caption>
          {run.device} · {formatDateLong(run.date)} · {run.circuit}
        </caption>
        <thead>
          <tr>
            <th scope="col">Source of shots</th>
            <th scope="col" className="num">
              Shots with exactly {n} sensors
            </th>
            <th scope="col" className="num">
              Shots that were the optimal placement
            </th>
            <th scope="col" className="num">
              Average score vs optimum
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Measured on {run.device}</th>
            <StatsCells
              s={run.measured}
              optimalNote={`${formatInt(run.optimal_shots)} of ${formatInt(run.shots)}`}
            />
          </tr>
          <tr>
            <th scope="row">Same circuit, noiseless simulation</th>
            <StatsCells s={run.noiseless} />
          </tr>
          <tr>
            <th scope="row">Random bitstrings</th>
            <StatsCells s={baselines.uniform_bitstrings} />
          </tr>
          <tr>
            <th scope="row">Random {n}-sensor placement</th>
            <StatsCells s={baselines.random_placement} />
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function RunFacts({ run, n }: { run: HardwareRun; n: number }) {
  return (
    <div className="run-facts">
      <h3>Measured, {formatDateLong(run.date)}</h3>
      <dl>
        <div>
          <dt>Device</dt>
          <dd>
            {run.device} · {run.access}
          </dd>
        </div>
        <div>
          <dt>Circuit</dt>
          <dd>{run.circuit}</dd>
        </div>
        <div>
          <dt>Shots</dt>
          <dd>{formatInt(run.shots)}</dd>
        </div>
        <div>
          <dt>With exactly {n} sensors</dt>
          <dd>{formatShare(run.measured.p_feasible)}</dd>
        </div>
        <div>
          <dt>Optimal placement</dt>
          <dd>
            {formatShare(run.measured.p_optimal)} ({formatInt(run.optimal_shots)} of {formatInt(run.shots)})
          </dd>
        </div>
      </dl>
      <p className="muted">{run.note}</p>
    </div>
  )
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`

/** One sentence for the whole section; empty unless every measured run scores below a random valid placement. */
function advantageSentence(runs: HardwareRun[], randomMean: number): string {
  if (!runs.every((r) => r.measured.mean_ratio < randomMean)) return ''
  if (runs.length === 1) return 'This run shows no quantum advantage.'
  if (runs.length === 2) return 'Neither run shows a quantum advantage.'
  return 'None of the runs shows a quantum advantage.'
}

export default function Hardware({ data }: { data: HardwareData }) {
  const { instance, baselines } = data
  const n = instance.n_sensors
  const runs = [...data.runs].sort((a, b) => a.date.localeCompare(b.date))
  const [ref, width] = useWidth<HTMLDivElement>()
  const plotW = Math.max(MIN_PLOT, width - HARDWARE_PAD)

  const first = runs[0]
  const last = runs[runs.length - 1]
  const randomMean = baselines.random_placement.mean_ratio
  const improved =
    runs.length >= 2 && first.measured.mean_ratio > 0 && last.measured.mean_ratio > first.measured.mean_ratio
  const allBelowRandom = runs.every((r) => r.measured.mean_ratio < randomMean)
  const headline =
    improved && allBelowRandom
      ? `On real quantum hardware every run scored below a random valid placement; the latest scored ${(last.measured.mean_ratio / first.measured.mean_ratio).toFixed(1)}× the first.`
      : `Results from ${plural(runs.length, 'run', 'runs')} on real quantum hardware.`

  const rows: HardwareDotRow[] = [
    ...runs.flatMap<HardwareDotRow>((r) => [
      {
        key: `${r.id}-measured`,
        label: `Measured, ${formatDateLong(r.date)}`,
        series: 'quantum',
        group: r.id,
        value: r.measured.mean_ratio,
      },
      {
        key: `${r.id}-noiseless`,
        label: 'Same circuit, noiseless simulation',
        series: 'quantum',
        hollow: true,
        group: r.id,
        value: r.noiseless.mean_ratio,
      },
    ]),
    {
      key: 'uniform',
      label: 'Random bitstrings',
      series: 'baseline',
      group: 'baselines',
      value: baselines.uniform_bitstrings.mean_ratio,
    },
    {
      key: 'random',
      label: `Random ${n}-sensor placement`,
      series: 'baseline',
      group: 'baselines',
      value: baselines.random_placement.mean_ratio,
    },
  ]
  const chartLabel =
    'Average score of one shot as a share of the optimum. ' +
    rows.map((r) => `${r.label}: ${formatScore(r.value)}.`).join(' ')

  return (
    <section className="hardware" aria-labelledby="hardware-title">
      <div className="sec-head">
        <h2 id="hardware-title">{headline}</h2>
        <p className="lede">
          The instance is the small {instance.n_sites}-site, {n}-sensor {instance.terrain} case, not the
          map above, and scores are per shot with no classical repair. {advantageSentence(runs, randomMean)}
        </p>
      </div>
      <div className="chart-block">
        <h3>Average score of one shot, as a share of the optimum</h3>
        <ChartLegend
          items={[
            { label: 'Measured on hardware', series: 'quantum', mark: 'dot' },
            { label: 'Noiseless simulation', series: 'quantum', mark: 'hollow' },
            { label: 'Random baseline', series: 'baseline', mark: 'dot' },
          ]}
        />
        <div ref={ref} className="chart-box">
          {width > 0 && <HardwareDots rows={rows} plotW={plotW} label={chartLabel} formatValue={formatScore} />}
        </div>
      </div>
      <p className="method-note">
        Each dot is the average score of one shot, where 1 is the optimal placement (the line). A random
        valid placement is the optimum {formatShare(baselines.random_placement.p_optimal)} of the time.
      </p>
      <div className="run-grid">
        {runs.map((r) => (
          <RunFacts key={r.id} run={r} n={n} />
        ))}
      </div>
      <details>
        <summary>Show the numbers</summary>
        <div className="run-tables">
          {runs.map((r) => (
            <RunTable key={r.id} run={r} data={data} />
          ))}
        </div>
      </details>
    </section>
  )
}
