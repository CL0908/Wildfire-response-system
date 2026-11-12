import { formatDateLong, formatInt, formatScore, formatShare } from '../lib/format'
import type { HardwareData, HardwareRun, RunStats } from '../lib/hardware'

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

function Run({ run, data }: { run: HardwareRun; data: HardwareData }) {
  const { instance, baselines } = data
  const n = instance.n_sensors
  return (
    <article className="run">
      <h3>{run.device}</h3>
      <p className="muted">
        {run.access} · {formatDateLong(run.date)} · {run.circuit} ·{' '}
        {formatInt(run.shots)} shots
      </p>
      <div
        className="table-wrap"
        tabIndex={0}
        role="region"
        aria-label={`Hardware results, ${run.device}`}
      >
        <table className="hw-table">
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
      <p className="muted">{run.note}</p>
      <p>
        A random valid placement is the optimum{' '}
        {formatShare(baselines.random_placement.p_optimal)} of the time; of the shots
        measured on {run.device}, {formatShare(run.measured.p_optimal)} were the
        optimal placement. This run shows no quantum advantage.
      </p>
    </article>
  )
}

export default function Hardware({ data }: { data: HardwareData }) {
  const { instance } = data
  return (
    <section className="hardware" aria-labelledby="hardware-title">
      <h2 id="hardware-title">Hardware runs</h2>
      <p>
        The instance is the small {instance.n_sites}-site, {instance.n_sensors}-sensor{' '}
        {instance.terrain} case, not the map above.
      </p>
      {data.runs.map((run) => (
        <Run key={run.id} run={run} data={data} />
      ))}
    </section>
  )
}
