import { formatInt, formatPercent } from '../lib/format'
import { randomBestHitProbability } from '../lib/sampling'
import type { Method, Results } from '../types'

interface Props {
  results: Results
  method: Method
}

export default function MethodDetail({ results, method }: Props) {
  const { grid, total_priority } = results
  const totalCells = grid.rows * grid.cols
  const s = method.sampling

  return (
    <div className="detail">
      <dl className="stats">
        <div>
          <dt>Sensors placed</dt>
          <dd>{method.chosen.length}</dd>
        </div>
        <div>
          <dt>Cells covered</dt>
          <dd>
            {formatInt(method.covered_cells.length)} of {formatInt(totalCells)}
          </dd>
        </div>
        <div>
          <dt>Covered priority</dt>
          <dd>
            {method.objective.toFixed(2)} of {total_priority.toFixed(2)}
          </dd>
        </div>
      </dl>
      {s && (
        <p className="note">
          Mean score per simulated shot: {formatPercent(s.mean_ratio)} of the optimum (infeasible
          shots count 0). Random {results.n_sensors}-sensor placement: {formatPercent(s.mean_ratio_random)}.
          {s.start_mean_ratio !== undefined && (
            <> Warm-start state before any circuit layers: {formatPercent(s.start_mean_ratio)}.</>
          )}{' '}
          {formatPercent(s.p_feasible)} of shots place exactly {results.n_sensors} sensors. A single
          shot is optimal {formatPercent(s.p_optimal)} of the time (random placement:{' '}
          {formatPercent(s.p_optimal_random)}). The placement drawn on the map is the best of{' '}
          {formatInt(s.shots)} shots, which is not evidence the circuit works: random draws reach
          the optimum with probability {formatPercent(randomBestHitProbability(s))} in the same
          number of shots.
        </p>
      )}
    </div>
  )
}
