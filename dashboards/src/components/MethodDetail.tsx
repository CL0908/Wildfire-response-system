import { formatInt, formatPercent } from '../lib/format'
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
          Best of {formatInt(s.shots)} simulated shots. A single shot is optimal{' '}
          {formatPercent(s.p_optimal)} of the time (random placement:{' '}
          {formatPercent(s.p_optimal_random)}); {formatPercent(s.p_feasible)} of
          shots place exactly {results.n_sensors} sensors. On average one shot
          reaches {formatPercent(s.mean_ratio)} of the optimum.
        </p>
      )}
    </div>
  )
}
