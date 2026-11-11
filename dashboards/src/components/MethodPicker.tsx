import { formatInt, formatPercent } from '../lib/format'
import type { Results } from '../types'

interface Props {
  results: Results
  selected: number
  onSelect: (index: number) => void
}

export default function MethodPicker({ results, selected, onSelect }: Props) {
  return (
    <fieldset className="methods">
      <legend>Method</legend>
      <div className="method-head" aria-hidden="true">
        <span />
        <span className="num">% of optimum</span>
        <span className="num">Objective</span>
      </div>
      {results.methods.map((m, i) => (
        <label key={m.name} className="method-row">
          <input
            type="radio"
            name="method"
            className="sr-only"
            checked={selected === i}
            onChange={() => onSelect(i)}
          />
          <span className="method-name">
            <strong>{m.name}</strong>
            <span className="muted">
              {m.sampling
                ? `${m.kind}, best of ${formatInt(m.sampling.shots)} shots; average shot ${formatPercent(m.sampling.mean_ratio)}`
                : m.kind}
            </span>
          </span>
          <span className="num">{formatPercent(m.objective / results.optimum)}</span>
          <span className="num">{m.objective.toFixed(2)}</span>
        </label>
      ))}
    </fieldset>
  )
}
