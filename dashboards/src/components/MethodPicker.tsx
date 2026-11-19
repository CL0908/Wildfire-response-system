import { formatPercent } from '../lib/format'
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
                ? `${m.kind} · mean per shot (random ${formatPercent(m.sampling.mean_ratio_random)})`
                : m.kind}
            </span>
          </span>
          <span className="num">
            {formatPercent(m.sampling ? m.sampling.mean_ratio : m.objective / results.optimum)}
            <span className="sr-only"> of optimum{m.sampling ? ', mean per shot' : ''}</span>
          </span>
        </label>
      ))}
    </fieldset>
  )
}
