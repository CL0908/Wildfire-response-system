import type { LayerInfo } from '../lib/layers'
import type { LayerKey } from '../types'

interface Props {
  layers: LayerInfo[]
  value: LayerKey
  onChange: (key: LayerKey) => void
}

export default function LayerSwitch({ layers, value, onChange }: Props) {
  // Equal-width grid: 3 columns for 3 or 6 layers, otherwise 2 with a full-width last
  // button when the count is odd, so the control never ends in a lone short button.
  const cols = layers.length % 3 === 0 ? 3 : 2
  return (
    <div
      className="segmented"
      role="radiogroup"
      aria-label="Map layer"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {layers.map((l, i) => (
        <label
          key={l.key}
          className={
            cols === 2 && layers.length % 2 === 1 && i === layers.length - 1
              ? 'wide'
              : undefined
          }
        >
          <input
            type="radio"
            name="layer"
            className="sr-only"
            checked={value === l.key}
            onChange={() => onChange(l.key)}
          />
          <span>{l.label}</span>
        </label>
      ))}
    </div>
  )
}
