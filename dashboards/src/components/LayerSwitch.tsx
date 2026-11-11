import { LAYERS } from '../lib/layers'
import type { LayerKey } from '../types'

interface Props {
  value: LayerKey
  onChange: (key: LayerKey) => void
}

export default function LayerSwitch({ value, onChange }: Props) {
  return (
    <div className="segmented" role="radiogroup" aria-label="Map layer">
      {LAYERS.map((l) => (
        <label key={l.key}>
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
