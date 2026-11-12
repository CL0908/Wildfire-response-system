import type { StageLayer } from '../lib/layers'

interface Props {
  layers: { key: StageLayer; label: string }[]
  value: StageLayer
  onChange: (key: StageLayer) => void
}

export default function LayerSwitch({ layers, value, onChange }: Props) {
  // Landscape (not a data layer) has the first row to itself. The data layers share an
  // equal-width grid: 3 columns for 3 or 6, otherwise 2 with a full-width last button when the
  // count is odd, so the control never ends in a lone short button.
  const lead = layers[0]?.key === 'landscape' ? 1 : 0
  const n = layers.length - lead
  const cols = n % 3 === 0 ? 3 : 2
  return (
    <div
      className="segmented"
      role="radiogroup"
      aria-label="Map layer"
      style={{ gridTemplateColumns: `repeat(${lead ? 6 : cols}, minmax(0, 1fr))` }}
    >
      {layers.map((l, i) => {
        const span = i < lead ? 6 : 6 / cols
        const wide = i >= lead && cols === 2 && n % 2 === 1 && i === layers.length - 1
        return (
          <label
            key={l.key}
            className={i < lead || wide ? 'wide' : undefined}
            style={!lead || i < lead || wide ? undefined : { gridColumn: `span ${span}` }}
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
        )
      })}
    </div>
  )
}
