import type { Scale } from '../lib/scale'

interface Props {
  value: Scale
  onChange: (scale: Scale) => void
}

const OPTIONS: { key: Scale; label: string }[] = [
  { key: 'window', label: '10 km window' },
  { key: 'park', label: 'Whole park' },
]

export default function ScaleSwitch({ value, onChange }: Props) {
  return (
    <div
      className="segmented"
      role="radiogroup"
      aria-label="Map scale"
      style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
    >
      {OPTIONS.map((o) => (
        <label key={o.key}>
          <input
            type="radio"
            name="scale"
            className="sr-only"
            checked={value === o.key}
            onChange={() => onChange(o.key)}
          />
          <span>{o.label}</span>
        </label>
      ))}
    </div>
  )
}
