export type ViewMode = '2d' | '3d'

interface Props {
  value: ViewMode
  onChange: (mode: ViewMode) => void
}

const OPTIONS: { key: ViewMode; label: string }[] = [
  { key: '2d', label: '2D' },
  { key: '3d', label: '3D' },
]

export default function ViewSwitch({ value, onChange }: Props) {
  return (
    <div
      className="segmented view-switch"
      role="radiogroup"
      aria-label="Map view"
      style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}
    >
      {OPTIONS.map((o) => (
        <label key={o.key}>
          <input
            type="radio"
            name="view"
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
