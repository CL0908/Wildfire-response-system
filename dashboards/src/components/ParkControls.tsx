import { PARK_K_VALUES, PARK_RULES, type ParkK, type ParkMetrics, type ParkRule } from '../lib/park'

interface Option<K extends string | number> {
  key: K
  label: string
}

function Segmented<K extends string | number>({
  name,
  label,
  options,
  value,
  onChange,
}: {
  name: string
  label: string
  options: Option<K>[]
  value: K
  onChange: (key: K) => void
}) {
  return (
    <div className="card-section">
      <span className="card-label" aria-hidden="true">
        {label}
      </span>
      <div
        className="segmented"
        role="radiogroup"
        aria-label={label}
        style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
      >
        {options.map((o) => (
          <label key={o.key}>
            <input
              type="radio"
              name={name}
              className="sr-only"
              checked={value === o.key}
              onChange={() => onChange(o.key)}
            />
            <span>{o.label}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

interface Props {
  k: ParkK
  rule: ParkRule
  onK: (k: ParkK) => void
  onRule: (rule: ParkRule) => void
}

/** Sensor count and placement rule for the whole-park view. */
export function ParkControls({ k, rule, onK, onRule }: Props) {
  return (
    <>
      <Segmented
        name="park-k"
        label="Sensors"
        options={PARK_K_VALUES.map((v) => ({ key: v, label: String(v) }))}
        value={k}
        onChange={onK}
      />
      <Segmented name="park-rule" label="Placed by" options={PARK_RULES} value={rule} onChange={onRule} />
    </>
  )
}

const pct = (share: number) => `${Math.round(share * 100)}%`

/** The four numbers for the chosen placement. Priority is modelled, detection time simulated. */
export function ParkStats({ metrics }: { metrics: ParkMetrics }) {
  return (
    <div className="card-section">
      <dl className="stats stats-park">
        <div>
          <dt>Priority covered, modelled</dt>
          <dd>{pct(metrics.priority_share)}</dd>
        </div>
        <div>
          <dt>Park area covered</dt>
          <dd>{pct(metrics.area_share)}</dd>
        </div>
        <div>
          <dt>Past fires in view</dt>
          <dd>{pct(metrics.fire_share)}</dd>
        </div>
        <div>
          <dt>Time to detection, simulated</dt>
          <dd>{metrics.mean_detection_h.toFixed(1)} h</dd>
        </div>
      </dl>
    </div>
  )
}
