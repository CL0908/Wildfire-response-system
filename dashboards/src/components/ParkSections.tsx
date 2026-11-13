import { formatRange } from '../lib/format'
import {
  PARK_K_VALUES,
  PARK_RULES,
  RULE_LABEL,
  parkPlacement,
  type ParkData,
  type ParkK,
  type ParkRule,
  type ParkState,
} from '../lib/park'
import CoverageCurve, { type CurveLine } from './charts/CoverageCurve'

const pct = (share: number) => `${Math.round(share * 100)}%`
const hours = (h: number) => h.toFixed(1)
const sensors = (k: number) => `${k} sensor${k === 1 ? '' : 's'}`

const LINES: CurveLine[] = [
  { key: 'priority_share', label: 'Priority, modelled', color: 'var(--c-ink)' },
  { key: 'area_share', label: 'Park area', color: 'var(--s-park-area)', dash: '6 4' },
  { key: 'fire_share', label: 'Past fires, 2012 to 2025', color: 'var(--s-park-fire)' },
]

interface SectionProps {
  park: ParkData
  k: ParkK
  rule: ParkRule
}

function PlacementSection({ park, k, rule }: SectionProps) {
  const m = parkPlacement(park, k, rule).metrics
  const detection = [
    { label: 'Seen at ignition', value: m.share_at_ignition },
    { label: 'Seen after spread', value: m.share_after_spread },
    { label: 'Left to the satellite', value: m.share_satellite },
  ]
  return (
    <section aria-labelledby="park-placement-title">
      <div className="sec-head">
        <h2 id="park-placement-title">
          {`${sensors(k)} placed by ${RULE_LABEL[rule]} have ${pct(m.fire_share)} of past fires in view and cover ${pct(m.area_share)} of the park.`}
        </h2>
      </div>
      <div className="chart-block">
        <h3>Where a simulated fire is first detected</h3>
        <dl className="figures">
          {detection.map((d) => (
            <div key={d.label}>
              <dt>{d.label}</dt>
              <dd>{pct(d.value)}</dd>
            </div>
          ))}
        </dl>
        <div
          className="share-bar"
          role="img"
          aria-label={`Simulated fires: ${detection.map((d) => `${d.label.toLowerCase()} ${pct(d.value)}`).join(', ')}.`}
        >
          {detection.map((d, i) => (
            <span key={d.label} className={`share-${i}`} style={{ flexGrow: d.value }} />
          ))}
        </div>
      </div>
      <p className="method-note">
        Past fires are the real 2012 to 2025 satellite detections inside the park; for fire history they are also what the sites were fitted to. Detection times come
        from 2,000 simulated fires, with the satellite assumed to detect every fire 6 hours after ignition. Mean time to detection with these sensors is{' '}
        {hours(m.mean_detection_h)} h, simulated.
      </p>
    </section>
  )
}

function CurveSection({ park }: { park: ParkData }) {
  const curve = park.coverage_curve
  const last = curve[curve.length - 1]
  const half = curve.find((c) => c.priority_share >= 0.5)
  const headline = half
    ? `It takes ${half.k} sensors to put half of the park’s modelled priority in view, and ${last.k} sensors reach ${pct(last.priority_share)}.`
    : `${last.k} sensors put ${pct(last.priority_share)} of the park’s modelled priority in view.`
  const ks = PARK_K_VALUES.filter((v) => curve.some((c) => c.k === v))
  const rows = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60].map((n) => curve.find((c) => c.k === n)).filter((c) => c !== undefined)
  const label = `Share of the park in view against the number of sensors, placed one at a time by priority coverage. With ${last.k} sensors: priority ${pct(last.priority_share)}, park area ${pct(last.area_share)}, past fires ${pct(last.fire_share)}.`
  return (
    <section aria-labelledby="park-curve-title">
      <div className="sec-head">
        <h2 id="park-curve-title">{headline}</h2>
      </div>
      <div className="chart-block">
        <ul className="chart-legend">
          {LINES.map((l) => (
            <li key={l.key}>
              <svg width="20" height="12" viewBox="0 0 20 12" aria-hidden="true">
                <line x1="1" x2="19" y1="6" y2="6" stroke={l.color} strokeWidth="2" strokeDasharray={l.dash} />
              </svg>
              {l.label}
            </li>
          ))}
        </ul>
        <CoverageCurve curve={curve} lines={LINES} marked={ks} label={label} />
        <div className="sr-only">
          <table>
            <caption>Share of the park in view for the priority coverage placement, by number of sensors</caption>
            <thead>
              <tr>
                <th scope="col">Sensors</th>
                <th scope="col">Priority, modelled</th>
                <th scope="col">Park area</th>
                <th scope="col">Past fires</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((c) => (
                <tr key={c.k}>
                  <th scope="row">{c.k}</th>
                  <td>{pct(c.priority_share)}</td>
                  <td>{pct(c.area_share)}</td>
                  <td>{pct(c.fire_share)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="method-note">
        Each sensor is added where it brings the most new priority into view; the marked counts are the sensor counts in the table below. Visibility is modelled from terrain;
        the fire detections are real.
      </p>
    </section>
  )
}

/** What the data says about time to detection against fire history, per sensor count. */
function finding(park: ParkData): string {
  const diffs = PARK_K_VALUES.map((k) => ({
    k,
    d: parkPlacement(park, k, 'timed_greedy').metrics.mean_detection_h - parkPlacement(park, k, 'fire_history').metrics.mean_detection_h,
  }))
  const later = diffs.filter((x) => x.d >= 0.05)
  const earlier = diffs.filter((x) => x.d <= -0.05)
  const same = diffs.filter((x) => Math.abs(x.d) < 0.05)
  const list = (xs: typeof diffs) => xs.map((x) => x.k).join(', ').replace(/, (\d+)$/, ' and $1')
  const range = (xs: typeof diffs) => {
    const v = xs.map((x) => Math.abs(x.d))
    const lo = hours(Math.min(...v))
    const hi = hours(Math.max(...v))
    return lo === hi ? `${lo} h` : `${lo} to ${hi} h`
  }
  if (earlier.length === 0 && same.length === 0) {
    return `Placing sensors by time to detection is not earlier than placing them by fire history: its simulated mean time to detection is ${range(later)} later at every sensor count in the default setting.`
  }
  const parts: string[] = []
  if (earlier.length) parts.push(`earlier by ${range(earlier)} at ${list(earlier)} sensors`)
  if (same.length) parts.push(`within 0.1 h at ${list(same)} sensors`)
  if (later.length) parts.push(`later by ${range(later)} at ${list(later)} sensors`)
  return `Placing sensors by time to detection against placing them by fire history gives a simulated mean time to detection that is ${parts.join(', ')}.`
}

function TableSection({ park, k, rule, onSelect }: SectionProps & { onSelect: (k: ParkK, rule: ParkRule) => void }) {
  return (
    <section aria-labelledby="park-table-title">
      <div className="sec-head">
        <h2 id="park-table-title">{finding(park)}</h2>
      </div>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Placement comparison table">
        <table className="park-table">
          <thead>
            <tr>
              <th scope="col">Placed by</th>
              <th scope="col" className="num">
                Sensors
              </th>
              <th scope="col" className="num">
                Priority, modelled
              </th>
              <th scope="col" className="num">
                Park area
              </th>
              <th scope="col" className="num">
                Past fires in view
              </th>
              <th scope="col" className="num">
                Time to detection, simulated
              </th>
            </tr>
          </thead>
          <tbody>
            {PARK_K_VALUES.flatMap((kk) =>
              PARK_RULES.map(({ key, label }) => {
                const m = parkPlacement(park, kk, key).metrics
                const selected = kk === k && key === rule
                return (
                  <tr key={`${kk}-${key}`} className={selected ? 'is-selected' : undefined}>
                    <th scope="row">
                      <button type="button" className="row-pick" aria-pressed={selected} onClick={() => onSelect(kk, key)}>
                        {label}
                      </button>
                    </th>
                    <td className="num">{kk}</td>
                    <td className="num">{pct(m.priority_share)}</td>
                    <td className="num">{pct(m.area_share)}</td>
                    <td className="num">{pct(m.fire_share)}</td>
                    <td className="num">{hours(m.mean_detection_h)} h</td>
                  </tr>
                )
              }),
            )}
          </tbody>
        </table>
      </div>
      <p className="method-note">
        Select a row to show that placement on the map. At 20 sensors, over nine settings of spread rate and satellite delay, fire history detected earlier in 6, time
        to detection in 1, and 2 fell inside the bootstrap interval. Both rules are chosen and scored on the same fire-history distribution, so
        neither result is a test on independent fires.
      </p>
    </section>
  )
}

function LimitsSection({ park }: { park: ParkData }) {
  const { mast_height_m, range_m } = park.meta
  return (
    <section aria-labelledby="park-limits-title">
      <div className="sec-head">
        <h2 id="park-limits-title">Limits</h2>
      </div>
      <ul className="limits">
        <li>Every in-park cell is treated as buildable. Road access, land tenure, power and maintenance are ignored, so this is not a siting plan.</li>
        <li>
          Visibility is terrain line of sight from a {mast_height_m} m mast with a {formatRange(range_m)} range. It has no tree-canopy height, smoke or weather.
        </li>
        <li>
          Fire spread is a simplified model that has not been calibrated or checked against any real fire. Its spread rate comes from deciduous-forest burns elsewhere
          in Thailand.
        </li>
        <li>One wind value per simulated fire applies to the whole map.</li>
        <li>The 6 hour satellite delay is an assumption, not a measurement. A sensor is assumed to detect a fire the moment it enters a cell it sees, which favours sensors.</li>
        <li>
          Ignitions are drawn only inside the park. About 12 times more satellite detections lie in the margin around it, and fires spreading in from there
          are not simulated.
        </li>
      </ul>
    </section>
  )
}

interface Props {
  state: ParkState
  k: ParkK
  rule: ParkRule
  onSelect: (k: ParkK, rule: ParkRule) => void
}

/** The sections below the stage when the view is the whole park. */
export default function ParkSections({ state, k, rule, onSelect }: Props) {
  return (
    <main id="results" className="results" tabIndex={-1}>
      {state.status === 'ready' ? (
        <>
          <PlacementSection park={state.park} k={k} rule={rule} />
          <CurveSection park={state.park} />
          <TableSection park={state.park} k={k} rule={rule} onSelect={onSelect} />
          <LimitsSection park={state.park} />
        </>
      ) : (
        <section aria-label="Whole-park results">
          <p role={state.status === 'error' ? 'alert' : 'status'} className="muted">
            {state.status === 'error' ? state.message : 'Loading placements…'}
          </p>
        </section>
      )}
    </main>
  )
}
