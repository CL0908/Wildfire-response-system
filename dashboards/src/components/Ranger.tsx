import { useId, useState } from 'react'
import { rampColor } from '../lib/color'
import { formatInt } from '../lib/format'
import {
  FAILED,
  INK,
  RAMP_LOW,
  RANGER_METHODS,
  UNREACHED,
  arrivalColor,
  crewLetter,
  formatHM,
  type CrewRun,
  type CrewSummary,
  type Defence,
  type FeaturedScenario,
  type MethodRun,
  type RangerData,
  type RangerMethod,
} from '../lib/ranger'
import { useMediaQuery } from '../lib/useMediaQuery'

/** Cell size in SVG units; the SVG scales to its box. */
const CELL = 24

const METHOD_NAME: Record<RangerMethod, string> = {
  greedy: 'Greedy',
  lookahead: 'Lookahead',
  exact: 'Exact search',
}
const ROLE_NAME: Record<string, string> = {
  'most-fire-prone': 'Most fire-prone',
  'median-gap': 'Median gap',
  'largest-gap': 'Largest gap',
}
const ROLE_ORDER = ['most-fire-prone', 'median-gap', 'largest-gap']
const MOBILE = '(max-width: 719px)'

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`
const crewWord = (n: number) => (n === 1 ? 'one crew' : n === 2 ? 'two crews' : `${n} crews`)
const oneDecimal = (x: number) => x.toFixed(1)
const roleOf = (s: FeaturedScenario) => s.roles.map((r) => ROLE_NAME[r] ?? r).join(', ')
const isTypical = (s: FeaturedScenario) => !s.roles.includes('largest-gap')

function Segmented<T extends string | number>({
  name,
  label,
  value,
  options,
  onChange,
}: {
  name: string
  label: string
  value: T
  options: { key: T; label: string; flag?: string }[]
  onChange: (key: T) => void
}) {
  return (
    <div className="ranger-ctl">
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
            <span>
              {o.label}
              {o.flag && <small className="ranger-flag">{o.flag}</small>}
            </span>
          </label>
        ))}
      </div>
    </div>
  )
}

interface FigureProps {
  title: string
  arrival: (number | null)[]
  defences: Defence[]
  stations: [number, number][]
  ignition: [number, number]
  rows: number
  cols: number
  horizon: number
  burned: number
  /** Square window of cells to draw: top-left row and column, and side length. */
  crop: { r0: number; c0: number; side: number }
}

/** The smallest square of cells (plus a margin) holding every reached cell, firebreak and the ignition. */
function cropOf(cells: [number, number][], rows: number, cols: number) {
  const MARGIN = 2
  const MIN_SIDE = 10
  const rs = cells.map((c) => c[0])
  const cs = cells.map((c) => c[1])
  const rLo = Math.min(...rs) - MARGIN
  const rHi = Math.max(...rs) + MARGIN
  const cLo = Math.min(...cs) - MARGIN
  const cHi = Math.max(...cs) + MARGIN
  const side = Math.min(rows, cols, Math.max(MIN_SIDE, rHi - rLo + 1, cHi - cLo + 1))
  const place = (lo: number, hi: number, max: number) =>
    Math.max(0, Math.min(max - side, Math.round((lo + hi) / 2 - side / 2 + 0.01)))
  return { r0: place(rLo, rHi, rows), c0: place(cLo, cHi, cols), side }
}

function Figure({ title, arrival, defences, stations, ignition, rows, cols, horizon, burned, crop }: FigureProps) {
  const hatch = `hatch-${useId().replace(/:/g, '')}`
  const held = defences.filter((d) => d.held).length
  const label =
    `${title}. ${burned} of ${rows * cols} cells burn within ${horizon} hours. ` +
    `Zoomed to the ${crop.side} by ${crop.side} cells around the fire. ` +
    `Ignition at row ${ignition[0] + 1}, column ${ignition[1] + 1}, counted from the north-west corner. ` +
    (defences.length > 0
      ? `${plural(defences.length, 'firebreak', 'firebreaks')}, ${held} hold. The list below gives each one.`
      : 'No firebreaks.')
  const stationKeys = new Set(stations.map(([r, c]) => `${r},${c}`))
  const inCrop = (r: number, c: number) =>
    r >= crop.r0 && r < crop.r0 + crop.side && c >= crop.c0 && c < crop.c0 + crop.side
  const stationCells = [...stationKeys]
    .map((k) => k.split(',').map(Number) as [number, number])
    .filter(([r, c]) => inCrop(r, c))
  const ctr = (n: number) => n * CELL + CELL / 2
  return (
    <figure className="ranger-fig">
      <figcaption>{title}</figcaption>
      <svg
        className="ranger-svg"
        viewBox={`${crop.c0 * CELL} ${crop.r0 * CELL} ${crop.side * CELL} ${crop.side * CELL}`}
        role="img"
        aria-label={label}
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="#ffffff" fillOpacity="0.55" />
            <line x1="0" y1="0" x2="0" y2="6" stroke={FAILED} strokeWidth="2" />
          </pattern>
        </defs>
        {arrival.map((h, i) => {
          const r = Math.floor(i / cols)
          const c = i % cols
          const fill = arrivalColor(h, horizon)
          return <rect key={i} x={c * CELL} y={r * CELL} width={CELL} height={CELL} fill={fill} stroke="#ffffff" strokeWidth="0.75" />
        })}
        {defences.map((d) => {
          const [r, c] = d.cell
          const x = c * CELL
          const y = r * CELL
          const text = `${crewLetter(d.crew)}${d.order + 1}${d.held ? '' : '×'}`
          return (
            <g key={`${d.crew}-${d.order}`}>
              {d.held ? (
                <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill="none" stroke={INK} strokeWidth="3" />
              ) : (
                <>
                  <rect x={x + 1.5} y={y + 1.5} width={CELL - 3} height={CELL - 3} fill={`url(#${hatch})`} />
                  <rect
                    x={x + 1.5}
                    y={y + 1.5}
                    width={CELL - 3}
                    height={CELL - 3}
                    fill="none"
                    stroke={FAILED}
                    strokeWidth="3"
                    strokeDasharray="5 3"
                  />
                </>
              )}
              <rect x={x + 2} y={y + CELL / 2 - 7} width={CELL - 4} height="14" rx="2" fill={d.held ? INK : '#ffffff'} />
              <text
                className="ranger-chip"
                x={x + CELL / 2}
                y={y + CELL / 2 + 4}
                textAnchor="middle"
                fill={d.held ? '#ffffff' : FAILED}
              >
                {text}
              </text>
            </g>
          )
        })}
        {stationCells.map(([r, c]) => (
          <polygon
            key={`s${r}-${c}`}
            points={`${ctr(c)},${ctr(r) - 8} ${ctr(c) + 8},${ctr(r) + 7} ${ctr(c) - 8},${ctr(r) + 7}`}
            fill="#ffffff"
            stroke={INK}
            strokeWidth="2"
            strokeLinejoin="round"
          />
        ))}
        <polygon
          points={`${ctr(ignition[1])},${ctr(ignition[0]) - 9} ${ctr(ignition[1]) + 9},${ctr(ignition[0])} ${ctr(ignition[1])},${ctr(ignition[0]) + 9} ${ctr(ignition[1]) - 9},${ctr(ignition[0])}`}
          fill="var(--s-quantum)"
          stroke="#ffffff"
          strokeWidth="2"
        />
        <rect x={crop.c0 * CELL + 0.5} y={crop.r0 * CELL + 0.5} width={crop.side * CELL - 1} height={crop.side * CELL - 1} fill="none" stroke="var(--s-ref)" strokeWidth="1" />
      </svg>
    </figure>
  )
}

function Mark({ children }: { children: React.ReactNode }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      {children}
    </svg>
  )
}

function FigureKey({ horizon }: { horizon: number }) {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((f) => rampColor(RAMP_LOW + (1 - RAMP_LOW) * (1 - f))).join(', ')
  return (
    <div className="ranger-key">
      <div className="ranger-ramp">
        <span>Fire arrives</span>
        <span className="num">0 h</span>
        <i style={{ background: `linear-gradient(to right, ${stops})` }} aria-hidden="true" />
        <span className="num">{horizon} h</span>
      </div>
      <ul className="chart-legend">
        <li>
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <rect x="0.5" y="0.5" width="13" height="13" fill={UNREACHED} stroke="var(--s-ref)" />
          </svg>
          Not reached in {horizon} h
        </li>
        <li>
          <Mark>
            <polygon points="7,0.5 13.5,7 7,13.5 0.5,7" fill="var(--s-quantum)" stroke="#ffffff" strokeWidth="1" />
          </Mark>
          Ignition
        </li>
        <li>
          <Mark>
            <polygon points="7,1.5 13,12 1,12" fill="#ffffff" stroke={INK} strokeWidth="1.5" strokeLinejoin="round" />
          </Mark>
          Crew station
        </li>
        <li>
          <Mark>
            <rect x="1.5" y="1.5" width="11" height="11" fill="none" stroke={INK} strokeWidth="3" />
          </Mark>
          Firebreak that holds (crew letter, order)
        </li>
        <li>
          <Mark>
            <rect x="1.5" y="1.5" width="11" height="11" fill="#fbe9dd" stroke={FAILED} strokeWidth="2.5" strokeDasharray="3 2" />
          </Mark>
          Firebreak that fails (marked ×)
        </li>
      </ul>
    </div>
  )
}

function proofLine(method: RangerMethod, run: MethodRun): string {
  if (method !== 'exact') return `${METHOD_NAME[method]} is a fast rule, so there is no claim that this plan is the best possible.`
  if (run.proven_optimal) return 'The exact search finished: this plan is proven optimal in the model.'
  return `Best found, not proven optimal${run.stop_reason ? ` (the search stopped at its ${run.stop_reason})` : ''}.`
}

function DefenceTable({ defences, caption }: { defences: Defence[]; caption: string }) {
  const sorted = [...defences].sort((a, b) => a.crew - b.crew || a.order - b.order)
  return (
    <>
    <p className="method-note">{caption}</p>
    <div className="table-wrap" tabIndex={0} role="region" aria-label={caption}>
      <table className="hw-table">
        <thead>
          <tr>
            <th scope="col">Crew</th>
            <th scope="col" className="num">Order</th>
            <th scope="col">Cell (row, column)</th>
            <th scope="col" className="num">Crew arrives</th>
            <th scope="col" className="num">Break finished</th>
            <th scope="col" className="num">Fire arrives</th>
            <th scope="col">Result</th>
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 && (
            <tr>
              <td colSpan={7}>This plan cuts no firebreaks.</td>
            </tr>
          )}
          {sorted.map((d) => (
            <tr key={`${d.crew}-${d.order}`}>
              <th scope="row">{crewLetter(d.crew)}</th>
              <td className="num">{d.order + 1}</td>
              <td>
                {d.cell[0] + 1}, {d.cell[1] + 1}
              </td>
              <td className="num">{formatHM(d.arrive_h)}</td>
              <td className="num">{formatHM(d.finish_h)}</td>
              <td className="num">{d.fire_arrival_h === null ? 'not reached' : formatHM(d.fire_arrival_h)}</td>
              <td>{d.held ? 'Holds' : 'Fails (finished after the fire)'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
    </>
  )
}

function Explorer({ data }: { data: RangerData }) {
  const { rows, cols } = data.grid
  const horizon = Number(data.parameters.horizon_h?.value) || 24
  const scenarios = [...data.featured].sort(
    (a, b) =>
      Math.min(...a.roles.map((r) => ROLE_ORDER.indexOf(r)).map((i) => (i < 0 ? 9 : i))) -
      Math.min(...b.roles.map((r) => ROLE_ORDER.indexOf(r)).map((i) => (i < 0 ? 9 : i))),
  )
  const firstTypical = scenarios.find((s) => s.roles.includes('median-gap')) ?? scenarios[0]
  const [scenarioId, setScenarioId] = useState(firstTypical.scenario)
  const [crewsPick, setCrewsPick] = useState(1)
  const [methodPick, setMethodPick] = useState<RangerMethod>('exact')
  const [viewPick, setViewPick] = useState<'plan' | 'none'>('plan')
  const mobile = useMediaQuery(MOBILE)

  const scenario = scenarios.find((s) => s.scenario === scenarioId) ?? scenarios[0]
  const runs = [...scenario.runs].sort((a, b) => a.crews - b.crews)
  const run: CrewRun = runs.find((r) => r.crews === crewsPick) ?? runs[0]
  const present = RANGER_METHODS.filter((m) => run.methods[m])
  const method = present.includes(methodPick) ? methodPick : present[present.length - 1]
  const result = run.methods[method]!
  const showPlan = !mobile || viewPick === 'plan'
  const showNone = !mobile || viewPick === 'none'
  const total = rows * cols
  const fireCells: [number, number][] = [scenario.ignition, ...result.defences.map((d) => d.cell)]
  for (const arr of [run.baseline.arrival_h, result.arrival_h]) {
    arr.forEach((h, i) => {
      if (h !== null) fireCells.push([Math.floor(i / cols), i % cols])
    })
  }
  const crop = cropOf(fireCells, rows, cols)
  const stationShown = run.stations.some(([r, c]) => r >= crop.r0 && r < crop.r0 + crop.side && c >= crop.c0 && c < crop.c0 + crop.side)
  const common = { stations: run.stations, ignition: scenario.ignition, rows, cols, horizon, crop }
  const saved = run.baseline.cells_burned - result.cells_burned

  return (
    <div className="ranger-explorer">
      <div className="ranger-controls">
        <Segmented
          name="ranger-scenario"
          label="Fire scenario"
          value={scenario.scenario}
          options={scenarios.map((s) => ({
            key: s.scenario,
            label: roleOf(s),
            flag: isTypical(s) ? undefined : 'not typical',
          }))}
          onChange={setScenarioId}
        />
        <Segmented
          name="ranger-crews"
          label="Crews"
          value={run.crews}
          options={runs.map((r) => ({ key: r.crews, label: String(r.crews) }))}
          onChange={setCrewsPick}
        />
        <Segmented
          name="ranger-method"
          label="Method"
          value={method}
          options={present.map((m) => ({ key: m, label: METHOD_NAME[m] }))}
          onChange={setMethodPick}
        />
        {mobile && (
          <Segmented
            name="ranger-view"
            label="Map shows"
            value={viewPick}
            options={[
              { key: 'plan', label: 'With plan' },
              { key: 'none', label: 'No response' },
            ]}
            onChange={setViewPick}
          />
        )}
      </div>
      <p className="method-note">
        Scenario {scenario.scenario}: ignition at row {scenario.ignition[0] + 1}, column {scenario.ignition[1] + 1}; wind{' '}
        {scenario.wind_speed_ms.toFixed(1)} m/s from {Math.round(scenario.wind_from_deg)}°; {crewWord(run.crews)} start
        {run.crews === 1 ? 's' : ''} at the window edge.{' '}
        {!isTypical(scenario) &&
          'This scenario has the largest gap between greedy and exact of the twenty. It is shown so the top of the range is visible, and it is not typical. '}
        Rows and columns count from the north-west corner. The maps are zoomed to the {crop.side} by {crop.side} cells around the fire (the window is{' '}
        {rows} by {cols}); north is up.
        {!stationShown && ' The crew station, at row ' + (run.stations[0][0] + 1) + ', column ' + (run.stations[0][1] + 1) + ', is outside the zoomed area.'}
      </p>
      <FigureKey horizon={horizon} />
      <div className="ranger-figs">
        {showNone && (
          <Figure
            title="No response"
            arrival={run.baseline.arrival_h}
            defences={[]}
            burned={run.baseline.cells_burned}
            {...common}
          />
        )}
        {showPlan && (
          <Figure
            title={`With plan: ${METHOD_NAME[method].toLowerCase()}, ${crewWord(run.crews)}`}
            arrival={result.arrival_h}
            defences={result.defences}
            burned={result.cells_burned}
            {...common}
          />
        )}
      </div>
      <dl className="figures">
        <div>
          <dt>Cells burned within {horizon} h</dt>
          <dd>
            {result.cells_burned} <span className="of">of {total} (no response: {run.baseline.cells_burned})</span>
          </dd>
        </div>
        <div>
          <dt>Value saved, share of what would burn</dt>
          <dd>
            {oneDecimal(result.pct_saved)}%{' '}
            <span className="of">
              ({result.value_saved.toFixed(2)} of {run.baseline.value_burned.toFixed(2)})
            </span>
          </dd>
        </div>
        <div>
          <dt>Firebreaks that hold</dt>
          <dd>
            {result.defences.filter((d) => d.held).length}{' '}
            <span className="of">of {result.defences.length}</span>
          </dd>
        </div>
      </dl>
      <p className="method-note">
        {proofLine(method, result)} {saved > 0 ? `${plural(saved, 'cell', 'cells')} fewer burn than with no response.` : 'No cell is saved.'}
      </p>
      <DefenceTable
        defences={result.defences}
        caption={`Firebreaks in this plan: ${METHOD_NAME[method].toLowerCase()}, ${crewWord(run.crews)}, scenario ${scenario.scenario}. Times are hours:minutes after ignition.`}
      />
    </div>
  )
}

const crewKeys = (s: Record<string, CrewSummary>) => Object.keys(s).sort((a, b) => Number(a) - Number(b))

function AggregateTables({ data }: { data: RangerData }) {
  const { summary, n_scenarios: n } = data.experiment
  const keys = crewKeys(summary)
  const gapPairs: [string, string][] = [
    ['exact', 'greedy'],
    ['exact', 'lookahead'],
  ]
  const unproven = keys.filter((k) => {
    const p = summary[k].methods.exact?.proven_optimal
    return p !== null && p !== undefined && p < summary[k].n
  })
  const base = summary[keys[0]].baseline
  return (
    <>
      <div className="table-wrap" tabIndex={0} role="region" aria-label={`Value saved, all ${n} scenarios`}>
        <table className="hw-table">
          <caption>
            Value saved as a share of the value that burns with no response, all {n} scenarios
          </caption>
          <thead>
            <tr>
              <th scope="col">Crews and method</th>
              <th scope="col" className="num">Mean saved</th>
              <th scope="col" className="num">Median saved</th>
              <th scope="col">Proven optimal</th>
            </tr>
          </thead>
          <tbody>
            {keys.flatMap((k) =>
              RANGER_METHODS.filter((m) => summary[k].methods[m]).map((m) => {
                const s = summary[k].methods[m]!
                const p = s.proven_optimal
                const partial = m === 'exact' && p !== null && p < summary[k].n
                return (
                  <tr key={`${k}-${m}`}>
                    <th scope="row">
                      {k} {Number(k) === 1 ? 'crew' : 'crews'}, {METHOD_NAME[m].toLowerCase()}
                      {partial && <span className="muted"> (best found, not proven optimal in {summary[k].n - p!})</span>}
                    </th>
                    <td className="num">{oneDecimal(s.mean_pct_saved)}%</td>
                    <td className="num">{oneDecimal(s.median_pct_saved)}%</td>
                    <td>{p === null ? 'no claim (fast rule)' : `${p} of ${summary[k].n}`}</td>
                  </tr>
                )
              }),
            )}
          </tbody>
        </table>
      </div>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Gaps between methods, in percentage points">
        <table className="hw-table">
          <caption>Gap between methods, in percentage points of the value that burns (scenarios where exact is better, equal, worse)</caption>
          <thead>
            <tr>
              <th scope="col">Crews and comparison</th>
              <th scope="col" className="num">Mean gap</th>
              <th scope="col" className="num">Median gap</th>
              <th scope="col" className="num">Better / equal / worse</th>
            </tr>
          </thead>
          <tbody>
            {keys.flatMap((k) =>
              gapPairs
                .filter(([a, b]) => summary[k].gaps[`${a} - ${b}`])
                .map(([a, b]) => {
                  const g = summary[k].gaps[`${a} - ${b}`]
                  return (
                    <tr key={`${k}-${a}-${b}`}>
                      <th scope="row">
                        {k} {Number(k) === 1 ? 'crew' : 'crews'}, {a} minus {b}
                      </th>
                      <td className="num">{oneDecimal(g.mean_pct_points)} points</td>
                      <td className="num">{oneDecimal(g.median_pct_points)} points</td>
                      <td className="num">
                        {g.n_positive} / {g.n_zero} / {g.n_negative}
                      </td>
                    </tr>
                  )
                }),
            )}
          </tbody>
        </table>
      </div>
      <p className="method-note">
        With no response a fire burns {oneDecimal(base.mean_cells_burned)} cells on average within the horizon, carrying{' '}
        {base.mean_value_burned.toFixed(1)} of the window&rsquo;s {base.mean_total_value.toFixed(1)} priority value; the percentages above are of
        that burning value.
        {unproven.length > 0 &&
          ` With ${unproven.map((k) => crewWord(Number(k))).join(' and ')} the exact search ran out of time in some scenarios. Those results are the best plan found, so the true gaps are at least as large as shown.`}
      </p>
    </>
  )
}

function Sensitivity({ data }: { data: RangerData }) {
  const entries = Object.entries(data.experiment.sensitivity)
  if (entries.length === 0) return null
  return (
    <details>
      <summary>Show how the result changes with the model&rsquo;s weakest inputs</summary>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Sensitivity of value saved to crew cutting rate and spread rate">
        <table className="hw-table">
          <caption>Mean value saved (% of the burning value), same {data.experiment.n_scenarios} scenarios, one input changed at a time</caption>
          <thead>
            <tr>
              <th scope="col">Input</th>
              <th scope="col" className="num">Crews</th>
              <th scope="col" className="num">Greedy</th>
              <th scope="col" className="num">Lookahead</th>
              <th scope="col" className="num">Exact (best found)</th>
              <th scope="col" className="num">Exact proven</th>
            </tr>
          </thead>
          <tbody>
            {entries.flatMap(([name, byCrews]) =>
              crewKeys(byCrews).map((k) => {
                const m = byCrews[k].methods
                const cell = (x?: { mean_pct_saved: number }) => (x ? `${oneDecimal(x.mean_pct_saved)}%` : 'n/a')
                return (
                  <tr key={`${name}-${k}`}>
                    <th scope="row">{name.startsWith('default') ? 'Default settings' : name}</th>
                    <td className="num">{k}</td>
                    <td className="num">{cell(m.greedy)}</td>
                    <td className="num">{cell(m.lookahead)}</td>
                    <td className="num">{cell(m.exact)}</td>
                    <td className="num">{m.exact?.proven_optimal == null ? 'n/a' : `${m.exact.proven_optimal} of ${byCrews[k].n}`}</td>
                  </tr>
                )
              }),
            )}
          </tbody>
        </table>
      </div>
    </details>
  )
}

function headlines(data: RangerData): { headline: string; lede: string } {
  const { summary, n_scenarios: n } = data.experiment
  const keys = crewKeys(summary)
  const horizon = Number(data.parameters.horizon_h?.value) || 24
  const one = summary['1'] ?? summary[keys[0]]
  const crewsOne = Number(summary['1'] ? 1 : keys[0])
  const present = RANGER_METHODS.filter((m) => one.methods[m])
  const best = present[present.length - 1]
  const bestStats = one.methods[best]!
  const gap = one.gaps[`${best} - greedy`]
  const head = `In the model, ${crewWord(crewsOne)} working from the park edge keeps ${oneDecimal(bestStats.mean_pct_saved)}% of the value that would burn in the first ${horizon} hours`
  const headline =
    best !== 'greedy' && gap
      ? `${head}; the best plan found beats a simple greedy plan by ${oneDecimal(gap.mean_pct_points)} points on average.`
      : `${head}.`

  const closes = (k: string): string | null => {
    const s = summary[k]
    const full = s?.gaps['exact - greedy']
    const part = s?.gaps['lookahead - greedy']
    if (!full || !part || full.mean_pct_points <= 0) return null
    return `${Math.round((part.mean_pct_points / full.mean_pct_points) * 100)}% of the ${Number(k) === 1 ? 'one-crew' : Number(k) === 2 ? 'two-crew' : `${k}-crew`} gap`
  }
  const look = one.methods.lookahead
  const closed = keys.map(closes).filter((x): x is string => x !== null)
  const two = summary['2']
  const twoBest = two?.methods.exact ?? (two ? two.methods[RANGER_METHODS.filter((m) => two.methods[m]).pop()!] : undefined)
  const unproven = two?.methods.exact && two.methods.exact.proven_optimal !== null && two.methods.exact.proven_optimal < two.n
  const parts: string[] = []
  if (look && closed.length > 0) {
    parts.push(
      `A cheap two-step lookahead rule reaches ${oneDecimal(look.mean_pct_saved)}% and closes ${closed.join(' and ')}.`,
    )
  }
  if (twoBest && two) {
    parts.push(
      `With two crews the best plan found keeps ${oneDecimal(twoBest.mean_pct_saved)}%${
        unproven && two.methods.exact
          ? `, but the exact search proved its plan optimal in only ${two.methods.exact.proven_optimal} of ${two.n} scenarios, so the best possible may be higher`
          : ''
      }.`,
    )
  }
  parts.push(`These are model results for ${n} sampled fires, not a forecast, and nothing here is checked against a real fire.`)
  return { headline, lede: parts.join(' ') }
}

function paramLine(data: RangerData): string[] {
  const p = data.parameters
  const lines: string[] = []
  const cut = p.cut_rate_m_min_per_person
  const hPerCell = p.cut_h_per_cell
  const size = p.crew_size
  if (cut && hPerCell && size) {
    lines.push(
      `Cutting a ${formatInt(data.grid.cell_size_m)} m cell takes ${Number(hPerCell.value).toFixed(1)} h: ${size.value} people at ${cut.value} ${cut.unit} each (crew size is an assumption). Source: ${cut.source}.`,
    )
  }
  const walk = p.walk_kmh
  if (walk) lines.push(`Walking speed is ${walk.value} ${walk.unit}, an assumption: ${walk.source.replace(/^assumption\s*/, '').replace(/^\(|\)$/g, '')}.`)
  return lines
}

export default function Ranger({ data }: { data: RangerData }) {
  const { headline, lede } = headlines(data)
  const lines = paramLine(data)
  return (
    <section id="ranger" className="ranger" aria-labelledby="ranger-title">
      <div className="sec-head">
        <p className="ranger-kicker">Ranger response (modelled)</p>
        <h2 id="ranger-title">{headline}</h2>
        <p className="lede">{lede}</p>
      </div>
      <Explorer data={data} />
      <div className="chart-block">
        <h3>All {data.experiment.n_scenarios} scenarios</h3>
        <AggregateTables data={data} />
      </div>
      <Sensitivity data={data} />
      <div className="chart-block" id="ranger-caveats">
        <h3>What this model leaves out</h3>
        <ul className="limits">
          {data.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
        {lines.map((l) => (
          <p key={l} className="method-note">
            {l}
          </p>
        ))}
      </div>
    </section>
  )
}
