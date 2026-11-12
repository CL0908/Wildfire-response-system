import { SERIES_COLOR, type Series } from './series'
import { PANEL_PAD, stackRows } from './layout'

export interface DotRow {
  key: string
  label: string
  series: Series
  /** Drawn open: a state that is not a result of the series' method. */
  hollow?: boolean
  mean: number | null
  std: number | null
}

const TOP = 8
const ROW_H = 28
const GROUP_GAP = 12
const AXIS_H = 28
const TICKS = [0, 0.25, 0.5, 0.75, 1]

const tickText = (t: number) => (t === 0 ? '0' : t === 1 ? '1' : String(t))

interface Props {
  rows: DotRow[]
  /** Width of the 0 to 1 axis in px. */
  plotW: number
  /** Draw the method names in a gutter on the left. */
  labels: boolean
  label: string
  formatValue: (v: number) => string
}

export default function DotPanel({ rows, plotW, labels, label, formatValue }: Props) {
  const x0 = labels ? PANEL_PAD.label : PANEL_PAD.left
  const width = x0 + plotW + PANEL_PAD.right
  const x = (v: number) => x0 + Math.max(0, Math.min(1, v)) * plotW

  const stacked = stackRows(rows, TOP, ROW_H, GROUP_GAP)
  const placed = stacked.items.map((r) => ({ ...r, cy: r.top + ROW_H / 2 }))
  const axisY = stacked.end
  const height = axisY + AXIS_H

  return (
    <svg
      className="chart"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
    >
      {TICKS.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={TOP} y2={axisY} className="grid" />
          <text x={x(t)} y={axisY + 18} textAnchor="middle">
            {tickText(t)}
          </text>
        </g>
      ))}
      <line x1={x(1)} x2={x(1)} y1={TOP} y2={axisY} className="ref" />
      {placed.map((r) => {
        const color = SERIES_COLOR[r.series]
        const hasMean = r.mean !== null
        const lo = hasMean && r.std !== null ? (r.mean as number) - r.std : null
        const hi = hasMean && r.std !== null ? (r.mean as number) + r.std : null
        return (
          <g key={r.key}>
            {labels && (
              <text x={x0 - 12} y={r.cy} textAnchor="end" dominantBaseline="central" className="ink">
                {r.label}
              </text>
            )}
            {hasMean && lo !== null && hi !== null && r.std! > 0 && (
              <>
                <line x1={x(lo)} x2={x(hi)} y1={r.cy} y2={r.cy} stroke={color} strokeWidth="2" />
                {lo >= 0 && <line x1={x(lo)} x2={x(lo)} y1={r.cy - 4} y2={r.cy + 4} stroke={color} strokeWidth="2" />}
                {hi <= 1 && <line x1={x(hi)} x2={x(hi)} y1={r.cy - 4} y2={r.cy + 4} stroke={color} strokeWidth="2" />}
              </>
            )}
            {hasMean && (
              <circle
                cx={x(r.mean as number)}
                cy={r.cy}
                r="5"
                fill={r.hollow ? 'var(--c-page)' : color}
                stroke={r.hollow ? color : 'var(--c-page)'}
                strokeWidth="2"
              />
            )}
            <text x={x0 + plotW + 8} y={r.cy} dominantBaseline="central" className="ink">
              {hasMean ? formatValue(r.mean as number) : '–'}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
