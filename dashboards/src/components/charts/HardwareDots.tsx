import { SERIES_COLOR, type Series } from './series'

export interface HardwareDotRow {
  key: string
  label: string
  series: Series
  /** Drawn open: a simulation, not a measurement. */
  hollow?: boolean
  /** Rows with the same group sit close together; a gap separates groups. */
  group: string
  value: number
}

const TOP = 8
const ROW_H = 46
const LABEL_CY = 9
const DOT_CY = 31
const GROUP_GAP = 14
const AXIS_H = 28
const PAD_LEFT = 8
const PAD_RIGHT = 48
const TICKS = [0, 0.25, 0.5, 0.75, 1]

/** Gutters around the 0 to 1 axis in px, for sizing it from the container width. */
export const HARDWARE_PAD = PAD_LEFT + PAD_RIGHT

interface Props {
  rows: HardwareDotRow[]
  /** Width of the 0 to 1 axis in px. */
  plotW: number
  label: string
  formatValue: (v: number) => string
}

/** Dot rows on a 0 to 1 axis; each name sits above its dot so long names fit at any width. */
export default function HardwareDots({ rows, plotW, label, formatValue }: Props) {
  const width = PAD_LEFT + plotW + PAD_RIGHT
  const x = (v: number) => PAD_LEFT + Math.max(0, Math.min(1, v)) * plotW

  // Each row starts after the rows above it, plus one gap per change of group.
  const gapsBefore = rows.map((_, i) => rows.slice(1, i + 1).filter((r, j) => rows[j].group !== r.group).length)
  const placed = rows.map((r, i) => ({ ...r, top: TOP + i * ROW_H + gapsBefore[i] * GROUP_GAP }))
  const axisY = TOP + rows.length * ROW_H + gapsBefore[rows.length - 1] * GROUP_GAP
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
            {t === 0 ? '0' : t === 1 ? '1' : String(t)}
          </text>
        </g>
      ))}
      <line x1={x(1)} x2={x(1)} y1={TOP} y2={axisY} className="ref" />
      {placed.map((r) => {
        const color = SERIES_COLOR[r.series]
        const cy = r.top + DOT_CY
        return (
          <g key={r.key}>
            <text x={0} y={r.top + LABEL_CY} dominantBaseline="central" className="ink halo">
              {r.label}
            </text>
            <line x1={x(r.value) + 8} x2={x(1)} y1={cy} y2={cy} className="lead" />
            <circle
              cx={x(r.value)}
              cy={cy}
              r="5"
              fill={r.hollow ? 'var(--c-page)' : color}
              stroke={r.hollow ? color : 'var(--c-page)'}
              strokeWidth="2"
            />
            <text x={PAD_LEFT + plotW + 8} y={cy} dominantBaseline="central" className="ink">
              {formatValue(r.value)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
