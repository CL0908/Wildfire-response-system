import { useWidth } from './useWidth'
import { SERIES_COLOR, type Series } from './series'
import { stackRows } from './layout'

export interface BarRow {
  name: string
  series: Series
  /** The bar: a fraction of the optimum. */
  value: number
  /** Open marker on the same row: a fraction of the optimum (QAOA: best of the shots). */
  best?: number
  selected: boolean
}

interface Props {
  rows: BarRow[]
  /** Text for the marker's value labels, e.g. "best of 2,000 shots". */
  bestTitle: string
  label: string
  formatValue: (fraction: number) => string
}

const LABEL_W = 96
const PAD_R = 60
const TOP = 28
const ROW_H = 36
const BAR_H = 16
const GROUP_GAP = 16
const AXIS_H = 28
const TICKS = [0, 0.25, 0.5, 0.75, 1]
const CHAR_W = 7.4

/** Bar with a 4 px rounded end, square at the baseline. */
function barPath(x: number, y: number, w: number, h: number): string {
  const r = Math.min(4, w / 2, h / 2)
  return `M${x} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h - r}Q${x + w} ${y + h} ${x + w - r} ${y + h}H${x}Z`
}

export default function MethodBars({ rows, bestTitle, label, formatValue }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const x0 = LABEL_W
  const x1 = Math.max(x0 + 80, width - PAD_R)
  const plotW = x1 - x0
  const x = (v: number) => x0 + Math.max(0, Math.min(1, v)) * plotW

  const stacked = stackRows(rows, TOP, ROW_H, GROUP_GAP)
  const placed = stacked.items.map((r) => {
    const top = r.top + (ROW_H - BAR_H) / 2
    return { ...r, top, cy: top + BAR_H / 2 }
  })
  const axisY = stacked.end
  const height = axisY + AXIS_H

  return (
    <div ref={ref} className="chart-box">
      {width > 0 && (
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
              <line x1={x(t)} x2={x(t)} y1={TOP - 4} y2={axisY} className="grid" />
              <text x={x(t)} y={axisY + 18} textAnchor="middle">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          <line x1={x1} x2={x1} y1={TOP - 8} y2={axisY} className="ref" />
          <text x={x1} y={12} textAnchor="middle">
            optimum
          </text>
          {placed.map((r) => {
            const color = SERIES_COLOR[r.series]
            const barEnd = x(r.value)
            const barW = Math.max(0, barEnd - x0)
            const valueText = formatValue(r.value)
            const tw = valueText.length * CHAR_W
            const limit = r.best !== undefined ? x(r.best) - 5 - 8 : x1 - 6
            let valueX = barEnd + 8
            let inside = false
            let anchor: 'start' | 'end' = 'start'
            if (valueX + tw > limit) {
              if (r.best !== undefined && barW >= tw + 12) {
                inside = true
                anchor = 'end'
                valueX = barEnd - 6
              } else if (r.best === undefined) {
                valueX = x1 + 12
              }
            }
            const bestText = r.best !== undefined ? formatValue(r.best) : ''
            let bestX = r.best !== undefined ? x(r.best) + 9 : 0
            if (r.best !== undefined && bestX + bestText.length * CHAR_W > x1 - 4) bestX = x1 + 12
            return (
              <g key={r.name}>
                <title>
                  {r.best !== undefined
                    ? `${r.name}: average shot ${valueText}; ${bestTitle} ${bestText}`
                    : `${r.name}: ${valueText}`}
                </title>
                <text
                  x={x0 - 12}
                  y={r.cy}
                  textAnchor="end"
                  dominantBaseline="central"
                  className={r.selected ? 'strong' : 'ink'}
                >
                  {r.name}
                </text>
                <path d={barPath(x0, r.top, barW, BAR_H)} fill={color} opacity={r.selected ? 1 : 0.6} />
                {r.best !== undefined && (
                  <>
                    <line
                      opacity={r.selected ? 1 : 0.6}
                      x1={inside ? barEnd : valueX + tw + 4}
                      x2={x(r.best) - 5}
                      y1={r.cy}
                      y2={r.cy}
                      stroke={color}
                      strokeWidth="1"
                      strokeDasharray="2 3"
                    />
                    <circle
                      cx={x(r.best)}
                      cy={r.cy}
                      r="5"
                      fill="var(--c-page)"
                      stroke={color}
                      strokeWidth="2"
                      opacity={r.selected ? 1 : 0.6}
                    />
                    <text x={bestX} y={r.cy} dominantBaseline="central" className="ink">
                      {bestText}
                    </text>
                  </>
                )}
                <text
                  x={valueX}
                  y={r.cy}
                  textAnchor={anchor}
                  dominantBaseline="central"
                  className={inside && r.selected ? 'on-bar' : 'ink'}
                >
                  {valueText}
                </text>
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
