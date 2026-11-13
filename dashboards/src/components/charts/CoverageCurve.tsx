import type { ParkCurvePoint } from '../../lib/park'
import { useWidth } from './useWidth'

export interface CurveLine {
  key: 'priority_share' | 'area_share' | 'fire_share'
  label: string
  color: string
  dash?: string
}

interface Props {
  curve: ParkCurvePoint[]
  lines: CurveLine[]
  /** Sensor counts to mark with a guide and a dot on every line. */
  marked: number[]
  label: string
}

const PAD = { left: 44, right: 14, top: 12, bottom: 52 }
const PLOT_H = 220
const XTICK = 10
const YTICKS = [0, 0.25, 0.5, 0.75, 1]

/** Line chart of shares (0 to 1) against the number of sensors, hand-drawn SVG. */
export default function CoverageCurve({ curve, lines, marked, label }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const kMax = curve[curve.length - 1].k
  const x0 = PAD.left
  const x1 = Math.max(x0 + 120, width - PAD.right)
  const y0 = PAD.top + PLOT_H
  const x = (k: number) => x0 + (k / kMax) * (x1 - x0)
  const y = (v: number) => y0 - Math.max(0, Math.min(1, v)) * PLOT_H
  const height = y0 + PAD.bottom
  const xTicks: number[] = []
  for (let k = 0; k <= kMax; k += XTICK) xTicks.push(k)
  const points = curve.filter((c) => marked.includes(c.k))

  return (
    <div ref={ref} className="chart-box">
      {width > 0 && (
        <svg className="chart" width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
          {YTICKS.map((t) => (
            <g key={t}>
              <line x1={x0} x2={x1} y1={y(t)} y2={y(t)} className="grid" />
              <text x={x0 - 8} y={y(t)} textAnchor="end" dominantBaseline="central">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {marked.map((k) => (
            <line key={k} x1={x(k)} x2={x(k)} y1={PAD.top} y2={y0} className="lead" />
          ))}
          {xTicks.map((k) => (
            <text key={k} x={x(k)} y={y0 + 18} textAnchor="middle" className={marked.includes(k) ? 'strong' : undefined}>
              {k}
            </text>
          ))}
          <text x={(x0 + x1) / 2} y={y0 + 42} textAnchor="middle">
            Sensors
          </text>
          {lines.map((l) => (
            <polyline
              key={l.key}
              points={curve.map((c) => `${x(c.k).toFixed(1)},${y(c[l.key]).toFixed(1)}`).join(' ')}
              fill="none"
              stroke={l.color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeDasharray={l.dash}
            />
          ))}
          {lines.map((l) =>
            points.map((c) => (
              <circle key={`${l.key}-${c.k}`} cx={x(c.k)} cy={y(c[l.key])} r="4" fill={l.color} stroke="var(--c-page)" strokeWidth="1.5">
                <title>{`${l.label}, ${c.k} sensors: ${Math.round(c[l.key] * 100)}%`}</title>
              </circle>
            )),
          )}
        </svg>
      )}
    </div>
  )
}
