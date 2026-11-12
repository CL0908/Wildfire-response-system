const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const MONTHS_LONG = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

const HEAD_H = 24
const ROW_H = 26
const LABEL_W = 36
const TOTAL_W = 36
const GAP = 2
const MAX_CELL = 56
/** Below this cell width the month names shrink to initials. */
const SHORT_MIN = 34
const MIN_FILL = 0.18

const PAD = LABEL_W + TOTAL_W

interface Props {
  years: number[]
  /** counts[i][m]: detections in years[i], month m + 1. */
  counts: number[][]
  /** Container width in px. */
  width: number
  label: string
}

/** One row per year, one column per month. Fill darkness follows the square root of the count; non-zero cells carry their count. */
export default function FireCalendar({ years, counts, width, label }: Props) {
  const cell = Math.min(MAX_CELL, Math.max(16, (width - PAD) / 12))
  const gridW = cell * 12
  const totalX = LABEL_W + gridW + 8
  const svgW = LABEL_W + gridW + TOTAL_W
  const height = HEAD_H + years.length * ROW_H
  const max = Math.max(1, ...counts.flat())
  const initials = cell < SHORT_MIN

  return (
    <svg
      className="chart"
      width={svgW}
      height={height}
      viewBox={`0 0 ${svgW} ${height}`}
      role="img"
      aria-label={label}
    >
      {MONTHS_SHORT.map((m, i) => (
        <text key={m} x={LABEL_W + (i + 0.5) * cell} y={HEAD_H - 8} textAnchor="middle">
          {initials ? m[0] : m}
        </text>
      ))}
      <text x={totalX} y={HEAD_H - 8} className="ink">
        Total
      </text>
      {years.map((year, r) => {
        const top = HEAD_H + r * ROW_H
        const total = counts[r].reduce((a, b) => a + b, 0)
        return (
          <g key={year}>
            <text x={0} y={top + ROW_H / 2} dominantBaseline="central" className="ink">
              {year}
            </text>
            {counts[r].map((n, m) => {
              const x = LABEL_W + m * cell + GAP / 2
              const fill = n === 0 ? 0 : MIN_FILL + (1 - MIN_FILL) * Math.sqrt(n / max)
              return (
                <g key={m}>
                  <rect
                    x={x}
                    y={top + GAP / 2}
                    width={cell - GAP}
                    height={ROW_H - GAP}
                    rx="3"
                    fill={n === 0 ? 'var(--c-hair)' : 'var(--accent)'}
                    fillOpacity={n === 0 ? 0.45 : fill}
                  >
                    <title>{`${MONTHS_LONG[m]} ${year}: ${n} detection${n === 1 ? '' : 's'}`}</title>
                  </rect>
                  {n > 0 && (
                    <text
                      x={x + (cell - GAP) / 2}
                      y={top + ROW_H / 2}
                      textAnchor="middle"
                      dominantBaseline="central"
                      className={fill > 0.85 ? 'fire-count on-bar' : 'fire-count ink'}
                    >
                      {n}
                    </text>
                  )}
                </g>
              )
            })}
            <text x={totalX} y={top + ROW_H / 2} dominantBaseline="central" className={total === 0 ? '' : 'strong'}>
              {total}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
