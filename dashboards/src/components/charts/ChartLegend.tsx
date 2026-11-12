import { SERIES_COLOR, type Series } from './series'

export interface LegendItem {
  label: string
  series: Series
  /** bar: filled square; dot: filled circle; hollow: open circle. */
  mark: 'bar' | 'dot' | 'hollow'
}

export default function ChartLegend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="chart-legend">
      {items.map((item) => (
        <li key={item.label}>
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            {item.mark === 'bar' && <rect x="1" y="1" width="10" height="10" rx="2" fill={SERIES_COLOR[item.series]} />}
            {item.mark === 'dot' && <circle cx="6" cy="6" r="5" fill={SERIES_COLOR[item.series]} />}
            {item.mark === 'hollow' && (
              <circle cx="6" cy="6" r="4" fill="var(--c-page)" stroke={SERIES_COLOR[item.series]} strokeWidth="2" />
            )}
          </svg>
          {item.label}
        </li>
      ))}
    </ul>
  )
}
