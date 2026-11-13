import { formatDate, formatRange } from './format'
import type { ParkData } from './park'

const words = (key: string) => key.replace(/_/g, ' ')

/** The title card's description line for the whole-park scale. */
export function parkTitleMeta(park: ParkData | null): string {
  const facts = park
    ? ` · ${park.meta.rows} × ${park.meta.cols} grid of ${park.meta.cell_m} m cells · ${park.meta.mast_height_m} m masts, ${formatRange(park.meta.range_m)} range · generated ${formatDate(park.meta.generated)}`
    : ''
  const keys = (g: Record<string, string> | undefined, fallback: string) => (g ? Object.keys(g).map(words).join(', ') : fallback)
  const labels = park?.meta.labels
  return (
    `Whole park${facts}. Real: ${keys(labels?.real, 'elevation, land cover, fire detections')}. ` +
    `Modelled: ${keys(labels?.modelled, 'priority, visibility')}. Simulated: ${keys(labels?.simulated, 'detection times')}.`
  )
}
