const MONTHS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
]

/** "2026-11-10" (or a longer ISO string) -> "10 Nov 2026". */
export function formatDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!m) return iso
  const month = MONTHS[Number(m[2]) - 1]
  return month ? `${Number(m[3])} ${month} ${m[1]}` : iso
}

export function formatInt(n: number): string {
  return n.toLocaleString('en-US')
}

/** Fraction -> percent string, with extra precision for small values. */
export function formatPercent(fraction: number): string {
  const p = fraction * 100
  if (fraction >= 1 - 1e-9) return '100%'
  if (p >= 99.95) return '99.9%'
  if (p >= 1) return `${p.toFixed(1)}%`
  return `${p.toFixed(2)}%`
}

export function formatRange(m: number): string {
  return m % 1000 === 0 ? `${m / 1000} km` : `${formatInt(m)} m`
}

const MONTHS_LONG = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
]

/** "2026-06-01" -> "1 June 2026". */
export function formatDateLong(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!m) return iso
  const month = MONTHS_LONG[Number(m[2]) - 1]
  return month ? `${Number(m[3])} ${month} ${m[1]}` : iso
}

/** Round half up with a little slack for binary floating point (0.2245 -> 22.5). */
function roundTo(x: number, digits: number): number {
  const f = 10 ** digits
  return Math.round(x * f + 1e-7) / f
}

/** Fraction -> "22.5%", "0.5%", "100%": one decimal from 1%, two below, no trailing zeros. */
export function formatShare(fraction: number): string {
  const p = fraction * 100
  return `${roundTo(p, p >= 1 ? 1 : 2)}%`
}

/** 0.6665 -> "0.667". */
export function formatScore(v: number): string {
  return roundTo(v, 3).toFixed(3)
}

/** Window bounds -> "14.32 to 14.41° N, 101.12 to 101.21° E". */
export function formatBounds(b: {
  north: number
  south: number
  west: number
  east: number
}): string {
  return `${b.south.toFixed(2)} to ${b.north.toFixed(2)}° N, ${b.west.toFixed(2)} to ${b.east.toFixed(2)}° E`
}
