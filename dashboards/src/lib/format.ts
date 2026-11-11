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
