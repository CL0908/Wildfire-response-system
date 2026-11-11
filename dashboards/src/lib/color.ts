const STOPS: [number, number, number][] = [
  [245, 243, 236],
  [201, 219, 214],
  [134, 180, 196],
  [74, 130, 174],
  [36, 63, 110],
]

/** Sequential scale: t in [0, 1] -> rgb() string. */
export function rampColor(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(x))
  const f = x - i
  const [a, b] = [STOPS[i], STOPS[i + 1]]
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}
