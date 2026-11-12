/** The three series colours (defined as CSS variables in styles.css). */
export type Series = 'classical' | 'quantum' | 'baseline'

export const SERIES_COLOR: Record<Series, string> = {
  classical: 'var(--s-classical)',
  quantum: 'var(--s-quantum)',
  baseline: 'var(--s-baseline)',
}
