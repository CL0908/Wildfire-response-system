import type { Sampling } from '../types'

/**
 * Chance that `shots` uniformly random N-sensor placements include the optimum:
 * 1 - (1 - p)^shots. This is the no-circuit control for a best-of-shots figure.
 */
export function randomBestHitProbability(s: Sampling): number {
  return -Math.expm1(s.shots * Math.log1p(-s.p_optimal_random))
}
