"""
Greedy (CELF lazy-greedy) for weighted maximum coverage.
Guarantee: f(A) >= (1 - 1/e) * OPT ≈ 0.632 * OPT.
"""

import heapq
from typing import Any


def greedy_coverage(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
) -> tuple[list[int], float]:
    """
    Returns (chosen_site_indices, total_covered_priority).

    Uses lazy greedy (CELF): negated marginal gain stored in a max-heap.
    A site's gain is only recomputed when it reaches the top of the heap.
    """
    covered: set[int] = set()
    chosen: list[int] = []
    sites = list(viewsheds.keys())

    # initialise heap: (-marginal, site_idx, last_updated_round)
    heap: list[tuple[float, int, int]] = []
    for s in sites:
        gain = sum(weights.get(c, 0.0) for c in viewsheds[s])
        heapq.heappush(heap, (-gain, s, 0))

    for rnd in range(n_sensors):
        while True:
            neg_gain, s, updated = heapq.heappop(heap)
            if updated == rnd:
                # still valid
                chosen.append(s)
                covered |= viewsheds[s]
                break
            # recompute marginal gain
            new_gain = sum(weights.get(c, 0.0) for c in viewsheds[s] if c not in covered)
            heapq.heappush(heap, (-new_gain, s, rnd))

    total = sum(weights.get(c, 0.0) for c in covered)
    return chosen, total


def greedy_coverage_simple(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
) -> tuple[list[int], float]:
    """Naive greedy (no CELF) — correct but O(n²) per round. Used for sanity checks."""
    covered: set[int] = set()
    chosen: list[int] = []
    remaining = set(viewsheds.keys())

    for _ in range(n_sensors):
        best_s, best_gain = -1, -1.0
        for s in remaining:
            gain = sum(weights.get(c, 0.0) for c in viewsheds[s] if c not in covered)
            if gain > best_gain:
                best_gain, best_s = gain, s
        if best_s < 0:
            break
        chosen.append(best_s)
        covered |= viewsheds[best_s]
        remaining.remove(best_s)

    total = sum(weights.get(c, 0.0) for c in covered)
    return chosen, total
