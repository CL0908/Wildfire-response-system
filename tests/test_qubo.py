"""The QUBO must agree with the true coverage objective on a small instance."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.qubo_sa import build_qubo, _slack_coeffs

# 4 sites, 6 cells; cells 0 and 1 are seen by several sites, cell 5 by none.
VIEWSHEDS = {0: {0, 1, 2}, 1: {0, 1, 3}, 2: {0, 4}, 3: {1}}
WEIGHTS = {0: 3.0, 1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 4.0}


# Disjoint viewsheds: every extra sensor adds coverage, so only the cardinality
# penalty stops the QUBO from placing too many.
DISJOINT_VIEWSHEDS = {0: {0}, 1: {1}, 2: {2}}
DISJOINT_WEIGHTS = {0: 5.0, 1: 4.0, 2: 3.0}


def coverage(chosen, viewsheds=VIEWSHEDS, weights=WEIGHTS):
    covered = set().union(*(viewsheds[s] for s in chosen)) if chosen else set()
    return sum(weights[c] for c in covered)


def min_energy_by_placement(n_sensors, viewsheds=VIEWSHEDS, weights=WEIGHTS):
    """Brute-force every assignment; return {y tuple: min energy over z and slack}."""
    Q, labels = build_qubo(viewsheds, weights, n_sensors)
    n = len(labels)
    M = np.zeros((n, n))
    for (i, j), v in Q.items():
        M[i, j] = v
    y_vars = [i for i in range(n) if labels[i].startswith("y_")]
    assert y_vars == list(range(len(viewsheds)))
    best = {}
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=float)
        e = float(x @ M @ x)
        y = bits[: len(y_vars)]
        if e < best.get(y, np.inf):
            best[y] = e
    return best, Q


def test_slack_coeffs_cover_exact_range():
    for m in range(0, 12):
        coeffs = _slack_coeffs(m)
        sums = {sum(c for c, b in zip(coeffs, bits) if b)
                for bits in itertools.product((0, 1), repeat=len(coeffs))}
        assert sums == set(range(m + 1))


def test_energy_matches_coverage_for_feasible_placements():
    for n_sensors in (1, 2, 3):
        best, Q = min_energy_by_placement(n_sensors)
        # build_qubo drops the constant P1*N^2. Sites 2 and 3 share no cell, so
        # their coupling is the cardinality term alone: 2*P1.
        P1 = Q[(2, 3)] / 2
        offset = P1 * n_sensors ** 2
        for y, e in best.items():
            if sum(y) == n_sensors:
                chosen = [s for s, b in enumerate(y) if b]
                assert abs(e + offset + coverage(chosen)) < 1e-9, (n_sensors, y)


def test_ground_state_is_the_optimal_feasible_placement():
    for n_sensors in (1, 2, 3):
        best, _ = min_energy_by_placement(n_sensors)
        y_star = min(best, key=best.get)
        assert sum(y_star) == n_sensors
        optimum = max(coverage(c) for c in itertools.combinations(VIEWSHEDS, n_sensors))
        assert abs(coverage([s for s, b in enumerate(y_star) if b]) - optimum) < 1e-9


def test_wrong_sensor_count_never_beats_the_feasible_optimum():
    for viewsheds, weights in ((VIEWSHEDS, WEIGHTS), (DISJOINT_VIEWSHEDS, DISJOINT_WEIGHTS)):
        for n_sensors in (1, 2):
            best, _ = min_energy_by_placement(n_sensors, viewsheds, weights)
            feasible_min = min(e for y, e in best.items() if sum(y) == n_sensors)
            for y, e in best.items():
                if sum(y) != n_sensors:
                    assert e > feasible_min + 1e-9, (n_sensors, y)
