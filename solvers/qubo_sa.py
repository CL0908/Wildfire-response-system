"""
QUBO formulation + Simulated Annealing baseline (dimod/neal).

Variables: y_i (sensor placement), z_c (cell covered), and slack bits per cell.

H = -sum_c w_c*z_c
  + P1*(sum_i y_i - N)^2
  + P2*sum_c (z_c + slack_c - sum_{i:c in V_i} y_i)^2

The last term enforces z_c <= sum of the y_i that see c. slack_c is a bounded
integer (0..min(k_c, N)-1, binary-encoded) that absorbs the surplus when more
than one chosen sensor sees the cell, so overlap is neither rewarded nor
penalised. For any placement y with exactly N sensors, the minimum of H over
z and slack equals -(covered priority).
"""

import numpy as np


def _slack_coeffs(max_value: int) -> list[int]:
    """Coefficients of a bounded binary encoding of the integers 0..max_value."""
    coeffs = []
    remaining = max_value
    power = 1
    while remaining > 0:
        c = min(power, remaining)
        coeffs.append(c)
        remaining -= c
        power *= 2
    return coeffs


def build_qubo(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    P1: float | None = None,
    P2: float | None = None,
) -> tuple[dict[tuple[int, int], float], dict[int, str]]:
    """
    Returns (Q_dict, var_labels).
    Q_dict: {(i,j): coeff} upper-triangular QUBO matrix (i<=j). The constant
    P1*N^2 is dropped, so energy = H - P1*N^2.
    var_labels: {var_idx: 'y_<site>', 'z_<cell>' or 's_<cell>_<bit>'}
    """
    sites = sorted(viewsheds.keys())

    cell_to_sites: dict[int, list[int]] = {}
    for s in sites:
        for c in viewsheds[s]:
            cell_to_sites.setdefault(c, []).append(s)

    # Only cells that carry weight and that some site can see need a z variable.
    cells = sorted(c for c, w in weights.items() if w > 0 and c in cell_to_sites)

    # P1 must exceed the most one extra sensor can add; P2 must exceed the most
    # one falsely claimed cell can add.
    max_site_weight = max(
        (sum(weights.get(c, 0.0) for c in viewsheds[s]) for s in sites), default=0.0
    )
    max_cell_weight = max((weights[c] for c in cells), default=0.0)
    P1 = P1 or 2.0 * max_site_weight
    P2 = P2 or 2.0 * max_cell_weight

    var_labels: dict[int, str] = {}
    s_idx = {s: i for i, s in enumerate(sites)}
    for s, i in s_idx.items():
        var_labels[i] = f"y_{s}"

    Q: dict[tuple[int, int], float] = {}

    def add_q(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + val

    # cardinality: P1*(sum y_i - N)^2
    for s in sites:
        add_q(s_idx[s], s_idx[s], P1 * (1 - 2 * n_sensors))
    for a_pos, a in enumerate(sites):
        for b in sites[a_pos + 1:]:
            add_q(s_idx[a], s_idx[b], 2 * P1)

    next_var = len(sites)
    for c in cells:
        seeing = cell_to_sites[c]
        z_var = next_var
        next_var += 1
        var_labels[z_var] = f"z_{c}"

        # reward: -w_c * z_c
        add_q(z_var, z_var, -weights[c])

        # linkage: P2*(z_c + slack_c - sum y_i)^2, expanded for binary variables
        terms: list[tuple[int, float]] = [(z_var, 1.0)]
        for bit, coeff in enumerate(_slack_coeffs(min(len(seeing), n_sensors) - 1)):
            var_labels[next_var] = f"s_{c}_{bit}"
            terms.append((next_var, float(coeff)))
            next_var += 1
        terms.extend((s_idx[s], -1.0) for s in seeing)

        for pos, (i, a) in enumerate(terms):
            add_q(i, i, P2 * a * a)
            for j, b in terms[pos + 1:]:
                add_q(i, j, 2 * P2 * a * b)

    return Q, var_labels


def sa_solve(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    num_reads: int = 100,
    P1: float | None = None,
    P2: float | None = None,
    seed: int | None = 0,
) -> tuple[list[int], float, int]:
    """
    Solve via simulated annealing on the QUBO.
    Returns (chosen_site_indices, covered_priority, n_cardinality_violations).
    n_cardinality_violations = 1 if no feasible QUBO sample was found (repair used), else 0.
    """
    try:
        import dimod
        import neal
    except ImportError as e:
        raise ImportError("Install dimod and dwave-neal: pip install dimod dwave-neal") from e

    sites = sorted(viewsheds.keys())
    Q, var_labels = build_qubo(viewsheds, weights, n_sensors, P1, P2)

    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=10000, seed=seed)

    label_to_site = {f"y_{s}": s for s in sites}
    n_violations = 0

    # Find lowest-energy feasible sample
    chosen: list[int] = []
    for sample, _ in sampleset.data(["sample", "energy"]):
        cands = [label_to_site[var_labels[i]] for i, val in sample.items()
                 if val == 1 and var_labels[i].startswith("y_")]
        if len(cands) == n_sensors:
            chosen = cands
            break

    # Repair if no feasible QUBO sample was found
    if len(chosen) != n_sensors:
        n_violations = 1
        all_y = [(label_to_site[var_labels[i]], val)
                 for i, val in sampleset.first.sample.items()
                 if var_labels[i].startswith("y_")]
        all_y.sort(key=lambda x: -x[1])
        chosen = [s for s, _ in all_y[:n_sensors]]

    covered: set[int] = set()
    for s in chosen:
        covered |= viewsheds[s]
    obj = sum(weights.get(c, 0.0) for c in covered)

    return chosen, obj, n_violations
