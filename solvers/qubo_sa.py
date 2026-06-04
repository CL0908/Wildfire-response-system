"""
QUBO formulation + Simulated Annealing baseline (dimod/neal).

Variables: y_i (sensor placement) + z_c (cell covered).
Qubits = |S| + |C|.

H = -sum_c w_c*z_c
  + P1*(sum_i y_i - N)^2
  + P2*sum_c z_c*(1 - sum_{i:c in V_i} y_i)
"""

import numpy as np


def build_qubo(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    P1: float | None = None,
    P2: float | None = None,
) -> tuple[dict[tuple[int, int], float], dict[int, str]]:
    """
    Returns (Q_dict, var_labels).
    Q_dict: {(i,j): coeff} upper-triangular QUBO matrix (i<=j).
    var_labels: {qubit_idx: 'y_siteN' or 'z_cellN'}
    """
    sites = sorted(viewsheds.keys())
    cells = sorted(weights.keys())

    # P1 must exceed the maximum possible reward so the budget constraint dominates.
    max_reward = sum(weights.values())
    P1 = P1 or max_reward * 5
    P2 = P2 or max_reward * 5

    n_s, n_c = len(sites), len(cells)
    s_idx = {s: i for i, s in enumerate(sites)}
    c_idx = {c: n_s + j for j, c in enumerate(cells)}

    var_labels: dict[int, str] = {}
    for s, i in s_idx.items():
        var_labels[i] = f"y_{s}"
    for c, j in c_idx.items():
        var_labels[j] = f"z_{c}"

    Q: dict[tuple[int, int], float] = {}

    def add_q(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + val

    # reward: -w_c * z_c
    for c, w in weights.items():
        add_q(c_idx[c], c_idx[c], -w)

    # cardinality: P1*(sum y_i - N)^2
    for s in sites:
        add_q(s_idx[s], s_idx[s], P1 * (1 - 2 * n_sensors))
    for a_pos, a in enumerate(sites):
        for b in sites[a_pos + 1:]:
            add_q(s_idx[a], s_idx[b], 2 * P1)

    # coverage linkage: P2*z_c*(1 - sum_{i:c in V_i} y_i)
    cell_to_sites: dict[int, list[int]] = {}
    for s, vs in viewsheds.items():
        for c in vs:
            cell_to_sites.setdefault(c, []).append(s)

    for c in cells:
        add_q(c_idx[c], c_idx[c], P2)
        for s in cell_to_sites.get(c, []):
            add_q(c_idx[c], s_idx[s], -P2)

    return Q, var_labels


def sa_solve(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    num_reads: int = 100,
    P1: float | None = None,
    P2: float | None = None,
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
    sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=10000)

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
