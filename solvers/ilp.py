"""
Exact ILP for weighted maximum coverage (ground truth on small instances).
Uses PuLP + CBC solver.
"""

from typing import Optional


def ilp_coverage(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    time_limit_s: int = 120,
) -> tuple[list[int], float, str]:
    """
    Returns (chosen_site_indices, objective_value, solver_status).
    Raises ImportError if PuLP is not installed.
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError("Install PuLP: pip install pulp") from e

    sites = list(viewsheds.keys())
    cells = list(weights.keys())

    prob = pulp.LpProblem("WeightedMaxCoverage", pulp.LpMaximize)

    y = {s: pulp.LpVariable(f"y_{s}", cat="Binary") for s in sites}
    z = {c: pulp.LpVariable(f"z_{c}", cat="Binary") for c in cells}

    # objective
    prob += pulp.lpSum(weights[c] * z[c] for c in cells)

    # budget
    prob += pulp.lpSum(y[s] for s in sites) == n_sensors

    # coverage linkage: z_c <= sum of y_i for sites that can see c
    cell_to_sites: dict[int, list[int]] = {c: [] for c in cells}
    for s, vs in viewsheds.items():
        for c in vs:
            if c in cell_to_sites:
                cell_to_sites[c].append(s)

    for c in cells:
        seeing = cell_to_sites[c]
        if seeing:
            prob += z[c] <= pulp.lpSum(y[s] for s in seeing)
        else:
            prob += z[c] == 0  # no site can see it

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    chosen = [s for s in sites if pulp.value(y[s]) and pulp.value(y[s]) > 0.5]
    obj = pulp.value(prob.objective) or 0.0
    return chosen, float(obj), status
