"""
Greedy and exact (PuLP + CBC) solvers for the time-to-detection objective.

Both take the site detection-time matrix T (n_scenarios, n_sites; see
`detection.objective.site_detection_times`), the sensor budget k, the satellite baseline and
optional scenario weights, and return a `DetectionPlacement`. Both are scored through
`detection.objective.score_detection`, so the reported objective is that function's value for the
returned placement. The model is simplified and uncalibrated (see `detection.objective`); nothing
here predicts real fire behaviour or real detection times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .objective import (DEFAULT_BASELINE_MIN, as_weight_vector, saving_matrix,
                        score_detection)


@dataclass
class DetectionPlacement:
    chosen: list            # sorted site indices
    objective: float        # expected minutes saved against satellite only (== score[...])
    score: dict             # full output of score_detection for `chosen`
    method: str             # "greedy" or "exact"
    status: str             # "Greedy", or the solver status ("Optimal", "Not Solved", ...)
    proven_optimal: bool    # True only if the exact solver proved optimality
    details: dict = field(default_factory=dict)


def _check_k(k: int, n_sites: int) -> int:
    if int(k) != k or k < 0:
        raise ValueError("k must be a non-negative integer")
    if k > n_sites:
        raise ValueError(f"k = {k} exceeds the number of sites ({n_sites})")
    return int(k)


def _pad(chosen: list, k: int, n_sites: int) -> list:
    """Fill up to k sites with the lowest-index unused ones (never lowers a monotone objective)."""
    used = set(chosen)
    for j in range(n_sites):
        if len(used) >= k:
            break
        used.add(j)
    return sorted(used)


def greedy_detection(T, k: int, baseline_min=DEFAULT_BASELINE_MIN, weights=None) -> DetectionPlacement:
    """
    Greedy: k rounds, each adding the site with the largest marginal gain in expected minutes
    saved. Ties (within 1e-12 relative) go to the lowest site index; once no site gains anything
    the lowest-index unused sites fill the budget, so exactly k sites are returned.

    The objective is monotone and submodular, so the greedy value is at least (1 - 1/e) (about
    0.632) of the optimum (Nemhauser, Wolsey and Fisher 1978). That is a guarantee about this
    model's objective only; it says nothing about how well the layout would work against real fires.

    Vectorised over scenarios: each round is one (scenarios x sites) matrix-vector product.
    """
    V = saving_matrix(T, baseline_min)
    n_scen, n_sites = V.shape
    k = _check_k(k, n_sites)
    w = as_weight_vector(weights, n_scen)
    cur = np.zeros(n_scen)                 # minutes saved so far in each scenario
    chosen: list[int] = []
    taken = np.zeros(n_sites, dtype=bool)
    for _ in range(k):
        gains = w @ np.maximum(V - cur[:, None], 0.0)
        gains[taken] = -np.inf
        best = gains.max()
        j = int(np.flatnonzero(gains >= best - 1e-12 * max(1.0, abs(best)))[0])
        chosen.append(j)
        taken[j] = True
        cur = np.maximum(cur, V[:, j])
    sc = score_detection(T, chosen, baseline_min, weights)
    return DetectionPlacement(sc["chosen"], sc["expected_saved_min"], sc, "greedy", "Greedy", False,
                              dict(n_rounds=k))


def exact_detection(T, k: int, baseline_min=DEFAULT_BASELINE_MIN, weights=None,
                    exactly_k: bool = False, time_limit_s: int = 120) -> DetectionPlacement:
    """
    Exact maximiser of expected minutes saved with at most k sites (exactly k if `exactly_k`),
    as a facility-location integer programme solved with PuLP's CBC.

    Model, with v[s, j] = max(0, baseline_s - T[s, j]) and scenario weights w_s (sum 1):
        maximise   sum_s w_s * sum_j v[s, j] * x[s, j]
        subject to x[s, j] <= y[j]            (a scenario is served only by a chosen site)
                   sum_j x[s, j] <= 1         (each scenario takes one site's saving)
                   sum_j y[j] <= k            (budget)
                   y binary, 0 <= x <= 1
    At the optimum each scenario picks its best chosen site, so the value is the objective of
    `score_detection`. Reductions before building the model, none of which changes the optimum:
    sites whose column of v is all zero are dropped, scenarios whose row is all zero are dropped,
    scenarios with identical v rows are merged (weights summed), and x[s, j] variables exist only
    where v[s, j] > 0. `exactly_k` solves the "at most k" model on the remaining sites and fills
    the budget with the lowest-index unused sites, which cannot lower the value (monotone).

    The result's `details` holds n_scenarios, n_distinct_rows (rows in the model), n_active_sites,
    n_x_variables, and model_objective (the solver's own value, for cross-checking).
    `proven_optimal` is True only if CBC reports a proven optimum (not a time-limit stop). If the
    time limit stops CBC with an incumbent worse than greedy's, the greedy placement is returned
    instead (details["fell_back_to_greedy"]), so the result is never worse than greedy.
    CBC starts from the greedy placement. Run time grows steeply with how many sites see each
    scenario's fire (see detection/RESULTS.md), not just with scenario count.
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError("Install PuLP: pip install pulp") from e

    V = saving_matrix(T, baseline_min)
    n_scen, n_sites = V.shape
    k = _check_k(k, n_sites)
    w = as_weight_vector(weights, n_scen)

    act_sites = np.flatnonzero(V.any(axis=0))
    act_rows = np.flatnonzero(V.any(axis=1))
    details = dict(n_scenarios=n_scen, n_active_sites=int(act_sites.size), n_distinct_rows=0,
                   n_x_variables=0, model_objective=0.0)
    chosen: list[int] = []
    status, proven = "Optimal", True

    if act_sites.size and act_rows.size and k > 0:
        Va = V[np.ix_(act_rows, act_sites)]
        U, inv = np.unique(Va, axis=0, return_inverse=True)
        wu = np.bincount(inv.ravel(), weights=w[act_rows], minlength=U.shape[0])
        details["n_distinct_rows"] = int(U.shape[0])

        prob = pulp.LpProblem("TimeToDetection", pulp.LpMaximize)
        y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(act_sites.size)]
        terms = []
        x_of = []                          # x_of[r] = {column: x variable} for row r
        n_x = 0
        for r in range(U.shape[0]):
            cols = np.flatnonzero(U[r] > 0)
            xs = []
            x_of.append({})
            for j in cols:
                x = pulp.LpVariable(f"x_{r}_{j}", lowBound=0, upBound=1)
                prob += x <= y[j]
                terms.append(wu[r] * U[r, j] * x)
                xs.append(x)
                x_of[r][int(j)] = x
            prob += pulp.lpSum(xs) <= 1
            n_x += len(xs)
        details["n_x_variables"] = n_x
        prob += pulp.lpSum(terms)
        prob += pulp.lpSum(y) <= k

        # MIP start from the greedy placement: a starting incumbent only, the proof is CBC's
        g = greedy_detection(T, k, baseline_min, weights)
        in_start = np.isin(act_sites, g.chosen)
        for j in range(act_sites.size):
            y[j].setInitialValue(int(in_start[j]))
        for r in range(U.shape[0]):
            best = max((j for j in x_of[r] if in_start[j]), key=lambda j: U[r, j], default=None)
            for j, x in x_of[r].items():
                x.setInitialValue(1 if j == best else 0)
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s, warmStart=True))
        status = pulp.LpStatus[prob.status]
        proven = bool(prob.status == pulp.LpStatusOptimal
                      and prob.sol_status == pulp.LpSolutionOptimal)
        chosen = [int(act_sites[j]) for j in range(act_sites.size)
                  if (pulp.value(y[j]) or 0.0) > 0.5]
        details["model_objective"] = float(pulp.value(prob.objective) or 0.0)
        if not proven and g.objective > score_detection(T, chosen, baseline_min, weights)[
                "expected_saved_min"]:
            chosen = list(g.chosen)        # time limit hit with a worse incumbent: keep greedy's
            details["fell_back_to_greedy"] = True

    if exactly_k:
        chosen = _pad(chosen, k, n_sites)
    sc = score_detection(T, chosen, baseline_min, weights)
    return DetectionPlacement(sc["chosen"], sc["expected_saved_min"], sc, "exact", status, proven,
                              details)
