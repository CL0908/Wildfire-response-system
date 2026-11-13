"""
Sparse solvers and scorers for the whole-park placement problem.

The existing solvers were written for 16 candidate sites and a few hundred cells: `solvers.greedy`
takes dicts of Python sets and `detection.objective.site_detection_times` a dense boolean matrix.
At park scale (8,874 sites, 24,497 cells, 0.8 million visible site-cell pairs) these are the
same algorithms on a scipy CSR matrix:

    greedy_cover                  greedy weighted maximum coverage (same rule as solvers.greedy)
    coverage_value                total weight of the cells seen by a set of sites
    coverage_bound                LP relaxation (an upper bound) and, within a time limit, the
                                  integer optimum of the maximum-coverage programme (HiGHS)
    site_detection_times_sparse   same output as detection.objective.site_detection_times

Conventions as in park.instance: `sees` is a (n_sites, n_cells) CSR matrix, sees[i, c] iff site i
sees cell c; `weights` has one non-negative entry per cell.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, Bounds, milp

TIE_TOL = 1e-12


def _as_csr(sees) -> sparse.csr_matrix:
    m = sparse.csr_matrix(sees)
    if m.nnz and not m.data.all():          # stored zeros / False would count as visible
        m = m.copy()
        m.eliminate_zeros()
    return m


def coverage_value(sees, weights, chosen) -> float:
    """Total weight of the cells seen by at least one site in `chosen` (0.0 for no sites)."""
    sees = _as_csr(sees)
    idx = np.asarray(list(chosen), dtype=np.int64)
    if idx.size == 0:
        return 0.0
    covered = np.zeros(sees.shape[1], dtype=bool)
    for i in idx:
        covered[sees.indices[sees.indptr[i]:sees.indptr[i + 1]]] = True
    return float(np.asarray(weights, dtype=np.float64)[covered].sum())


def greedy_cover(sees, weights, k: int) -> tuple[list[int], list[float]]:
    """
    Greedy weighted maximum coverage: k rounds, each adding the site whose newly seen cells carry
    the most weight. Ties (within 1e-12 relative) go to the lowest site index. Once no site adds
    weight the lowest-index unused sites fill the budget, so exactly k sites are returned.

    Returns (chosen site indices in the order picked, cumulative covered weight after each pick).
    The objective is monotone and submodular, so the value is at least (1 - 1/e) of the optimum.
    """
    A = _as_csr(sees).astype(np.float64)
    n_sites, n_cells = A.shape
    if int(k) != k or k < 0 or k > n_sites:
        raise ValueError(f"k = {k} must be an integer between 0 and the number of sites ({n_sites})")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_cells,) or (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weights must be finite and non-negative, one per cell")
    uncovered = w.copy()
    taken = np.zeros(n_sites, dtype=bool)
    chosen: list[int] = []
    cumulative: list[float] = []
    total = 0.0
    for _ in range(int(k)):
        gains = A @ uncovered
        gains[taken] = -np.inf
        best = gains.max()
        j = int(np.flatnonzero(gains >= best - TIE_TOL * max(1.0, abs(best)))[0])
        chosen.append(j)
        taken[j] = True
        cells = A.indices[A.indptr[j]:A.indptr[j + 1]]
        total += float(uncovered[cells].sum())
        uncovered[cells] = 0.0
        cumulative.append(total)
    return chosen, cumulative


def coverage_bound(sees, weights, k: int, integer: bool = True, time_limit_s: float = 300.0) -> dict:
    """
    Upper bound (and, if it finishes, the optimum) of the maximum-coverage programme

        maximise sum_c w_c z_c   s.t.  z_c <= sum over sites i that see c of y_i,
                                       sum_i y_i <= k,  0 <= z_c <= 1,  y_i in {0, 1}.

    Cells with no weight or no site seeing them are dropped, and cells seen by exactly the same
    sites are merged into one row with their weights added; neither changes the optimum. The LP
    relaxation (y_i in [0, 1]) is always solved and is a valid upper bound on the integer optimum.
    The integer programme is solved by HiGHS under `time_limit_s` and a relative gap of 1e-9.

    Returns dict: lp_value, integer_value (None unless proven optimal), integer_proven (bool),
    integer_status (HiGHS message, "not run" or "time limit"), lp_seconds, integer_seconds (None
    if not run), n_rows (merged cell groups) and n_sites.
    """
    A = _as_csr(sees)
    n_sites, n_cells = A.shape
    w = np.asarray(weights, dtype=np.float64)
    csc = A.tocsc()
    groups: dict[bytes, int] = {}
    g_w: list[float] = []
    rows, cols = [], []
    for c in np.flatnonzero(w > 0):
        idx = csc.indices[csc.indptr[c]:csc.indptr[c + 1]]
        if idx.size == 0:
            continue
        key = idx.tobytes()
        g = groups.get(key)
        if g is None:
            g = groups[key] = len(g_w)
            g_w.append(0.0)
            rows.append(np.full(idx.size, g, dtype=np.int64))
            cols.append(idx.astype(np.int64))
        g_w[g] += float(w[c])
    n_g = len(g_w)
    out = dict(lp_value=0.0, integer_value=None, integer_proven=False, integer_status="not run",
               lp_seconds=0.0, integer_seconds=None, n_rows=n_g, n_sites=n_sites)
    if n_g == 0 or k <= 0:
        out["integer_value"], out["integer_proven"], out["integer_status"] = 0.0, True, "trivial"
        return out
    M = sparse.csr_matrix((np.ones(sum(len(r) for r in rows)),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(n_g, n_sites))
    # variables: y (n_sites) then z (n_g); constraint z_g - M y <= 0 and sum y <= k
    cons = LinearConstraint(
        sparse.vstack([sparse.hstack([-M, sparse.identity(n_g, format="csr")]),
                       sparse.hstack([sparse.csr_matrix(np.ones((1, n_sites))),
                                      sparse.csr_matrix((1, n_g))])], format="csr"),
        -np.inf, np.concatenate([np.zeros(n_g), [float(k)]]))
    c_vec = -np.concatenate([np.zeros(n_sites), np.asarray(g_w)])
    bounds = Bounds(np.zeros(n_sites + n_g), np.ones(n_sites + n_g))

    t0 = time.time()
    lp = milp(c_vec, constraints=cons, bounds=bounds, integrality=np.zeros(n_sites + n_g))
    out["lp_seconds"] = time.time() - t0
    if not lp.success:
        raise RuntimeError(f"LP relaxation failed: {lp.message}")
    out["lp_value"] = float(-lp.fun)
    if integer:
        t0 = time.time()
        mip = milp(c_vec, constraints=cons, bounds=bounds,
                   integrality=np.concatenate([np.ones(n_sites), np.zeros(n_g)]),
                   options=dict(time_limit=float(time_limit_s), mip_rel_gap=1e-9))
        out["integer_seconds"] = time.time() - t0
        if mip.status == 0:
            out["integer_value"], out["integer_proven"] = float(-mip.fun), True
            out["integer_status"] = "optimal"
        else:
            out["integer_status"] = "time limit" if mip.status == 1 else str(mip.message)
    return out


def site_detection_times_sparse(arrival_min, sees) -> np.ndarray:
    """
    Same result as detection.objective.site_detection_times for a CSR `sees`: T[s, j] = earliest
    arrival_min[s, c] over the cells c that site j sees (inf if it sees none), float64
    (n_scenarios, n_sites). Works on the transposed arrival matrix so each site gathers whole
    contiguous rows, and never builds a dense (n_sites, n_cells) matrix.
    """
    arrival_min = np.asarray(arrival_min)
    sees = _as_csr(sees)
    if arrival_min.ndim != 2 or sees.shape[1] != arrival_min.shape[1]:
        raise ValueError("arrival_min must be (n_scenarios, n_cells) and sees (n_sites, n_cells)")
    n_scen, n_sites = arrival_min.shape[0], sees.shape[0]
    at = np.ascontiguousarray(arrival_min.T)            # (n_cells, n_scenarios)
    out = np.full((n_sites, n_scen), np.inf)
    for j in range(n_sites):
        cols = sees.indices[sees.indptr[j]:sees.indptr[j + 1]]
        if cols.size:
            out[j] = at[cols].min(axis=0)
    return np.ascontiguousarray(out.T)
