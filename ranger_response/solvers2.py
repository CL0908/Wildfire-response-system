"""Solvers for model v2: greedy, greedy with 2-step lookahead, and exact branch and bound.

Every plan is scored by model2.simulate2. Finish times are computed in the same order of
floating-point operations as simulate2 (arrive = t + travel; finish = arrive + cut) so that "holds"
comparisons agree bit for bit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import Plan
from .model2 import INF, Instance2, SimResult2, simulate2

EPS = 1e-9


@dataclass
class SolveResult2:
    method: str
    plan: Plan
    result: SimResult2
    proven_optimal: Optional[bool]   # None for heuristics
    nodes: int
    elapsed: float
    stop_reason: str

    @property
    def objective(self) -> float:
        return self.result.objective


def _crew_state(inst: Instance2, routes, ci: int):
    pos, t = inst.stations[ci], inst.delay_h
    for cell in routes[ci]:
        t = (t + inst.travel_h(pos, cell)) + inst.cut_h
        pos = cell
    return pos, t


def _planned_flat(inst: Instance2, routes) -> set:
    return {inst.idx(c) for route in routes for c in route}


def _candidates(inst: Instance2, res: SimResult2, planned: set, pos, t):
    """Cells this crew could still usefully defend in time: (flat idx, finish time) with finish <= fire arrival.

    A firebreak is only useful if it is on a cell the fire reaches, or is one of the two corner cells of a
    diagonal spread step the fire currently attempts (a break on a cell the fire never enters can still
    close a diagonal squeeze). Cells the fire does not reach and that are no such corner cannot change the
    outcome. The set only shrinks further down the search tree, because added firebreaks only delay the fire.
    """
    cand = set(res.order)
    horizon = inst.horizon_h
    for u in res.order:
        tu = res.burn_flat[u]
        for v, w, ca, cb in inst.adj[u]:
            if ca >= 0 and tu + w <= horizon:
                cand.add(ca)
                cand.add(cb)
    burnable = inst.burnable
    out = []
    for q in sorted(cand):
        if q in planned or not burnable.flat[q]:
            continue
        fin = (t + inst.travel_h(pos, inst.cell(q))) + inst.cut_h
        if fin <= res.threat_flat[q]:
            out.append((q, fin))
    return out


def solve_greedy2(inst: Instance2) -> SolveResult2:
    """Append the single (crew, cell) defence that raises the simulated objective most; stop when none does."""
    t0 = time.perf_counter()
    routes = [[] for _ in range(inst.n_crews)]
    res = simulate2(inst, Plan.of(routes))
    evals = 1
    while True:
        planned = _planned_flat(inst, routes)
        best = None
        for ci in range(inst.n_crews):
            pos, t = _crew_state(inst, routes, ci)
            for q, fin in _candidates(inst, res, planned, pos, t):
                trial = [list(r) for r in routes]
                trial[ci].append(inst.cell(q))
                obj = simulate2(inst, Plan.of(trial)).objective
                evals += 1
                key = (obj, -fin, -ci, -q)
                if best is None or key > best:
                    best = key
        if best is None or best[0] <= res.objective + EPS:
            break
        routes[-best[2]].append(inst.cell(-best[3]))
        res = simulate2(inst, Plan.of(routes))
    plan = Plan.of(routes)
    return SolveResult2("greedy", plan, simulate2(inst, plan), None, evals, time.perf_counter() - t0, "")


def solve_lookahead2(inst: Instance2, radius: int = 2) -> SolveResult2:
    """Greedy with a 2-step lookahead (the 'strong heuristic' baseline).

    Each round it scores every single defence and every pair of defences, where the second of the pair
    is within `radius` cells (Chebyshev) of the first, by the simulated objective after both. A pair
    is committed (both defences) if it beats the best single defence, otherwise the best single is
    committed if it improves the objective, otherwise it stops. This finds two-cell lines whose first
    cell saves nothing alone; lines of three or more cells whose partial versions save nothing are still
    invisible to it.
    """
    t0 = time.perf_counter()
    routes = [[] for _ in range(inst.n_crews)]
    res = simulate2(inst, Plan.of(routes))
    evals = 1
    while True:
        planned = _planned_flat(inst, routes)
        best1 = None     # (obj, -fin, -ci, -q)
        best2 = None     # (obj, -fin2, -ci1, -q1, -ci2, -q2)
        for ci in range(inst.n_crews):
            pos, t = _crew_state(inst, routes, ci)
            for q, fin in _candidates(inst, res, planned, pos, t):
                r1 = [list(r) for r in routes]
                r1[ci].append(inst.cell(q))
                res1 = simulate2(inst, Plan.of(r1))
                evals += 1
                key1 = (res1.objective, -fin, -ci, -q)
                if best1 is None or key1 > best1:
                    best1 = key1
                q1r, q1c = inst.cell(q)
                planned1 = planned | {q}
                for cj in range(inst.n_crews):
                    pos2, t2 = _crew_state(inst, r1, cj)
                    for q2, fin2 in _candidates(inst, res1, planned1, pos2, t2):
                        r2c = inst.cell(q2)
                        if max(abs(r2c[0] - q1r), abs(r2c[1] - q1c)) > radius:
                            continue
                        r2 = [list(r) for r in r1]
                        r2[cj].append(r2c)
                        obj = simulate2(inst, Plan.of(r2)).objective
                        evals += 1
                        key2 = (obj, -max(fin, fin2), -ci, -q, -cj, -q2)
                        if best2 is None or key2 > best2:
                            best2 = key2
        b1 = best1[0] if best1 else -INF
        b2 = best2[0] if best2 else -INF
        if best2 is not None and b2 > b1 + EPS and b2 > res.objective + EPS:
            routes[-best2[2]].append(inst.cell(-best2[3]))
            routes[-best2[4]].append(inst.cell(-best2[5]))
        elif best1 is not None and b1 > res.objective + EPS:
            routes[-best1[2]].append(inst.cell(-best1[3]))
        else:
            break
        res = simulate2(inst, Plan.of(routes))
    plan = Plan.of(routes)
    return SolveResult2("lookahead", plan, simulate2(inst, plan), None, evals, time.perf_counter() - t0, "")


def _subtree_values(inst: Instance2, res: SimResult2) -> dict:
    sub = {i: float(inst.value.flat[i]) for i in res.order}
    for i in reversed(res.order):
        p = res.parent_flat[i]
        if p >= 0:
            sub[p] += sub[i]
    return sub


def upper_bound2(inst: Instance2, res: SimResult2, planned: set, crew_states, last_finish: float) -> float:
    """Objective of this node plus the value of every burning cell that some later defence could still save.

    Every defence added below this node finishes at time >= last_finish. A burning cell x can only be
    saved if the fire's current shortest-path tree path to x (from an ignition) is cut by a new
    firebreak. Take the first place where the new world differs from the current one along that path;
    up to there the arrival times are unchanged. Two ways to differ:
      * a new firebreak on a path cell q: it needs earliest_finish(q) <= current arrival at q (the
        arrival at q in the new world equals the current one, because the path before q is unchanged);
      * a diagonal step p -> x whose squeeze is closed: one of the two corner cells c must be a new
        firebreak finished by the (unchanged) arrival time at x, earliest_finish(c) <= dist[x]. c's own
        arrival time may change in the new world, so no condition on it is used (a relaxation).
    earliest_finish is the best over crews of (current crew state -> cell travel + cut), by the
    triangle inequality never improved by detours, and at least last_finish.
    """
    table = inst.travel_table()
    earliest = None
    for pos, t in crew_states:
        e = t + table[inst.idx(pos)]
        earliest = e if earliest is None else np.minimum(earliest, e)
    earliest = np.maximum(earliest + inst.cut_h, last_finish).tolist()
    burnable = inst.burnable.ravel().tolist()
    cols = inst.cols
    good = {}
    gain = 0.0
    for i in res.order:                       # parents come before children in this order
        p = res.parent_flat[i]
        g = p >= 0 and good[p]
        if not g and p >= 0:
            if i not in planned and earliest[i] <= res.threat_flat[i] + EPS:
                g = True
            else:
                pr, pc = divmod(p, cols)
                ir, ic = divmod(i, cols)
                if pr != ir and pc != ic:
                    for c in (ir * cols + pc, pr * cols + ic):
                        if c not in planned and burnable[c] and earliest[c] <= res.burn_flat[i] + EPS:
                            g = True
                            break
        good[i] = g
        if g:
            gain += float(inst.value.flat[i])
    return res.objective + gain


def solve_exact2(inst: Instance2, node_limit: int = 5_000_000, time_limit: float = 20.0,
                 warm_start: Optional[Plan] = None) -> SolveResult2:
    """Depth-first branch and bound over joint plans, as in v1 (defences added in order of finish time
    across crews, each only if it holds against the earlier-finishing ones). `warm_start` is an
    incumbent plan (default: greedy). Ties between exactly equal finish times are not enumerated in
    both orders (probability zero with continuous times). `proven_optimal` is True only if the search
    finished before the node or time limit."""
    t0 = time.perf_counter()
    k = inst.n_crews
    evals0 = 0
    if warm_start is None:
        g = solve_greedy2(inst)
        warm_start, evals0 = g.plan, g.nodes
    best_plan = warm_start
    best_obj = simulate2(inst, best_plan).objective
    state = dict(nodes=0, stop="", best_plan=best_plan, best_obj=best_obj)

    def dfs(routes, states, last, res):
        if state["stop"]:
            return
        state["nodes"] += 1
        if state["nodes"] > node_limit:
            state["stop"] = "node limit"
            return
        if time.perf_counter() - t0 > time_limit:
            state["stop"] = "time limit"
            return
        if res.objective > state["best_obj"] + EPS:
            state["best_obj"] = res.objective
            state["best_plan"] = Plan.of(routes)
        planned = _planned_flat(inst, routes)
        if upper_bound2(inst, res, planned, states, last[0]) <= state["best_obj"] + EPS:
            return
        sub = _subtree_values(inst, res)
        moves = []
        for c in range(k):
            pos, t = states[c]
            for q, fin in _candidates(inst, res, planned, pos, t):
                if (fin, c) > last:
                    moves.append((q, fin, c))
        moves.sort(key=lambda m: (-sub.get(m[0], 0.0), m[1], m[2], m[0]))
        for q, fin, c in moves:
            cell = inst.cell(q)
            routes[c].append(cell)
            old = states[c]
            states[c] = (cell, fin)
            dfs(routes, states, (fin, c), simulate2(inst, Plan.of(routes)))
            states[c] = old
            routes[c].pop()
            if state["stop"]:
                return

    routes = [[] for _ in range(k)]
    dfs(routes, [(inst.stations[c], inst.delay_h) for c in range(k)], (-INF, -1),
        simulate2(inst, Plan.of(routes)))
    plan = state["best_plan"]
    return SolveResult2("exact", plan, simulate2(inst, plan), state["stop"] == "",
                        state["nodes"] + evals0, time.perf_counter() - t0, state["stop"])
