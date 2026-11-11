"""Greedy and exact (branch and bound) solvers. Every plan is scored by model.simulate."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import INF, Instance, Plan, SimResult, simulate, travel_time

EPS = 1e-12


@dataclass
class SolveResult:
    method: str
    plan: Plan
    result: SimResult
    proven_optimal: Optional[bool]   # None for heuristics
    nodes: int
    elapsed: float
    stop_reason: str                 # "" when finished normally

    @property
    def objective(self) -> float:
        return self.result.objective


def _crew_state(inst: Instance, routes, ci: int):
    pos, t = inst.stations[ci], 0
    for cell in routes[ci]:
        t += travel_time(pos, cell, inst.speed) + inst.work_time
        pos = cell
    return pos, t


def _candidates(inst: Instance, res: SimResult, planned: set, pos, t):
    """Cells this crew could still defend in time, as (flat idx, finish step)."""
    out = []
    for q, fa in enumerate(res.threat_flat):
        if fa == INF or q in planned:
            continue
        fin = t + travel_time(pos, inst.cell(q), inst.speed) + inst.work_time
        if fin <= fa:
            out.append((q, fin))
    return out


def _planned_flat(inst: Instance, routes) -> set:
    return {inst.idx(c) for route in routes for c in route}


def solve_greedy(inst: Instance) -> SolveResult:
    """Repeatedly add the (crew, cell) append that raises the simulated objective most; stop when none does."""
    t0 = time.perf_counter()
    routes = [[] for _ in range(inst.n_crews)]
    res = simulate(inst, Plan.of(routes))
    evals = 1
    while True:
        planned = _planned_flat(inst, routes)
        best = None   # (objective, -finish, -crew, -idx)
        for ci in range(inst.n_crews):
            pos, t = _crew_state(inst, routes, ci)
            for q, fin in _candidates(inst, res, planned, pos, t):
                trial = [list(r) for r in routes]
                trial[ci].append(inst.cell(q))
                obj = simulate(inst, Plan.of(trial)).objective
                evals += 1
                key = (obj, -fin, -ci, -q)
                if best is None or key > best:
                    best = key
        if best is None or best[0] <= res.objective + EPS:
            break
        _, _, nci, nq = best
        routes[-nci].append(inst.cell(-nq))
        res = simulate(inst, Plan.of(routes))
    plan = Plan.of(routes)
    return SolveResult("greedy", plan, simulate(inst, plan), None, evals,
                       time.perf_counter() - t0, "")


def _subtree_values(inst: Instance, res: SimResult) -> dict:
    """Value of every burned cell plus everything downstream of it in the fire's shortest-path tree."""
    sub = {i: float(inst.value.flat[i]) for i in res.order}
    for i in reversed(res.order):
        p = res.parent_flat[i]
        if p >= 0:
            sub[p] += sub[i]
    return sub


def _upper_bound(inst: Instance, res: SimResult, planned: set, crew_states, last_finish: int) -> float:
    """Objective of this node plus the value of cells that any extension could still save.

    Every defence added below this node finishes at step >= last_finish, after all defences already
    in the plan. A burning cell x can only be saved if some new firebreak lands on its tree path from
    an ignition (inclusive of x). The first new firebreak q on that path sees fire at the same step as
    now (the path before q has no new firebreak), so it needs earliest_finish(q) <= current fire
    arrival at q. earliest_finish is the best over all crews, each starting from its current state;
    by the triangle inequality, going via other cells first never gets there sooner.
    """
    table = inst.travel_table()
    earliest = None
    for pos, t in crew_states:
        e = t + table[inst.idx(pos)]
        earliest = e if earliest is None else np.minimum(earliest, e)
    earliest = np.maximum(earliest + inst.work_time, last_finish).tolist()
    good = {}
    gain = 0.0
    for i in res.order:                       # parents come before children in this order
        p = res.parent_flat[i]
        g = (p >= 0 and good[p]) or (i not in planned and earliest[i] <= res.threat_flat[i])
        good[i] = g
        if g:
            gain += float(inst.value.flat[i])
    return res.objective + gain


def solve_exact(inst: Instance, node_limit: int = 200_000, time_limit: float = 20.0,
                warm_start: bool = True) -> SolveResult:
    """Depth-first branch and bound over joint defence plans (any number of crews).

    Defences are added in non-decreasing order of (finish step, crew index), i.e. in the order they
    are completed across all crews. Each is only tried if it holds against the defences already
    added, which are exactly the ones that finish earlier (or tie, and ties cannot interact: a
    firebreak that holds finishes no later than the fire arrives, so fire passing through it
    arrives strictly later). Every joint plan whose defences all hold is therefore visited exactly
    once, including crew A defences that only hold because crew B finished earlier. Plans with
    defences that fail are never needed: removing a failed defence only frees crew time.
    `proven_optimal` is True only if the search finished without hitting the node or time limit.
    """
    t0 = time.perf_counter()
    k = inst.n_crews
    if warm_start:
        g = solve_greedy(inst)
        best_plan, best_obj = g.plan, g.objective
        evals0 = g.nodes
    else:
        best_plan, best_obj, evals0 = Plan.empty(k), simulate(inst, Plan.empty(k)).objective, 1
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
        if _upper_bound(inst, res, planned, states, last[0]) <= state["best_obj"] + EPS:
            return
        sub = _subtree_values(inst, res)
        moves = []
        for c in range(k):
            pos, t = states[c]
            for q, fin in _candidates(inst, res, planned, pos, t):
                if (fin, c) > last:
                    moves.append((q, fin, c))
        moves.sort(key=lambda m: (-sub[m[0]], m[1], m[2], m[0]))
        for q, fin, c in moves:
            cell = inst.cell(q)
            routes[c].append(cell)
            old = states[c]
            states[c] = (cell, fin)
            dfs(routes, states, (fin, c), simulate(inst, Plan.of(routes)))
            states[c] = old
            routes[c].pop()
            if state["stop"]:
                return

    routes = [[] for _ in range(k)]
    dfs(routes, [(inst.stations[c], 0) for c in range(k)], (0, -1), simulate(inst, Plan.of(routes)))
    plan = state["best_plan"]
    return SolveResult("exact", plan, simulate(inst, plan), state["stop"] == "",
                       state["nodes"] + evals0, time.perf_counter() - t0, state["stop"])
