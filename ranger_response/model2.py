"""Ranger-response model v2: physical spread times, walking crews, timed firebreaks.

Fire spread reuses detection/fire_sim.py: its 8-neighbour graph, land-cover fuel factors, slope factor,
wind factor and base rate r0 (its `_Graph` supplies the edge geometry and the wind-free edge times; the
wind factor formula is repeated here and a test checks the no-response arrival times against
fire_sim.simulate_scenarios). Times are hours.

Rules
  * Fire starts at the ignition cell(s) at t = 0 and reaches cell v at the minimum travel time over
    8-neighbour edges (Dijkstra). A cell with a finished firebreak blocks every arrival at or after the
    finish time. Arrivals later than `horizon_h` are ignored (a cell counts as burned only if the fire
    reaches it within the horizon).
  * Diagonal squeeze rule. A diagonal move u -> v is forbidden at arrival time t if both cells that
    share its corner (the two orthogonal neighbours) are blocked at t, where a cell is blocked if it
    is not burnable or has a finished firebreak, and at least one of the two is a firebreak. Natural
    gaps between two non-burnable cells are left as fire_sim has them. Without this rule a fire
    would slip diagonally between two firebreak cells that touch at a corner.
  * A crew starts at its station at t = `delay_h` and walks in a straight line between cell centres at
    `walk_kmh`, then needs `cut_h` hours to cut a firebreak across a cell. It executes its ordered list
    whether or not an earlier firebreak held; a failed firebreak still costs the crew its time.
  * A firebreak at cell x finished at time f holds iff f <= the earliest time fire would reach x given
    the other firebreaks that hold (same semantics as model v1). A failed firebreak is inert: it does
    not block spread and does not count for the squeeze rule.
  * Objective: summed value of burnable cells the fire does not reach within the horizon.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from detection.fire_sim import CELL_M, K_WIND, WIND_CAP_MS, _Graph

from .model import Cell, Plan

INF = math.inf
DEFAULT_WALK_KMH = 2.0
DEFAULT_HORIZON_H = 24.0


@dataclass
class Instance2:
    rows: int
    cols: int
    burnable: np.ndarray        # (rows, cols) bool
    value: np.ndarray           # (rows, cols) float
    dem_m: np.ndarray
    fuel_factor: np.ndarray     # fire_sim relative spread multiplier, 0 = not burnable
    ignitions: tuple
    stations: tuple             # one per crew
    wind_speed_ms: float
    wind_from_deg: float
    r0_m_min: float
    horizon_h: float
    walk_kmh: float
    cut_h: float
    delay_h: float
    cell_m: float
    adj: list = field(default_factory=list, repr=False)   # flat -> [(v, hours, corner_a, corner_b)]
    params: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_crews(self) -> int:
        return len(self.stations)

    @property
    def fuel(self) -> np.ndarray:
        return self.burnable

    @property
    def n_fuel(self) -> int:
        return int(self.burnable.sum())

    @property
    def horizon(self) -> float:
        return self.horizon_h

    def idx(self, cell: Cell) -> int:
        return cell[0] * self.cols + cell[1]

    def cell(self, i: int) -> Cell:
        return divmod(i, self.cols)

    def travel_h(self, a: Cell, b: Cell) -> float:
        if a == b:
            return 0.0
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.cell_m / 1000.0 / self.walk_kmh

    def travel_table(self) -> np.ndarray:
        t = self._cache.get("travel")
        if t is None:
            n = self.rows * self.cols
            rr, cc = np.divmod(np.arange(n), self.cols)
            d = np.hypot(rr[:, None] - rr[None, :], cc[:, None] - cc[None, :])
            t = d * self.cell_m / 1000.0 / self.walk_kmh
            self._cache["travel"] = t
        return t

    def with_params(self, **kw) -> "Instance2":
        """Copy with cut_h / walk_kmh / delay_h / horizon_h changed (the spread graph is shared)."""
        import copy
        new = copy.copy(self)
        for k, v in kw.items():
            if k not in ("cut_h", "walk_kmh", "delay_h", "horizon_h"):
                raise ValueError(k)
            setattr(new, k, v)
        new._cache = {}
        new.params = dict(self.params)
        return new


def wind_edge_hours(g: _Graph, speed_ms: float, from_deg: float) -> np.ndarray:
    """Per-edge spread time in hours: fire_sim's wind factor applied to its wind-free edge times."""
    toward = np.radians(from_deg + 180.0)
    d_row, d_col = -np.cos(toward), np.sin(toward)
    cos_angle = g.ur * d_row + g.uc * d_col
    wf = np.exp(K_WIND * min(max(speed_ms, 0.0), WIND_CAP_MS) * cos_angle)
    return g.base_time / wf / 60.0


def make_instance2(fuel_factor, value, dem_m, ignitions: Sequence[Cell], stations: Sequence[Cell],
                   wind_speed_ms: float = 0.0, wind_from_deg: float = 0.0, r0_m_min: float = 2.0,
                   horizon_h: float = DEFAULT_HORIZON_H, walk_kmh: float = DEFAULT_WALK_KMH,
                   cut_h: float = 1.0, delay_h: float = 0.0, cell_m: float = CELL_M) -> Instance2:
    fuel_factor = np.asarray(fuel_factor, dtype=float)
    value = np.asarray(value, dtype=float)
    dem_m = np.asarray(dem_m, dtype=float)
    rows, cols = fuel_factor.shape
    if value.shape != fuel_factor.shape or dem_m.shape != fuel_factor.shape:
        raise ValueError("fuel_factor, value and dem_m must have the same shape")
    if walk_kmh <= 0 or cut_h <= 0 or horizon_h <= 0 or r0_m_min <= 0 or delay_h < 0:
        raise ValueError("walk_kmh, cut_h, horizon_h, r0_m_min must be > 0 and delay_h >= 0")
    ignitions = tuple((int(r), int(c)) for r, c in ignitions)
    stations = tuple((int(r), int(c)) for r, c in stations)
    if not ignitions or not stations:
        raise ValueError("need at least one ignition and one station")
    burnable = fuel_factor > 0
    for r, c in ignitions:
        if not (0 <= r < rows and 0 <= c < cols) or not burnable[r, c]:
            raise ValueError(f"ignition {(r, c)} must be a burnable cell inside the grid")
    for r, c in stations:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"station {(r, c)} outside the grid")

    g = _Graph(fuel_factor, dem_m, r0_m_min, cell_m)
    hours = wind_edge_hours(g, wind_speed_ms, wind_from_deg)
    n = rows * cols
    adj = [[] for _ in range(n)]
    for u in range(n):
        ur_, uc_ = divmod(u, cols)
        for k in range(g.indptr[u], g.indptr[u + 1]):
            v = int(g.indices[k])
            vr, vc = divmod(v, cols)
            if vr != ur_ and vc != uc_:
                ca, cb = vr * cols + uc_, ur_ * cols + vc
            else:
                ca = cb = -1
            adj[u].append((v, float(hours[k]), ca, cb))
    return Instance2(rows, cols, burnable, value, dem_m, fuel_factor, ignitions, stations,
                     float(wind_speed_ms), float(wind_from_deg), float(r0_m_min), float(horizon_h),
                     float(walk_kmh), float(cut_h), float(delay_h), float(cell_m), adj)


@dataclass
class DefenceRecord2:
    crew: int
    order: int
    cell: Cell
    arrive: float
    finish: float
    fire_arrival: float
    held: bool
    reason: str


@dataclass
class SimResult2:
    instance: Instance2
    burn_flat: list
    threat_flat: list
    parent_flat: list
    order: list
    defences: list
    objective: float
    baseline_objective: float
    baseline_burned: int

    @property
    def burn_time(self) -> np.ndarray:
        return np.array(self.burn_flat, dtype=float).reshape(self.instance.rows, self.instance.cols)

    @property
    def burned(self) -> np.ndarray:
        return np.isfinite(self.burn_time)

    @property
    def cells_burned(self) -> int:
        return len(self.order)

    @property
    def value_saved(self) -> float:
        return self.objective - self.baseline_objective

    @property
    def held_cells(self) -> list:
        return [d.cell for d in self.defences if d.held]


def _as_plan(inst: Instance2, plan) -> Plan:
    if not isinstance(plan, Plan):
        plan = Plan.of(plan)
    if len(plan.routes) > inst.n_crews:
        raise ValueError("plan has more routes than the instance has crews")
    if len(plan.routes) < inst.n_crews:
        plan = Plan(plan.routes + tuple(() for _ in range(inst.n_crews - len(plan.routes))))
    return plan


def spread2(inst: Instance2, fin: list):
    """Dijkstra. `fin[i]` is the finish time of the firebreak on cell i (INF = none). Returns
    (dist, threat, parent, order). threat[v] = earliest time fire would reach v, ignoring v's own firebreak."""
    n = inst.rows * inst.cols
    horizon = inst.horizon_h
    dist = [INF] * n
    threat = [INF] * n
    parent = [-1] * n
    nbl = (~inst.burnable).ravel().tolist()
    heap = []
    for ig in inst.ignitions:
        i = inst.idx(ig)
        dist[i] = threat[i] = 0.0
        heap.append((0.0, i))
    heapq.heapify(heap)
    order = []
    settled = [False] * n
    adj = inst.adj
    while heap:
        t, u = heapq.heappop(heap)
        if settled[u]:
            continue
        settled[u] = True
        order.append(u)
        for v, w, ca, cb in adj[u]:
            tv = t + w
            if tv > horizon:
                continue
            if ca >= 0:
                fa, fb = fin[ca] <= tv, fin[cb] <= tv
                if (fa or nbl[ca]) and (fb or nbl[cb]) and (fa or fb):
                    continue                      # squeeze between blocked corner cells
            if tv < threat[v]:
                threat[v] = tv
            if fin[v] <= tv:
                continue
            if tv < dist[v]:
                dist[v] = tv
                parent[v] = u
                heapq.heappush(heap, (tv, v))
    return dist, threat, parent, order


def _baseline2(inst: Instance2):
    b = inst._cache.get("baseline")
    if b is None:
        n = inst.rows * inst.cols
        _, _, _, order = spread2(inst, [INF] * n)
        total = float(inst.value[inst.burnable].sum())
        b = (total - sum(float(inst.value.flat[i]) for i in order), len(order))
        inst._cache["baseline"] = b
    return b


def crew_records(inst: Instance2, plan: Plan) -> list:
    """(finish, crew, order, cell, arrive) for every defence, in crew order."""
    out = []
    for ci, route in enumerate(plan.routes):
        pos, t = inst.stations[ci], inst.delay_h
        for oi, cell in enumerate(route):
            if not (0 <= cell[0] < inst.rows and 0 <= cell[1] < inst.cols):
                raise ValueError(f"cell {cell} outside the grid")
            arrive = t + inst.travel_h(pos, cell)
            t = arrive + inst.cut_h
            out.append((t, ci, oi, cell, arrive))
            pos = cell
    return out


def simulate2(inst: Instance2, plan) -> SimResult2:
    """Score a plan. Single source of truth for arrival times, held firebreaks and the objective."""
    plan = _as_plan(inst, plan)
    n = inst.rows * inst.cols
    records = sorted(crew_records(inst, plan))
    active = {}
    for f, ci, oi, cell, arrive in records:          # earliest firebreak on a burnable cell counts
        i = inst.idx(cell)
        if inst.burnable[cell] and i not in active:
            active[i] = f
    while True:
        fin = [INF] * n
        for i, f in active.items():
            fin[i] = f
        dist, threat, parent, order = spread2(inst, fin)
        failing = [i for i, f in active.items() if f > threat[i]]
        if not failing:
            break
        for i in failing:                            # failed firebreaks are inert; rerun without them
            del active[i]

    seen = set()
    defences = []
    for f, ci, oi, cell, arrive in records:
        i = inst.idx(cell)
        fa = threat[i]
        if not inst.burnable[cell]:
            held, why = False, "no fuel at this cell"
        elif i in seen:
            held, why = False, "cell already defended by an earlier firebreak"
        elif f <= fa:
            held, why = True, ("held" if fa < INF else "held (fire does not reach this cell within the horizon)")
        else:
            held, why = False, "fire arrives before the firebreak is finished"
        seen.add(i)
        defences.append(DefenceRecord2(ci, oi, cell, arrive, f, fa, held, why))
    defences.sort(key=lambda d: (d.crew, d.order))
    total = float(inst.value[inst.burnable].sum())
    burned_val = sum(float(inst.value.flat[i]) for i in order)
    base_obj, base_burned = _baseline2(inst)
    return SimResult2(inst, dist, threat, parent, order, defences, total - burned_val, base_obj, base_burned)
