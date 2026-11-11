"""Instance, plan and simulator for the ranger-response model.

Cells are (row, col) tuples. Internally every cell has a flat index r * cols + c.

Rules (all deterministic):
  * Fire spreads over 4-neighbour fuel cells. Spread time u -> v is 1 step, or
    `downhill_time` (2) when v is lower than u by more than `downhill_drop_m` metres.
  * A crew walks in a straight line at `speed` cells per step: travel time between two
    different cells is max(1, ceil(euclid / speed)), 0 for the same cell. After arriving
    it needs `work_time` steps to finish a firebreak.
  * A firebreak finished at step f on cell x holds iff f <= the earliest step at which
    fire would reach x (given the other firebreaks that hold). A held firebreak is never
    burned and blocks spread through it. A crew follows its list regardless of whether an
    earlier defence held; a failed defence still costs the crew its time.
  * Only fuel cells that burn at a step <= horizon count as burned (horizon None = no limit).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

Cell = tuple[int, int]
INF = math.inf

DEFAULT_FUEL_THRESHOLD = 0.55
DEFAULT_DOWNHILL_DROP_M = 50.0
DEFAULT_DOWNHILL_TIME = 2


def travel_time(a: Cell, b: Cell, speed: float) -> int:
    """Steps to walk from cell a to cell b: 0 if equal, else max(1, ceil(euclid / speed))."""
    if a == b:
        return 0
    d = math.hypot(a[0] - b[0], a[1] - b[1])
    return max(1, math.ceil(d / speed - 1e-9))


@dataclass
class Instance:
    rows: int
    cols: int
    fuel: np.ndarray            # (rows, cols) bool
    value: np.ndarray           # (rows, cols) float, summed over cells that never burn
    dem_m: np.ndarray           # (rows, cols) elevation, kept for plotting / reference
    ignitions: tuple[Cell, ...]
    stations: tuple[Cell, ...]  # one per crew
    speed: float = 3.0
    work_time: int = 1
    horizon: Optional[int] = None
    adj: list = field(default_factory=list, repr=False)   # flat idx -> [(nbr flat idx, spread steps)]
    params: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_crews(self) -> int:
        return len(self.stations)

    @property
    def n_fuel(self) -> int:
        return int(self.fuel.sum())

    def idx(self, cell: Cell) -> int:
        return cell[0] * self.cols + cell[1]

    def cell(self, i: int) -> Cell:
        return divmod(i, self.cols)

    def travel_table(self) -> np.ndarray:
        """(n, n) int array of travel_time between flat indices (same formula as travel_time)."""
        t = self._cache.get("travel")
        if t is None:
            n = self.rows * self.cols
            rr, cc = np.divmod(np.arange(n), self.cols)
            d = np.hypot(rr[:, None] - rr[None, :], cc[:, None] - cc[None, :])
            t = np.maximum(1, np.ceil(d / self.speed - 1e-9)).astype(np.int64)
            np.fill_diagonal(t, 0)
            self._cache["travel"] = t
        return t


@dataclass(frozen=True)
class Plan:
    """routes[i] is the ordered list of cells crew i defends."""
    routes: tuple[tuple[Cell, ...], ...]

    @staticmethod
    def of(routes: Iterable[Iterable[Cell]]) -> "Plan":
        return Plan(tuple(tuple((int(r), int(c)) for r, c in route) for route in routes))

    @staticmethod
    def empty(n_crews: int) -> "Plan":
        return Plan(tuple(() for _ in range(n_crews)))


@dataclass
class DefenceRecord:
    crew: int
    order: int           # position in that crew's route (0-based)
    cell: Cell
    arrive: int          # step the crew reaches the cell
    finish: int          # step the firebreak is finished
    fire_arrival: float  # earliest step fire would reach the cell (inf = never within horizon)
    held: bool
    reason: str


@dataclass
class SimResult:
    instance: Instance
    burn_flat: list      # burn step per flat index, inf if never burned
    threat_flat: list    # earliest step fire would reach (also for held firebreaks)
    parent_flat: list    # predecessor on the fire's shortest-path tree, -1 for ignitions
    order: list          # flat indices of burned cells in the order they ignite
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
    def cells_saved(self) -> int:
        return self.baseline_burned - self.cells_burned

    @property
    def value_saved(self) -> float:
        return self.objective - self.baseline_objective

    @property
    def held_cells(self) -> list:
        return [d.cell for d in self.defences if d.held]


def make_instance(fuel, value, dem_m, ignitions: Sequence[Cell], stations: Sequence[Cell],
                  speed: float = 3.0, work_time: int = 1, horizon: Optional[int] = None,
                  downhill_drop_m: float = DEFAULT_DOWNHILL_DROP_M,
                  downhill_time: int = DEFAULT_DOWNHILL_TIME) -> Instance:
    """Build an Instance from raw arrays (used directly by tests for hand-built graphs)."""
    fuel = np.asarray(fuel, dtype=bool)
    value = np.asarray(value, dtype=float)
    dem_m = np.asarray(dem_m, dtype=float)
    rows, cols = fuel.shape
    if value.shape != fuel.shape or dem_m.shape != fuel.shape:
        raise ValueError("fuel, value and dem_m must have the same shape")
    if speed <= 0 or work_time < 1:
        raise ValueError("speed must be > 0 and work_time >= 1")
    ignitions = tuple((int(r), int(c)) for r, c in ignitions)
    stations = tuple((int(r), int(c)) for r, c in stations)
    if not ignitions or not stations:
        raise ValueError("need at least one ignition and one station")
    for r, c in ignitions:
        if not (0 <= r < rows and 0 <= c < cols) or not fuel[r, c]:
            raise ValueError(f"ignition {(r, c)} must be a fuel cell inside the grid")
    for r, c in stations:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"station {(r, c)} outside the grid")

    adj = [[] for _ in range(rows * cols)]
    for r in range(rows):
        for c in range(cols):
            if not fuel[r, c]:
                continue
            u = r * cols + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and fuel[rr, cc]:
                    t = downhill_time if dem_m[r, c] - dem_m[rr, cc] > downhill_drop_m else 1
                    adj[u].append((rr * cols + cc, t))
    return Instance(rows, cols, fuel, value, dem_m, ignitions, stations, float(speed),
                    int(work_time), horizon, adj,
                    params=dict(downhill_drop_m=downhill_drop_m, downhill_time=downhill_time))


def build_instance(surf: dict, ignitions: Sequence[Cell], stations: Sequence[Cell],
                   fuel_threshold: float = DEFAULT_FUEL_THRESHOLD,
                   value_key: str = "priority", speed: float = 3.0, work_time: int = 1,
                   horizon: Optional[int] = None,
                   downhill_drop_m: float = DEFAULT_DOWNHILL_DROP_M,
                   downhill_time: int = DEFAULT_DOWNHILL_TIME) -> Instance:
    """Instance from a priority_surface dict: fuel = hazard >= fuel_threshold, value = surf[value_key]."""
    inst = make_instance(surf["hazard"] >= fuel_threshold, surf[value_key], surf["dem_m"],
                         ignitions, stations, speed, work_time, horizon,
                         downhill_drop_m, downhill_time)
    inst.params.update(fuel_threshold=fuel_threshold, value_key=value_key)
    return inst


def _as_plan(inst: Instance, plan) -> Plan:
    if not isinstance(plan, Plan):
        plan = Plan.of(plan)
    if len(plan.routes) > inst.n_crews:
        raise ValueError("plan has more routes than the instance has crews")
    if len(plan.routes) < inst.n_crews:
        plan = Plan(plan.routes + tuple(() for _ in range(inst.n_crews - len(plan.routes))))
    return plan


def _spread(inst: Instance, finish_at: dict):
    """Dijkstra from the ignitions; a cell with a firebreak finished at f blocks any arrival t >= f."""
    n = inst.rows * inst.cols
    horizon = INF if inst.horizon is None else inst.horizon
    dist = [INF] * n
    threat = [INF] * n
    parent = [-1] * n
    heap = []
    for ig in inst.ignitions:
        i = inst.idx(ig)
        dist[i] = threat[i] = 0
        heapq.heappush(heap, (0, i))
    order = []
    settled = [False] * n
    adj = inst.adj
    while heap:
        t, u = heapq.heappop(heap)
        if settled[u]:
            continue
        settled[u] = True
        order.append(u)
        for v, w in adj[u]:
            tv = t + w
            if tv > horizon:
                continue
            if tv < threat[v]:
                threat[v] = tv
            f = finish_at.get(v)
            if f is not None and f <= tv:
                continue
            if tv < dist[v]:
                dist[v] = tv
                parent[v] = u
                heapq.heappush(heap, (tv, v))
    return dist, threat, parent, order


def _baseline(inst: Instance):
    b = inst._cache.get("baseline")
    if b is None:
        _, _, _, order = _spread(inst, {})
        total = float(inst.value[inst.fuel].sum())
        burned_val = sum(float(inst.value.flat[i]) for i in order)
        b = (total - burned_val, len(order))
        inst._cache["baseline"] = b
    return b


def simulate(inst: Instance, plan) -> SimResult:
    """Score a plan. Single source of truth for burn times, held defences and the objective."""
    plan = _as_plan(inst, plan)
    work = inst.work_time
    records = []   # (finish, crew, order, cell, arrive)
    for ci, route in enumerate(plan.routes):
        pos, t = inst.stations[ci], 0
        for oi, cell in enumerate(route):
            if not (0 <= cell[0] < inst.rows and 0 <= cell[1] < inst.cols):
                raise ValueError(f"cell {cell} outside the grid")
            arrive = t + travel_time(pos, cell, inst.speed)
            t = arrive + work
            records.append((t, ci, oi, cell, arrive))
            pos = cell

    # Earliest firebreak per fuel cell is the one that counts; later ones on the same cell are redundant.
    records.sort()
    finish_at = {}
    for f, ci, oi, cell, arrive in records:
        i = inst.idx(cell)
        if inst.fuel[cell] and i not in finish_at:
            finish_at[i] = f

    dist, threat, parent, order = _spread(inst, finish_at)

    seen = set()
    defences = []
    for f, ci, oi, cell, arrive in records:
        i = inst.idx(cell)
        fa = threat[i]
        if not inst.fuel[cell]:
            held, why = False, "no fuel at this cell"
        elif i in seen:
            held, why = False, "cell already defended by an earlier firebreak"
        elif f <= fa:
            if fa < INF:
                held, why = True, "held"
            elif inst.horizon is None:
                held, why = True, "held (fire never reaches this cell)"
            else:
                held, why = True, "held (no fire arrival within the horizon)"
        else:
            held, why = False, "fire arrives before the firebreak is finished"
        seen.add(i)
        defences.append(DefenceRecord(ci, oi, cell, arrive, f, fa, held, why))
    defences.sort(key=lambda d: (d.crew, d.order))

    total = float(inst.value[inst.fuel].sum())
    burned_val = sum(float(inst.value.flat[i]) for i in order)
    base_obj, base_burned = _baseline(inst)
    return SimResult(inst, dist, threat, parent, order, defences,
                     total - burned_val, base_obj, base_burned)
