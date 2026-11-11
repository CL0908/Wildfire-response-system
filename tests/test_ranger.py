"""Tests for ranger_response. Expected values are worked out by hand or by an independent reference."""
import itertools
import math

import numpy as np
import pytest

from ranger_response import (Plan, make_instance, simulate, solve_exact, solve_greedy,
                             travel_time)

INF = math.inf


def corridor(dem, ignition=0, station=None, speed=3, work_time=1, **kw):
    n = len(dem)
    return make_instance(np.ones((1, n), bool), np.ones((1, n)), np.array([dem], float),
                         [(0, ignition)], [(0, n - 1 if station is None else station)],
                         speed=speed, work_time=work_time, **kw)


def burn_row(res):
    return list(res.burn_time[0])


# 1. fire arrival times --------------------------------------------------------------------

def test_arrival_times_on_corridor_with_slow_downhill_edge():
    # Drop of 80 m between columns 2 and 3 (> 50 m) costs 2 steps going down, 1 going up.
    dem = [100, 100, 100, 20, 20, 20]
    down = simulate(corridor(dem, ignition=0), Plan.empty(1))
    assert burn_row(down) == [0, 1, 2, 4, 5, 6]
    up = simulate(corridor(dem, ignition=5), Plan.empty(1))
    assert burn_row(up) == [5, 4, 3, 2, 1, 0]


def test_drop_exactly_at_threshold_is_not_slow():
    res = simulate(corridor([100, 50, 50], ignition=0), Plan.empty(1))   # drop == 50, not > 50
    assert burn_row(res) == [0, 1, 2]


def test_arrival_times_around_a_hole_and_with_two_ignitions():
    fuel = np.array([[1, 0, 1], [1, 1, 1]], bool)
    kw = dict(value=np.ones((2, 3)), dem_m=np.zeros((2, 3)), stations=[(0, 1)])
    res = simulate(make_instance(fuel, ignitions=[(0, 0)], **kw), Plan.empty(1))
    assert res.burn_time.tolist() == [[0, INF, 4], [1, 2, 3]]
    both = simulate(make_instance(fuel, ignitions=[(0, 0), (0, 2)], **kw), Plan.empty(1))
    assert both.burn_time.tolist() == [[0, INF, 0], [1, 2, 1]]


def test_horizon_cuts_the_fire_off():
    res = simulate(corridor([0] * 6, ignition=0, horizon=2), Plan.empty(1))
    assert burn_row(res) == [0, 1, 2, INF, INF, INF]
    assert res.cells_burned == 3 and res.objective == 3


# 2. defences ----------------------------------------------------------------------------------

def test_held_defence_never_burns_and_blocks_spread():
    inst = corridor([0] * 6, ignition=0, station=3)        # fire reaches column 3 at step 3
    res = simulate(inst, Plan.of([[(0, 3)]]))               # travel 0 + work 1 -> finished at 1
    assert burn_row(res) == [0, 1, 2, INF, INF, INF]
    d = res.defences[0]
    assert d.held and d.finish == 1 and d.fire_arrival == 3
    assert res.objective == 3 and res.value_saved == 3 and res.cells_saved == 3


def test_defence_finished_exactly_at_fire_arrival_holds_one_step_later_fails():
    inst = corridor([0] * 6, ignition=0, station=3, speed=1)
    ok = simulate(inst, Plan.of([[(0, 2)]]))        # travel 1 + work 1 = 2, fire at 2 -> holds
    assert ok.defences[0].held and ok.defences[0].finish == 2
    assert burn_row(ok) == [0, 1, INF, INF, INF, INF]
    late = simulate(corridor([0] * 6, ignition=0, station=3, speed=1, work_time=2),
                    Plan.of([[(0, 2)]]))            # finishes at 3 > 2
    d = late.defences[0]
    assert not d.held and "before" in d.reason
    assert burn_row(late) == [0, 1, 2, 3, 4, 5]     # failed firebreak changes nothing


def test_ignition_nonfuel_and_duplicate_defences_fail():
    fuel = np.array([[1, 1, 1, 0]], bool)
    inst = make_instance(fuel, np.ones((1, 4)), np.zeros((1, 4)), [(0, 0)], [(0, 3)], speed=3)
    res = simulate(inst, Plan.of([[(0, 2), (0, 0), (0, 3), (0, 2)]]))
    # (0,2): travel 1 + work 1 = 2 <= fire at 2 -> holds; then ignition, non-fuel and a repeat all fail.
    assert [d.held for d in res.defences] == [True, False, False, False]
    assert "no fuel" in res.defences[2].reason and "already" in res.defences[3].reason


def test_failed_defence_still_costs_the_crew_its_time():
    inst = corridor([0] * 8, ignition=0, station=7, speed=1)
    # (0,1) is lost (finish 7 > 1); the crew then walks 1 step to (0,2): finish 9 > 2 -> also lost.
    res = simulate(inst, Plan.of([[(0, 1), (0, 2)]]))
    assert [(d.finish, d.held) for d in res.defences] == [(7, False), (9, False)]


def test_defences_of_two_crews_interact():
    # Ring: row 0 (cols 0-3), column 0 and 3 down to row 2, row 2 (cols 0-3). Ignition (0,0).
    # Direct route to x=(0,3) takes 3 steps; with (0,1) blocked the fire goes round the ring:
    # (1,0)=1 (2,0)=2 (2,1)=3 (2,2)=4 (2,3)=5 (1,3)=6 (0,3)=7.
    fuel = np.zeros((3, 4), bool)
    fuel[0, :] = fuel[2, :] = fuel[:, 0] = fuel[:, 3] = True
    inst = make_instance(fuel, np.ones((3, 4)), np.zeros((3, 4)), [(0, 0)], [(2, 0), (0, 1)],
                         speed=1, work_time=1)
    # Crew 0 reaches x after 4 + 1 = 5 steps: too late alone (fire at 3) ...
    alone = simulate(inst, Plan.of([[(0, 3)], []]))
    assert not alone.defences[0].held and alone.defences[0].finish == 5
    # ... but crew 1 holds (0,1) at step 1, which pushes the fire's arrival at x to 7.
    both = simulate(inst, Plan.of([[(0, 3)], [(0, 1)]]))
    assert both.defences[0].held and both.defences[0].fire_arrival == 7
    assert both.defences[1].held
    assert both.burn_time[0, 3] == INF and both.burn_time[2, 3] == 5


# 3. choke point ---------------------------------------------------------------------------------

def choke_instance(speed):
    fuel = np.zeros((3, 7), bool)
    fuel[:, 0:3] = True          # region A (ignition side)
    fuel[1, 3] = True            # corridor cell
    fuel[:, 4:7] = True          # region B
    return make_instance(fuel, np.ones((3, 7)), np.zeros((3, 7)), [(1, 0)], [(2, 6)],
                         speed=speed, work_time=1)


def test_exact_defends_corridor_and_saves_far_region():
    # Fire reaches corridor (1,3) at step 3. Crew at (2,6): distance sqrt(10)=3.16, speed 3 -> 2 steps
    # + 1 work = 3 <= 3.
    inst = choke_instance(speed=3)
    sol = solve_exact(inst)
    assert sol.proven_optimal
    assert sol.plan.routes == (((1, 3),),)
    res = sol.result
    assert res.baseline_burned == 19 and res.cells_burned == 9      # A (9) burns; corridor + B (10) saved
    assert res.value_saved == 10
    assert not res.burned[:, 4:].any() and not res.burned[1, 3]
    assert res.burned[:, :3].all()
    assert res.burned.sum() == 9


def test_slow_crew_cannot_reach_corridor():
    # Speed 1: sqrt(10) -> 4 steps + 1 = 5 > 3, so the corridor is out of reach.
    inst = choke_instance(speed=1)
    sol = solve_exact(inst)
    assert sol.proven_optimal
    assert (1, 3) not in sol.plan.routes[0]
    assert not simulate(inst, Plan.of([[(1, 3)]])).defences[0].held
    # Best available by hand: (1,4), arrival 4, sqrt(5)=2.24 -> 3 steps + 1 = 4 <= 4, saves the 9 cells of B.
    assert sol.plan.routes == (((1, 4),),)
    assert sol.result.value_saved == 9


# 4. solvers against brute force --------------------------------------------------------------

def random_instance(seed, max_fuel=7, n_crews=1):
    rng = np.random.default_rng(seed)
    while True:
        fuel = rng.random((3, 4)) < 0.6
        if 3 <= fuel.sum() <= max_fuel:
            break
    cells = [tuple(map(int, c)) for c in np.argwhere(fuel)]
    ig = cells[rng.integers(len(cells))]
    stations = [(int(rng.integers(3)), int(rng.integers(4))) for _ in range(n_crews)]
    return make_instance(fuel, rng.uniform(0.1, 1.0, fuel.shape), rng.uniform(0, 130, fuel.shape),
                         [ig], stations, speed=float(rng.choice([1, 2, 3, 5])),
                         work_time=int(rng.integers(1, 3)), downhill_drop_m=20, downhill_time=3)


def sequences(cells):
    for k in range(len(cells) + 1):
        yield from itertools.permutations(cells, k)


def brute_force_best(inst):
    cells = [tuple(map(int, c)) for c in np.argwhere(inst.fuel)]
    best = -INF
    if inst.n_crews == 1:
        for seq in sequences(cells):
            best = max(best, simulate(inst, Plan.of([seq])).objective)
    else:
        for s0 in sequences(cells):
            rest = [c for c in cells if c not in s0]
            for s1 in sequences(rest):
                best = max(best, simulate(inst, Plan.of([s0, s1])).objective)
    return best


@pytest.mark.parametrize("seed", range(16))
def test_exact_matches_brute_force_one_crew(seed):
    inst = random_instance(seed)
    ex = solve_exact(inst, warm_start=bool(seed % 2))
    assert ex.proven_optimal
    assert ex.objective == pytest.approx(brute_force_best(inst))
    assert solve_greedy(inst).objective <= ex.objective + 1e-9


def joint_instance(seed):
    """Random small two-crew instance (2-4 x 2-5 grid, 3-9 fuel cells, speed/work time varied).

    Returns (instance, cells that burn with no response, excluding the ignition); cells the fire never
    reaches can never matter and the ignition cell cannot be defended, so enumerating only these is
    still every joint plan that could change the outcome.
    """
    rng = np.random.default_rng(seed)
    while True:
        rows, cols = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        fuel = rng.random((rows, cols)) < 0.65
        if 3 <= fuel.sum() <= 9:
            break
    cells = [tuple(map(int, c)) for c in np.argwhere(fuel)]
    ig = cells[rng.integers(len(cells))]
    inst = make_instance(
        fuel, rng.uniform(0.1, 1.0, fuel.shape), rng.uniform(0, 150, fuel.shape), [ig],
        [(int(rng.integers(rows)), int(rng.integers(cols))) for _ in range(2)],
        speed=float(rng.choice([1, 1, 1.5, 2, 3])), work_time=int(rng.integers(1, 4)),
        downhill_drop_m=float(rng.choice([20, 50])), downhill_time=int(rng.choice([2, 3])))
    cand = [inst.cell(i) for i in simulate(inst, Plan.empty(2)).order if inst.cell(i) != ig]
    return inst, cand


# Seeds on which the earlier exact solver (crew 0's route built first, each defence judged only
# against its own prefix) returned a worse plan while claiming optimality. Found by scanning 60000 seeds.
OLD_SOLVER_FAILED = [6229, 8337, 16327, 20850, 28932, 35921, 38900, 43176, 45292, 52110, 59452]


def joint_seeds(n_random=30):
    seeds, s = [], 0
    while len(seeds) < n_random:      # first usable random seeds: enumeration kept to <= 6 cells
        if 3 <= len(joint_instance(s)[1]) <= 6:
            seeds.append(s)
        s += 1
    return seeds + OLD_SOLVER_FAILED


def all_joint_plans(cand):
    """Every assignment of ordered, disjoint sequences to crew 0 and crew 1 (failing defences included)."""
    for s0 in sequences(cand):
        rest = [c for c in cand if c not in s0]
        for s1 in sequences(rest):
            yield s0, s1


@pytest.fixture(scope="module")
def joint_optima():
    out = []
    for seed in joint_seeds():
        inst, cand = joint_instance(seed)
        best = max(simulate(inst, Plan.of(p)).objective for p in all_joint_plans(cand))
        out.append((inst, best, simulate(inst, Plan.empty(2)).objective))
    return out


@pytest.mark.parametrize("warm_start", [True, False])
def test_exact_matches_brute_force_over_all_joint_two_crew_plans(joint_optima, warm_start):
    for inst, best, base in joint_optima:
        ex = solve_exact(inst, warm_start=warm_start)
        assert ex.proven_optimal
        assert ex.objective == pytest.approx(best)
        assert solve_greedy(inst).objective <= ex.objective + 1e-9


def test_joint_instances_are_not_trivial(joint_optima):
    saving = [best - base for _, best, base in joint_optima if best - base > 1e-9]
    assert len(joint_optima) >= 40 and len(saving) >= 0.5 * len(joint_optima)
    # Instances whose optimum needs both crews, and where crew 1 is not just a spare:
    # the best plan with crew 1 idle is strictly worse than the joint optimum.
    needs_both = 0
    for inst, best, _ in joint_optima:
        base_cells = [inst.cell(i) for i in simulate(inst, Plan.empty(2)).order if inst.cell(i) not in inst.ignitions]
        solo = max(simulate(inst, Plan.of([s0, ()])).objective for s0 in sequences(base_cells))
        needs_both += best > solo + 1e-9
    assert needs_both >= 8


def test_crew_defence_that_only_holds_because_the_other_crew_finished_earlier():
    # Regression: 4x2 grid, ignition (3,0), stations (2,0) and (1,1), speed 1, work 3.
    # Crew 1 finishes (1,1) at step 3; that delays the fire so crew 0's (0,1), finished at step 6,
    # arrives exactly as the fire does. Alone, crew 0's firebreak is too late.
    value = [[0.34339115875500953, 0.860481407625948], [0.7075777272017354, 0.7469933234947539],
             [0.7022488896440046, 0.5095836039084085], [0.6996701572580967, 0.9767275583610499]]
    dem = [[85.11464096243722, 5.473841847944655], [48.154082087325754, 36.85634581393821],
           [2.799105461887491, 144.36227516039511], [147.5794209316727, 38.08832187583761]]
    inst = make_instance(np.ones((4, 2)), value, dem, [(3, 0)], [(2, 0), (1, 1)], speed=1,
                         work_time=3, downhill_drop_m=50, downhill_time=2)
    assert not simulate(inst, Plan.of([[(0, 1)], []])).defences[0].held
    joint = simulate(inst, Plan.of([[(0, 1)], [(1, 1)]]))
    assert [d.held for d in joint.defences] == [True, True]
    assert [(d.finish, d.fire_arrival) for d in joint.defences] == [(6, 6), (3, 4)]
    assert joint.value_saved == pytest.approx(1.6074747311207016)
    for warm in (True, False):
        ex = solve_exact(inst, warm_start=warm)
        assert ex.proven_optimal
        assert ex.result.value_saved == pytest.approx(1.6074747311207016)


def test_reported_objective_equals_simulate_and_independent_sum():
    for seed in range(6):
        inst = random_instance(seed, n_crews=1 + seed % 2)
        for sol in (solve_greedy(inst), solve_exact(inst)):
            again = simulate(inst, sol.plan)
            assert sol.objective == again.objective == sol.result.objective
            burned_value = inst.value[again.burned].sum()
            assert again.objective == pytest.approx(inst.value[inst.fuel].sum() - burned_value)
            assert again.value_saved == pytest.approx(again.objective - simulate(inst, Plan.empty(inst.n_crews)).objective)
            assert all(d.held for d in again.defences)     # solvers only plan defences that hold


def reference_burn_times(inst, plan):
    """Step-by-step fire spread written independently of the Dijkstra in model.py."""
    rows, cols = inst.fuel.shape
    finish = {}
    for ci, route in enumerate(plan.routes):
        pos, t = inst.stations[ci], 0
        for cell in route:
            t += travel_time(pos, cell, inst.speed) + inst.work_time
            pos = cell
            if inst.fuel[cell]:
                finish[cell] = min(finish.get(cell, INF), t)
    burn = {c: 0 for c in inst.ignitions}
    for t in range(1, 200):
        new = {}
        for (r, c), bt in burn.items():
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                v = (r + dr, c + dc)
                if not (0 <= v[0] < rows and 0 <= v[1] < cols) or not inst.fuel[v] or v in burn:
                    continue
                w = inst.params["downhill_time"] if inst.dem_m[r, c] - inst.dem_m[v] > inst.params["downhill_drop_m"] else 1
                if bt + w == t and finish.get(v, INF) > t:
                    new[v] = t
        burn.update(new)
    return burn


def test_simulate_matches_step_by_step_reference_on_random_plans():
    rng = np.random.default_rng(7)
    for seed in range(15):
        inst = random_instance(seed, max_fuel=9, n_crews=1 + seed % 2)
        cells = [tuple(map(int, c)) for c in np.argwhere(inst.fuel)]
        for _ in range(10):
            plan = Plan.of([[cells[i] for i in rng.integers(len(cells), size=rng.integers(0, 4))]
                            for _ in range(inst.n_crews)])
            ref = reference_burn_times(inst, plan)
            got = simulate(inst, plan).burn_time
            for r in range(3):
                for c in range(4):
                    assert got[r, c] == ref.get((r, c), INF)


def test_two_crews_beat_one_on_a_split_fire():
    fuel = np.ones((1, 9), bool)
    mk = lambda stations: make_instance(fuel, np.ones((1, 9)), np.zeros((1, 9)), [(0, 4)], stations, speed=2)
    one = solve_exact(mk([(0, 2)]))
    two = solve_exact(mk([(0, 2), (0, 6)]))
    assert one.proven_optimal and two.proven_optimal
    assert one.result.value_saved == 3 and two.result.value_saved == 6


def test_exact_reports_unproven_when_limit_hit():
    fuel = np.ones((4, 5), bool)
    inst = make_instance(fuel, np.random.default_rng(1).uniform(0.1, 1, (4, 5)), np.zeros((4, 5)),
                         [(0, 0)], [(3, 4)], speed=1)
    sol = solve_exact(inst, node_limit=3)
    assert sol.proven_optimal is False and sol.stop_reason == "node limit"
    assert sol.objective >= solve_greedy(inst).objective


# 5. travel time and the moving constraint --------------------------------------------------

@pytest.mark.parametrize("a,b,speed,expected", [
    ((0, 0), (0, 0), 3, 0),
    ((0, 0), (3, 4), 1, 5),
    ((0, 0), (3, 4), 2, 3),      # 5 / 2 = 2.5 -> 3
    ((0, 0), (3, 4), 5, 1),      # exactly 1
    ((0, 0), (3, 0), 3, 1),
    ((0, 0), (0, 1), 100, 1),    # minimum 1 for a different cell
    ((0, 0), (6, 8), 3, 4),      # 10 / 3 -> 4
    ((2, 2), (1, 1), 1, 2),      # sqrt(2) -> 2
    ((0, 0), (1, 2), 1.5, 2),    # sqrt(5) = 2.236 / 1.5 = 1.49 -> 2
])
def test_travel_time_is_ceil_of_euclid_over_speed(a, b, speed, expected):
    assert travel_time(a, b, speed) == expected


def test_travel_table_agrees_with_travel_time():
    inst2 = make_instance(np.ones((4, 5), bool), np.ones((4, 5)), np.zeros((4, 5)), [(0, 0)], [(3, 4)], speed=2)
    table = inst2.travel_table()
    for a in range(20):
        for b in range(20):
            assert table[a, b] == travel_time(inst2.cell(a), inst2.cell(b), inst2.speed)


def test_moving_constraint_binds():
    def inst(speed):
        return corridor([0] * 8, ignition=0, station=7, speed=speed)
    plan = Plan.of([[(0, 3)]])       # fire reaches column 3 at step 3; crew starts 4 columns away
    fast = simulate(inst(3), plan)   # ceil(4/3) = 2 steps + 1 work = 3 <= 3
    slow = simulate(inst(1), plan)   # 4 steps + 1 work = 5 > 3
    assert fast.defences[0].held and fast.defences[0].finish == 3
    assert not slow.defences[0].held and slow.defences[0].finish == 5
    assert fast.objective == 5 and slow.objective == 0
