"""Tests for ranger_response model v2 (physical spread times, walking crews, squeeze rule) and its solvers."""
import itertools
import json
import math

import numpy as np
import pytest

from detection.fire_sim import (daytime_wind_pool, load_real_window, load_wind, simulate_scenarios)
from ranger_response import experiment as ex
from ranger_response.model import Plan
from ranger_response.model2 import make_instance2, simulate2
from ranger_response.solvers2 import (_candidates, _crew_state, _planned_flat, solve_exact2, solve_greedy2,
                                      solve_lookahead2, upper_bound2)

INF = math.inf


def grid_inst(fuel_factor, ignition, stations, wind=(0.0, 0.0), cut_h=0.5, r0=20.0, horizon_h=50.0,
              walk_kmh=2.0, dem=None, value=None, delay_h=0.0):
    ff = np.asarray(fuel_factor, float)
    return make_instance2(ff, np.ones_like(ff) if value is None else value,
                          np.zeros_like(ff) if dem is None else dem, [ignition], stations, wind[0], wind[1],
                          r0_m_min=r0, horizon_h=horizon_h, walk_kmh=walk_kmh, cut_h=cut_h, delay_h=delay_h)


# 1. the spread agrees with fire_sim ------------------------------------------------------------

def test_no_response_arrival_equals_fire_sim_on_real_window():
    s = ex.site_inputs()
    w = s["window"]
    sp, dr = daytime_wind_pool(load_wind())
    n = 6
    ss = simulate_scenarios(w["ignition_weights"], w["fuel"], w["elevation"], sp, dr, n, 7, r0=1.5,
                            horizon_hours=24.0)
    for k in range(n):
        ig = divmod(int(ss.ignition_cell[k]), 20)
        inst = make_instance2(w["fuel"], s["value"], s["dem"], [ig], [(0, 10)], ss.wind_speed_ms[k],
                              ss.wind_dir_deg[k], r0_m_min=1.5, horizon_h=24.0, cut_h=1.0)
        mine = np.array(simulate2(inst, Plan.empty(1)).burn_flat)
        ref = ss.arrival_min[k].astype(float) / 60.0
        assert np.array_equal(np.isfinite(mine), np.isfinite(ref))
        assert np.allclose(mine[np.isfinite(mine)], ref[np.isfinite(ref)], rtol=2e-6)


def test_hand_computed_arrival_times_with_fuel_and_wind():
    # 1 x 3 corridor, flat, r0 = 10 m/min; fuel factors 1.0, 0.5, 1.0.
    # time(0->1) = 0.5 * 500 * (1/1.0 + 1/0.5) / 10 = 75 min; time(1->2) the same.
    inst = grid_inst([[1.0, 0.5, 1.0]], (0, 0), [(0, 2)], r0=10.0)
    t = simulate2(inst, Plan.empty(1)).burn_time[0]
    assert t == pytest.approx([0.0, 1.25, 2.5])
    # wind blowing towards +col at 5 m/s: factor exp(0.12 * 5) = 1.8221 downwind (from the west = 270 deg)
    inst = grid_inst([[1.0, 1.0]], (0, 0), [(0, 1)], wind=(5.0, 270.0), r0=10.0)
    t = simulate2(inst, Plan.empty(1)).burn_time[0]
    assert t[1] == pytest.approx(500 / (10 * math.exp(0.6)) / 60)
    inst = grid_inst([[1.0, 1.0]], (0, 0), [(0, 1)], wind=(5.0, 90.0), r0=10.0)   # from the east: upwind
    assert simulate2(inst, Plan.empty(1)).burn_time[0][1] == pytest.approx(500 / (10 * math.exp(-0.6)) / 60)


# 2. crews: timing and the 'holds' rule ------------------------------------------------------------

def test_crew_timing_walking_and_cutting():
    inst = grid_inst(np.ones((1, 8)), (0, 0), [(0, 7)], cut_h=1.0, walk_kmh=2.0, r0=1.0)
    res = simulate2(inst, Plan.of([[(0, 5), (0, 3)]]))
    a, b = res.defences
    assert a.arrive == pytest.approx(2 * 0.5 / 2.0) and a.finish == pytest.approx(0.5 + 1.0)
    assert b.arrive == pytest.approx(1.5 + 2 * 0.5 / 2.0) and b.finish == pytest.approx(3.0)
    inst = grid_inst(np.ones((1, 8)), (0, 0), [(0, 7)], cut_h=1.0, delay_h=2.0)
    assert simulate2(inst, Plan.of([[(0, 7)]])).defences[0].finish == pytest.approx(3.0)


def corridor(n=6, cut_h=0.5, r0=10.0, station=5, **kw):
    return grid_inst(np.ones((1, n)), (0, 0), [(0, station)], cut_h=cut_h, r0=r0, **kw)


def test_firebreak_holds_blocks_spread_and_failure_is_inert():
    inst = corridor()                        # fire needs 500 / 10 = 50 min = 0.8333 h per cell
    res = simulate2(inst, Plan.of([[(0, 3)]]))      # walk 2 cells = 0.5 h, cut 0.5 -> done at 1.0 h < 2.5 h
    d = res.defences[0]
    assert d.held and d.finish == pytest.approx(1.0) and d.fire_arrival == pytest.approx(2.5)
    assert list(res.burned[0]) == [True, True, True, False, False, False]
    late = corridor(cut_h=2.5)               # finishes at 3.0 h > 2.5 h arrival: fails, fire passes
    res = simulate2(late, Plan.of([[(0, 3)]]))
    assert not res.defences[0].held and res.burned.all()
    exact_tie = corridor(cut_h=2.0)          # finishes at 2.5 h = arrival: holds
    assert simulate2(exact_tie, Plan.of([[(0, 3)]])).defences[0].held


def test_duplicates_and_nonburnable_defences_fail():
    ff = np.ones((1, 5))
    ff[0, 2] = 0
    inst = grid_inst(ff, (0, 0), [(0, 4)])
    res = simulate2(inst, Plan.of([[(0, 2)]]))
    assert not res.defences[0].held
    inst = corridor()
    res = simulate2(inst, Plan.of([[(0, 3), (0, 3)]]))
    assert [d.held for d in res.defences] == [True, False]


# 3. blocking, including diagonals --------------------------------------------------------------

def diag_inst(corner_ff=(1, 1), stations=((2, 2), (2, 2)), cut_h=0.2):
    """3 x 3 grid, ignition (0,0). Its diagonal neighbour (1,1) can be reached directly or via (0,1)/(1,0)."""
    ff = np.ones((3, 3))
    ff[0, 1], ff[1, 0] = corner_ff
    return grid_inst(ff, (0, 0), list(stations), cut_h=cut_h, r0=2.0)


def test_diagonal_squeeze_between_two_held_firebreaks_is_forbidden():
    inst = diag_inst()
    base = simulate2(inst, Plan.empty(2))
    assert base.burned.all()
    # Crew A blocks (0,1), crew B blocks (1,0). Their corner touches (0,0) and (1,1): fire in (0,0)
    # cannot slip diagonally to (1,1) between them, so only the ignition cell burns.
    res = simulate2(inst, Plan.of([[(0, 1)], [(1, 0)]]))
    assert all(d.held for d in res.defences)
    assert res.cells_burned == 1 and res.burned[0, 0] and not res.burned[1, 1]
    # one firebreak alone does not stop the diagonal: (1,1) and (1,0) still burn
    one = simulate2(inst, Plan.of([[(0, 1)], []]))
    assert one.defences[0].held and one.burned[1, 1] and one.burned[1, 0]


def test_failed_firebreak_does_not_close_the_diagonal():
    inst = diag_inst()
    # B's cut finishes after the fire is already in (1,0): it fails, so the squeeze stays open
    slow = inst.with_params(cut_h=0.2)
    res = simulate2(slow, Plan.of([[(0, 1)], [(1, 0)]]))
    assert all(d.held for d in res.defences)           # both finish ~1.2 h; fire needs far longer (r0 = 2)
    fast_fire = grid_inst(np.ones((3, 3)), (0, 0), [(2, 2), (2, 2)], cut_h=3.0, r0=40.0)
    res = simulate2(fast_fire, Plan.of([[(0, 1)], [(1, 0)]]))
    assert not any(d.held for d in res.defences) and res.burned.all()
    # one held break + one failed break: the failed one must not count as blocking the corner
    mixed = grid_inst(np.ones((3, 3)), (0, 0), [(0, 2), (2, 2)], cut_h=0.2, r0=12.0)
    res = simulate2(mixed, Plan.of([[(0, 1)], [(2, 1), (1, 0)]]))
    d = {x.cell: x for x in res.defences}
    assert d[(0, 1)].held and not d[(1, 0)].held and d[(1, 0)].finish > d[(1, 0)].fire_arrival
    assert res.burned[1, 1] and res.burned[1, 0]


def test_diagonal_squeeze_counts_a_nonburnable_corner_but_natural_gaps_stay_open():
    ff = np.ones((3, 3))
    ff[0, 1] = 0                                        # water north-east of the ignition's east neighbour
    inst = grid_inst(ff, (0, 0), [(2, 2)], cut_h=0.2, r0=2.0)
    res = simulate2(inst, Plan.of([[(1, 0)]]))          # firebreak on the other corner
    assert res.defences[0].held and res.cells_burned == 1
    ff[1, 0] = 0                                        # two natural gaps: fire_sim lets fire slip through
    res = simulate2(grid_inst(ff, (0, 0), [(2, 2)]), Plan.empty(1))
    assert res.burned[1, 1] and res.cells_burned == 7


def test_arrival_at_a_held_cell_does_not_count_a_squeezed_arrival_as_threat():
    # If both corners around a diagonal are firebreaks, a break placed on the diagonal target (1,1)
    # sees no fire at all (never reached), so it is 'held (fire does not reach...)'.
    inst = diag_inst()
    res = simulate2(inst, Plan.of([[(0, 1), (1, 1)], [(1, 0)]]))
    d = {x.cell: x for x in res.defences}
    assert d[(0, 1)].held and d[(1, 0)].held and d[(1, 1)].fire_arrival == INF and d[(1, 1)].held


# 4. B&B vs brute force on tiny instances; bound validity -----------------------------------------------

def tiny(seed, n_crews=1, shape=(3, 3), holes=True):
    rng = np.random.default_rng(seed)
    ff = rng.choice([0.5, 0.6, 0.9, 1.0], size=shape)
    if holes:
        ff[rng.random(shape) < 0.15] = 0.0
    burn = np.argwhere(ff > 0)
    ig = tuple(int(x) for x in burn[rng.integers(len(burn))])
    st = [tuple(int(x) for x in rng.integers(0, s, 2)) if False else (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
          for _ in range(n_crews)]
    return grid_inst(ff, ig, st, wind=(float(rng.uniform(0, 10)), float(rng.uniform(0, 360))),
                     cut_h=float(rng.uniform(0.15, 0.7)), r0=float(rng.uniform(10, 25)),
                     horizon_h=float(rng.uniform(2.0, 5.0)), walk_kmh=float(rng.uniform(1.0, 3.0)),
                     dem=rng.uniform(0, 80, shape), value=rng.uniform(0.5, 3.0, shape))


def sequences(cells, max_len):
    out = [()]
    for L in range(1, max_len + 1):
        out += list(itertools.permutations(cells, L))
    return out


def brute_best(inst, max_len=3):
    cells = [tuple(c) for c in np.argwhere(inst.burnable)]
    seqs = sequences(cells, max_len)
    if inst.n_crews == 1:
        return max(simulate2(inst, Plan.of([s])).objective for s in seqs)
    seqs = sequences(cells, max_len)
    return max(simulate2(inst, Plan.of([a, b])).objective for a in seqs for b in seqs if not set(a) & set(b))


def assert_matches_brute_force(inst, max_len, time_limit):
    e = solve_exact2(inst, time_limit=time_limit)
    best = brute_best(inst, max_len)
    assert e.proven_optimal
    assert e.objective >= best - 1e-9                      # exact never loses to any enumerated plan
    if max(len(r) for r in e.plan.routes) <= max_len:      # the enumeration covers exact's own plan size
        assert e.objective == pytest.approx(best, abs=1e-9)
    assert e.objective >= solve_greedy2(inst).objective - 1e-9
    assert e.objective >= solve_lookahead2(inst).objective - 1e-9


@pytest.mark.parametrize("seed", range(30))
def test_exact_matches_brute_force_one_crew(seed):
    assert_matches_brute_force(tiny(seed, shape=(3, 3) if seed % 2 else (3, 4)), 4 if seed % 2 else 3, 60)


@pytest.mark.parametrize("seed", range(8))
def test_exact_matches_brute_force_two_crews(seed):
    inst = tiny(100 + seed, n_crews=2, shape=(3, 3))
    e = solve_exact2(inst, time_limit=60)
    best = brute_best(inst, 3)
    assert e.proven_optimal and e.objective >= best - 1e-9
    if max(len(r) for r in e.plan.routes) <= 3:
        assert e.objective == pytest.approx(best, abs=1e-9)


def test_unreached_cell_firebreak_can_close_a_diagonal_squeeze():
    # Regression found by brute force (tiny seed 463, 3 x 4): a firebreak on a cell the fire never
    # reaches within the horizon still saves cells, because with the non-burnable cell (0,0) it blocks the
    # diagonal step (1,0) -> (0,1). The exact solver must therefore consider corner cells.
    inst = tiny(463, shape=(3, 4))
    best_plan = Plan.of([[(2, 1), (1, 1), (2, 3)]])
    assert not simulate2(inst, Plan.empty(1)).burned[1, 1]
    with_corner = simulate2(inst, Plan.of([[(2, 1), (1, 1)]]))
    without = simulate2(inst, Plan.of([[(2, 1)]]))
    assert with_corner.objective > without.objective + 0.5 and with_corner.defences[1].held
    e = solve_exact2(inst, time_limit=30)
    assert e.proven_optimal and e.objective >= simulate2(inst, best_plan).objective - 1e-9


def test_tiny_instances_are_nontrivial_and_use_diagonals():
    gains, wins = 0, 0
    for seed in range(14):
        inst = tiny(seed)
        base = simulate2(inst, Plan.empty(1)).objective
        e = solve_exact2(inst, time_limit=30)
        gains += e.objective > base + 1e-9
        wins += e.objective > solve_greedy2(inst).objective + 1e-9
    assert gains >= 6                                    # many instances where defending helps
    assert wins >= 1                                     # and at least one where greedy is not optimal


@pytest.mark.parametrize("seed", range(10))
def test_upper_bound_is_valid_at_root_and_children_one_crew(seed):
    inst = tiny(seed + 40, shape=(3, 4))
    cells = [tuple(c) for c in np.argwhere(inst.burnable)]
    seqs = sequences(cells, 3)
    objs = {s: simulate2(inst, Plan.of([s])).objective for s in seqs}
    root = simulate2(inst, Plan.of([[]]))
    planned = set()
    state = [(inst.stations[0], inst.delay_h)]
    assert upper_bound2(inst, root, planned, state, -INF) >= max(objs.values()) - 1e-9
    for q, fin in _candidates(inst, root, planned, *state[0]):
        cell = inst.cell(q)
        child = simulate2(inst, Plan.of([[cell]]))
        best_ext = max(o for s, o in objs.items() if s[:1] == (cell,))
        ub = upper_bound2(inst, child, {q}, [(cell, fin)], fin)
        assert ub >= best_ext - 1e-9


def test_upper_bound_accounts_for_a_diagonal_squeeze_closed_by_a_corner_cell():
    # Fire at (0,0) of a 2 x 2 grid. Cells (0,1) and (1,0) are firebreaks planned by a crew; the
    # diagonal target (1,1) is saved only through the squeeze rule, never by a break on its own tree path.
    inst = grid_inst(np.ones((2, 2)), (0, 0), [(1, 1), (1, 1)], cut_h=0.2, r0=2.0)
    root = simulate2(inst, Plan.of([[], []]))
    child = simulate2(inst, Plan.of([[(0, 1)], []]))
    states = [((0, 1), 1.0 + 0.0), ((1, 1), 0.0)]
    # after the first firebreak the tree path of (1,1) is (0,0) -> (1,1); only the corner cell (1,0) can save it
    ub = upper_bound2(inst, child, {1}, [((1, 1), 0.0), ((1, 1), 0.0)], 0.0)
    assert ub >= simulate2(inst, Plan.of([[(0, 1)], [(1, 0)]])).objective - 1e-9
    assert root.objective <= child.objective <= ub


# 5. heuristics --------------------------------------------------------------------------------------

def test_greedy_lookahead_exact_ordering_on_known_instances():
    # Seeds found by a search over tiny(seed, shape=(3, 4)); all three methods are scored by simulate2 and
    # exact is proven optimal (and equal to brute force in the other tests).
    inst = tiny(226, shape=(3, 4))
    g, la, e = solve_greedy2(inst), solve_lookahead2(inst), solve_exact2(inst, time_limit=20)
    assert la.result.value_saved > g.result.value_saved + 10 and e.proven_optimal
    assert e.objective == pytest.approx(la.objective)
    inst = tiny(284, shape=(3, 4))
    g, la, e = solve_greedy2(inst), solve_lookahead2(inst), solve_exact2(inst, time_limit=20)
    assert g.objective < la.objective < e.objective - 1.0 and e.proven_optimal


# 6. experiment and dashboard exporter -----------------------------------------------------------------

from ranger_response import export_dashboard as xd


def test_default_cut_time_is_the_documented_derivation():
    assert ex.cut_hours(ex.BASE_CONFIG) == pytest.approx(500 / (0.16 * 10 * 60))      # 5.208 h
    assert ex.cut_hours(ex.BASE_CONFIG, 2.0) == pytest.approx(ex.cut_hours(ex.BASE_CONFIG) / 2)


def test_scenarios_are_seeded_inside_the_window_and_drawn_from_the_daytime_wind_pool():
    a, b = ex.make_scenarios(20, 42), ex.make_scenarios(20, 42)
    assert a == b and len(a) == 20 and ex.make_scenarios(20, 43) != a
    sp, dr = daytime_wind_pool(load_wind())
    pool = set(zip(np.round(sp, 6), np.round(dr, 6)))
    for s in a:
        r, c = s["ignition"]
        assert 0 <= r < 20 and 0 <= c < 20 and s["ignition_weight"] > 0
        assert (round(s["wind_speed_ms"], 6), round(s["wind_from_deg"], 6)) in pool


def test_site_window_matches_results_json_surface():
    import os
    s = ex.site_inputs()
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboards", "public", "results.json")
    if os.path.exists(path):
        site = json.load(open(path))
        if site.get("terrain") == "real":
            assert np.allclose(np.array(site["layers"]["elevation_m"]), s["dem"], atol=0.1)
            assert np.array_equal(np.array(site["layers"]["fire_count"]), s["surf"]["fire_count"])
    assert s["window"]["burnable"].all() and s["fuel_factor"].min() > 0


def test_pick_featured_rules_on_hand_made_results():
    scens = [dict(id=i, ignition_weight=w) for i, w in enumerate([0.5, 1.2, 1.2, 0.3])]
    def rec(i, g, e):
        return dict(scenario=i, crews=1, methods=dict(greedy=dict(pct_saved=g), exact=dict(pct_saved=e)))
    results = [rec(0, 1.0, 5.0), rec(1, 2.0, 2.0), rec(2, 0.0, 9.0), rec(3, 4.0, 5.0)]   # gaps 4, 0, 9, 1
    got = xd.pick_featured(scens, results)
    assert got == {"most-fire-prone": 1, "median-gap": 3, "largest-gap": 2}               # lower median of [0,1,4,9] = 1 -> id 3


@pytest.fixture(scope="module")
def small_export(tmp_path_factory):
    exp = ex.run_experiment(n_scenarios=4, seed=42, exact_limit=2.0, sens_n=0, workers=1, log=lambda *_: None)
    path = tmp_path_factory.mktemp("rr") / "exp.json"
    path.write_text(json.dumps(exp))
    return exp, json.loads(json.dumps(xd.build(str(path))))


def test_export_schema(small_export):
    exp, d = small_export
    assert d["model"] == "v2" and d["terrain"] == "real" and d["grid"]["rows"] == d["grid"]["cols"] == 20
    assert d["grid"]["window"] == [15, 3] and set(d["grid"]["bounds"]) == {"north", "south", "west", "east"}
    assert len(d["generated"]) == 10 and len(d["value"]) == 400
    assert d["caveats"] and all(isinstance(c, str) for c in d["caveats"])
    assert any("24 hours" in c for c in d["caveats"]) and any("validated" in c for c in d["caveats"])
    for k in ("horizon_h", "r0_m_min", "walk_kmh", "cut_rate_m_min_per_person", "cut_h_per_cell", "crew_size"):
        assert {"value", "unit", "source"} <= set(d["parameters"][k])
    e = d["experiment"]
    assert e["n_scenarios"] == 4 and len(e["per_scenario"]) == 4 and set(e["summary"]) == {"1", "2"}
    for m in xd.METHODS:
        assert "mean_pct_saved" in e["summary"]["1"]["methods"][m]
    assert {"exact - greedy", "exact - lookahead"} <= set(e["summary"]["1"]["gaps"])
    assert d["featured"] and all(fs["roles"] for fs in d["featured"])
    for fs in d["featured"]:
        assert [r["crews"] for r in fs["runs"]] == [1, 2]
        for r in fs["runs"]:
            assert r["best"] in xd.METHODS and len(r["baseline"]["arrival_h"]) == 400
            for m in r["methods"].values():
                assert len(m["arrival_h"]) == 400
                assert set(m) >= {"value_saved", "pct_saved", "cells_burned", "proven_optimal", "stop_reason", "elapsed_s", "defences"}
                for df in m["defences"]:
                    assert set(df) == {"crew", "order", "cell", "arrive_h", "finish_h", "fire_arrival_h", "held"}


def test_export_is_consistent_with_its_own_arrays_and_replays(small_export):
    exp, d = small_export
    value = np.array(d["value"])
    for fs in d["featured"]:
        scen = exp["scenarios"][fs["scenario"]]
        for r in fs["runs"]:
            base = np.array([v is None for v in r["baseline"]["arrival_h"]])
            assert (~base).sum() == r["baseline"]["cells_burned"]
            assert value[~base].sum() == pytest.approx(r["baseline"]["value_burned"], abs=0.01)
            for name, m in r["methods"].items():
                never = np.array([v is None for v in m["arrival_h"]])
                assert (~never).sum() == m["cells_burned"]
                kept_base = value.sum() - value[~base].sum()
                assert value[never].sum() - kept_base == pytest.approx(m["value_saved"], abs=0.02)
                assert m["pct_saved"] == pytest.approx(100 * m["value_saved"] / r["baseline"]["value_burned"], abs=0.01)
                for df in m["defences"]:
                    cell = df["cell"][0] * 20 + df["cell"][1]
                    if df["held"]:
                        assert never[cell] and (df["fire_arrival_h"] is None or df["finish_h"] <= df["fire_arrival_h"] + 0.011)
                    assert df["finish_h"] > df["arrive_h"]
                # replay the stored plan from the experiment file
                rec = next(x for x in exp["results"] if x["scenario"] == fs["scenario"] and x["crews"] == r["crews"])
                inst = ex.build_instance(scen, r["crews"])
                res = simulate2(inst, Plan.of([[tuple(c) for c in rt] for rt in rec["methods"][name]["plan"]]))
                assert res.value_saved == pytest.approx(m["value_saved"], abs=1e-3)
            # the labelled best has the highest value saved, and the exact result is never below either heuristic
            best_val = max(m["value_saved"] for m in r["methods"].values())
            assert r["methods"][r["best"]]["value_saved"] == pytest.approx(best_val, abs=1e-3)
            assert r["methods"]["exact"]["value_saved"] >= max(r["methods"]["greedy"]["value_saved"],
                                                                r["methods"]["lookahead"]["value_saved"]) - 1e-6


def test_experiment_summary_counts_and_stays_small(small_export):
    exp, d = small_export
    assert len(json.dumps(d, separators=(",", ":"))) < 120 * 1024
    for crews in ("1", "2"):
        s = exp["summary"][crews]
        assert s["n"] == 4
        for g in s["gaps"].values():
            assert g["n_positive"] + g["n_zero"] + g["n_negative"] == 4
        assert s["gaps"]["exact - greedy"]["n_negative"] == 0 and s["gaps"]["exact - lookahead"]["n_negative"] == 0


def test_cli_v2_runs_a_scenario(tmp_path, capsys):
    from ranger_response import run
    out = tmp_path / "plan.png"
    run.main(["--scenario", "0", "--crews", "1", "--exact-limit", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert "model v2" in text and "lookahead" in text and "exact B&B" in text and out.exists()
