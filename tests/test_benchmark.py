"""Benchmark aggregation and the real-terrain window picker (the full benchmark is not run here)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark as bm

NAN = float("nan")


def row(n_sites, n_sensors, seed, ilp=10.0, **vals):
    r = {c: NAN for c in bm.RUN_COLS}
    r.update(terrain="real", n_sites=n_sites, n_sensors=n_sensors, seed=seed,
             window_row=0, window_col=0, ilp_objective=ilp)
    r.update(vals)
    return r


def test_aggregate_means_std_shares_and_counts():
    rows = [
        row(8, 2, 0, greedy=1.0, sa=1.0, qaoa_mean=0.2, qaoa_p_optimal=0.1),
        row(8, 2, 1, greedy=0.9, sa=1.0, qaoa_mean=0.4, qaoa_p_optimal=0.3),
        row(8, 2, 2, greedy=1.0, sa=0.8, qaoa_mean=0.6, qaoa_p_optimal=0.2),
        row(12, 3, 0, greedy=0.5, sa=0.5, qaoa_mean=0.1),
        row(12, 3, 1, greedy=0.7, sa=1.0, qaoa_mean=0.3),
    ]
    s = bm.aggregate_runs(rows)
    assert [x["instance"] for x in s] == ["|S|=8, N=2", "|S|=12, N=3"]
    a, b = s
    assert a["terrain"] == "real" and a["n_instances"] == 3
    assert a["greedy"] == pytest.approx(np.mean([1.0, 0.9, 1.0]))
    assert a["greedy_std"] == pytest.approx(np.std([1.0, 0.9, 1.0], ddof=1))   # n - 1
    assert a["qaoa_mean"] == pytest.approx(0.4)
    assert a["qaoa_mean_std"] == pytest.approx(0.2)
    assert a["qaoa_p_optimal"] == pytest.approx(0.2)
    assert a["greedy_optimal_share"] == pytest.approx(2 / 3)
    assert a["sa_optimal_share"] == pytest.approx(2 / 3)
    assert b["n_instances"] == 2
    assert b["sa_optimal_share"] == pytest.approx(0.5)
    assert b["qaoa_mean_std"] == pytest.approx(np.std([0.1, 0.3], ddof=1))


def test_aggregate_ignores_nan_and_invalid_ilp():
    rows = [
        row(8, 2, 0, greedy=1.0, sa=NAN, qaoa_mean=0.2),
        row(8, 2, 1, greedy=0.5, sa=0.6, qaoa_mean=NAN),
        row(8, 2, 2, ilp=NAN, greedy=0.0, sa=0.0, qaoa_mean=0.9),   # ILP failed: excluded
        row(8, 2, 3, greedy=1.0 - 1e-9, sa=0.6, qaoa_mean=0.4),     # within tolerance of optimal
    ]
    (s,) = bm.aggregate_runs(rows)
    assert s["n_instances"] == 3
    assert s["greedy"] == pytest.approx((1.0 + 0.5 + (1.0 - 1e-9)) / 3)
    assert s["sa"] == pytest.approx(0.6)
    assert s["sa_std"] == pytest.approx(0.0)
    assert s["qaoa_mean"] == pytest.approx(0.3)
    assert s["greedy_optimal_share"] == pytest.approx(2 / 3)
    assert s["sa_optimal_share"] == pytest.approx(0.0)
    # a column with no values at all, and a single value, give NaN rather than a crash
    assert math.isnan(s["xy_qaoa_mean"]) and math.isnan(s["xy_qaoa_mean_std"])
    rows = [row(8, 2, 0, greedy=0.7)]
    (s,) = bm.aggregate_runs(rows)
    assert s["greedy"] == pytest.approx(0.7) and math.isnan(s["greedy_std"])


@pytest.fixture(scope="module")
def inside_park():
    return bm._real_inside_park()


def test_window_picker_deterministic_and_obeys_rule(inside_park):
    assert inside_park.shape == (40, 40)
    cands = bm.real_window_candidates(inside_park)
    assert 0 < len(cands) < 21 * 21
    picks = [bm.pick_real_window(s, inside_park) for s in range(20)]
    assert picks == [bm.pick_real_window(s, inside_park) for s in range(20)]
    for r0, c0 in picks:
        assert 0 <= r0 <= 20 and 0 <= c0 <= 20
        assert inside_park[r0:r0 + 20, c0:c0 + 20].mean() >= 0.5
    # the rule is not vacuous: some windows are rejected
    rejected = [(r, c) for r in range(21) for c in range(21) if (r, c) not in set(cands)]
    for r0, c0 in rejected[:5]:
        assert inside_park[r0:r0 + 20, c0:c0 + 20].mean() < 0.5


def test_different_seeds_can_give_different_windows(inside_park):
    assert len({bm.pick_real_window(s, inside_park) for s in range(20)}) > 1
