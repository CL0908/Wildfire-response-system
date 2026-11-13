"""Time-to-detection objective and its greedy / exact solvers (detection/objective.py,
detection/solvers.py): hand-computed cases, exact vs brute force, the greedy guarantee, and one
end-to-end case through the fire simulator. All inputs are synthetic; no network, no data files."""
import itertools
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection import fire_sim as fs  # noqa: E402
from detection.objective import score_detection, site_detection_times  # noqa: E402
from detection.solvers import exact_detection, greedy_detection  # noqa: E402

INF = np.inf


def random_instance(seed, n_scen=30, n_sites=7):
    r = np.random.default_rng(seed)
    T = r.gamma(2.0, 100.0, size=(n_scen, n_sites))
    T[r.random(T.shape) < 0.3] = INF
    T[r.random(T.shape) < 0.05] = 0.0
    baseline = r.uniform(150.0, 500.0, size=n_scen) if seed % 2 else 360.0
    weights = r.uniform(0.1, 3.0, size=n_scen) if seed % 3 == 0 else None
    return T, baseline, weights


def brute_force(T, k, baseline, weights):
    best = max(score_detection(T, c, baseline, weights)["expected_saved_min"]
               for c in itertools.combinations(range(T.shape[1]), k))
    return best


# --- site_detection_times ---------------------------------------------------------------------

def test_site_detection_times_hand_case():
    arrival = np.array([[0, 10, 20, INF],
                        [INF, 5, 15, 30],
                        [INF, INF, INF, INF]], dtype=np.float32)
    sees = np.array([[1, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 0],      # sees nothing
                     [0, 0, 0, 1]], dtype=bool)
    T = site_detection_times(arrival, sees)
    expected = np.array([[0, 20, INF, INF],
                         [5, 15, INF, 30],
                         [INF, INF, INF, INF]])
    assert T.shape == (3, 4)
    np.testing.assert_array_equal(T, expected)


# --- score_detection --------------------------------------------------------------------------

def test_score_detection_hand_case_scalar_baseline():
    T = np.array([[0.0, 100.0], [50.0, 400.0], [INF, 200.0]])
    s = score_detection(T, [0], 360.0)          # d = [0, 50, 360]
    np.testing.assert_allclose(s["detection_min"], [0, 50, 360])
    assert s["expected_saved_min"] == pytest.approx((360 + 310 + 0) / 3)
    assert s["mean_detection_min"] == pytest.approx(410 / 3)
    assert s["share_detected_at_time_0"] == pytest.approx(1 / 3)
    assert s["share_detected_by_sensor_after_spread"] == pytest.approx(1 / 3)
    assert s["share_left_to_satellite"] == pytest.approx(1 / 3)

    s2 = score_detection(T, [1, 0], 360.0)      # d = [0, 50, 200]; order of `chosen` is irrelevant
    assert s2["chosen"] == [0, 1]
    assert s2["expected_saved_min"] == pytest.approx((360 + 310 + 160) / 3)
    assert s2["share_left_to_satellite"] == pytest.approx(0.0)

    empty = score_detection(T, [], 360.0)
    assert empty["expected_saved_min"] == 0.0 and empty["share_left_to_satellite"] == 1.0


def test_score_detection_per_scenario_baseline_and_weights():
    T = np.array([[0.0, 100.0], [50.0, 400.0], [INF, 200.0]])
    base = np.array([100.0, 360.0, 150.0])
    w = np.array([1.0, 2.0, 1.0])               # normalised to 0.25, 0.5, 0.25
    s = score_detection(T, [0], base, w)        # d = [0, 50, 150]; saved = [100, 310, 0]
    assert s["expected_saved_min"] == pytest.approx(0.25 * 100 + 0.5 * 310)
    assert s["share_detected_at_time_0"] == pytest.approx(0.25)
    assert s["share_detected_by_sensor_after_spread"] == pytest.approx(0.5)
    assert s["share_left_to_satellite"] == pytest.approx(0.25)
    # a site that is only slower than the baseline saves nothing
    assert score_detection(T, [1], base, w)["expected_saved_min"] == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_shares_sum_to_one(seed):
    T, base, w = random_instance(seed)
    s = score_detection(T, [0, 3], base, w)
    total = (s["share_detected_at_time_0"] + s["share_detected_by_sensor_after_spread"]
             + s["share_left_to_satellite"])
    assert total == pytest.approx(1.0)


# --- exact and greedy against brute force -----------------------------------------------------

CASES = [(seed, k) for seed in range(5) for k in (1, 2, 3, 4)]


@pytest.mark.parametrize("seed, k", CASES)
def test_exact_matches_brute_force(seed, k):
    T, base, w = random_instance(seed)
    res = exact_detection(T, k, base, w)
    assert res.proven_optimal and res.status == "Optimal"
    assert len(res.chosen) <= k
    assert res.objective == pytest.approx(brute_force(T, k, base, w), rel=1e-7, abs=1e-7)
    # reported objective is the shared scorer's value for the returned placement
    assert res.objective == score_detection(T, res.chosen, base, w)["expected_saved_min"]
    assert res.details["model_objective"] == pytest.approx(res.objective, rel=1e-7, abs=1e-7)


@pytest.mark.parametrize("seed, k", CASES)
def test_greedy_bounds(seed, k):
    T, base, w = random_instance(seed)
    ex = exact_detection(T, k, base, w).objective
    gr = greedy_detection(T, k, base, w)
    assert len(gr.chosen) == k
    assert gr.objective == score_detection(T, gr.chosen, base, w)["expected_saved_min"]
    assert gr.objective <= ex + 1e-9
    assert gr.objective >= (1 - 1 / math.e) * ex - 1e-9
    if k == 1:
        assert gr.objective == pytest.approx(ex, rel=1e-9)


def test_greedy_tie_break_lowest_index_and_fills_budget():
    T = np.full((5, 4), INF)
    T[:, 2] = 10.0
    T[:, 3] = 10.0                              # sites 2 and 3 tie; 0 and 1 are useless
    assert greedy_detection(T, 1).chosen == [2]
    # after site 2 nothing gains; the budget is filled with the lowest unused indices
    assert greedy_detection(T, 3).chosen == [0, 1, 2]


def test_exactly_k_pads_without_changing_objective():
    T, base, w = random_instance(1)
    at_most = exact_detection(T, 4, base, w)
    exactly = exact_detection(T, 4, base, w, exactly_k=True)
    assert len(exactly.chosen) == 4
    assert exactly.objective == pytest.approx(at_most.objective)


# --- properties of the objective --------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_monotone_and_diminishing_returns(seed):
    T, base, w = random_instance(seed, n_scen=40, n_sites=8)
    rng = np.random.default_rng(100 + seed)

    def f(s):
        return score_detection(T, sorted(s), base, w)["expected_saved_min"]

    for _ in range(25):
        order = [int(i) for i in rng.permutation(T.shape[1])]
        j = order[0]
        sizes = sorted(rng.choice(np.arange(0, 7), size=2, replace=False))
        small, large = set(order[1:1 + sizes[0]]), set(order[1:1 + sizes[1]])  # small is a subset
        gain_small = f(small | {j}) - f(small)
        gain_large = f(large | {j}) - f(large)
        assert gain_small >= -1e-9 and gain_large >= -1e-9          # monotone
        assert gain_small >= gain_large - 1e-9                       # diminishing returns


def test_merging_duplicate_scenarios_keeps_the_optimum():
    r = np.random.default_rng(7)
    base_rows = r.gamma(2.0, 100.0, size=(10, 6))
    base_rows[r.random(base_rows.shape) < 0.3] = INF
    pick = r.integers(0, 10, size=60)                   # 60 scenarios drawn from 10 distinct rows
    T = base_rows[pick]
    w = r.uniform(0.5, 2.0, size=60)
    res = exact_detection(T, 2, 360.0, w)
    assert res.details["n_distinct_rows"] <= 10
    assert res.objective == pytest.approx(brute_force(T, 2, 360.0, w), rel=1e-7, abs=1e-7)

    T0, base, w0 = random_instance(2)
    one = exact_detection(T0, 3, base, w0)
    two = exact_detection(np.vstack([T0, T0]), 3, np.tile(base, 2) if np.ndim(base) else base,
                          None if w0 is None else np.tile(w0, 2))
    assert two.objective == pytest.approx(one.objective, rel=1e-9)
    assert two.details["n_distinct_rows"] == one.details["n_distinct_rows"]


def test_input_validation():
    T = np.ones((3, 2))
    with pytest.raises(ValueError):
        exact_detection(T, 3)
    with pytest.raises(ValueError):
        greedy_detection(T, 1, baseline_min=np.ones(4))
    with pytest.raises(ValueError):
        score_detection(T, [0, 0])


# --- end to end through the simulator ---------------------------------------------------------

def test_end_to_end_with_fire_sim_on_a_synthetic_grid():
    rows = cols = 12
    shape = (rows, cols)
    ign = np.zeros(shape)
    ign[6, 2] = 1.0                                     # every scenario ignites at (6, 2)
    sc = fs.simulate_scenarios(ign, np.ones(shape), np.zeros(shape), 0.0, 0.0, 5, seed=0,
                               r0=2.0, horizon_hours=500.0)
    assert sc.arrival_min.dtype == np.float32 and sc.arrival_min.shape == (5, rows * cols)

    cell = lambda r, c: r * cols + c                    # noqa: E731
    sees = np.zeros((3, rows * cols), dtype=bool)
    sees[0, cell(6, 6)] = True                          # 4 cells due east of the ignition
    sees[1, cell(6, 2)] = True                          # the ignition cell itself
    sees[2, :] = False                                  # sees nothing
    T = site_detection_times(sc.arrival_min, sees)
    assert T.shape == (5, 3)
    np.testing.assert_allclose(T[:, 0], 4 * 500.0 / 2.0, rtol=1e-5)    # 4 cells x 500 m at 2 m/min
    np.testing.assert_array_equal(T[:, 1], 0.0)
    assert np.isinf(T[:, 2]).all()

    base = 2000.0
    assert score_detection(T, [0], base)["expected_saved_min"] == pytest.approx(base - 1000.0)
    assert score_detection(T, [1], base)["share_detected_at_time_0"] == 1.0
    assert score_detection(T, [2], base)["share_left_to_satellite"] == 1.0
    assert exact_detection(T, 1, base).chosen == [1]
    assert greedy_detection(T, 1, base).chosen == [1]
