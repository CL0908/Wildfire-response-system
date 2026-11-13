"""Whole-park placement: sparse solvers against brute force and the small-instance solvers, the
coverage scorer on a hand case, and one smoke test of the quick path. No network, no visibility cache."""
import itertools

import numpy as np
import pytest
from scipy import sparse

from detection.objective import site_detection_times
from park.instance import build_park_instance, load_park_data
from park.run import (_permille, coverage_shares, load_inputs, run_analysis)
from park.solvers import (coverage_bound, coverage_value, greedy_cover,
                          site_detection_times_sparse)
from solvers.greedy import greedy_coverage


def random_instance(seed, n_sites=7, n_cells=14, density=0.35):
    rng = np.random.default_rng(seed)
    dense = rng.random((n_sites, n_cells)) < density
    dense[np.arange(n_sites), rng.integers(0, n_cells, n_sites)] = True   # every site sees something
    return sparse.csr_matrix(dense), rng.random(n_cells) + 0.01


def brute_force(sees, w, k):
    return max(coverage_value(sees, w, c) for c in itertools.combinations(range(sees.shape[0]), k))


@pytest.mark.parametrize("seed", range(25))
def test_greedy_k1_is_the_brute_force_optimum(seed):
    sees, w = random_instance(seed)
    chosen, cum = greedy_cover(sees, w, 1)
    assert cum[0] == pytest.approx(brute_force(sees, w, 1))
    assert coverage_value(sees, w, chosen) == pytest.approx(cum[0])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("k", [2, 3])
def test_greedy_is_feasible_and_within_its_guarantee(seed, k):
    sees, w = random_instance(seed)
    chosen, cum = greedy_cover(sees, w, k)
    opt = brute_force(sees, w, k)
    assert len(set(chosen)) == k
    assert cum[-1] == pytest.approx(coverage_value(sees, w, chosen))
    assert cum[-1] <= opt + 1e-9
    assert cum[-1] >= (1 - 1 / np.e) * opt - 1e-9


def test_greedy_matches_the_small_instance_solver():
    sees, w = random_instance(3, n_sites=30, n_cells=80, density=0.1)
    views = {i: set(map(int, sees.indices[sees.indptr[i]:sees.indptr[i + 1]])) for i in range(30)}
    ref_chosen, ref_total = greedy_coverage(views, {c: float(w[c]) for c in range(80)}, 6)
    chosen, cum = greedy_cover(sees, w, 6)
    assert sorted(chosen) == sorted(ref_chosen)
    assert cum[-1] == pytest.approx(ref_total)


@pytest.mark.parametrize("seed", range(6))
def test_bound_is_at_least_the_optimum_and_integer_solve_finds_it(seed):
    sees, w = random_instance(seed, n_sites=8, n_cells=16)
    for k in (1, 3):
        opt = brute_force(sees, w, k)
        b = coverage_bound(sees, w, k, integer=True, time_limit_s=20)
        assert b["lp_value"] >= opt - 1e-7
        assert b["integer_proven"] and b["integer_value"] == pytest.approx(opt, abs=1e-6)
        assert b["integer_value"] <= b["lp_value"] + 1e-7


def test_coverage_shares_hand_case():
    # 4 cells; cell 3 is outside the park. Site 0 sees cells 0, 1; site 1 sees cells 1, 3; site 2 sees cell 2.
    sees = sparse.csr_matrix(np.array([[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=bool))
    weights = np.array([1.0, 2.0, 3.0, 10.0])
    inside = np.array([True, True, True, False])
    fires = np.array([4, 0, 1, 50])
    s = coverage_shares(sees, weights, inside, fires, [0, 1])
    assert s["priority_share"] == pytest.approx(3.0 / 6.0)     # cells 0, 1 of the in-park total 6
    assert s["area_share"] == pytest.approx(2.0 / 3.0)
    assert s["fire_share"] == pytest.approx(4.0 / 5.0)          # the outside cell's 50 detections are ignored
    none = coverage_shares(sees, weights, inside, fires, [])
    assert none == dict(priority_share=0.0, area_share=0.0, fire_share=0.0)


def test_sparse_detection_times_equal_the_dense_function():
    rng = np.random.default_rng(5)
    arrival = rng.uniform(0, 500, size=(9, 40)).astype(np.float32)
    arrival[rng.random(arrival.shape) < 0.5] = np.inf
    dense = rng.random((12, 40)) < 0.15
    dense[3] = False                                            # a site that sees nothing
    expected = site_detection_times(arrival, dense)
    got = site_detection_times_sparse(arrival, sparse.csr_matrix(dense))
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_rounded_shares_still_sum_to_one():
    for shares in ([1 / 3, 1 / 3, 1 / 3], [0.0005, 0.9985, 0.001], [0.8555, 0.1005, 0.044]):
        assert sum(_permille(shares)) == pytest.approx(1.0, abs=1e-12)


def test_quick_path_is_consistent():
    cells = np.flatnonzero(load_park_data()["inside_park"].ravel() > 0.5)
    park = build_park_instance(site_cells=cells[::220][:40], use_cache=False)
    out = run_analysis(quick=True, n_scenarios=25, inp=load_inputs(park), ks=(3,), log=lambda *_: None)
    res = out["results"]
    h = res["headline"]
    assert h["satellite_only"]["saved_h"] == 0.0
    assert h["satellite_only"]["share_satellite"] == 1.0
    obj = [c["objective_share"] for c in res["coverage_curve"]]
    assert obj == sorted(obj) and 0 <= obj[0] and obj[-1] <= 1
    shares = ("priority_share", "area_share", "fire_share", "share_at_ignition", "share_after_spread", "share_satellite")
    for key, p in h["3"].items():
        assert all(0.0 <= p[s] <= 1.0 for s in shares), key
        assert p["share_at_ignition"] + p["share_after_spread"] + p["share_satellite"] == pytest.approx(1.0)
        assert 0.0 <= p["saved_h"] <= 6.0
        assert p["saved_h"] + p["mean_detection_h"] == pytest.approx(6.0)
    b = res["coverage_bounds"]["3"]
    assert b["greedy_over_lp_bound"] <= 1.0
