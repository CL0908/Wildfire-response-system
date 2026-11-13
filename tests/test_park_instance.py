"""Whole-park placement instance. Visibility is built for small site subsets only;
nothing here needs the cache or the network."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

import park.instance as pi
import viewshed as vs
from park.instance import (build_park_instance, build_visibility, candidate_cells,
                           load_park_data, park_layers, window_cells_in_park)

RESULTS = Path(__file__).resolve().parent.parent / "dashboards" / "public" / "results.json"


@pytest.fixture(scope="module")
def data():
    return load_park_data()


@pytest.fixture(scope="module")
def layers(data):
    return park_layers(data)


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text())


def _has_half_cell_tie(dr, dc):
    """True if the ray (dr, dc) crosses a cell boundary exactly at a half cell, where
    _los_clear's round() goes to the even absolute index and so depends on the site's
    absolute row/column parity."""
    st = max(abs(dr), abs(dc))
    return any((2 * k * abs(dr)) % (2 * st) == st or (2 * k * abs(dc)) % (2 * st) == st
               for k in range(1, st))


def _window_ids(cells, cols, r0, c0, n=20):
    """Park flat cell ids -> flat ids inside the n x n window at (r0, c0); others dropped."""
    out = set()
    for c in cells:
        r, k = divmod(int(c), cols)
        if 0 <= r - r0 < n and 0 <= k - c0 < n:
            out.add((r - r0) * n + (k - c0))
    return out


# --- consistency with the 20 x 20 pipeline -----------------------------------------

def test_window_dem_matches_results_json(layers, results):
    r0, c0 = window_cells_in_park()
    win = layers["dem_m"][r0:r0 + 20, c0:c0 + 20]
    assert np.abs(win - np.array(results["layers"]["elevation_m"])).max() < 0.06  # json rounds to 0.1 m


def test_results_json_viewsheds_reproduced_up_to_half_cell_ties(layers, results):
    # The park-grid viewshed of each of the 16 results.json sites, restricted to the 20 x 20
    # window, differs from the stored one only at cells whose ray has an exact half-cell
    # tie (see _has_half_cell_tie). Elsewhere it is identical.
    r0, c0 = window_cells_in_park()
    dem = layers["dem_m"]
    cols = dem.shape[1]
    n_diff = 0
    for s in results["sites"]:
        cell = (r0 + s["row"]) * cols + (c0 + s["col"])
        got = _window_ids(build_visibility(dem, [cell])[0].indices, cols, r0, c0)
        exp = set(results["viewsheds"][str(s["id"])])
        for f in got ^ exp:
            assert _has_half_cell_tie(f // 20 - s["row"], f % 20 - s["col"]), (s["id"], f)
            n_diff += 1
    assert n_diff > 0  # documents that the tie effect is real, not that it is zero


def test_results_json_viewsheds_exact_when_window_has_even_offset(layers, results):
    # Same DEM, same sites, window placed at an even row and column: identical to results.json.
    r0, c0 = window_cells_in_park()
    win = layers["dem_m"][r0:r0 + 20, c0:c0 + 20]
    big = np.zeros((60, 60))
    big[8:28, 6:26] = win
    for s in results["sites"]:
        cell = (8 + s["row"]) * 60 + (6 + s["col"])
        got = _window_ids(build_visibility(big, [cell])[0].indices, 60, 8, 6)
        assert got == set(results["viewsheds"][str(s["id"])]), s["id"]


# --- visibility geometry ---------------------------------------------------------------

def test_sites_see_own_cell_and_nothing_beyond_range(layers):
    dem = layers["dem_m"]
    rows, cols = dem.shape
    rng = np.random.default_rng(1)
    cells = rng.choice(candidate_cells(layers["inside_park"]), 60, replace=False)
    sees = build_visibility(dem, cells)
    for i, cell in enumerate(cells):
        seen = sees[i].indices
        assert cell in seen
        r, c = np.divmod(seen, cols)
        dist = np.hypot(r - cell // cols, c - cell % cols)
        assert dist.max() <= vs.MAX_RANGE_CELLS


def test_visibility_not_transposed_on_a_nonsquare_ridge():
    # 9 x 15 grid, flat. A tall wall in column 5 runs along every row. A sensor at (4, 3)
    # cannot see east of the wall but sees the cells north and south of it in its own column.
    dem = np.zeros((9, 15))
    dem[:, 5] = 500.0
    sees = build_visibility(dem, [4 * 15 + 3])[0].indices
    seen = {(int(c // 15), int(c % 15)) for c in sees}
    assert (4, 5) in seen                       # the wall face itself is visible
    assert not any(c > 5 for _, c in seen)      # nothing behind it
    assert (0, 3) in seen and (8, 3) in seen    # along the column, rows 4 away
    assert (4, 0) in seen


# --- instance ----------------------------------------------------------------------------

def test_weights_are_finite_nonnegative_and_flat(layers):
    p = layers["priority"]
    assert p.shape == (131, 187)
    assert np.isfinite(p).all() and (p >= 0).all() and p.max() <= 1.0 + 1e-12


def test_candidates_are_exactly_cells_inside_park(data, monkeypatch):
    inside = data["inside_park"].ravel()
    cand = candidate_cells(data["inside_park"])
    assert set(cand) == set(np.flatnonzero(inside > 0.5))

    # the default instance uses that set (visibility stubbed out to keep this fast)
    monkeypatch.setattr(pi, "build_visibility",
                        lambda elev, sites, *a, **k: sparse.csr_matrix((len(sites), elev.size), dtype=bool))
    inst = build_park_instance(data, use_cache=False)
    assert np.array_equal(inst.site_cells, cand)
    assert inst.sees.shape == (len(cand), 131 * 187)
    assert inst.weights.shape == (inst.n_cells,)
    assert np.array_equal(inst.weights, inst.priority.ravel())


def test_cache_roundtrip_returns_identical_matrix(data, tmp_path, monkeypatch):
    monkeypatch.setenv("Q_PITHAK_CACHE", str(tmp_path))
    sites = candidate_cells(data["inside_park"])[:7]
    first = build_park_instance(data, site_cells=sites)
    assert len(list(tmp_path.glob("sees_*.npz"))) == 1

    def boom(*a, **k):
        raise AssertionError("visibility recomputed instead of loaded from cache")
    monkeypatch.setattr(pi, "build_visibility", boom)
    second = build_park_instance(data, site_cells=sites)
    assert (first.sees != second.sees).nnz == 0
    assert second.sees.dtype == bool
