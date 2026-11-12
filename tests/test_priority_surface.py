"""Priority surface: synthetic mode must not change; real mode must be consistent and correct."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import priority_surface as ps
from priority_surface import build_priority_surface

# Arrays recorded before real-terrain mode was added. Compared with a tolerance, not by hash:
# floating-point results differ in the last bits between macOS and Linux.
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "synthetic_surfaces.npz")
REFERENCE_CASES = [(20, 20, 42), (20, 20, 7), (20, 20, 13), (15, 15, 42)]


LAYER_KEYS = ["priority", "dem_m", "hazard", "value"]


@pytest.fixture(scope="module")
def reference():
    with np.load(REFERENCE_PATH) as z:
        return {k: z[k] for k in z.files}


def _same(a, b):
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-9)


@pytest.fixture(scope="module")
def region():
    with np.load(ps.REAL_DATA_PATH) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def full():
    return build_priority_surface(40, 40, window=(0, 0), source="real")


@pytest.mark.parametrize("args", REFERENCE_CASES)
def test_synthetic_unchanged(args, reference):
    rows, cols, seed = args
    surf = build_priority_surface(rows, cols, seed, source="synthetic")
    for k in LAYER_KEYS:
        assert _same(surf[k], reference[f"{rows}_{cols}_{seed}_{k}"]), k
    assert surf["source"] == "synthetic" and surf["window"] is None


def test_synthetic_positional_call_unchanged(reference):
    a = build_priority_surface(20, 20, 42, None, 500.0, source="synthetic")
    assert _same(a["priority"], reference["20_20_42_priority"])


def test_synthetic_weights_names_still_work(reference):
    w = dict(arson=0.5, fuel=0.2, dryness=0.2, terrain=0.1)
    a = build_priority_surface(source="synthetic", weights=w)
    assert not _same(a["hazard"], reference["20_20_42_hazard"])


def test_real_shapes_ranges_and_keys():
    real = build_priority_surface(source="real")
    syn = build_priority_surface(source="synthetic")
    assert real["source"] == "real" and real["window"] == (15, 3)
    assert real["grid_shape"] == (20, 20)
    for k in ("hazard", "value", "priority"):
        assert real[k].min() >= 0.0 and real[k].max() <= 1.0, k
    for k in ("dem", "slope", "aspect", "arson", "fuel", "dryness", "spread"):
        assert real[k].min() >= 0.0 and real[k].max() <= 1.0, k
    for k, v in syn.items():
        assert k in real, k
        if isinstance(v, np.ndarray):
            assert real[k].shape == v.shape, k
            assert real[k].dtype.kind == v.dtype.kind, k
    for k in ("lat", "lon", "inside_park", "fire_count", "fire", "source", "window"):
        assert k in real
    for k in ("lat", "lon", "inside_park", "fire_count", "dem_m", "hazard"):
        assert real[k].shape == (20, 20), k


def test_real_window_does_not_change_values(full):
    w = build_priority_surface(15, 12, window=(5, 7), source="real")
    for k in ("priority", "hazard", "value", "dem_m"):
        assert np.array_equal(w[k], full[k][5:20, 7:19]), k


def test_real_orientation_and_dem_from_npz(region):
    s = build_priority_surface(10, 12, window=(4, 6), source="real")
    assert np.all(np.diff(s["lat"], axis=0) < 0)
    assert np.all(np.diff(s["lon"], axis=1) > 0)
    assert np.allclose(s["dem_m"], region["dem_m"][4:14, 6:18])
    assert np.array_equal(s["fire_count"], region["fire_count"][4:14, 6:18])
    assert np.allclose(s["lat"], region["lat"][4:14, 6:18])


def _flat_region(n=20):
    """Uniform terrain and land cover, so only what a test changes can matter."""
    z = np.zeros((n, n))
    r = np.arange(n)[:, None] * np.ones((1, n))
    data = {k: z.copy() for k in
            ["fire_count", "lc_shrub", "lc_grass", "lc_crop", "lc_bare", "lc_built",
             "lc_water", "lc_wetland", "lc_other"]}
    data.update(dem_m=z + 300.0, lc_tree=z + 1.0, inside_park=z + 1.0, lat=r, lon=r.T)
    return data


def _layers(data):
    return ps._real_layers(data, ps.REAL_DEFAULT_WEIGHTS, 500.0)


def test_more_fires_means_higher_hazard():
    d = _flat_region()
    d["fire_count"][5, 5] = 2
    d["fire_count"][14, 14] = 9
    h = _layers(d)["hazard"]
    assert h[14, 14] > h[5, 5] > h[10, 10]


def test_fuel_ranks_grass_over_tree_over_built():
    d = _flat_region()
    d["lc_tree"][:] = 0.0
    d["lc_grass"][3, 3] = 1.0
    d["lc_tree"][6, 6] = 1.0
    d["lc_built"][9, 9] = 1.0
    fuel = _layers(d)["fuel"]
    assert fuel[3, 3] > fuel[6, 6] > fuel[9, 9]


def test_value_lower_outside_park():
    d = _flat_region()
    d["inside_park"][:, 10:] = 0.0
    v = _layers(d)["value"]
    assert v[:, :10].min() > v[:, 10:].max()
    assert np.allclose(v[:, :10], 1.0) and np.allclose(v[:, 10:], 0.1)


def test_value_on_real_surface_lower_outside_park(full):
    inside = full["inside_park"] == 1.0
    outside = full["inside_park"] == 0.0
    assert inside.any() and outside.any()
    assert full["value"][inside].mean() > full["value"][outside].mean()
    assert full["value"][outside].max() <= 0.1 + 1e-12


def test_real_weights_aliases_and_validation():
    base = build_priority_surface(source="real")
    named = build_priority_surface(source="real", weights=dict(fire=0.4, fuel=0.25, dryness=0.2, spread=0.15))
    legacy = build_priority_surface(source="real", weights=dict(arson=0.4, fuel=0.25, dryness=0.2, terrain=0.15))
    assert np.array_equal(base["hazard"], named["hazard"])
    assert np.array_equal(base["hazard"], legacy["hazard"])
    with pytest.raises(ValueError):
        build_priority_surface(source="real", weights=dict(fire=1.0))


def test_seed_does_not_affect_real_terrain():
    a = build_priority_surface(source="real", seed=1)
    b = build_priority_surface(source="real", seed=99)
    assert np.array_equal(a["priority"], b["priority"])


@pytest.mark.parametrize("rows,cols,window", [
    (20, 20, (21, 0)), (20, 20, (0, 21)), (20, 20, (-1, 0)), (41, 20, (0, 0)), (20, 20, (0,)),
])
def test_bad_window_raises(rows, cols, window):
    with pytest.raises(ValueError, match="window"):
        build_priority_surface(rows, cols, source="real", window=window)


def test_missing_data_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "REAL_DATA_PATH", str(tmp_path / "nope.npz"))
    with pytest.raises(ValueError, match="synthetic"):
        build_priority_surface(source="real")
    # synthetic mode does not need the file
    assert build_priority_surface(source="synthetic")["source"] == "synthetic"


def test_real_cell_size_must_be_500():
    with pytest.raises(ValueError, match="cell_size"):
        build_priority_surface(source="real", cell_size=250.0)


def test_unknown_source_raises():
    with pytest.raises(ValueError, match="source"):
        build_priority_surface(source="mock")


def test_pipeline_runs_on_real_surface():
    from solvers.greedy import greedy_coverage
    from viewshed import compute_viewsheds, generate_candidate_sites

    surf = build_priority_surface(source="real")
    sites = generate_candidate_sites((20, 20), n_candidates=16, priority=surf["priority"],
                                     dem_norm=surf["dem"], seed=42)
    viewsheds = compute_viewsheds(surf["dem_m"], sites)
    weights = {r * 20 + c: float(surf["priority"][r, c]) for r in range(20) for c in range(20)}
    chosen, objective = greedy_coverage(viewsheds, weights, 4)
    assert len(sites) == 16 and len(chosen) == 4
    assert objective > 0.0
