"""Priority surface: synthetic mode must not change; real mode must be consistent and correct."""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import priority_surface as ps
from priority_surface import build_priority_surface

# sha256 of the raw bytes of each array, recorded before real-terrain mode was added.
FINGERPRINTS = {
    (20, 20, 42): dict(
        priority="afd85258ebf57af982f24e65f5d0b70dd53f90b4707754756bd12d92f59494ec",
        dem_m="36bda75d4441081ab20af6b035e95d33e5aa8712012f29bb72dd7acb73a2e37e",
        hazard="26bf534b28cbd80532db4de4456256c1a8fec0b0744fcabbc9bbbf761523f72e",
        value="12472222cc179f7c24d28ab674d7e550d5e5b631265814132eaf1c2ad08bfc5a",
    ),
    (20, 20, 7): dict(
        priority="0f3ad085b7e0f777b54203d42a0985f28862e34f8dc1242e5443b2067cf902f6",
        dem_m="fa9ebf1eac552a74e52f43789dd61ad561675c8ca0aa9cbe8139485f571fdc67",
        hazard="e2d095bed0d0529aece4dfe4bb8ac47d05aec43c53b6b2401fe9c88bc1ccdd1b",
        value="4642a49f34a1499ad89d8989884242d503aee62946fdb08e1a191d350a751958",
    ),
    (20, 20, 13): dict(
        priority="8e6133c20a4ad547d4fce4aa164b6fed19bee185ced61a4d8d9d1523b0a5f532",
        dem_m="8d31d2086e1e443443269fd664187448e7f61c0b99b8400d34f1f960789f47d9",
        hazard="34bbf074e0af3ed6a1ac565a7ca8bfbc4887e327e04bd2719c7d574b736b1806",
        value="4e90e15591de6b352d608d0122550fabbc4025f37de01878ed27364d6b2fccb2",
    ),
    (15, 15, 42): dict(
        priority="a7083c891888bec1a98dd32671bfd3584c787791dde371600052a0c9fe15f0af",
        dem_m="ad380085d1016fedf0de1b1e2fe195416f912bd4d450fb2a4e6888c94964538d",
        hazard="4a16e54d39498bd9156be8d9b11e77b8d14ea2f79c475b6ca1d12c88dcf87c98",
        value="83ab2dd9594f8f2e4b4fb7b1a8b73a9f7447268ad70e1e95009b672880f7bc62",
    ),
}


LAYER_KEYS = ["priority", "dem_m", "hazard", "value"]


def _digest(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def region():
    with np.load(ps.REAL_DATA_PATH) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def full():
    return build_priority_surface(40, 40, window=(0, 0), source="real")


@pytest.mark.parametrize("args", list(FINGERPRINTS))
def test_synthetic_unchanged(args):
    rows, cols, seed = args
    surf = build_priority_surface(rows, cols, seed, source="synthetic")
    for k, expected in FINGERPRINTS[args].items():
        assert _digest(surf[k]) == expected, k
    assert surf["source"] == "synthetic" and surf["window"] is None


def test_synthetic_positional_call_unchanged():
    a = build_priority_surface(20, 20, 42, None, 500.0, source="synthetic")
    assert _digest(a["priority"]) == FINGERPRINTS[(20, 20, 42)]["priority"]


def test_synthetic_weights_names_still_work():
    w = dict(arson=0.5, fuel=0.2, dryness=0.2, terrain=0.1)
    a = build_priority_surface(source="synthetic", weights=w)
    assert _digest(a["hazard"]) != FINGERPRINTS[(20, 20, 42)]["hazard"]


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
