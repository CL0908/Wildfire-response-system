"""Park dashboard export: covered cells, row/col order, window corner, terrain and land cover grids.

The park.json builder and the raster resamplers are tested in memory; the committed
park_terrain files are only compared with the 500 m elevation when they exist. No network.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

import export_park_dashboard as epd  # noqa: E402
from park.instance import load_park_instance, window_cells_in_park  # noqa: E402

PUBLIC = os.path.join(ROOT, "dashboards", "public")


@pytest.fixture(scope="module")
def inst():
    return load_park_instance()


@pytest.fixture(scope="module")
def results():
    with open(epd.RESULTS) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def grid_meta():
    with open(epd.PARK_META) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def doc(inst, results, grid_meta):
    return epd.build_park_json(inst, results, grid_meta, window_cells_in_park(), (20, 20))


def test_covered_is_union_of_visibility_rows(inst, results, doc):
    cols = inst.shape[1]
    sites = results["headline"]["10"]["coverage"]["sites"]
    expect = set()
    for r, c in sites:
        i = int(np.searchsorted(inst.site_cells, r * cols + c))
        expect |= set(inst.sees[i].indices.tolist())
    covered = doc["placements"]["10"]["coverage"]["covered"]
    assert covered == sorted(expect)


@pytest.mark.parametrize("k,rule", [("10", "coverage"), ("10", "fire_history"), ("10", "timed_greedy"),
                                    ("40", "coverage")])
def test_in_park_shares_recomputed_from_covered_match_results(inst, results, doc, k, rule):
    """Pins [row, col] order: swapped coordinates would pick other cells and change the shares."""
    p = doc["placements"][k][rule]
    mask = np.zeros(inst.n_cells, dtype=bool)
    mask[p["covered"]] = True
    inside = inst.inside_park.ravel() > 0.5
    cov = mask & inside
    area = cov.sum() / inside.sum()
    fire = inst.fire_count[cov].sum() / inst.fire_count[inside].sum()
    prio = inst.weights[cov].sum() / inst.weights[inside].sum()
    ref = results["headline"][k][rule]
    assert round(area, 3) == ref["area_share"]
    assert round(fire, 3) == ref["fire_share"]
    assert round(prio, 3) == ref["priority_share"]
    assert p["metrics"] == {key: v for key, v in ref.items() if key != "sites"}
    assert p["sites"] == ref["sites"]


def test_every_site_covers_its_own_cell(inst, doc):
    cols = inst.shape[1]
    for k, rules in doc["placements"].items():
        for rule, p in rules.items():
            assert len(p["sites"]) == int(k)
            covered = set(p["covered"])
            assert p["covered"] == sorted(covered)
            for r, c in p["sites"]:
                assert r * cols + c in covered, (k, rule, r, c)


def test_covered_rejects_a_non_candidate_site(inst):
    outside = int(np.flatnonzero(inst.inside_park.ravel() < 0.5)[0])
    cols = inst.shape[1]
    with pytest.raises(ValueError):
        epd.covered_cells(inst.sees, inst.site_cells, [[outside // cols, outside % cols]], cols)


def test_window_corner_matches_instance_helper(doc):
    corner = window_cells_in_park()
    w = doc["meta"]["window"]
    assert (w["row0"], w["col0"]) == corner
    assert (w["n_rows"], w["n_cols"]) == (20, 20)


def test_layers_have_one_value_per_cell(inst, doc):
    n = inst.n_cells
    assert {k: len(v) for k, v in doc["layers"].items()} == {
        "inside_park": n, "priority": n, "fire_count": n, "elevation_m": n}
    assert all(isinstance(v, int) for v in doc["layers"]["fire_count"][:50])
    assert doc["layers"]["fire_count"] == inst.fire_count.tolist()


# ---- terrain and land cover grids -----------------------------------------------

def _fine_grid(b, step_deg=1 / 3600):
    """Pixel-centre latitudes (north to south) and longitudes of a 1 arc-second grid over b."""
    lats = np.arange(b["north"] - step_deg / 2, b["south"], -step_deg)
    lons = np.arange(b["west"] + step_deg / 2, b["east"], step_deg)
    return lats, lons


def test_terrain_grid_and_orientation(grid_meta):
    b = grid_meta["bounds_deg"]
    rows, cols, bounds = epd.grid_150m(b)
    assert rows * cols == pytest.approx(270_000, rel=0.05)
    lats, lons = _fine_grid(b)
    # a surface rising to the north and to the east: row 0 must be higher than the last row,
    # the last column higher than column 0, and values must follow the surface
    f = lambda la, lo: (100 + 1000 * (la - b["south"]) / (b["north"] - b["south"])  # noqa: E731
                        + 500 * (lo - b["west"]) / (b["east"] - b["west"]))
    dem = f(lats[:, None], lons[None, :]).astype(np.float32)
    dm = epd.resample_terrain(dem, lons, lats, bounds, rows, cols)
    assert dm.shape == (rows, cols) and dm.dtype == np.dtype("<u2")
    assert len(dm.tobytes()) == rows * cols * 2
    assert dm[0, 0] > dm[-1, 0] and dm[0, -1] > dm[0, 0]
    olat = b["north"] - (np.arange(rows) + 0.5) * (b["north"] - b["south"]) / rows
    olon = b["west"] + (np.arange(cols) + 0.5) * (b["east"] - b["west"]) / cols
    expect = f(olat[:, None], olon[None, :])
    assert np.abs(dm / 10 - expect).max() < 1.0
    meta = epd.terrain_meta(dm, bounds, "test", 150.0)
    assert (meta["rows"], meta["cols"]) == (rows, cols)
    assert meta["bounds"] == b
    assert meta["width_m"] == pytest.approx(93_500, rel=0.01)
    assert meta["min_m"] == dm.min() / 10 and meta["max_m"] == dm.max() / 10


def test_terrain_rejects_elevation_beyond_uint16(grid_meta):
    b = grid_meta["bounds_deg"]
    rows, cols, bounds = epd.grid_150m(b)
    lats, lons = _fine_grid(b, 1 / 360)
    with pytest.raises(ValueError):
        epd.resample_terrain(np.full((len(lats), len(lons)), 7000.0), lons, lats, bounds, rows, cols)


def test_landcover_majority_ties_and_tree_fraction():
    b = {"north": 2.0, "south": 0.0, "west": 0.0, "east": 2.0}
    rows = cols = 2
    step = 2.0 / 8
    lats = b["north"] - (np.arange(8) + 0.5) * step
    lons = b["west"] + (np.arange(8) + 0.5) * step
    arr = np.full((8, 8), 10, np.uint8)
    arr[:4, 4:] = 40                 # NE output pixel: all cropland
    arr[4:, :4] = 30                 # SW: 8 of 16 grassland, 8 of 16 tree (tie -> lowest code, 10)
    arr[4:, :4][:, :2] = 10
    arr[4:, 4:] = 80                 # SE: water, 12 of 16
    arr[4:, 4:][:1, :] = 10
    counts = np.zeros((rows * cols, epd.N_SLOTS), np.int64)
    epd.accumulate_landcover(counts, arr[:4], lons, lats[:4], b, rows, cols)
    epd.accumulate_landcover(counts, arr[4:], lons, lats[4:], b, rows, cols)   # strips add up
    cls, tree = epd.finish_landcover(counts, rows, cols)
    assert cls.tolist() == [10, 40, 10, 80]        # NW, NE, SW, SE; row 0 north, column 0 west
    assert tree.tolist() == [255, 0, 128, 64]


def test_landcover_nodata_majority_is_rejected():
    counts = np.zeros((1, epd.N_SLOTS), np.int64)
    counts[0, -1] = 5
    counts[0, 0] = 1
    with pytest.raises(ValueError):
        epd.finish_landcover(counts, 1, 1)


def _committed(name):
    path = os.path.join(PUBLIC, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not generated")
    return path


def test_committed_terrain_agrees_with_file_size_and_park_bounds(grid_meta):
    with open(_committed("park_terrain.json")) as f:
        meta = json.load(f)
    assert os.path.getsize(_committed("park_terrain.bin")) == meta["rows"] * meta["cols"] * 2
    assert meta["bounds"] == grid_meta["bounds_deg"]
    with open(_committed("park_landcover.json")) as f:
        lc = json.load(f)
    assert (lc["rows"], lc["cols"]) == (meta["rows"], meta["cols"])
    assert os.path.getsize(_committed("park_landcover.bin")) == lc["rows"] * lc["cols"] * 2


def test_committed_terrain_block_means_match_500m_elevation(inst):
    """The mean of the 150 m pixels inside a 500 m cell reproduces the cell's mean DEM height."""
    with open(_committed("park_terrain.json")) as f:
        meta = json.load(f)
    t = np.fromfile(_committed("park_terrain.bin"), "<u2").reshape(meta["rows"], meta["cols"]) / 10
    R, C = inst.shape
    ri = np.floor((np.arange(meta["rows"]) + 0.5) / meta["rows"] * R).astype(int)
    ci = np.floor((np.arange(meta["cols"]) + 0.5) / meta["cols"] * C).astype(int)
    rr, cc = np.meshgrid(ri, ci, indexing="ij")
    total, count = np.zeros((R, C)), np.zeros((R, C))
    np.add.at(total, (rr, cc), t)
    np.add.at(count, (rr, cc), 1)
    block = total / count
    diff = np.abs(block - inst.elevation)
    assert diff.mean() < 4.0
    assert np.corrcoef(block.ravel(), inst.elevation.ravel())[0, 1] > 0.999
    for r, c in [(0, 0), (30, 40), (65, 93), (90, 150), (130, 186)]:      # a few cells, incl. corners
        assert diff[r, c] < 60.0, (r, c)
