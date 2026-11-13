"""Checks that data/park_500m.npz agrees with the older extracts on the same lattice and
covers the whole park. Reads committed files only (no network)."""
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
LC = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water",
      "lc_wetland", "lc_other"]


@pytest.fixture(scope="module")
def park():
    with np.load(DATA / "park_500m.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def meta():
    return json.loads((DATA / "park_500m_meta.json").read_text())


@pytest.fixture(scope="module")
def wide():
    with np.load(DATA / "hazard_training_500m.npz") as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def small():
    with np.load(DATA / "khao_yai_500m.npz") as z:
        return {k: z[k] for k in z.files}


def locate(big, small_lat, small_lon):
    """Row/col in `big` of the cell whose centre is (small_lat, small_lon), found from the
    coordinates alone; raises if the lattices do not coincide."""
    r = int(np.argmin(np.abs(big["lat"][:, 0] - small_lat)))
    c = int(np.argmin(np.abs(big["lon"][0, :] - small_lon)))
    assert abs(big["lat"][r, 0] - small_lat) < 1e-9 and abs(big["lon"][0, c] - small_lon) < 1e-9
    return r, c


def test_shapes_years_and_values(park, meta):
    nr, nc = meta["grid_shape"]
    for k in ["lat", "lon", "dem_m", "dem_max_m", "inside_park", "dist_park_edge_m",
              "fire_count", "fire_years"] + LC:
        assert park[k].shape == (nr, nc), k
    assert park["fire_by_year"].shape == (14, nr, nc)
    assert park["fire_by_year"].dtype == np.int16
    assert list(park["years"]) == list(range(2012, 2026))
    for k, v in park.items():
        assert np.isfinite(v).all(), k
    assert np.allclose(sum(park[k].astype(np.float64) for k in LC), 1.0, atol=1e-5)
    assert (park["dem_max_m"] >= park["dem_m"] - 1e-3).all()
    assert np.array_equal(park["fire_count"], park["fire_by_year"].sum(0))


def test_overlap_with_200_grid(park, wide, meta):
    # offsets found from coordinates, then compared with what the meta file says
    r0, c0 = locate(wide, park["lat"][0, 0], park["lon"][0, 0])
    row0, col0 = meta["offset_in_hazard_training_grid_row_col"]
    assert (r0, c0) == (row0, col0)
    nr, nc = park["lat"].shape
    rows = slice(0, min(nr, 200 - row0))
    cols = slice(0, min(nc, 200 - col0))
    wr, wc = slice(row0, row0 + rows.stop), slice(col0, col0 + cols.stop)
    # the overlap must be a real block, and the park must extend east of the 200 grid
    assert meta["overlap_with_hazard_training_grid"]["shape"] == [rows.stop, cols.stop]
    assert rows.stop > 100 and cols.stop > 100 and col0 + nc > 200
    np.testing.assert_allclose(park["lat"][rows, cols], wide["lat"][wr, wc], rtol=0, atol=1e-9)
    np.testing.assert_allclose(park["lon"][rows, cols], wide["lon"][wr, wc], rtol=0, atol=1e-9)
    for k in ["dem_m"] + LC + ["inside_park"]:
        np.testing.assert_allclose(park[k][rows, cols], wide[k][wr, wc], rtol=0, atol=1e-5, err_msg=k)
    assert np.array_equal(park["fire_by_year"][:, rows, cols], wide["fire_by_year"][:, wr, wc])
    assert np.array_equal(park["years"], wide["years"])


def test_overlap_with_40_block(park, small, meta):
    r, c = locate(park, small["lat"][0, 0], small["lon"][0, 0])
    assert [r, c] == meta["khao_yai_block_offset_row_col"]
    sl = (slice(r, r + 40), slice(c, c + 40))
    np.testing.assert_allclose(park["lat"][sl], small["lat"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(park["lon"][sl], small["lon"], rtol=0, atol=1e-9)
    for k in ["dem_m", "dem_max_m", "inside_park"] + LC:
        np.testing.assert_allclose(park[k][sl], small[k], rtol=0, atol=1e-5, err_msg=k)
    assert np.array_equal(park["fire_count"][sl], small["fire_count"])
    assert np.array_equal(park["fire_years"][sl], small["fire_years"])


def test_whole_park_is_inside_the_grid(park, meta):
    ins = park["inside_park"]
    ring = np.zeros(ins.shape, bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    assert (ins[ring] == 0).all()
    # park area from the grid (0.25 km2 per cell) against the polygon's own area from the geojson
    grid_km2 = float(ins.sum()) * 0.25
    poly_km2 = meta["park_polygon_area_km2"]["lattice"]
    assert abs(grid_km2 - poly_km2) / poly_km2 < 0.02
    # the polygon's bounding box lies at least 10 cells from every grid edge
    b, g = meta["park_polygon_bbox_deg"], meta["bounds_deg"]
    dlat, dlon = meta["cell_size_deg"]["lat"], meta["cell_size_deg"]["lon"]
    assert g["north"] - b["north"] >= 10 * dlat and b["south"] - g["south"] >= 10 * dlat
    assert b["west"] - g["west"] >= 10 * dlon and g["east"] - b["east"] >= 10 * dlon


def test_distance_to_edge_consistent_with_membership(park):
    d, ins = park["dist_park_edge_m"], park["inside_park"]
    assert (d >= 0).all()
    assert d[ins == 0].max() > 4500    # the 5 km margin is really there
