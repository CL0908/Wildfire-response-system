"""Terrain mesh export: file integrity and orientation against the 500 m elevation layer.

Uses only the committed dashboards/public files (no rasterio, no network).
"""
import json
import os

import numpy as np
import pytest

PUBLIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboards", "public")


@pytest.fixture(scope="module")
def terrain():
    with open(os.path.join(PUBLIC, "terrain.json")) as f:
        meta = json.load(f)
    raw = np.fromfile(os.path.join(PUBLIC, "terrain.bin"), dtype="<u2")
    return meta, raw


@pytest.fixture(scope="module")
def results():
    with open(os.path.join(PUBLIC, "results.json")) as f:
        d = json.load(f)
    return d["bounds"], np.array(d["layers"]["elevation_m"], dtype=float)


def _block_mean(fine_m, meta, bounds, n):
    """Mean of the fine grid per coarse cell, mapping each pixel centre into the results.json
    bounds (row 0 = north, column 0 = west). Pixels outside the bounds are ignored."""
    rows, cols = fine_m.shape
    b = meta["bounds"]
    lat = b["north"] - (np.arange(rows) + 0.5) * (b["north"] - b["south"]) / rows
    lon = b["west"] + (np.arange(cols) + 0.5) * (b["east"] - b["west"]) / cols
    ri = np.floor((bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * n).astype(int)
    ci = np.floor((lon - bounds["west"]) / (bounds["east"] - bounds["west"]) * n).astype(int)
    rv = (ri >= 0) & (ri < n)
    cv = (ci >= 0) & (ci < n)
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    sub = fine_m[np.ix_(rv, cv)]
    rr, cc = np.meshgrid(ri[rv], ci[cv], indexing="ij")
    np.add.at(total, (rr, cc), sub)
    np.add.at(count, (rr, cc), 1)
    assert count.min() > 0
    return total / count


def _agreement(coarse, ref):
    corr = np.corrcoef(coarse.ravel(), ref.ravel())[0, 1]
    return corr, np.abs(coarse - ref).mean()


def test_file_matches_meta(terrain):
    meta, raw = terrain
    assert raw.size == meta["rows"] * meta["cols"]
    assert os.path.getsize(os.path.join(PUBLIC, "terrain.bin")) == meta["rows"] * meta["cols"] * 2
    assert raw.min() / 10 == pytest.approx(meta["min_m"])
    assert raw.max() / 10 == pytest.approx(meta["max_m"])


def test_orientation_matches_500m_layer(terrain, results):
    meta, raw = terrain
    bounds, ref = results
    fine = raw.reshape(meta["rows"], meta["cols"]) / 10.0
    n = ref.shape[0]
    corr, mad = _agreement(_block_mean(fine, meta, bounds, n), ref)
    assert corr > 0.999
    assert mad < 0.5  # measured 0.03 m: the fine grid is the same DEM as the 500 m layer

    # Any flip must be clearly worse, which is what pins row 0 = north and column 0 = west.
    for name, flipped in (("north-south", fine[::-1, :]), ("east-west", fine[:, ::-1])):
        f_corr, f_mad = _agreement(_block_mean(flipped, meta, bounds, n), ref)
        assert f_corr < corr - 0.1, name
        assert f_mad > 5 * mad, name
