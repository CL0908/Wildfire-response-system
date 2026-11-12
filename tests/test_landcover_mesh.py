"""Land-cover mesh export: file integrity and orientation against the 500 m tree-cover layer.

Uses only the committed dashboards/public and data files (no rasterio, no network).
"""
import json
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "dashboards", "public")


@pytest.fixture(scope="module")
def terrain_meta():
    with open(os.path.join(PUBLIC, "terrain.json")) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def landcover(terrain_meta):
    with open(os.path.join(PUBLIC, "landcover.json")) as f:
        meta = json.load(f)
    raw = np.fromfile(os.path.join(PUBLIC, "landcover.bin"), dtype=np.uint8)
    n = terrain_meta["rows"] * terrain_meta["cols"]
    return meta, raw[:n], raw[n:]


@pytest.fixture(scope="module")
def reference():
    """results.json bounds and the 500 m tree share for the same 20 x 20 window."""
    with open(os.path.join(PUBLIC, "results.json")) as f:
        bounds = json.load(f)["bounds"]
    with open(os.path.join(ROOT, "data", "khao_yai_meta.json")) as f:
        w = json.load(f)["default_window"]
    tree = np.load(os.path.join(ROOT, "data", "khao_yai_500m.npz"))["lc_tree"]
    r0, c0, nr, nc = w["row0"], w["col0"], w["n_rows"], w["n_cols"]
    return bounds, tree[r0:r0 + nr, c0:c0 + nc].astype(float)


def _block_mean(fine, tmeta, bounds, n):
    """Mean of the fine grid per coarse cell, mapping each pixel centre into the results.json
    bounds (row 0 = north, column 0 = west). Pixels outside the bounds are ignored."""
    rows, cols = fine.shape
    b = tmeta["bounds"]
    lat = b["north"] - (np.arange(rows) + 0.5) * (b["north"] - b["south"]) / rows
    lon = b["west"] + (np.arange(cols) + 0.5) * (b["east"] - b["west"]) / cols
    ri = np.floor((bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * n).astype(int)
    ci = np.floor((lon - bounds["west"]) / (bounds["east"] - bounds["west"]) * n).astype(int)
    rv = (ri >= 0) & (ri < n)
    cv = (ci >= 0) & (ci < n)
    total = np.zeros((n, n))
    count = np.zeros((n, n))
    rr, cc = np.meshgrid(ri[rv], ci[cv], indexing="ij")
    np.add.at(total, (rr, cc), fine[np.ix_(rv, cv)])
    np.add.at(count, (rr, cc), 1)
    assert count.min() > 0
    return total / count


def _agreement(coarse, ref):
    return np.corrcoef(coarse.ravel(), ref.ravel())[0, 1], np.abs(coarse - ref).mean()


def test_file_matches_terrain_grid(terrain_meta, landcover):
    meta, cls, tree = landcover
    assert (meta["rows"], meta["cols"]) == (terrain_meta["rows"], terrain_meta["cols"])
    n = meta["rows"] * meta["cols"]
    assert os.path.getsize(os.path.join(PUBLIC, "landcover.bin")) == 2 * n
    assert cls.size == n and tree.size == n


def test_classes_and_shares_match_json(landcover):
    meta, cls, _ = landcover
    codes, counts = np.unique(cls, return_counts=True)
    assert 0 not in codes
    assert {str(c) for c in codes} == set(meta["classes"])
    for code, count in zip(codes, counts):
        assert meta["classes"][str(code)]["share"] == pytest.approx(count / cls.size, abs=1e-5)


def test_tree_fraction_consistent_with_class(landcover):
    _, cls, tree = landcover
    # A tree-cover pixel has at least 1 of at most 16 pixels in the class (>= 15/255); a pixel
    # with more than half tree pixels can only have tree cover as its majority.
    assert tree[cls == 10].min() >= 15
    assert tree[cls != 10].max() <= 0.5 * 255 + 1


def test_orientation_matches_500m_tree_layer(terrain_meta, landcover, reference):
    _, _, tree = landcover
    bounds, ref = reference
    fine = tree.reshape(terrain_meta["rows"], terrain_meta["cols"]) / 255.0
    n = ref.shape[0]
    corr, mad = _agreement(_block_mean(fine, terrain_meta, bounds, n), ref)
    assert corr > 0.995   # measured 0.9995: same WorldCover pixels, differently binned
    assert mad < 0.01     # measured 0.0013 (tree share, 0..1)

    # Any flip must be clearly worse (measured corr about -0.1, MAD about 0.075), which is what
    # pins row 0 = north and column 0 = west.
    for name, flipped in (("north-south", fine[::-1, :]), ("east-west", fine[:, ::-1])):
        f_corr, f_mad = _agreement(_block_mean(flipped, terrain_meta, bounds, n), ref)
        assert f_corr < corr - 0.5, name
        assert f_mad > 10 * mad, name
