"""Whole-park fire history export: shared cell mapping and the committed fires_park.json.

Reads only committed files (dashboards/public/fires_park.json, data/park_500m.npz). No raw
FIRMS CSVs, no network.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "dashboards", "public")
PARK_JSON = os.path.join(PUBLIC, "fires_park.json")
PARK_NPZ = os.path.join(ROOT, "data", "park_500m.npz")
sys.path.insert(0, os.path.join(ROOT, "data"))

import export_fire_history as efh  # noqa: E402

BOUNDS = {"north": 14.4, "south": 14.3, "west": 101.1, "east": 101.2}
ROWS, COLS = 131, 187
EPS = 1e-9


def test_shared_cell_mapping_corners_and_outside():
    nw = efh.cell_of(BOUNDS["north"] - EPS, BOUNDS["west"] + EPS, BOUNDS, ROWS, COLS)
    se = efh.cell_of(BOUNDS["south"] + EPS, BOUNDS["east"] - EPS, BOUNDS, ROWS, COLS)
    assert nw == (0, 0)
    assert se == (ROWS - 1, COLS - 1)
    assert efh.cell_of(BOUNDS["north"] + 0.01, 101.15, BOUNDS, ROWS, COLS) is None
    assert efh.cell_of(14.35, BOUNDS["east"], BOUNDS, ROWS, COLS) is None
    assert efh.cell_of(14.35, BOUNDS["west"] - EPS, BOUNDS, ROWS, COLS) is None


@pytest.fixture(scope="module")
def park():
    if not os.path.exists(PARK_JSON):
        pytest.skip("dashboards/public/fires_park.json not present")
    with open(PARK_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def inside_and_counts():
    z = np.load(PARK_NPZ)
    return z["inside_park"] > 0.5, z["fire_count"]


def test_park_count_matches_fire_count_total(park, inside_and_counts):
    _, fire_count = inside_and_counts
    assert park["count"] == len(park["detections"]) == 16365
    assert int(fire_count.sum()) == 16365


def test_park_fields_and_grid(park):
    assert park["fields"] == ["date", "lon", "lat", "frp", "row", "col", "night", "inside"]
    assert park["grid"] == {"rows": ROWS, "cols": COLS}


def test_park_per_cell_counts_match_fire_count(park, inside_and_counts):
    _, fire_count = inside_and_counts
    per_cell = Counter((d[4], d[5]) for d in park["detections"])
    for r in range(ROWS):
        for c in range(COLS):
            assert per_cell.get((r, c), 0) == fire_count[r, c], (r, c)


def test_park_inside_flag_matches_inside_park(park, inside_and_counts):
    inside, _ = inside_and_counts
    for d in park["detections"]:
        assert d[7] == int(inside[d[4], d[5]])


def test_park_detections_sorted_by_date(park):
    dates = [d[0] for d in park["detections"]]
    assert dates == sorted(dates)
