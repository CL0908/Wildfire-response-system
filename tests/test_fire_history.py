"""Fire history export: cell mapping logic and integrity of the committed fires.json.

Uses only the committed dashboards/public files (no raw FIRMS CSVs, no network).
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, "dashboards", "public")
sys.path.insert(0, os.path.join(ROOT, "data"))

import export_fire_history as efh  # noqa: E402

BOUNDS = {"north": 14.4, "south": 14.3, "west": 101.1, "east": 101.2}
EPS = 1e-9


def test_cell_mapping_corners_and_outside():
    nw = efh.cell_of(BOUNDS["north"] - EPS, BOUNDS["west"] + EPS, BOUNDS, 20, 20)
    se = efh.cell_of(BOUNDS["south"] + EPS, BOUNDS["east"] - EPS, BOUNDS, 20, 20)
    assert nw == (0, 0)
    assert se == (19, 19)
    assert efh.cell_of(BOUNDS["north"] + 0.01, 101.15, BOUNDS, 20, 20) is None
    assert efh.cell_of(14.35, BOUNDS["east"], BOUNDS, 20, 20) is None


@pytest.fixture(scope="module")
def fires():
    with open(os.path.join(PUBLIC, "fires.json")) as f:
        return json.load(f)


def test_fires_count_and_order(fires):
    det = fires["detections"]
    assert fires["count"] == len(det)
    dates = [d[0] for d in det]
    assert dates == sorted(dates)
    if det:
        assert fires["first_date"] == dates[0]
        assert fires["last_date"] == dates[-1]


def test_fires_points_inside_bounds_and_grid(fires):
    b = fires["bounds"]
    rows, cols = fires["grid"]["rows"], fires["grid"]["cols"]
    for _, lon, lat, _, row, col, night in fires["detections"]:
        assert b["south"] < lat < b["north"]
        assert b["west"] < lon < b["east"]
        assert 0 <= row < rows and 0 <= col < cols
        assert night in (0, 1)
