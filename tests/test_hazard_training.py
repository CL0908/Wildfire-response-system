"""Committed hazard-training grid (data/hazard_training_500m.npz): alignment with the Khao Yai
extract, fire counts, distances, land-cover ranges, and the segment-distance helper."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))

import fetch_hazard_training as fht

NEW = np.load(os.path.join(ROOT, "data", "hazard_training_500m.npz"))
OLD = np.load(os.path.join(ROOT, "data", "khao_yai_500m.npz"))
O, N0 = 80, 40
SUB = (slice(O, O + N0), slice(O, O + N0))
LC = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water", "lc_wetland"]


def test_shapes():
    for k in ["lat", "lon", "dem_m", "inside_park", "dist_park_edge_m", "dist_road_m",
              "dist_major_road_m", "dist_settlement_m"] + LC:
        assert NEW[k].shape == (200, 200), k
    assert NEW["fire_by_year"].shape == (14, 200, 200)
    assert NEW["fire_by_year"].dtype == np.int16
    assert list(NEW["years"]) == list(range(2012, 2026))


def test_existing_region_is_aligned_subblock():
    assert np.abs(NEW["lat"][SUB] - OLD["lat"]).max() < 1e-9
    assert np.abs(NEW["lon"][SUB] - OLD["lon"]).max() < 1e-9
    assert (np.diff(NEW["lat"], axis=0) < 0).all() and (np.diff(NEW["lon"], axis=1) > 0).all()


def test_fire_matches_existing_extract():
    assert np.array_equal(NEW["fire_by_year"][(slice(None),) + SUB].sum(0), OLD["fire_count"])
    per_year = NEW["fire_by_year"][(slice(None),) + SUB].sum((1, 2))
    assert list(per_year) == [30, 29, 82, 24, 223, 13, 4, 26, 416, 5, 3, 152, 7, 3]


def test_distances_finite_nonnegative():
    for k in ["dist_park_edge_m", "dist_road_m", "dist_major_road_m", "dist_settlement_m"]:
        assert np.isfinite(NEW[k]).all() and (NEW[k] >= 0).all(), k
    assert (NEW["dist_road_m"] <= NEW["dist_major_road_m"] + 1e-3).all()


def test_landcover_fractions_in_unit_interval():
    for k in LC:
        assert NEW[k].min() >= 0 and NEW[k].max() <= 1, k
    assert np.allclose(sum(NEW[k].astype(float) for k in LC) + NEW["lc_other"], 1.0, atol=1e-5)


def test_segment_distance_hand_case():
    a, b = np.array([[0.0, 0.0]]), np.array([[1000.0, 0.0]])
    p = np.array([[500.0, 30.0]])
    assert fht.point_segment_distance(p, a, b)[0] == pytest.approx(30.0)
    # beyond the end the nearest point is the end point
    assert fht.point_segment_distance(np.array([[1040.0, 30.0]]), a, b)[0] == pytest.approx(50.0)
    # densified KD-tree route agrees with the exact value
    d = fht.distance_to_lines(np.array([[500.0, 30.0], [-40.0, 30.0]]), a, b)
    assert d[0] == pytest.approx(30.0, abs=1e-6) and d[1] == pytest.approx(50.0, abs=1e-6)
    pts, seg = fht.densify(a, b, 50.0)
    assert len(pts) == 20 and (seg == 0).all() and pts[:, 0].max() <= 1000
