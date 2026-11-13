"""Fire-spread scenario simulator (detection/fire_sim.py): symmetry, wind and slope direction
conventions, impassable cells, seeding, ignition sampling and one hand-computed case.
All inputs are synthetic; no network and no wind file needed."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection import fire_sim as fs  # noqa: E402


def one_hot(shape, cell):
    w = np.zeros(shape)
    w[cell] = 1.0
    return w


def run(shape, cell, fuel=None, elev=None, speed=0.0, from_deg=0.0, r0=2.0, hours=500.0, n=1, seed=0):
    fuel = np.ones(shape) if fuel is None else fuel
    elev = np.zeros(shape) if elev is None else elev
    return fs.simulate_scenarios(one_hot(shape, cell), fuel, elev, speed, from_deg, n, seed,
                                 r0=r0, horizon_hours=hours)


def test_zero_wind_flat_is_symmetric_and_starts_at_zero():
    n = 11
    c = n // 2
    a = run((n, n), (c, c)).arrival_grid(0)
    assert a[c, c] == 0.0
    assert np.isfinite(a).all()
    for t in (np.rot90(a), np.rot90(a, 2), np.rot90(a, 3), a[::-1], a[:, ::-1], a.T):
        np.testing.assert_allclose(t, a, rtol=1e-5)


@pytest.mark.parametrize("from_deg, down, up", [
    (0, (1, 0), (-1, 0)),      # wind from north blows towards south: larger row is downwind
    (90, (0, -1), (0, 1)),     # wind from east blows towards west: smaller column is downwind
    (180, (-1, 0), (1, 0)),
    (270, (0, 1), (0, -1)),
])
def test_wind_direction_convention(from_deg, down, up):
    n, c, k = 21, 10, 4
    a = run((n, n), (c, c), speed=5.0, from_deg=from_deg).arrival_grid(0)
    d, u = (c + k * down[0], c + k * down[1]), (c + k * up[0], c + k * up[1])
    assert a[d] < a[u]


def test_uphill_reached_before_downhill():
    n, c, k = 21, 10, 4
    rows = np.arange(n)[:, None] * np.ones((1, n))
    elev = (n - 1 - rows) * 50.0           # 10% grade rising towards the north (row 0)
    a = run((n, n), (c, c), elev=elev).arrival_grid(0)
    assert a[c - k, c] < a[c + k, c]
    assert np.isclose(a[c, c - k], a[c, c + k], rtol=1e-5)   # cross-slope is symmetric


def test_impassable_cells_are_never_reached_and_block_spread():
    shape = (5, 7)
    fuel = np.ones(shape)
    fuel[:, 3] = 0.0                        # a full wall
    a = run(shape, (2, 1), fuel=fuel).arrival_grid(0)
    assert np.isinf(a[:, 3]).all()
    assert np.isinf(a[:, 4:]).all()
    assert np.isfinite(a[:, :3]).all()

    fuel[0, 3] = 1.0                        # open a gap in the top row: spread detours through it
    detour = run(shape, (2, 1), fuel=fuel).arrival_grid(0)
    free = run(shape, (2, 1)).arrival_grid(0)
    assert np.isinf(detour[1:, 3]).all()
    assert np.isfinite(detour[:, 4:]).all()
    assert detour[2, 5] > free[2, 5]


def test_same_seed_identical_and_different_seed_differs():
    rng = np.random.default_rng(3)
    shape = (12, 12)
    w, fuel, elev = rng.random(shape), 0.5 + rng.random(shape), rng.random(shape) * 100
    sp, di = rng.random(50) * 8, rng.random(50) * 360
    a = fs.simulate_scenarios(w, fuel, elev, sp, di, 20, seed=7)
    b = fs.simulate_scenarios(w, fuel, elev, sp, di, 20, seed=7)
    c = fs.simulate_scenarios(w, fuel, elev, sp, di, 20, seed=8)
    np.testing.assert_array_equal(a.arrival_min, b.arrival_min)
    np.testing.assert_array_equal(a.ignition_cell, b.ignition_cell)
    np.testing.assert_array_equal(a.wind_speed_ms, b.wind_speed_ms)
    assert not np.array_equal(a.ignition_cell, c.ignition_cell)
    assert not np.array_equal(a.wind_dir_deg, c.wind_dir_deg)


def test_ignition_sampling_respects_weights_and_burnability():
    w = np.array([[1.0, 0.0, 3.0],
                  [5.0, 2.0, 0.0]])
    fuel = np.ones((2, 3))
    fuel[1, 0] = 0.0                        # positive weight but non-burnable
    s = fs.simulate_scenarios(w, fuel, np.zeros((2, 3)), 0.0, 0.0, 6000, seed=1, horizon_hours=1)
    freq = np.bincount(s.ignition_cell, minlength=6) / 6000
    assert freq[1] == 0 and freq[5] == 0    # zero weight
    assert freq[3] == 0                     # non-burnable
    np.testing.assert_allclose(freq[[0, 2, 4]], np.array([1, 3, 2]) / 6, atol=0.02)
    assert np.isfinite(s.arrival_min[np.arange(6000), s.ignition_cell]).all()


def test_ignition_with_no_usable_weight_raises():
    with pytest.raises(ValueError):
        fs.simulate_scenarios(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]]), np.zeros((1, 2)),
                              0.0, 0.0, 1, seed=0)


def test_hand_computed_line_and_diagonal():
    r0 = 2.0
    a = run((1, 5), (0, 0), r0=r0).arrival_grid(0)[0]
    np.testing.assert_allclose(a, [0, 250, 500, 750, 1000], rtol=1e-6)    # 500 m / 2 m/min per step

    d = run((3, 3), (0, 0), r0=r0).arrival_grid(0)
    assert d[0, 1] == pytest.approx(250.0)
    assert d[1, 1] == pytest.approx(250.0 * np.sqrt(2.0), rel=1e-6)
    assert d[2, 2] == pytest.approx(500.0 * np.sqrt(2.0), rel=1e-6)

    h = run((1, 5), (0, 0), r0=r0, hours=500 / 60.0 + 1e-3).arrival_grid(0)[0]   # horizon 500 min
    assert np.isfinite(h[:3]).all() and np.isinf(h[3:]).all()


def test_fuel_factor_slows_spread_by_mean_of_the_two_cells():
    fuel = np.array([[1.0, 0.5]])
    a = run((1, 2), (0, 0), fuel=fuel, r0=2.0).arrival_grid(0)[0]
    # half a cell at 2 m/min plus half a cell at 1 m/min: 125 + 250 min
    assert a[1] == pytest.approx(375.0, rel=1e-6)


def test_daytime_wind_pool_uses_local_hours():
    t = np.arange(24, dtype=np.int64) + 24 * 365       # one day, hour 0 UTC upwards
    w = fs.WindRecord(t, np.arange(24, dtype=float), np.zeros(24))
    speed, _ = fs.daytime_wind_pool(w)                 # local 10:00 up to 18:00 = 03:00 to 11:00 UTC
    np.testing.assert_array_equal(speed, np.arange(3, 11))


def test_real_window_inputs_are_consistent():
    d = fs.load_real_window()
    assert d["fuel"].shape == d["elevation"].shape == d["ignition_weights"].shape == (40, 40)
    assert (d["ignition_weights"][d["fuel"] == 0] == 0).all()
    assert d["ignition_weights"].sum() > 0
    sub = fs.load_real_window(window=(100, 90), shape=(10, 12))
    assert sub["fuel"].shape == (10, 12)
    np.testing.assert_allclose(sub["elevation"], d["elevation"][20:30, 10:22])
