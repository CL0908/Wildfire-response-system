"""
Fire-spread scenario simulator for sensor-placement analysis.

SIMPLIFIED AND UNCALIBRATED. This is a minimum-travel-time (Dijkstra) spread model on an
8-neighbour grid with hand-set rate multipliers for fuel, slope and wind. It has not been
fitted to or validated against any observed fire. Its outputs are scenario arrival times for
comparing sensor layouts under the same stated assumptions; they are not predictions of real
fire behaviour, and absolute times scale directly with the assumed base rate `r0`.

Conventions: arrays have shape (rows, cols); row 0 is the north edge, column 0 the west edge;
flat index = row * cols + col; cells are `cell_m` (default 500 m) square. Nothing here is tied
to a particular grid size.

Model
-----
Edge between 8-neighbour cells i -> j (both burnable), length d (500 m or 500*sqrt(2) m):

    rate_ij = r0 * slope_factor_ij * wind_factor_ij          (m/min, before fuel)
    time_ij = 0.5 * d * (1/(fuel_i*rate_ij) + 1/(fuel_j*rate_ij))

i.e. the fire crosses half of cell i at i's fuel factor and half of cell j at j's (the mean of
the two, taken as travel time, which is symmetric in the fuel and exact for a centre-to-centre
path). Arrival time of a cell is the shortest path time from the ignition cell.

    slope_factor = clip(exp(k_slope * s), slope_min, slope_max), s = (z_j - z_i) / d
        s > 0 is uphill. k_slope = ln 2 / tan(10 deg) = 3.93, from the rule of thumb that
        forward spread doubles per 10 degrees of uphill slope (McArthur's slope correction
        for Australian forest fires, as commonly quoted; a rule of thumb that is not fully
        validated, not a Thai figure, and not fitted here). Bounds 0.5 and 4.0 (assumption).
    wind_factor  = exp(k_wind * min(U, wind_cap) * cos(angle)), U in m/s
        angle = angle between the spread direction and the downwind direction. Equals 1 at
        zero wind or when spreading cross-wind; larger downwind, smaller upwind; bounded
        between exp(-k_wind * wind_cap) and exp(k_wind * wind_cap). k_wind = 0.12 s/m and
        wind_cap = 15 m/s (assumptions; they fold the reduction from 10 m wind to mid-flame
        wind into one constant).

Wind is one (speed, direction) pair per scenario, constant over the map and over the
scenario. Direction is meteorological: degrees the wind blows FROM, 0 = north, 90 = east, so a
wind from the north pushes fire towards increasing row index.

Fuel: `fuel` is a float array, 0 = impassable (water, built-up, other non-fuel), > 0 is a
relative spread multiplier (1.0 = base rate `r0`).

Ignition: one cell per scenario, drawn with probability proportional to `ignition_weights`
(zeroed on non-burnable cells). `ignition_weights_from_counts` implements the smoothed
past-fire-count score used by the validated persistence predictor
(hazard_model.history_feature: gaussian_filter(log1p(count), sigma=1 cell, mode="nearest")).

Rate of spread, r0. Published experimental burns in Thai deciduous forest report:
  - 0.51 to 2.55 m/min over 80 plots in deciduous forests of northern Thailand, 2008 to 2009
    (Junpen, Garivait, Bonnet and Pongpullponsak 2013, ScienceAsia 39(5): 535-545,
    doi:10.2306/scienceasia1513-1874.2013.39.535);
  - a mean of 1.89 m/min over 21 plots in dry deciduous dipterocarp forest at Huai Kha Khaeng
    (Wiriya and Kaitpraneet 2009, Agriculture and Natural Resources 43(5): 41-47);
  - means of 0.47 and 0.44 m/min in two successive years over 12 plots at the same site
    (Himmapan and Kaitpraneet 2006, Thai Journal of Forestry 25(1-2): 112-124).
No published rate was found for evergreen forest in Thailand or for Khao Yai, which is mostly
evergreen, so these plots do not represent it. The default r0 = 2.0 m/min is applied to fuel
factor 1.0 (grass), so the tree class (0.6) gets 1.2 m/min. This is an ASSUMPTION inside the
deciduous-forest range above, not a calibrated value for Khao Yai. Sensitivity runs use about
0.5 to 6 m/min; the upper end is an assumption with no source checked.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hazard_model import history_feature  # noqa: E402  (the persistence smoothing)

CELL_M = 500.0
DEFAULT_R0_M_MIN = 2.0          # m/min at fuel factor 1.0; assumption within the 0.44-2.55 range of the cited burns
DEFAULT_HORIZON_HOURS = 24.0
K_SLOPE = float(np.log(2.0) / np.tan(np.radians(10.0)))   # doubling per 10 degrees uphill
SLOPE_MIN, SLOPE_MAX = 0.5, 4.0
K_WIND = 0.12                   # per m/s, assumption
WIND_CAP_MS = 15.0
UTC_OFFSET_H = 7                # Thailand
DAY_HOURS_LOCAL = (10, 18)      # assumption: scenario wind drawn from local 10:00 up to 18:00
MIN_FLAMMABLE_SHARE = 0.5       # a cell is burnable if >= this share of it is fuel classes

# Relative spread multiplier by land-cover class: the same factors as
# priority_surface.FUEL_FACTORS (assumption there too). Water, built-up, wetland and other
# classes are impassable.
FUEL_FACTORS = dict(lc_grass=1.0, lc_shrub=0.9, lc_tree=0.6, lc_crop=0.5, lc_bare=0.1)

HAZARD_TRAINING_PATH = os.path.join(ROOT, "data", "hazard_training_500m.npz")
DEFAULT_WINDOW = (80, 80)       # row/col offset of the 40 x 40 project window in the 200 x 200 file
DEFAULT_WINDOW_SHAPE = (40, 40)
DEFAULT_WIND_PATH = os.path.join("data", "wind_fire_season.npz")


# ------------------------------------------------------------------------------ inputs

@dataclass(frozen=True)
class WindRecord:
    """Hourly wind record: time_utc int64 hours since 1970-01-01T00:00Z; direction is FROM."""
    time_utc: np.ndarray
    speed_ms: np.ndarray
    direction_deg: np.ndarray


def load_wind(path: str = DEFAULT_WIND_PATH) -> WindRecord:
    """Read time_utc, speed_ms, direction_deg from an npz (path relative to the repo root)."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with np.load(path) as z:
        return WindRecord(np.asarray(z["time_utc"], dtype=np.int64),
                          np.asarray(z["speed_ms"], dtype=np.float64),
                          np.asarray(z["direction_deg"], dtype=np.float64))


def daytime_wind_pool(wind: WindRecord, hours_local: tuple = DAY_HOURS_LOCAL,
                      utc_offset_h: int = UTC_OFFSET_H) -> tuple[np.ndarray, np.ndarray]:
    """
    (speed_ms, direction_deg) of the records whose local hour h satisfies
    hours_local[0] <= h < hours_local[1] (default 10:00 up to 18:00 at UTC+7). Drawing a scenario
    wind uniformly from this pool is the assumption "one random daytime hour per scenario".
    """
    h = (wind.time_utc + utc_offset_h) % 24
    keep = (h >= hours_local[0]) & (h < hours_local[1])
    if not keep.any():
        raise ValueError("no wind records inside the requested local hours")
    return wind.speed_ms[keep], wind.direction_deg[keep]


def ignition_weights_from_counts(counts: np.ndarray, burnable: np.ndarray | None = None) -> np.ndarray:
    """
    Ignition weights from per-cell fire counts (2-D, all years summed): the persistence score
    gaussian_filter(log1p(counts), sigma=1, mode="nearest"), zeroed on non-burnable cells.
    Smooth on the largest grid available and slice afterwards, to avoid edge effects.
    """
    w = history_feature(counts)
    if burnable is not None:
        w = np.where(burnable, w, 0.0)
    return w


def load_real_window(window: tuple = DEFAULT_WINDOW, shape: tuple = DEFAULT_WINDOW_SHAPE,
                     path: str = HAZARD_TRAINING_PATH,
                     min_flammable_share: float = MIN_FLAMMABLE_SHARE) -> dict:
    """
    Real inputs for a (rows, cols) window starting at (row0, col0) of the 200 x 200, 500 m
    hazard-training grid (the project 40 x 40 window is the default (80, 80); this layer is
    cell-for-cell identical to data/khao_yai_500m.npz there, which priority_surface loads).

    Returns dict: fuel (float, 0 = impassable), elevation (m), ignition_weights (smoothed 2012 to
    2025 counts, 0 on non-burnable cells), inside_park (share 0-1), burnable (bool), window, cell_m.
    Fuel factor of a cell = share-weighted mean of FUEL_FACTORS over its fuel classes
    (tree, shrub, grass, crop, bare); the cell is non-burnable if those classes cover less than
    `min_flammable_share` of it.
    """
    row0, col0 = window
    rows, cols = shape
    with np.load(path) as z:
        n_r, n_c = z["dem_m"].shape
        if row0 < 0 or col0 < 0 or row0 + rows > n_r or col0 + cols > n_c:
            raise ValueError(f"window {window} with shape {shape} does not fit in {n_r} x {n_c}")
        sl = (slice(row0, row0 + rows), slice(col0, col0 + cols))
        share = {k: np.asarray(z[k], dtype=np.float64)[sl] for k in FUEL_FACTORS}
        flam = sum(share.values())
        weighted = sum(FUEL_FACTORS[k] * share[k] for k in FUEL_FACTORS)
        burnable = flam >= min_flammable_share
        fuel = np.where(burnable, weighted / np.maximum(flam, 1e-12), 0.0)
        counts = np.asarray(z["fire_by_year"], dtype=np.float64).sum(axis=0)
        weights = ignition_weights_from_counts(counts)[sl]
        weights = np.where(burnable, weights, 0.0)
        return dict(fuel=fuel, elevation=np.asarray(z["dem_m"], dtype=np.float64)[sl],
                    ignition_weights=weights,
                    inside_park=np.asarray(z["inside_park"], dtype=np.float64)[sl],
                    burnable=burnable, window=(row0, col0), cell_m=CELL_M)


# ------------------------------------------------------------------------------ graph

_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


class _Graph:
    """Static part of the spread graph: CSR structure, fuel/slope base times, edge unit vectors."""

    def __init__(self, fuel, elevation, r0, cell_m):
        rows, cols = fuel.shape
        idx = np.arange(rows * cols).reshape(rows, cols)
        burn = fuel > 0
        src, dst, dlen, ur, uc = [], [], [], [], []
        for dr, dc in _OFFSETS:
            r_lo, r_hi = max(0, -dr), rows - max(0, dr)
            c_lo, c_hi = max(0, -dc), cols - max(0, dc)
            a = (slice(r_lo, r_hi), slice(c_lo, c_hi))
            b = (slice(r_lo + dr, r_hi + dr), slice(c_lo + dc, c_hi + dc))
            ok = burn[a] & burn[b]
            s, t = idx[a][ok], idx[b][ok]
            norm = np.hypot(dr, dc)
            src.append(s)
            dst.append(t)
            dlen.append(np.full(s.size, norm * cell_m))
            ur.append(np.full(s.size, dr / norm))
            uc.append(np.full(s.size, dc / norm))
        src, dst, dlen, ur, uc = (np.concatenate(x) for x in (src, dst, dlen, ur, uc))
        order = np.argsort(src, kind="stable")
        src, dst, dlen, ur, uc = src[order], dst[order], dlen[order], ur[order], uc[order]
        f, z = fuel.ravel(), elevation.ravel()
        slope = (z[dst] - z[src]) / dlen
        slope_factor = np.clip(np.exp(K_SLOPE * slope), SLOPE_MIN, SLOPE_MAX)
        # time at wind factor 1, minutes: half a cell at each end's fuel factor
        self.base_time = 0.5 * dlen * (1.0 / f[src] + 1.0 / f[dst]) / (r0 * slope_factor)
        self.ur, self.uc = ur, uc
        self.indices = dst.astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=rows * cols))]).astype(np.int32)
        self.n = rows * cols
        self.mat = csr_matrix((np.ones(dst.size), self.indices, self.indptr), shape=(self.n, self.n))

    def arrival(self, ignition: int, speed: float, from_deg: float, limit_min: float) -> np.ndarray:
        # Downwind unit vector in (row, col): wind FROM compass angle a blows towards a + 180;
        # north is -row, east is +col.
        toward = np.radians(from_deg + 180.0)
        d_row, d_col = -np.cos(toward), np.sin(toward)
        cos_angle = self.ur * d_row + self.uc * d_col
        wind_factor = np.exp(K_WIND * min(max(speed, 0.0), WIND_CAP_MS) * cos_angle)
        self.mat.data = self.base_time / wind_factor
        return dijkstra(self.mat, directed=True, indices=ignition, limit=limit_min)


# ------------------------------------------------------------------------------ entry point

@dataclass
class ScenarioSet:
    arrival_min: np.ndarray        # float32 (n_scenarios, n_cells); inf = not reached / unreachable
    ignition_cell: np.ndarray      # int64 (n_scenarios,) flat index
    wind_speed_ms: np.ndarray      # float64 (n_scenarios,)
    wind_dir_deg: np.ndarray       # float64 (n_scenarios,), FROM direction
    shape: tuple                   # (rows, cols)
    params: dict = field(default_factory=dict)

    def arrival_grid(self, i: int) -> np.ndarray:
        return self.arrival_min[i].reshape(self.shape)


def simulate_scenarios(ignition_weights, fuel, elevation, wind_speed, wind_dir, n_scenarios: int,
                       seed: int, r0: float = DEFAULT_R0_M_MIN,
                       horizon_hours: float = DEFAULT_HORIZON_HOURS,
                       cell_m: float = CELL_M) -> ScenarioSet:
    """
    Per-cell fire arrival times for `n_scenarios` seeded ignition + wind scenarios.

    SIMPLIFIED, UNCALIBRATED scenario model (see the module docstring): the times compare
    sensor layouts under stated assumptions and are not predictions of real fire behaviour.

    ignition_weights (rows, cols) >= 0: ignition probability weights; cells with fuel == 0 are
        forced to zero. One ignition cell per scenario.
    fuel (rows, cols): relative spread multiplier, 0 = impassable.
    elevation (rows, cols): metres.
    wind_speed (m/s), wind_dir (degrees the wind blows FROM, 0 = north, 90 = east): scalars
        (constant wind) or equal-length 1-D pools; each scenario draws one (speed, direction)
        pair uniformly from the pool and keeps it for the whole scenario.
    r0: base rate of spread in m/min at fuel factor 1 (plausible 0.5 to 6; default 2.0).
    horizon_hours: arrival times beyond this are reported as inf.
    Deterministic for a given seed. Returns a ScenarioSet; arrival_min is float32, in minutes.
    """
    fuel = np.asarray(fuel, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    w = np.asarray(ignition_weights, dtype=np.float64)
    if fuel.ndim != 2 or elevation.shape != fuel.shape or w.shape != fuel.shape:
        raise ValueError("ignition_weights, fuel and elevation must be 2-D arrays of one shape")
    if r0 <= 0 or horizon_hours <= 0 or cell_m <= 0 or n_scenarios < 1:
        raise ValueError("r0, horizon_hours, cell_m must be > 0 and n_scenarios >= 1")
    if (fuel < 0).any() or (w < 0).any():
        raise ValueError("fuel and ignition_weights must be non-negative")
    w = np.where(fuel > 0, w, 0.0).ravel()
    if w.sum() <= 0:
        raise ValueError("no burnable cell has a positive ignition weight")
    speed_pool = np.atleast_1d(np.asarray(wind_speed, dtype=np.float64))
    dir_pool = np.atleast_1d(np.asarray(wind_dir, dtype=np.float64))
    if speed_pool.ndim != 1 or speed_pool.shape != dir_pool.shape or speed_pool.size == 0:
        raise ValueError("wind_speed and wind_dir must be scalars or equal-length 1-D arrays")

    ss_ign, ss_wind = np.random.SeedSequence(seed).spawn(2)
    ignition = np.random.default_rng(ss_ign).choice(w.size, size=n_scenarios, p=w / w.sum())
    pick = np.random.default_rng(ss_wind).integers(speed_pool.size, size=n_scenarios)
    speeds, dirs = speed_pool[pick], dir_pool[pick]

    # Own Dijkstra rather than ranger_response.model._spread: that one is pure Python heapq on a
    # 4-neighbour graph with integer edge times fixed at build time; here the 8-neighbour float
    # edge times change with every scenario's wind, and 5,000 scenarios need scipy's C Dijkstra.
    g = _Graph(fuel, elevation, r0, cell_m)
    limit = horizon_hours * 60.0
    out = np.empty((n_scenarios, g.n), dtype=np.float32)
    for k in range(n_scenarios):
        out[k] = g.arrival(int(ignition[k]), float(speeds[k]), float(dirs[k]), limit)
    return ScenarioSet(out, ignition.astype(np.int64), speeds, dirs, fuel.shape,
                       dict(r0=r0, horizon_hours=horizon_hours, cell_m=cell_m, seed=seed))
