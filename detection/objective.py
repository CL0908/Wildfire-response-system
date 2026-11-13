"""
Time-to-detection objective for sensor placement (one shared scoring function).

SIMPLIFIED AND UNCALIBRATED. The fire arrival times come from `detection.fire_sim`, a
hand-parameterised spread model that has not been fitted to or validated against any observed
fire. Every number computed here compares sensor layouts under the stated assumptions; none of
it is a prediction of real fire behaviour or of real detection times.

Conventions: grids are (rows, cols), row 0 is the north edge, flat cell index = row * cols + col.
`ScenarioSet.arrival_min` is float32 (n_scenarios, n_cells), minutes after ignition, inf = the
fire never reaches that cell inside the simulated horizon.

Assumptions of the detection model
----------------------------------
1. A sensor detects a fire the moment the fire enters any cell the sensor sees (`sees`
   is a boolean matrix (n_sites, n_cells), True where a sensor at that site sees the cell). A fire
   igniting inside a seen cell is therefore detected at time 0. There is no detection delay,
   no smoke or flame-size threshold, no sensor failure and no false-alarm handling.
2. Satellite baseline: every fire is detected by satellite `baseline_min` minutes after ignition
   at the latest (scalar, or one value per scenario so that a delay can be drawn per scenario).
   A fire the sensors never reach is detected at exactly `baseline_min`.
3. The detection time of scenario s under a placement is
       d_s = min(baseline_s, min over chosen sites j of T[s, j]),
   T[s, j] = min over cells c seen by site j of arrival_min[s, c]   (inf if none is reached).

Objective (to MAXIMISE): expected time saved against satellite only,
       sum_s w_s * (baseline_s - d_s)        with w_s >= 0 normalised to sum 1 (default uniform).
It is monotone (more sites never hurt) and has diminishing returns (submodular).

DEFAULT_BASELINE_MIN = 360 (6 hours) is an ASSUMPTION, not a measured value. What NASA's FIRMS FAQ
(https://www.earthdata.nasa.gov/data/tools/firms/faq, read 13 November 2026 through a summarising
fetch of the first half of the page) says that bears on it:
  - VIIRS flies on Suomi NPP and NOAA-20/21. NOAA-20/21 cross the equator at "approximately 12:40
    p.m. and 12:40 a.m." and Suomi NPP at 1:30 p.m. and 1:30 a.m. mean local time, so a given
    satellite looks at a given place roughly twice a day, about 12 hours apart; several satellites
    shorten the gaps. (The page is not consistent about the spacing between the satellites.)
  - Near-real-time data are available "within 3 hours of a satellite observation (on a
    best-effort basis)"; the page also gives faster real-time (30 minutes to 1 hour) and
    ultra-real-time (5 minutes) tiers for some stations.
So a fire can wait several hours for the next overpass and then up to 3 hours for the data; 6 hours
is a round number inside that range, chosen only so the sensors have something to beat. It ignores
clouds, smoke, fires too small for a 375 m pixel, and the time a human needs to act on an alert.
Treat it as a parameter and vary it.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BASELINE_MIN = 360.0  # ASSUMPTION, see the module docstring


def as_baseline_vector(baseline_min, n_scenarios: int) -> np.ndarray:
    """Scalar or (n_scenarios,) baseline as a float64 vector; values must be finite and > 0."""
    b = np.asarray(baseline_min, dtype=np.float64)
    if b.ndim == 0:
        b = np.full(n_scenarios, float(b))
    elif b.shape != (n_scenarios,):
        raise ValueError(f"baseline_min must be a scalar or shape ({n_scenarios},), got {b.shape}")
    if not np.isfinite(b).all() or (b <= 0).any():
        raise ValueError("baseline_min must be finite and > 0")
    return b


def as_weight_vector(weights, n_scenarios: int) -> np.ndarray:
    """Scenario weights normalised to sum 1 (None = uniform)."""
    if weights is None:
        return np.full(n_scenarios, 1.0 / n_scenarios)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_scenarios,):
        raise ValueError(f"weights must have shape ({n_scenarios},), got {w.shape}")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, with a positive sum")
    return w / w.sum()


def _as_time_matrix(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 2:
        raise ValueError("T must be 2-D (n_scenarios, n_sites)")
    if np.isnan(T).any() or (T < 0).any():
        raise ValueError("T must be >= 0 (inf allowed) and contain no NaN")
    return T


def saving_matrix(T, baseline_min) -> np.ndarray:
    """v[s, j] = max(0, baseline_s - T[s, j]): minutes a lone sensor at j saves in scenario s."""
    T = _as_time_matrix(T)
    b = as_baseline_vector(baseline_min, T.shape[0])
    return np.maximum(b[:, None] - T, 0.0)


def site_detection_times(arrival_min, sees) -> np.ndarray:
    """
    T[s, j] = earliest arrival_min[s, c] over the cells c that site j sees (inf if the site sees
    no cell or the fire never reaches any of them). Returns float64 (n_scenarios, n_sites).

    Loops over sites and works on column slices, so no (scenarios x sites x cells) array is built;
    5,000 x 18,500 x 300 stays at the size of `arrival_min` plus a bounded temporary.
    """
    arrival_min = np.asarray(arrival_min)
    sees = np.asarray(sees)
    if arrival_min.ndim != 2 or sees.ndim != 2 or sees.shape[1] != arrival_min.shape[1]:
        raise ValueError("arrival_min must be (n_scenarios, n_cells) and sees (n_sites, n_cells)")
    if sees.dtype != np.bool_:
        raise ValueError("sees must be a boolean matrix")
    n_scen, n_sites = arrival_min.shape[0], sees.shape[0]
    T = np.full((n_scen, n_sites), np.inf)
    for j in range(n_sites):
        cols = np.flatnonzero(sees[j])
        if cols.size == 0:
            continue
        step = max(1, 4_000_000 // cols.size)  # bound the temporary at ~4M floats
        for a in range(0, n_scen, step):
            T[a:a + step, j] = arrival_min[a:a + step][:, cols].min(axis=1)
    return T


def score_detection(T, chosen, baseline_min=DEFAULT_BASELINE_MIN, weights=None) -> dict:
    """
    The one scoring function for a placement; solvers and tests all go through it.

    T: (n_scenarios, n_sites) from `site_detection_times`. chosen: distinct site indices (may be
    empty). Returns a dict:
      expected_saved_min   weighted mean of baseline_s - d_s (the objective, maximised)
      mean_detection_min   weighted mean of d_s
      mean_baseline_min    weighted mean of baseline_s
      detection_min        d_s per scenario, float64 array
      saved_min            baseline_s - d_s per scenario
      share_detected_at_time_0                 weighted share with d_s == 0 (fire ignites in a seen cell)
      share_detected_by_sensor_after_spread    weighted share with 0 < d_s < baseline_s
      share_left_to_satellite                  weighted share with d_s == baseline_s
      The three shares sum to 1.
      chosen               sorted list of site indices; n_sensors their count
    Simplified, uncalibrated model: see the module docstring.
    """
    T = _as_time_matrix(T)
    n_scen, n_sites = T.shape
    b = as_baseline_vector(baseline_min, n_scen)
    w = as_weight_vector(weights, n_scen)
    idx = np.asarray(list(chosen), dtype=np.int64).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= n_sites):
        raise ValueError("chosen contains a site index out of range")
    if np.unique(idx).size != idx.size:
        raise ValueError("chosen contains a repeated site index")
    idx = np.sort(idx)
    d = b.copy() if idx.size == 0 else np.minimum(b, T[:, idx].min(axis=1))
    at0 = d <= 0.0
    sat = d >= b
    mid = ~at0 & ~sat
    return dict(
        expected_saved_min=float(w @ (b - d)),
        mean_detection_min=float(w @ d),
        mean_baseline_min=float(w @ b),
        detection_min=d,
        saved_min=b - d,
        share_detected_at_time_0=float(w @ at0),
        share_detected_by_sensor_after_spread=float(w @ mid),
        share_left_to_satellite=float(w @ sat),
        chosen=[int(i) for i in idx],
        n_sensors=int(idx.size),
    )
