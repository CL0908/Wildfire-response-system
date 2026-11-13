"""
Learned wildfire hazard: feature builder and predictor for the logistic models fitted by
data/fit_hazard_model.py and stored in data/hazard_model.json.

No scikit-learn here. numpy for the model, scipy.ndimage (already a runtime dependency of
priority_surface.py) for the smoothing and neighbourhood filters. Prediction is a standardised
linear model followed by a sigmoid.

Features are neighbourhood-based (slope, 1.5 km crop share, fire-history smoothing), so build
them on the whole 200 x 200 training grid and slice the window afterwards. A small window
gets edge effects.

Read data/HAZARD_VALIDATION.md before using the output: it is a ranking of cells by the chance
of at least one satellite detection, not a calibrated ignition probability.
"""

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "data", "hazard_model.json")
CELL_SIZE_M = 500.0
HISTORY_FEATURE = "fire_history"

_LC = ["lc_tree", "lc_shrub", "lc_grass", "lc_crop", "lc_built", "lc_bare", "lc_water"]
STATIC_FEATURE_NAMES = (
    ["elevation_m", "slope_deg", "northness_x_slope", "eastness_x_slope"]
    + _LC
    + ["inside_park", "signed_dist_park_edge_km",
       "log1p_dist_road", "log1p_dist_major_road", "log1p_dist_settlement",
       "crop_share_1p5km"]
)


def _f64(arrays, key):
    return np.asarray(arrays[key], dtype=np.float64)


def build_features(arrays: dict, cell_size: float = CELL_SIZE_M):
    """
    Static features from npz-style arrays (2-D, row 0 north).
    Returns (X, names): X has shape (rows * cols, n_features), row-major cell order.
    """
    dem = _f64(arrays, "dem_m")
    dy, dx = np.gradient(dem, cell_size, cell_size)       # dy: change per row, i.e. towards south
    g = np.hypot(dx, dy)
    slope_deg = np.degrees(np.arctan(g))
    safe = np.where(g > 0, g, 1.0)
    # Direction the slope faces (downhill): north component = dy/|g|, east component = -dx/|g|.
    north = np.where(g > 0, dy / safe, 0.0) * slope_deg
    east = np.where(g > 0, -dx / safe, 0.0) * slope_deg

    inside = _f64(arrays, "inside_park")
    edge = _f64(arrays, "dist_park_edge_m")
    signed_edge_km = np.where(inside > 0.5, -edge, edge) / 1000.0
    crop_near = uniform_filter(_f64(arrays, "lc_crop"), size=3, mode="nearest")

    cols = [dem, slope_deg, north, east]
    cols += [_f64(arrays, k) for k in _LC]
    cols += [inside, signed_edge_km,
             np.log1p(_f64(arrays, "dist_road_m")),
             np.log1p(_f64(arrays, "dist_major_road_m")),
             np.log1p(_f64(arrays, "dist_settlement_m")),
             crop_near]
    X = np.stack([c.ravel() for c in cols], axis=1)
    return X, list(STATIC_FEATURE_NAMES)


def history_feature(counts: np.ndarray) -> np.ndarray:
    """Smoothed log1p of a per-cell detection count (2-D), the fire-history feature."""
    return gaussian_filter(np.log1p(np.asarray(counts, dtype=np.float64)), sigma=1.0,
                           mode="nearest")


def history_counts(arrays: dict, model: dict) -> np.ndarray:
    """
    Per-cell detection count over the model's history window, rescaled to the length the model
    was trained with. Needs arrays["fire_by_year"] (years, rows, cols) and arrays["years"].
    """
    spec = model["history"]
    years = np.asarray(arrays["years"]).astype(int)
    sel = (years >= spec["first_year"]) & (years <= spec["last_year"])
    if not sel.any():
        raise ValueError("no years of fire history inside the model's history window")
    counts = np.asarray(arrays["fire_by_year"], dtype=np.float64)[sel].sum(axis=0)
    return counts * (spec["rescale_to_years"] / float(sel.sum()))


def load_model(path: str = DEFAULT_MODEL_PATH, name: str = "validated") -> dict:
    """One exported model ('validated', 'deployment' or 'static') as a dict."""
    with open(path) as f:
        doc = json.load(f)
    if name not in doc["models"]:
        raise KeyError(f"model {name!r} not in {path}; has {sorted(doc['models'])}")
    return doc["models"][name]


def predict(arrays: dict, model: dict, fire_history=None) -> np.ndarray:
    """
    Probability of at least one detection in the model's label window, per cell, shaped like
    arrays["dem_m"].

    Models with a history window need fire history: pass `fire_history`, a 2-D per-cell count
    already summed over the window and rescaled (see history_counts), or leave it None and
    arrays["fire_by_year"] / arrays["years"] are used.
    """
    shape = np.asarray(arrays["dem_m"]).shape
    X, names = build_features(arrays)
    if model["history"] is not None:
        if fire_history is None:
            fire_history = history_counts(arrays, model)
        fire_history = np.asarray(fire_history, dtype=np.float64)
        if fire_history.shape != shape:
            raise ValueError(f"fire_history shape {fire_history.shape} != grid {shape}")
        X = np.concatenate([X, history_feature(fire_history).reshape(-1, 1)], axis=1)
        names = names + [HISTORY_FEATURE]
    if names != list(model["features"]):
        raise ValueError("feature names do not match the model's")
    z = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logit = z @ np.asarray(model["coef"]) + model["intercept"]
    return (1.0 / (1.0 + np.exp(-logit))).reshape(shape)
