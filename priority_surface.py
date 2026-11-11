"""
Step A: Build the priority surface (hazard × value) on a raster grid.

Physical setup: 20×20 grid, cell_size = 500 m → 10 km × 10 km.
Two terrain sources:
  source="synthetic": smooth synthetic DEM in metres (300–1 350 m, matching Khao Yai)
                      and random hazard/value layers (seeded).
  source="real":      a window of the committed 40 × 40 cell extract of real geodata in
                      data/khao_yai_500m.npz (Copernicus DEM, NASA FIRMS VIIRS fire
                      detections, ESA WorldCover, OpenStreetMap park boundary).
"""

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter

CELL_SIZE_M = 500.0          # metres per grid cell
GRID_ROWS = GRID_COLS = 20   # canonical grid size

_HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_PATH = os.path.join(_HERE, "data", "khao_yai_500m.npz")
REAL_META_PATH = os.path.join(_HERE, "data", "khao_yai_meta.json")

# Modelling assumptions for the real hazard model (not measurements).
REAL_DEFAULT_WEIGHTS = dict(fire=0.40, fuel=0.25, dryness=0.20, spread=0.15)
# Fuel flammability by land-cover class (modelling assumption).
FUEL_FACTORS = dict(lc_grass=1.0, lc_shrub=0.9, lc_tree=0.6, lc_crop=0.5, lc_bare=0.1)


# ---------------------------------------------------------------------------
# Smooth DEM (metres)
# ---------------------------------------------------------------------------

def build_smooth_dem(
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    seed: int = 42,
) -> np.ndarray:
    """
    Synthetic DEM in metres, approximately matching Khao Yai (300–1 350 m).

    Structure:
      base = 350 m + gentle regional NE-rising tilt
      + 4 Gaussian peaks (massif-style) with amplitudes up to ~1 000 m
      + light white noise (σ = 15 m)
      → Gaussian blur (σ = 1.5 cells) for realistic smoothness
      → clip to ≥ 300 m
    """
    rng = np.random.default_rng(seed)
    rows, cols = grid_rows, grid_cols
    rr = np.arange(rows)[:, None]   # shape (rows, 1)
    cc = np.arange(cols)[None, :]   # shape (1, cols)

    # Regional tilt: rises gently toward NE
    dem = 350.0 + 3.5 * rr + 2.0 * cc

    # Gaussian peaks — (row, col, amplitude_m, sigma_cells)
    peaks = [
        (7,  9,  1000, 2.5),   # main massif
        (5,  5,   750, 2.0),   # western ridge
        (12, 13,  600, 1.8),   # southern plateau
        (4,  14,  450, 1.5),   # eastern shoulder
    ]
    for pr, pc, amp, sig in peaks:
        dem = dem + amp * np.exp(-((rr - pr)**2 + (cc - pc)**2) / (2 * sig**2))

    # Light noise
    dem = dem + rng.normal(0, 15, (rows, cols))

    # Smooth
    dem = gaussian_filter(dem, sigma=1.5)

    # Hard floor
    dem = np.clip(dem, 300.0, None)

    return dem


# ---------------------------------------------------------------------------
# Priority surface (hazard × value), all layers [0, 1]
# ---------------------------------------------------------------------------

def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def _terrain_derivatives(dem_m: np.ndarray, cell_size: float):
    """Slope (degrees), slope [0, 1] and aspect [0, 1] from a DEM in metres."""
    dy_m, dx_m = np.gradient(dem_m, cell_size, cell_size)   # rise / run (m/m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx_m**2 + dy_m**2)))
    slope = _normalise(slope_deg)
    aspect = _normalise(np.arctan2(dy_m, dx_m) + np.pi)
    return slope_deg, slope, aspect


def _dryness_layer(dem: np.ndarray, jitter: np.ndarray | None = None) -> np.ndarray:
    """Hotter/drier at lower elevations. `dem` is normalised; `jitter` is optional noise."""
    base = (1 - dem) * 0.7
    if jitter is not None:
        base = base + 0.3 * jitter
    return _normalise(base)


def _spread_layer(slope: np.ndarray, aspect: np.ndarray) -> np.ndarray:
    """Spread terrain: steep south-facing slopes in prevailing wind."""
    return _normalise(slope * 0.6 + (1 - aspect) * 0.4)


# ---------------------------------------------------------------------------
# Real terrain
# ---------------------------------------------------------------------------

def _load_real_region():
    """Return (arrays as float64 / int, meta dict) for the committed extract."""
    hint = ("run `python data/fetch_khao_yai.py` to regenerate the extract "
            "or use source=\"synthetic\"")
    for path in (REAL_DATA_PATH, REAL_META_PATH):
        if not os.path.exists(path):
            raise ValueError(f"Real terrain data file not found: {path}; {hint}.")
    with np.load(REAL_DATA_PATH) as z:
        data = {k: z[k] for k in z.files}
    with open(REAL_META_PATH) as f:
        meta = json.load(f)
    return data, meta


def _resolve_real_weights(weights: dict | None) -> dict:
    """Map user weights to fire/fuel/dryness/spread; accept arson=fire and terrain=spread."""
    if weights is None:
        return dict(REAL_DEFAULT_WEIGHTS)
    alias = dict(fire=("fire", "arson"), fuel=("fuel",), dryness=("dryness",),
                 spread=("spread", "terrain"))
    out = {}
    for key, names in alias.items():
        found = [n for n in names if n in weights]
        if len(found) != 1:
            raise ValueError(
                f"weights for source='real' need exactly one of {list(names)} "
                f"(got {found}); keys are fire (or arson), fuel, dryness, spread (or terrain)")
        out[key] = weights[found[0]]
    return out


def _real_layers(data: dict, w: dict, cell_size: float) -> dict:
    """
    All real-terrain layers for the whole region in `data` (dict of 2-D arrays).
    Every normalisation happens here, over the whole region, so a window sliced
    out of the result does not depend on which window was chosen.
    """
    f64 = lambda k: np.asarray(data[k], dtype=np.float64)
    dem_m = f64("dem_m")
    dem = _normalise(dem_m)
    slope_deg, slope, aspect = _terrain_derivatives(dem_m, cell_size)

    fire = _normalise(gaussian_filter(np.log1p(f64("fire_count")), sigma=1.0, mode="nearest"))
    fuel = _normalise(sum(f * f64(k) for k, f in FUEL_FACTORS.items()))
    dryness = _dryness_layer(dem)
    spread = _spread_layer(slope, aspect)

    hazard = _normalise(
        w["fire"] * fire + w["fuel"] * fuel + w["dryness"] * dryness + w["spread"] * spread
    )

    inside = f64("inside_park")
    value = inside * (0.5 + 0.5 * f64("lc_tree")) + (1 - inside) * 0.1
    priority = _normalise(hazard * value)

    return dict(
        dem_m=dem_m, dem=dem, slope_deg=slope_deg, slope=slope, aspect=aspect,
        arson=fire, fire=fire, fuel=fuel, dryness=dryness, spread=spread,
        hazard=hazard, value=value, priority=priority,
        lat=f64("lat"), lon=f64("lon"), inside_park=inside,
        fire_count=np.asarray(data["fire_count"]),
    )


def real_window_bounds(surf: dict) -> dict:
    """North/south/west/east (cell edges, degrees) of a real-terrain surface's window."""
    if surf.get("source") != "real":
        raise ValueError("real_window_bounds needs a surface built with source='real'")
    _, meta = _load_real_region()
    dlat = meta["cell_size_deg"]["lat"]
    dlon = meta["cell_size_deg"]["lon"]
    lat, lon = surf["lat"], surf["lon"]
    return dict(north=float(lat[0, 0] + dlat / 2), south=float(lat[-1, 0] - dlat / 2),
                west=float(lon[0, 0] - dlon / 2), east=float(lon[0, -1] + dlon / 2))


def _build_real_surface(grid_rows, grid_cols, weights, cell_size, window) -> dict:
    data, meta = _load_real_region()
    reg_rows, reg_cols = data["dem_m"].shape
    meta_cell = float(meta["cell_size_m"])
    if cell_size != meta_cell:
        raise ValueError(
            f"source='real' is fixed at cell_size={meta_cell:g} m (got {cell_size}); "
            "use source=\"synthetic\" for other cell sizes")
    if window is None:
        dw = meta["default_window"]
        window = (dw["row0"], dw["col0"])
    try:
        row0, col0 = (int(v) for v in window)
    except (TypeError, ValueError):
        raise ValueError(f"window must be (row0, col0) integers, got {window!r}") from None
    if grid_rows < 1 or grid_cols < 1 or row0 < 0 or col0 < 0 \
            or row0 + grid_rows > reg_rows or col0 + grid_cols > reg_cols:
        raise ValueError(
            f"window (row0={row0}, col0={col0}) with grid {grid_rows} x {grid_cols} does not "
            f"fit inside the {reg_rows} x {reg_cols} real region (needs 0 <= row0, "
            f"row0 + rows <= {reg_rows}, 0 <= col0, col0 + cols <= {reg_cols})")

    layers = _real_layers(data, _resolve_real_weights(weights), cell_size)
    sl = (slice(row0, row0 + grid_rows), slice(col0, col0 + grid_cols))
    surf = {k: v[sl].copy() for k, v in layers.items()}
    return dict(grid_shape=(grid_rows, grid_cols), source="real", window=(row0, col0), **surf)


def build_priority_surface(
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    seed: int = 42,
    weights: dict | None = None,
    cell_size: float = CELL_SIZE_M,
    source: str = "synthetic",
    window: tuple | None = None,
) -> dict:
    """
    Returns a dict with keys:
        grid_shape, dem_m (elevation in metres), dem (normalised),
        slope_deg, slope, aspect,
        arson, fuel, dryness, spread,
        hazard, value, priority,
        source ("real" or "synthetic"), window ((row0, col0) or None)
    and, for source="real" only, also: fire (same array as arson), lat, lon,
    inside_park, fire_count.

    source="synthetic": seeded random terrain and layers. `weights` has keys
        arson, fuel, dryness, terrain (default 0.30, 0.25, 0.25, 0.20).
    source="real": a grid_rows x grid_cols window of the committed 40 x 40 cell
        (500 m) extract, starting at window=(row0, col0) (default: the extract's
        default_window). `seed` has no effect. All normalisation to 0-1 is done over
        the whole 40 x 40 region and then sliced, so a cell has the same values in any
        window. ValueError if the window does not fit, the data file is missing or
        cell_size != 500.
        Hazard = fire 0.40 + fuel 0.25 + dryness 0.20 + spread 0.15, where
          fire    = log1p(VIIRS detections 2012-2025), Gaussian-smoothed (sigma 1 cell);
          fuel    = grass 1.0, shrub 0.9, tree 0.6, crop 0.5, bare 0.1, others 0;
          dryness = synthetic dryness formula on the real DEM (lower is drier, no noise);
          spread  = synthetic slope/aspect formula on the real DEM.
        `weights` uses fire, fuel, dryness, spread; the synthetic names are accepted
        as aliases (arson = fire, terrain = spread). The layer returned as "arson"
        is the fire layer.
        value = inside_park * (0.5 + 0.5 * lc_tree) + (1 - inside_park) * 0.1.
        priority = normalise(hazard * value), as in synthetic mode.
        These weights and factors are modelling assumptions, not measurements; only
        the input layers (DEM, fire detections, land cover, park boundary) are real.
    """
    if source == "real":
        return _build_real_surface(grid_rows, grid_cols, weights, cell_size, window)
    if source != "synthetic":
        raise ValueError(f"source must be 'real' or 'synthetic', got {source!r}")

    rng = np.random.default_rng(seed)
    w = weights or dict(arson=0.30, fuel=0.25, dryness=0.25, terrain=0.20)

    # ── DEM ────────────────────────────────────────────────────────────────────
    dem_m = build_smooth_dem(grid_rows, grid_cols, seed)
    dem = _normalise(dem_m)  # [0, 1] for priority layers

    # Physical slope (degrees) using 500 m cell spacing
    slope_deg, slope, aspect = _terrain_derivatives(dem_m, cell_size)

    rows, cols = grid_rows, grid_cols

    # ── Hazard layers ─────────────────────────────────────────────────────────
    # arson_rate: high near boundary / road edges
    arson = np.zeros((rows, cols))
    arson[:3, :] = arson[-3:, :] = arson[:, :3] = arson[:, -3:] = 1.0
    arson = arson + 0.3 * rng.uniform(0, 1, (rows, cols))
    arson = _normalise(arson)

    # fuel: dense at mid-elevation flat zones (forest floor / peat)
    fuel = _normalise((1 - np.abs(dem - 0.4)) + 0.2 * rng.uniform(0, 1, (rows, cols)))

    # dryness: hotter/drier at lower elevations
    dryness = _dryness_layer(dem, rng.uniform(0, 1, (rows, cols)))

    # spread terrain: steep south-facing slopes in prevailing wind
    spread = _spread_layer(slope, aspect)

    hazard = _normalise(
        w["arson"] * arson
        + w["fuel"] * fuel
        + w["dryness"] * dryness
        + w["terrain"] * spread
    )

    # ── Value layer (ecological weight) ───────────────────────────────────────
    cx, cy = rows // 2, cols // 2
    r_dist = np.sqrt(
        (np.arange(rows)[:, None] - cx) ** 2
        + (np.arange(cols)[None, :] - cy) ** 2
    )
    core_zone = _normalise(np.exp(-r_dist / (rows * 0.25)))
    habitat = _normalise(fuel * 0.5 + core_zone * 0.5)
    value = _normalise(habitat + 0.3 * rng.uniform(0, 1, (rows, cols)))

    priority = _normalise(hazard * value)

    return dict(
        grid_shape=(rows, cols),
        dem_m=dem_m, dem=dem,
        slope_deg=slope_deg, slope=slope, aspect=aspect,
        arson=arson, fuel=fuel, dryness=dryness, spread=spread,
        hazard=hazard, value=value, priority=priority,
        source="synthetic", window=None,
    )


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf = build_priority_surface()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    layers = [
        ("DEM (m)", surf["dem_m"], "terrain"),
        ("Hazard",  surf["hazard"], "YlOrRd"),
        ("Value",   surf["value"], "Greens"),
        ("Priority", surf["priority"], "hot"),
        ("Slope (°)", surf["slope_deg"], "Greys"),
        ("Fuel load", surf["fuel"], "YlGn"),
    ]
    for ax, (title, data, cmap) in zip(axes.flat, layers):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle("Q-Pithak priority surface (synthetic Khao Yai)", fontsize=13)
    plt.tight_layout()
    plt.savefig("priority_surface.png", dpi=120)
    print("Saved priority_surface.png")
