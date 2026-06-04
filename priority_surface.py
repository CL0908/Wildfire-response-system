"""
Step A: Build the priority surface (hazard × value) on a raster grid.

Physical setup: 20×20 grid, cell_size = 500 m → 10 km × 10 km.
DEM is a smooth synthetic terrain in metres (300–1 350 m, matching Khao Yai).
Swap build_smooth_dem() for a real rasterio loader when a SRTM tile is available.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

CELL_SIZE_M = 500.0          # metres per grid cell
GRID_ROWS = GRID_COLS = 20   # canonical grid size


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


def build_priority_surface(
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    seed: int = 42,
    weights: dict | None = None,
    cell_size: float = CELL_SIZE_M,
) -> dict:
    """
    Returns a dict with keys:
        grid_shape, dem_m (elevation in metres), dem (normalised),
        slope_deg, slope, aspect,
        arson, fuel, dryness, spread,
        hazard, value, priority
    """
    rng = np.random.default_rng(seed)
    w = weights or dict(arson=0.30, fuel=0.25, dryness=0.25, terrain=0.20)

    # ── DEM ────────────────────────────────────────────────────────────────────
    dem_m = build_smooth_dem(grid_rows, grid_cols, seed)
    dem = _normalise(dem_m)  # [0, 1] for priority layers

    # Physical slope (degrees) using 500 m cell spacing
    dy_m, dx_m = np.gradient(dem_m, cell_size, cell_size)   # rise / run (m/m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx_m**2 + dy_m**2)))
    slope = _normalise(slope_deg)
    aspect = _normalise(np.arctan2(dy_m, dx_m) + np.pi)

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
    dryness = _normalise((1 - dem) * 0.7 + 0.3 * rng.uniform(0, 1, (rows, cols)))

    # spread terrain: steep south-facing slopes in prevailing wind
    spread = _normalise(slope * 0.6 + (1 - aspect) * 0.4)

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
    plt.suptitle("QPreFire — Priority Surface (synthetic Khao Yai)", fontsize=13)
    plt.tight_layout()
    plt.savefig("priority_surface.png", dpi=120)
    print("Saved priority_surface.png")
