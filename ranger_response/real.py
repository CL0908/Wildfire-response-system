"""Real Khao Yai window for the ranger planner: the same 20 x 20, 500 m window main.py uses.

The surface is built exactly as main.py does (`build_priority_surface(20, 20, 42, source="real")`,
window (15, 3) of the committed data/khao_yai_500m.npz), so grid, DEM, hazard and priority match
the dashboard cell for cell. The land-cover shares and the 2012-2025 FIRMS counts come from the
same extract.

Fuel rules
  "landcover": a cell is fuel if tree + shrub + grass + crop cover at least 50% of it (the same
      burnable test detection/fire_sim.py uses, minus its 'bare' class, which cannot carry fire).
      Water, built-up, bare and wetland dominated cells are non-fuel.
  "hazard":   hazard >= threshold (the synthetic-terrain rule).
On this window every cell is at least 65% tree/shrub/grass/crop, so "landcover" marks all 400 cells
as fuel: the land is one unbroken forest/crop block. See README.md for what each rule does to the
results.
"""
from __future__ import annotations

import numpy as np

from priority_surface import _load_real_region, build_priority_surface

GRID = 20
SEED = 42
FUEL_CLASSES = ("lc_tree", "lc_shrub", "lc_grass", "lc_crop")
MIN_FUEL_SHARE = 0.5
DEFAULT_REAL_THRESHOLD = 0.55


def build_real_surface(grid: int = GRID, seed: int = SEED) -> dict:
    """Priority surface exactly as main.py builds it, plus lc_* shares and fire_count (already in surf)."""
    surf = build_priority_surface(grid, grid, seed, source="real")
    row0, col0 = surf["window"]
    data, _ = _load_real_region()
    sl = (slice(row0, row0 + grid), slice(col0, col0 + grid))
    for k in data:
        if k.startswith("lc_"):
            surf[k] = np.asarray(data[k], dtype=float)[sl].copy()
    surf["fuel_share"] = sum(surf[k] for k in FUEL_CLASSES)
    return surf


def real_fuel_mask(surf: dict, rule: str = "landcover", threshold: float = DEFAULT_REAL_THRESHOLD):
    if rule == "landcover":
        return surf["fuel_share"] >= MIN_FUEL_SHARE
    if rule == "hazard":
        return surf["hazard"] >= threshold
    raise ValueError(f"fuel rule must be 'landcover' or 'hazard', got {rule!r}")


def ignitions_by_fire_history(surf: dict, fuel, n: int = 4, min_sep: int = 3):
    """Fuel cells ranked by real fire_count (NASA FIRMS 2012-2025), ties by hazard, then row, col.
    Later picks must be at least `min_sep` cells (Chebyshev) from earlier ones so alternatives differ."""
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(fuel))]
    cells.sort(key=lambda rc: (-int(surf["fire_count"][rc]), -float(surf["hazard"][rc]), rc))
    out = []
    for rc in cells:
        if all(max(abs(rc[0] - o[0]), abs(rc[1] - o[1])) >= min_sep for o in out):
            out.append(rc)
        if len(out) == n:
            break
    return out
