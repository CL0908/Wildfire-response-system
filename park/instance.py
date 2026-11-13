"""
Whole-park sensor placement instance for Khao Yai National Park.

Builds the data a solver needs and nothing else: objective weights per cell,
candidate sites and a sparse visibility matrix. No solvers, no results.

Grid: data/park_500m.npz, 131 x 187 cells of 500 m (the park plus a 5 km margin).
Row 0 is the north edge, column 0 the west edge, flat index = row * cols + col.

Priority (weights)
    The same hand formula as the 20 x 20 pipeline, applied to the whole grid by
    calling priority_surface._real_layers on the park arrays:
        hazard   = normalise(0.40 fire + 0.25 fuel + 0.20 dryness + 0.15 spread)
        value    = inside_park * (0.5 + 0.5 * lc_tree) + (1 - inside_park) * 0.1
        priority = normalise(hazard * value)
    A cell's weight is its priority. The park weighting is the value term: cells
    outside the park keep a value of 0.1 instead of 0.5 to 1.0, so they still
    carry weight. Every term below is normalised (min-max) or filtered over the
    grid it is given, so a cell's value on the park grid differs from its value on
    the old 40 x 40 block: elevation, slope, aspect (via gradients at the grid edge),
    the Gaussian smoothing of fire counts (edge handling), and the min-max
    normalisation of fire, fuel, dryness, spread, hazard and priority. Weights from
    this module are therefore not comparable cell by cell with results.json.

Candidates
    Every cell with inside_park > 0.5, mast at the cell, 30 m mast and 4 km (8 cell)
    range as in viewshed.py.

Visibility
    viewshed.compute_viewsheds, unchanged, on the float64 elevation the 20 x 20
    pipeline uses. Sensors see cells outside the park and in the margin. A site sees
    its own cell.

Limitations
    Access is ignored (the road data stops at longitude 101.66), as are land tenure
    and power. Every in-park cell counts as buildable. The instance poses the
    geometric problem; it is not a siting plan.

Cache
    Visibility is cached outside the repository (default ../data_cache/park/ next to
    the repo, override with the Q_PITHAK_CACHE environment variable), keyed by a hash of the
    elevation, candidate list and viewshed parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

import priority_surface as ps
import viewshed as vs

REPO = Path(__file__).resolve().parent.parent
PARK_NPZ = REPO / "data" / "park_500m.npz"
PARK_META = REPO / "data" / "park_500m_meta.json"
OLD_META = REPO / "data" / "khao_yai_meta.json"
CACHE_VERSION = 1
INSIDE_THRESHOLD = 0.5


def cache_dir() -> Path:
    env = os.environ.get("Q_PITHAK_CACHE")
    return Path(env) if env else REPO.parent / "data_cache" / "park"


@dataclass
class ParkInstance:
    shape: tuple[int, int]            # (rows, cols)
    cell_m: float
    inside_park: np.ndarray           # (rows, cols) share of the cell inside the park, 0 to 1
    elevation: np.ndarray             # (rows, cols) metres, float64
    priority: np.ndarray              # (rows, cols) in [0, 1]
    weights: np.ndarray               # (rows * cols,) objective weight per cell = priority.ravel()
    site_cells: np.ndarray            # (n_sites,) flat cell index of each candidate, ascending
    sees: sparse.csr_matrix           # (n_sites, rows * cols) bool; sees[i, c] iff site i sees cell c
    fire_count: np.ndarray            # (rows * cols,) VIIRS detections 2012 to 2025
    lat: np.ndarray                   # (rows, cols) cell centres
    lon: np.ndarray
    params: dict = field(default_factory=dict)   # mast_height_m, max_range_m, max_range_cells, ...

    @property
    def n_sites(self) -> int:
        return len(self.site_cells)

    @property
    def n_cells(self) -> int:
        return self.shape[0] * self.shape[1]


def load_park_data() -> dict:
    with np.load(PARK_NPZ) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(PARK_META.read_text())
    if float(meta["cell_size_m"]) != vs.CELL_SIZE_M:
        raise ValueError("park grid cell size differs from viewshed.CELL_SIZE_M")
    data["_meta"] = meta
    return data


def park_layers(data: dict, weights: dict | None = None) -> dict:
    """All priority layers for the park grid (same formula as the 20 x 20 pipeline)."""
    arrays = {k: v for k, v in data.items() if not k.startswith("_")}
    return ps._real_layers(arrays, ps._resolve_real_weights(weights), vs.CELL_SIZE_M)


def candidate_cells(inside_park: np.ndarray) -> np.ndarray:
    """Flat indices of every cell with inside_park > 0.5, ascending."""
    return np.flatnonzero(np.asarray(inside_park).ravel() > INSIDE_THRESHOLD).astype(np.int64)


def build_visibility(elevation: np.ndarray, site_cells, mast_height_m: float = vs.MAST_HEIGHT_M,
                     cell_m: float = vs.CELL_SIZE_M,
                     max_range_cells: int = vs.MAX_RANGE_CELLS) -> sparse.csr_matrix:
    """Boolean CSR (n_sites, n_cells) from viewshed.compute_viewsheds, one row per site."""
    rows, cols = elevation.shape
    site_cells = np.asarray(site_cells, dtype=np.int64)
    sites = [(int(c // cols), int(c % cols)) for c in site_cells]
    views = vs.compute_viewsheds(elevation, sites, mast_height=mast_height_m,
                                 cell_size=cell_m, max_range_cells=max_range_cells)
    indptr = np.zeros(len(sites) + 1, dtype=np.int64)
    indices = []
    for i in range(len(sites)):
        idx = np.fromiter(sorted(views[i]), dtype=np.int64, count=len(views[i]))
        indices.append(idx)
        indptr[i + 1] = indptr[i] + len(idx)
    flat = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(len(flat), dtype=bool), flat, indptr),
                             shape=(len(sites), rows * cols))


def _cache_key(elevation, site_cells, params) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(elevation, dtype=np.float64).tobytes())
    h.update(np.asarray(site_cells, dtype=np.int64).tobytes())
    with open(vs.__file__, "rb") as f:     # a change to the line-of-sight rule must not reuse a stale cache
        h.update(f.read())
    h.update(json.dumps(dict(params, v=CACHE_VERSION, shape=list(elevation.shape)),
                        sort_keys=True).encode())
    return h.hexdigest()[:20]


def build_park_instance(data: dict | None = None, use_cache: bool = True,
                        site_cells=None) -> ParkInstance:
    """
    Build the instance. `site_cells` (flat indices) overrides the default candidate set
    (every cell with inside_park > 0.5); used by tests to keep visibility small.
    """
    data = load_park_data() if data is None else data
    layers = park_layers(data)
    elevation, inside, priority = layers["dem_m"], layers["inside_park"], layers["priority"]
    shape = elevation.shape
    sites = candidate_cells(inside) if site_cells is None else np.asarray(site_cells, np.int64)

    params = dict(mast_height_m=vs.MAST_HEIGHT_M, max_range_m=vs.MAX_RANGE_M,
                  max_range_cells=vs.MAX_RANGE_CELLS, cell_m=vs.CELL_SIZE_M)
    path = cache_dir() / f"sees_{_cache_key(elevation, sites, params)}.npz"
    sees = None
    if use_cache and path.exists():
        try:
            sees = sparse.load_npz(path).tocsr()
        except Exception:
            sees = None
    if sees is None:
        sees = build_visibility(elevation, sites, params["mast_height_m"], params["cell_m"],
                                params["max_range_cells"])
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npz")
            sparse.save_npz(tmp, sees, compressed=True)
            os.replace(tmp, path)

    return ParkInstance(
        shape=shape, cell_m=vs.CELL_SIZE_M, inside_park=inside, elevation=elevation,
        priority=priority, weights=priority.ravel().copy(), site_cells=sites, sees=sees,
        fire_count=np.asarray(data["fire_count"]).ravel(), lat=layers["lat"], lon=layers["lon"],
        params=params,
    )


def load_park_instance(use_cache: bool = True) -> ParkInstance:
    """The whole-park instance: all inside_park > 0.5 cells as candidates. Cached visibility."""
    return build_park_instance(use_cache=use_cache)


def window_cells_in_park(shape: tuple[int, int] = (20, 20)) -> tuple[int, int]:
    """(row, col) of the old 20 x 20 window's corner in the park grid, from the two meta files."""
    block = json.loads(PARK_META.read_text())["khao_yai_block_offset_row_col"]
    win = json.loads(OLD_META.read_text())["default_window"]
    return int(block[0]) + int(win["row0"]), int(block[1]) + int(win["col0"])
