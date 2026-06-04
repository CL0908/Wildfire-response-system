"""
Step B: Range-limited viewshed with physically correct line-of-sight.

Physical parameters:
  cell_size = 500 m, mast height h = 30 m, max detection range R = 4 km = 8 cells.

Algorithm (progressive-angle / reference-plane):
  Eye at sensor s: eye_height = dem_m[s] + h
  Cell t is visible iff BOTH:
    (a) ground_distance(s, t) ≤ R
    (b) elevation_angle_to_t ≥ max elevation_angle over all intermediate cells
        where elevation_angle(P) = (dem_m[P] - eye_height) / ground_dist(s, P)

This produces circle-ish footprints with terrain "bites" taken out behind ridges.
"""

import numpy as np

# Physical constants (can be overridden per call)
CELL_SIZE_M    = 500.0    # metres per grid cell
MAST_HEIGHT_M  = 30.0     # camera mast above ground (m)
MAX_RANGE_M    = 4_000.0  # smoke-detection range (m)
MAX_RANGE_CELLS = int(MAX_RANGE_M / CELL_SIZE_M)   # = 8


# ---------------------------------------------------------------------------
# Core LOS check
# ---------------------------------------------------------------------------

def _los_clear(
    dem_m: np.ndarray,
    sr: int, sc: int,
    tr: int, tc: int,
    eye_height: float,
    cell_size: float,
) -> bool:
    """
    Progressive-angle line-of-sight test.

    Walk the straight ray from (sr,sc) toward (tr,tc) in integer steps.
    At each intermediate grid cell i, compute the elevation angle from the eye:
        elev_angle_i = (dem_m[i] - eye_height) / ground_dist(sensor, i)
    Target t is visible iff
        (dem_m[t] - eye_height) / ground_dist(sensor, t)  >=  max(elev_angle_i)
    """
    dr = tr - sr
    dc = tc - sc
    steps = max(abs(dr), abs(dc))

    if steps == 0:
        return True   # same cell

    # Elevation angle to target (may be negative = looking down)
    dist_t = np.sqrt(dr**2 + dc**2) * cell_size
    target_angle = (dem_m[tr, tc] - eye_height) / dist_t

    max_block = -np.inf

    rows, cols = dem_m.shape
    for k in range(1, steps):           # intermediate cells, excluding target
        frac = k / steps
        ri = int(round(sr + frac * dr))
        ci = int(round(sc + frac * dc))
        ri = max(0, min(rows - 1, ri))
        ci = max(0, min(cols - 1, ci))

        dist_i = np.sqrt((ri - sr)**2 + (ci - sc)**2) * cell_size
        if dist_i < 1.0:
            continue   # guard against numerical zero at near-sensor cells

        angle_i = (dem_m[ri, ci] - eye_height) / dist_i
        if angle_i > max_block:
            max_block = angle_i

    return target_angle >= max_block


# ---------------------------------------------------------------------------
# Public viewshed computation
# ---------------------------------------------------------------------------

def compute_viewsheds(
    dem_m: np.ndarray,
    candidate_sites: list[tuple[int, int]],
    mast_height: float = MAST_HEIGHT_M,
    cell_size: float = CELL_SIZE_M,
    max_range_cells: int = MAX_RANGE_CELLS,
) -> dict[int, set[int]]:
    """
    Returns viewsheds[site_idx] = set of flat cell indices {row*cols + col}
    visible from that site within max_range_cells (range-limited viewshed).
    """
    rows, cols = dem_m.shape
    viewsheds: dict[int, set[int]] = {}

    for idx, (sr, sc) in enumerate(candidate_sites):
        eye = float(dem_m[sr, sc]) + mast_height
        visible: set[int] = set()

        # Only test cells within the circular range
        r_lo = max(0, sr - max_range_cells)
        r_hi = min(rows - 1, sr + max_range_cells)
        c_lo = max(0, sc - max_range_cells)
        c_hi = min(cols - 1, sc + max_range_cells)

        for tr in range(r_lo, r_hi + 1):
            for tc in range(c_lo, c_hi + 1):
                dist_cells = np.sqrt((tr - sr)**2 + (tc - sc)**2)
                if dist_cells > max_range_cells:
                    continue   # outside circular range
                if _los_clear(dem_m, sr, sc, tr, tc, eye, cell_size):
                    visible.add(tr * cols + tc)

        viewsheds[idx] = visible

    return viewsheds


# ---------------------------------------------------------------------------
# Candidate site generation
# ---------------------------------------------------------------------------

def generate_candidate_sites(
    grid_shape: tuple[int, int],
    n_candidates: int = 16,
    priority: np.ndarray | None = None,
    dem_norm: np.ndarray | None = None,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """
    Sample candidate tower/camera sites.

    Biases:
      - 60 % weight on normalised elevation (ridges = better viewpoints)
      - 40 % weight on priority (high-risk areas matter)
    Falls back gracefully if either layer is absent.
    """
    rng = np.random.default_rng(seed)
    rows, cols = grid_shape
    n_cells = rows * cols

    if priority is not None and dem_norm is not None:
        score = 0.6 * dem_norm.ravel() + 0.4 * priority.ravel()
    elif priority is not None:
        score = priority.ravel()
    elif dem_norm is not None:
        score = dem_norm.ravel()
    else:
        score = np.ones(n_cells)

    score = np.clip(score, 0, None)
    score = score / score.sum()

    chosen_flat = rng.choice(n_cells, size=n_candidates, replace=False, p=score)
    return [(int(i // cols), int(i % cols)) for i in chosen_flat]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def coverage_overlap_stats(viewsheds: dict[int, set[int]], n_cells: int) -> dict:
    """How many candidate sites can see each cell?"""
    cover_count = np.zeros(n_cells, dtype=int)
    for vs in viewsheds.values():
        for c in vs:
            cover_count[c] += 1
    return {
        "max_overlap":      int(cover_count.max()),
        "mean_overlap":     float(cover_count.mean()),
        "pct_seen_by_le2":  float((cover_count <= 2).mean()),
        "pct_unseen":       float((cover_count == 0).mean()),
        "median_vs_size":   float(np.median([len(v) for v in viewsheds.values()])),
    }


if __name__ == "__main__":
    from priority_surface import build_priority_surface
    import time

    surf = build_priority_surface()
    dem_m = surf["dem_m"]
    dem_n = surf["dem"]

    sites = generate_candidate_sites((20, 20), n_candidates=12,
                                     priority=surf["priority"], dem_norm=dem_n)
    print(f"Sites: {sites[:4]} ...")

    t0 = time.time()
    vs = compute_viewsheds(dem_m, sites)
    print(f"Viewsheds done in {time.time()-t0:.2f}s")
    print("Overlap stats:", coverage_overlap_stats(vs, 20 * 20))
    print("Viewshed sizes:", [len(v) for v in vs.values()])
