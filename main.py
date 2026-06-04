"""
QPreFire — Static Sensor Placement Optimizer for Khao Yai National Park.
Run this file to execute the full pipeline: Steps A→E.

Usage:
    python main.py [--quick]      # --quick skips QAOA and uses grid=15
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

from priority_surface import build_priority_surface
from viewshed import generate_candidate_sites, compute_viewsheds, coverage_overlap_stats
from solvers.greedy import greedy_coverage
from solvers.ilp import ilp_coverage
from solvers.qubo_sa import sa_solve
from viz.map_viz import plot_placement_static, plot_folium_map

OUT = os.path.dirname(os.path.abspath(__file__))


def main(quick=False):
    GRID = 15 if quick else 20
    N_SITES = 12 if quick else 16
    N_SENSORS = 3 if quick else 4
    SEED = 42

    print("=" * 60)
    print("QPreFire — Khao Yai Static Sensor Placement")
    print("=" * 60)

    # ── Step A ──────────────────────────────────────────────────
    print("\n[A] Building priority surface...")
    surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=SEED)
    print(f"    Grid {GRID}×{GRID} | priority range "
          f"[{surf['priority'].min():.3f}, {surf['priority'].max():.3f}]")

    # ── Step B ──────────────────────────────────────────────────
    print(f"\n[B] Generating {N_SITES} candidate sites and computing viewsheds...")
    sites = generate_candidate_sites(
        (GRID, GRID), n_candidates=N_SITES,
        priority=surf["priority"], dem_norm=surf["dem"], seed=SEED
    )
    viewsheds = compute_viewsheds(surf["dem_m"], sites, max_range_cells=GRID // 2)
    stats = coverage_overlap_stats(viewsheds, GRID * GRID)
    print(f"    {stats}")

    weights = {r * GRID + c: float(surf["priority"][r, c])
               for r in range(GRID) for c in range(GRID)}

    # ── Step C ──────────────────────────────────────────────────
    print(f"\n[C] Solving placement (N={N_SENSORS} sensors)...")

    results = {}

    # 1. Greedy
    chosen_g, obj_g = greedy_coverage(viewsheds, weights, N_SENSORS)
    results["Greedy"] = (chosen_g, obj_g)
    print(f"    Greedy:  {obj_g:.4f}  sites={chosen_g}")

    # 2. ILP
    try:
        chosen_ilp, obj_ilp, status = ilp_coverage(viewsheds, weights, N_SENSORS)
        results["ILP"] = (chosen_ilp, obj_ilp)
        print(f"    ILP:     {obj_ilp:.4f}  status={status}  sites={chosen_ilp}")
        approx = obj_g / obj_ilp if obj_ilp > 1e-9 else 0
        print(f"    Greedy approximation ratio: {approx:.3f} (theory ≥ {1-1/2.718:.3f})")
    except Exception as e:
        print(f"    ILP skipped: {e}")

    # 3. SA
    try:
        chosen_sa, obj_sa, _ = sa_solve(viewsheds, weights, N_SENSORS, num_reads=50)
        results["SA"] = (chosen_sa, obj_sa)
        print(f"    SA:      {obj_sa:.4f}  sites={chosen_sa}")
    except Exception as e:
        print(f"    SA skipped: {e}")

    # 4. QAOA family (skip if --quick)
    if not quick:
        try:
            from solvers.qaoa import run_qaoa
            for mode, label in [("plain", "QAOA"), ("ws", "WS-QAOA"), ("wsxy", "WSXY-QAOA")]:
                r = run_qaoa(viewsheds, weights, N_SENSORS, p_layers=2, mode=mode)
                results[label] = (r["chosen"], r["objective"])
                print(f"    {label}: {r['objective']:.4f}  depth={r['circuit_depth']}  "
                      f"iters={r['iterations']}")
        except Exception as e:
            print(f"    QAOA skipped: {e}")

    # ── Steps D + E ─────────────────────────────────────────────
    print("\n[D/E] Generating benchmark charts and map visualizations...")

    # Priority surface plot
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, data, cmap) in zip(axes, [
        ("Priority (hazard×value)", surf["priority"], "YlOrRd"),
        ("Hazard", surf["hazard"], "Reds"),
        ("Value (ecological)", surf["value"], "Greens"),
    ]):
        im = ax.imshow(data, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i, (r, c) in enumerate(sites):
            color = "#1a6dd4" if i in results["Greedy"][0] else "#888"
            ax.scatter(c, r, c=color, s=80 if i in results["Greedy"][0] else 20,
                       marker="^" if i in results["Greedy"][0] else "o", zorder=5)
    plt.suptitle("QPreFire — Priority Surface (Khao Yai, synthetic)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/priority_surface.png", dpi=120)
    print(f"    Saved priority_surface.png")
    plt.close()

    # Placement + viewshed map
    plot_placement_static(surf, sites, results["Greedy"][0], viewsheds,
                          out_path=f"{OUT}/placement_map.png")
    plot_folium_map(surf, sites, results["Greedy"][0], viewsheds,
                    out_path=f"{OUT}/placement_map.html")

    print("\n[Done] Output files:")
    for f in ["priority_surface.png", "placement_map.png", "placement_map.html"]:
        if os.path.exists(f"{OUT}/{f}"):
            print(f"    {OUT}/{f}")

    print("\nRun `python benchmark.py` for the full multi-instance benchmark charts.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Skip QAOA, use smaller grid")
    args = ap.parse_args()
    main(quick=args.quick)
