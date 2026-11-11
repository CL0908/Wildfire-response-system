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
from benchmarks.map_viz import plot_placement_static, plot_folium_map

OUT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DATA = os.path.join(OUT, "dashboards", "public", "results.json")


def write_results_json(surf, sites, viewsheds, weights, results, n_sensors, grid,
                       qaoa_stats=None, path=DASHBOARD_DATA):
    """
    Everything the dashboard draws, from this run. `results` is {method: (chosen, objective)}.
    For QAOA methods the placement is the best sampled shot, so `qaoa_stats` carries the
    per-shot probabilities that say how often the circuit actually produces it.
    """
    import csv, json
    from datetime import date
    from viewshed import CELL_SIZE_M, MAST_HEIGHT_M, MAX_RANGE_M

    def grid_list(a, digits):
        return [[round(float(v), digits) for v in row] for row in a]

    optimum = results["ILP"][1] if "ILP" in results else None
    methods = []
    for name, (chosen, objective) in results.items():
        covered = sorted(set().union(*(viewsheds[s] for s in chosen))) if chosen else []
        methods.append(dict(
            name=name,
            kind="quantum (simulated)" if "QAOA" in name else "classical",
            chosen=[int(s) for s in chosen],
            objective=round(float(objective), 4),
            ratio=round(float(objective) / optimum, 4) if optimum else None,
            covered_cells=[int(c) for c in covered],
            sampling=(qaoa_stats or {}).get(name),
        ))

    benchmark = []
    csv_path = os.path.join(OUT, "approx_ratio.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            benchmark = list(csv.DictReader(f))

    data = dict(
        generated=date.today().isoformat(),
        terrain="synthetic",
        grid=dict(rows=grid, cols=grid, cell_size_m=CELL_SIZE_M),
        mast_height_m=MAST_HEIGHT_M,
        range_m=MAX_RANGE_M,
        n_sensors=n_sensors,
        total_priority=round(sum(weights.values()), 4),
        optimum=round(float(optimum), 4) if optimum else None,
        layers=dict(
            priority=grid_list(surf["priority"], 4),
            hazard=grid_list(surf["hazard"], 4),
            value=grid_list(surf["value"], 4),
            elevation_m=grid_list(surf["dem_m"], 1),
        ),
        sites=[dict(id=i, row=int(r), col=int(c)) for i, (r, c) in enumerate(sites)],
        viewsheds={str(i): sorted(int(c) for c in vs) for i, vs in viewsheds.items()},
        methods=methods,
        benchmark=benchmark,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return path


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
    viewsheds = compute_viewsheds(surf["dem_m"], sites)  # default range: 4 km
    stats = coverage_overlap_stats(viewsheds, GRID * GRID)
    print(f"    {stats}")

    weights = {r * GRID + c: float(surf["priority"][r, c])
               for r in range(GRID) for c in range(GRID)}

    # ── Step C ──────────────────────────────────────────────────
    print(f"\n[C] Solving placement (N={N_SENSORS} sensors)...")

    results = {}
    qaoa_stats = {}

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
            for mode, label in [("plain", "QAOA"), ("xy", "XY-QAOA"), ("wsxy", "WSXY-QAOA")]:
                r = run_qaoa(viewsheds, weights, N_SENSORS, p_layers=2, mode=mode)
                if r["chosen"]:
                    results[label] = (r["chosen"], r["objective"])
                    qaoa_stats[label] = dict(
                        shots=r["shots"],
                        p_optimal=round(r["p_optimal"], 4),
                        p_optimal_random=round(r["p_optimal_random"], 4),
                        p_feasible=round(r["p_feasible"], 4),
                        mean_ratio=round(r["mean_ratio"], 4),
                    )
                print(f"    {label}: best shot {r['objective']:.4f}  "
                      f"P(optimal)={r['p_optimal']:.4f} "
                      f"(random placement {r['p_optimal_random']:.4f})  "
                      f"P(feasible)={r['p_feasible']:.3f}")
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

    path = write_results_json(surf, sites, viewsheds, weights, results, N_SENSORS, GRID, qaoa_stats)
    print(f"    Saved {os.path.relpath(path, OUT)} (dashboard data)")

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
