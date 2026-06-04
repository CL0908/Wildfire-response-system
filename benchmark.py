"""
Benchmark: ILP · Greedy · SA · QAOA · WS-QAOA on three Khao Yai instances.

One shared scoring function scores every method identically:
  enforce exactly N sensors → recompute covered-priority → ratio vs ILP optimal.

Outputs:
  benchmark.png   – grouped approximation-ratio bar chart
  stdout          – per-run sanity flags (QAOA only) + violations table
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from priority_surface import build_priority_surface
from viewshed import (generate_candidate_sites, compute_viewsheds,
                      coverage_overlap_stats, MAX_RANGE_CELLS)
from solvers.greedy import greedy_coverage
from solvers.ilp import ilp_coverage
from solvers.qubo_sa import sa_solve

OUT = os.path.dirname(os.path.abspath(__file__))

# Methods shown in bar chart (ILP is the reference line, not a bar)
BAR_METHODS = ["Greedy (CELF)", "Simulated Annealing", "QAOA", "WS-QAOA"]
BAR_COLORS  = ["#40916c",       "#e07a5f",              "#d62828", "#264653"]


# ---------------------------------------------------------------------------
# Shared scoring function (used by every method)
# ---------------------------------------------------------------------------

def score_placement(
    chosen_raw: list[int],
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    ilp_obj: float,
) -> tuple[list[int], float, float, bool]:
    """
    Enforce feasibility, recompute objective from scratch, compute ratio.

    - len > n_sensors : keep N sites with highest individual viewshed weight
    - len < n_sensors : infeasible → objective = 0, ratio = 0
    - len == n_sensors: use as-is

    Returns: (feasible_chosen, objective, ratio, was_violation)
    """
    was_violation = len(chosen_raw) != n_sensors

    if len(chosen_raw) < n_sensors:
        return [], 0.0, 0.0, True

    chosen = list(chosen_raw)
    if len(chosen) > n_sensors:
        site_w = {s: sum(weights.get(c, 0.0) for c in viewsheds[s]) for s in chosen}
        chosen = sorted(chosen, key=lambda s: site_w[s], reverse=True)[:n_sensors]

    covered: set[int] = set()
    for s in chosen:
        covered |= viewsheds[s]
    obj = sum(weights.get(c, 0.0) for c in covered)
    ratio = obj / ilp_obj if ilp_obj > 1e-9 else 0.0

    return chosen, obj, ratio, was_violation


# ---------------------------------------------------------------------------
# Per-instance runner
# ---------------------------------------------------------------------------

def run_instance(n_sites: int, n_sensors: int, seed: int) -> dict:
    """
    Fixed physical setup: 20×20 grid, 500 m cells, h=30 m mast, R=4 km range.

    Returns a dict keyed by method name, each value:
      {ratio, objective, chosen, violation, shot_violations (QAOA only),
       params_moved, best_ne_seed}
    Plus key 'ilp_obj' for the reference optimum.
    """
    GRID = 20
    surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=seed)

    # Bias candidates toward ridges (high elevation) in high-priority zones
    sites = generate_candidate_sites(
        (GRID, GRID), n_candidates=n_sites,
        priority=surf["priority"], dem_norm=surf["dem"], seed=seed,
    )
    # Physical range-limited viewshed with progressive-angle LOS
    viewsheds = compute_viewsheds(surf["dem_m"], sites,
                                  max_range_cells=MAX_RANGE_CELLS)
    weights = {r * GRID + c: float(surf["priority"][r, c])
               for r in range(GRID) for c in range(GRID)}

    stats = coverage_overlap_stats(viewsheds, GRID * GRID)
    print(f"  Viewshed overlap: {stats}")

    out: dict = {}

    # ── ILP (exact optimum — reference only, not a bar) ───────────────────────
    try:
        _, ilp_obj, status = ilp_coverage(viewsheds, weights, n_sensors)
        print(f"  ILP: {ilp_obj:.4f} ({status})")
    except Exception as e:
        print(f"  ILP failed: {e}")
        ilp_obj = 0.0
    out["ilp_obj"] = ilp_obj

    # ── Greedy ────────────────────────────────────────────────────────────────
    chosen_g, _ = greedy_coverage(viewsheds, weights, n_sensors)
    _, obj_g, ratio_g, viol_g = score_placement(chosen_g, viewsheds, weights, n_sensors, ilp_obj)
    out["Greedy (CELF)"] = dict(ratio=ratio_g, objective=obj_g, chosen=chosen_g,
                                 violation=viol_g, shot_violations=0)
    print(f"  Greedy: {obj_g:.4f}  ratio={ratio_g:.3f}")

    # ── Simulated Annealing ───────────────────────────────────────────────────
    try:
        chosen_sa, _, sa_raw_viol = sa_solve(viewsheds, weights, n_sensors, num_reads=100)
        _, obj_sa, ratio_sa, viol_sa = score_placement(
            chosen_sa, viewsheds, weights, n_sensors, ilp_obj
        )
        out["Simulated Annealing"] = dict(ratio=ratio_sa, objective=obj_sa, chosen=chosen_sa,
                                           violation=viol_sa or bool(sa_raw_viol),
                                           shot_violations=sa_raw_viol)
        print(f"  SA:     {obj_sa:.4f}  ratio={ratio_sa:.3f}  raw_violations={sa_raw_viol}")
    except Exception as e:
        print(f"  SA failed: {e}")
        out["Simulated Annealing"] = dict(ratio=0.0, objective=0.0, chosen=[],
                                           violation=True, shot_violations=1)

    # ── QAOA family ───────────────────────────────────────────────────────────
    try:
        from solvers.qaoa import run_qaoa
        for mode, label in [("plain", "QAOA"), ("ws", "WS-QAOA")]:
            t0 = time.time()
            r = run_qaoa(viewsheds, weights, n_sensors,
                         p_layers=4, mode=mode, max_iter=150, shots=2000)
            elapsed = time.time() - t0

            _, obj_q, ratio_q, viol_q = score_placement(
                r["chosen"], viewsheds, weights, n_sensors, ilp_obj
            )
            out[label] = dict(
                ratio=ratio_q, objective=obj_q, chosen=r["chosen"],
                violation=viol_q,
                shot_violations=r["constraint_violations"],
                params_moved=r["params_moved"],
                best_ne_seed=r["best_ne_seed"],
                iterations=r["iterations"],
                circuit_depth=r["circuit_depth"],
            )
            # Sanity flags
            pm  = "YES" if r["params_moved"]  else "NO ← params did not move"
            bns = "YES" if r["best_ne_seed"]  else "NO ← echoing initial state"
            print(f"  {label}: obj={obj_q:.4f}  ratio={ratio_q:.3f}  "
                  f"shots_violated={r['constraint_violations']}/{2000}  "
                  f"iters={r['iterations']}  depth={r['circuit_depth']}  "
                  f"time={elapsed:.1f}s")
            print(f"    [sanity] params moved from x0?         {pm}")
            print(f"    [sanity] best shot ≠ initial-state seed? {bns}")
    except Exception as e:
        print(f"  QAOA failed: {e}")
        for label in ("QAOA", "WS-QAOA"):
            out[label] = dict(ratio=0.0, objective=0.0, chosen=[],
                              violation=True, shot_violations=2000,
                              params_moved=False, best_ne_seed=False,
                              iterations=0, circuit_depth=0)

    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_benchmark(all_results: list[dict], instance_labels: list[str]):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_inst = len(instance_labels)
    n_meth = len(BAR_METHODS)
    width = 0.18
    group_gap = 0.1
    x = np.arange(n_inst) * (n_meth * width + group_gap)

    for mi, (method, color) in enumerate(zip(BAR_METHODS, BAR_COLORS)):
        ratios = [r.get(method, {}).get("ratio", 0.0) for r in all_results]
        bars = ax.bar(x + mi * width, ratios, width, label=method,
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, ratios):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#333")

    # Dashed optimal line
    ax.axhline(1.0, color="#222", linestyle="--", linewidth=1.2, label="Optimal (ILP)")

    ax.set_xticks(x + width * (n_meth - 1) / 2)
    ax.set_xticklabels(instance_labels, fontsize=10)
    ax.set_ylabel("Approximation ratio (vs exact optimum)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Khao Yai sensor placement — solution quality", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT, "benchmark.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"\nSaved {path}")


# ---------------------------------------------------------------------------
# Violations + ratio table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list[dict]):
    print("\n" + "=" * 62)
    print(f"{'Method':<22} {'Constraint violations':>22} {'Mean ratio':>10}")
    print("-" * 62)
    all_methods = ["Greedy (CELF)", "Simulated Annealing", "QAOA", "WS-QAOA"]
    for method in all_methods:
        total_viol = sum(r.get(method, {}).get("shot_violations", 0) for r in all_results)
        ratios = [r.get(method, {}).get("ratio", 0.0) for r in all_results]
        mean_r = float(np.mean(ratios))
        is_qaoa = method in ("QAOA", "WS-QAOA")
        denom = len(all_results) * 2000 if is_qaoa else len(all_results)
        viol_str = f"{total_viol}/{denom} shots" if is_qaoa else f"{total_viol}/{len(all_results)} runs"
        print(f"  {method:<20} {viol_str:>22} {mean_r:>10.3f}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # All instances use 20×20 / 500 m / 4 km range (physical model)
    instances = [
        dict(n_sites=8,  n_sensors=2, seed=42),
        dict(n_sites=10, n_sensors=3, seed=7),
        dict(n_sites=12, n_sensors=3, seed=13),
    ]
    labels = [f"|S|={d['n_sites']}, N={d['n_sensors']}" for d in instances]

    all_results = []
    for i, cfg in enumerate(instances):
        print(f"\n{'='*60}")
        print(f"Instance {i+1}: {labels[i]}  (20×20, 500 m cells, R=4 km)")
        print("=" * 60)
        res = run_instance(**cfg)
        all_results.append(res)

    plot_benchmark(all_results, labels)
    print_summary_table(all_results)

    # terrain_slope.png — generate once from the first instance's surface
    try:
        from viz.map_viz import plot_terrain_slope, plot_placement_static
        from viewshed import MAX_RANGE_CELLS
        surf = build_priority_surface(grid_rows=20, grid_cols=20, seed=42)
        sites = generate_candidate_sites(
            (20, 20), n_candidates=12,
            priority=surf["priority"], dem_norm=surf["dem"], seed=42,
        )
        vs = compute_viewsheds(surf["dem_m"], sites,
                               max_range_cells=MAX_RANGE_CELLS)
        weights = {r * 20 + c: float(surf["priority"][r, c])
                   for r in range(20) for c in range(20)}
        from solvers.greedy import greedy_coverage as _gc
        chosen, _ = _gc(vs, weights, n_sensors=4)

        plot_terrain_slope(surf, out_path=os.path.join(OUT, "terrain_slope.png"))
        plot_placement_static(surf, sites, chosen, vs,
                              out_path=os.path.join(OUT, "placement_map.png"))
    except Exception as e:
        print(f"  Visualization skipped: {e}")


if __name__ == "__main__":
    main()
