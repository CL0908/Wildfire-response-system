"""
Benchmark: ILP · Greedy · SA · QAOA · XY-QAOA · WSXY-QAOA on three Khao Yai instances.

Greedy and SA return one placement each, scored as covered priority / ILP optimum.
QAOA returns a distribution over placements, so it is scored per shot with no
classical repair (infeasible shots count as 0) and shown next to the baseline
of drawing N-sensor placements uniformly at random. WSXY-QAOA starts from the
LP relaxation, so its baseline is its own starting state ("WSXY start").

Outputs:
  benchmark.png     – grouped approximation-ratio bar chart
  approx_ratio.csv  – the numbers behind it
  stdout            – per-instance detail
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv
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
BAR_METHODS = ["Greedy (CELF)", "Simulated Annealing", "Random placement",
               "QAOA", "XY-QAOA", "WSXY start", "WSXY-QAOA"]
BAR_COLORS  = ["#40916c", "#e07a5f", "#9a9a9a", "#d62828", "#264653", "#c9b8e0", "#7b5ea7"]
QAOA_METHODS = ("QAOA", "XY-QAOA", "WSXY-QAOA")
QAOA_LAYERS, QAOA_SHOTS = 4, 2000


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
       params_moved}; QAOA entries also carry p_optimal, p_feasible,
      best_shot_ratio, and 'ratio' there is the mean over shots.
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

    # ── QAOA family (per-shot statistics, no repair) ──────────────────────────
    try:
        from solvers.qaoa import run_qaoa
        for mode, label in [("plain", "QAOA"), ("xy", "XY-QAOA"), ("wsxy", "WSXY-QAOA")]:
            t0 = time.time()
            r = run_qaoa(viewsheds, weights, n_sensors, p_layers=QAOA_LAYERS,
                         mode=mode, max_iter=150, shots=QAOA_SHOTS)
            elapsed = time.time() - t0

            # The enumerated optimum inside run_qaoa must agree with the ILP.
            if ilp_obj > 1e-9 and abs(r["optimum"] - ilp_obj) > 1e-6 * ilp_obj:
                print(f"  WARNING: enumerated optimum {r['optimum']:.4f} != ILP {ilp_obj:.4f}")

            out[label] = dict(
                ratio=r["mean_ratio"], objective=r["objective"], chosen=r["chosen"],
                best_shot_ratio=r["objective"] / r["optimum"] if r["optimum"] > 0 else 0.0,
                p_optimal=r["p_optimal"], p_feasible=r["p_feasible"],
                violation=not r["chosen"],
                shot_violations=r["constraint_violations"],
                params_moved=r["params_moved"],
                iterations=r["iterations"],
            )
            if mode == "wsxy":
                out["WSXY start"] = dict(
                    ratio=r["start_mean_ratio"], p_optimal=r["start_p_optimal"],
                    p_feasible=r["p_feasible"], shot_violations=0,
                )
            out["Random placement"] = dict(
                ratio=r["mean_ratio_random"], p_optimal=r["p_optimal_random"],
                p_feasible=1.0, shot_violations=0,
            )
            print(f"  {label}: mean ratio/shot={r['mean_ratio']:.3f}  "
                  f"P(optimal)={r['p_optimal']:.4f}  P(feasible)={r['p_feasible']:.3f}  "
                  f"best of {QAOA_SHOTS} shots={out[label]['best_shot_ratio']:.3f}  "
                  f"evals={r['iterations']}  time={elapsed:.1f}s"
                  + (f"  start: mean={r['start_mean_ratio']:.3f} P(optimal)={r['start_p_optimal']:.4f}"
                     if mode == "wsxy" else "")
                  + ("" if r["params_moved"] else "  ← params did not move"))
        rnd = out["Random placement"]
        print(f"  Random placement: mean ratio={rnd['ratio']:.3f}  "
              f"P(optimal)={rnd['p_optimal']:.4f}")
    except Exception as e:
        print(f"  QAOA failed: {e}")
        for label in QAOA_METHODS:
            out[label] = dict(ratio=0.0, objective=0.0, chosen=[], best_shot_ratio=0.0,
                              p_optimal=0.0, p_feasible=0.0,
                              violation=True, shot_violations=QAOA_SHOTS,
                              params_moved=False, iterations=0)

    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_benchmark(all_results: list[dict], instance_labels: list[str]):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_inst = len(instance_labels)
    n_meth = len(BAR_METHODS)
    width = 0.115
    group_gap = 0.12
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
    ax.set_ylabel("Covered priority / exact optimum", fontsize=10)
    ax.set_xlabel("Greedy, SA: the placement returned.  Random and the QAOA variants: "
                  "expected value of one shot, no repair.",
                  fontsize=8, color="#555")
    ax.set_ylim(0, 1.05)
    ax.set_title("Khao Yai sensor placement — solution quality", fontsize=12)
    ax.legend(fontsize=8, framealpha=0.9, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
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
    for method in BAR_METHODS:
        if method in ("Random placement", "WSXY start"):
            viol_str = "-"
        elif method in QAOA_METHODS:
            total = sum(r.get(method, {}).get("shot_violations", 0) for r in all_results)
            viol_str = f"{total}/{len(all_results) * QAOA_SHOTS} shots"
        else:
            total = sum(r.get(method, {}).get("shot_violations", 0) for r in all_results)
            viol_str = f"{total}/{len(all_results)} runs"
        mean_r = float(np.mean([r.get(method, {}).get("ratio", 0.0) for r in all_results]))
        print(f"  {method:<20} {viol_str:>22} {mean_r:>10.3f}")
    print("=" * 62)


def write_csv(all_results: list[dict], instance_labels: list[str]):
    """approx_ratio.csv: one row per instance, ratios to the ILP optimum."""
    path = os.path.join(OUT, "approx_ratio.csv")
    header = ["instance", "ilp_objective", "greedy", "sa",
              "random_mean", "qaoa_mean", "xy_qaoa_mean", "wsxy_start_mean", "wsxy_qaoa_mean",
              "random_p_optimal", "qaoa_p_optimal", "xy_qaoa_p_optimal",
              "wsxy_start_p_optimal", "wsxy_qaoa_p_optimal",
              "qaoa_p_feasible", "xy_qaoa_p_feasible", "wsxy_qaoa_p_feasible",
              "qaoa_best_shot", "xy_qaoa_best_shot", "wsxy_qaoa_best_shot"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for label, r in zip(instance_labels, all_results):
            get = lambda m, k: r.get(m, {}).get(k, 0.0)
            w.writerow([label, f"{r['ilp_obj']:.4f}"] + [f"{v:.4f}" for v in (
                get("Greedy (CELF)", "ratio"), get("Simulated Annealing", "ratio"),
                get("Random placement", "ratio"), get("QAOA", "ratio"), get("XY-QAOA", "ratio"),
                get("WSXY start", "ratio"), get("WSXY-QAOA", "ratio"),
                get("Random placement", "p_optimal"), get("QAOA", "p_optimal"),
                get("XY-QAOA", "p_optimal"),
                get("WSXY start", "p_optimal"), get("WSXY-QAOA", "p_optimal"),
                get("QAOA", "p_feasible"), get("XY-QAOA", "p_feasible"),
                get("WSXY-QAOA", "p_feasible"),
                get("QAOA", "best_shot_ratio"), get("XY-QAOA", "best_shot_ratio"),
                get("WSXY-QAOA", "best_shot_ratio"),
            )])
    print(f"Saved {path}")


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
    write_csv(all_results, labels)
    print_summary_table(all_results)

    # terrain_slope.png — generate once from the first instance's surface
    try:
        from benchmarks.map_viz import plot_terrain_slope, plot_placement_static
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
