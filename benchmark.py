"""
Benchmark: ILP, Greedy, SA, QAOA, XY-QAOA, WSXY-QAOA over many instances per size.

Greedy and SA return one placement each, scored as covered priority / ILP optimum.
QAOA returns a distribution over placements, so it is scored per shot with no
classical repair (infeasible shots count as 0) and shown next to the baseline
of drawing N-sensor placements uniformly at random. WSXY-QAOA starts from the
LP relaxation, so its baseline is its own starting state ("WSXY start").

Instances: for each size (n_sites, n_sensors), seeds 0..seeds-1. With --terrain real
(default) seed s picks a 20 x 20 window of the real 40 x 40 Khao Yai region (see
pick_real_window) and seeds the candidate sites; with --terrain synthetic, s is the
synthetic surface seed and the candidate-site seed. An instance is reproducible from
(terrain, size, seed) alone.

Outputs (repo root, optional --out-prefix):
  benchmark_runs.csv  - one row per instance
  approx_ratio.csv    - one row per size: mean and sample std over instances
  benchmark.png       - grouped bars (mean) with error bars (std)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import functools
import math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from priority_surface import build_priority_surface
from viewshed import generate_candidate_sites, compute_viewsheds, MAX_RANGE_CELLS
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
# Instances
# ---------------------------------------------------------------------------

GRID = 20                 # instance grid: 20 x 20 cells of 500 m
REAL_REGION = 40          # the real extract is 40 x 40 cells
PARK_MIN = 0.5            # a real window must have mean inside_park >= this
OPT_TOL = 1e-6            # ratio >= 1 - OPT_TOL counts as optimal
DEFAULT_SIZES = [(8, 2), (12, 3), (16, 4)]
LARGE_SIZE = (20, 5)


@functools.lru_cache(maxsize=1)
def _real_inside_park() -> np.ndarray:
    """inside_park of the full 40 x 40 real surface (built once)."""
    surf = build_priority_surface(REAL_REGION, REAL_REGION, 0, source="real", window=(0, 0))
    return np.asarray(surf["inside_park"], dtype=float)


def real_window_candidates(inside_park: np.ndarray, grid: int = GRID,
                           park_min: float = PARK_MIN) -> list[tuple[int, int]]:
    """All (row0, col0) offsets whose grid x grid window has mean inside_park >= park_min."""
    rows, cols = inside_park.shape
    out = []
    for r0 in range(rows - grid + 1):
        for c0 in range(cols - grid + 1):
            if inside_park[r0:r0 + grid, c0:c0 + grid].mean() >= park_min:
                out.append((r0, c0))
    return out


def pick_real_window(seed: int, inside_park: np.ndarray | None = None) -> tuple[int, int]:
    """Seed s picks one qualifying window uniformly at random (numpy default_rng(s))."""
    if inside_park is None:
        inside_park = _real_inside_park()
    cands = real_window_candidates(inside_park)
    if not cands:
        raise ValueError("no real-terrain window meets the inside_park rule")
    return cands[int(np.random.default_rng(seed).integers(len(cands)))]


def build_instance(terrain: str, n_sites: int, seed: int):
    """Returns (surface, candidate sites, window or None). Depends only on the arguments."""
    if terrain == "real":
        window = pick_real_window(seed)
        surf = build_priority_surface(GRID, GRID, seed, source="real", window=window)
    elif terrain == "synthetic":
        window = None
        surf = build_priority_surface(GRID, GRID, seed)
    else:
        raise ValueError(f"terrain must be 'real' or 'synthetic', got {terrain!r}")
    sites = generate_candidate_sites(
        (GRID, GRID), n_candidates=n_sites,
        priority=surf["priority"], dem_norm=surf["dem"], seed=seed,
    )
    return surf, sites, window


# ---------------------------------------------------------------------------
# Per-instance runner
# ---------------------------------------------------------------------------

NAN = float("nan")
RATIO_COLS = ["greedy", "sa", "random_mean", "qaoa_mean", "xy_qaoa_mean",
              "wsxy_start_mean", "wsxy_qaoa_mean"]
P_OPT_COLS = ["random_p_optimal", "qaoa_p_optimal", "xy_qaoa_p_optimal",
              "wsxy_start_p_optimal", "wsxy_qaoa_p_optimal"]
P_FEAS_COLS = ["qaoa_p_feasible", "xy_qaoa_p_feasible", "wsxy_qaoa_p_feasible"]
RUN_COLS = (["terrain", "n_sites", "n_sensors", "seed", "window_row", "window_col",
             "ilp_objective"] + RATIO_COLS + P_OPT_COLS + P_FEAS_COLS + ["seconds"])
QAOA_PREFIX = {"plain": "qaoa", "xy": "xy_qaoa", "wsxy": "wsxy_qaoa"}


def run_instance(terrain: str, n_sites: int, n_sensors: int, seed: int) -> dict:
    """
    Run every method on one instance. Returns a row dict (RUN_COLS keys) plus
    'failures' (list of method names that raised) and 'mismatches' (number of
    run_qaoa calls whose enumerated optimum differed from the ILP objective).
    Any method that fails leaves NaN in its columns.
    """
    t_start = time.time()
    row: dict = {c: NAN for c in RUN_COLS}
    row.update(terrain=terrain, n_sites=n_sites, n_sensors=n_sensors, seed=seed,
               window_row="", window_col="")
    failures: list[str] = []
    mismatches = 0

    surf, sites, window = build_instance(terrain, n_sites, seed)
    if window is not None:
        row["window_row"], row["window_col"] = window
    viewsheds = compute_viewsheds(surf["dem_m"], sites, max_range_cells=MAX_RANGE_CELLS)
    weights = {r * GRID + c: float(surf["priority"][r, c])
               for r in range(GRID) for c in range(GRID)}

    # ILP (exact optimum, reference only)
    ilp_obj = NAN
    try:
        _, ilp_obj, _ = ilp_coverage(viewsheds, weights, n_sensors)
        ilp_obj = float(ilp_obj)
    except Exception as e:
        print(f"    ILP failed: {e}")
        failures.append("ILP")
    row["ilp_objective"] = ilp_obj

    def ratio_of(chosen):
        return score_placement(chosen, viewsheds, weights, n_sensors, ilp_obj)[2]

    try:
        chosen_g, _ = greedy_coverage(viewsheds, weights, n_sensors)
        row["greedy"] = ratio_of(chosen_g)
    except Exception as e:
        print(f"    Greedy failed: {e}")
        failures.append("Greedy")

    try:
        chosen_sa, _, _ = sa_solve(viewsheds, weights, n_sensors, num_reads=100, seed=seed)
        row["sa"] = ratio_of(chosen_sa)
    except Exception as e:
        print(f"    SA failed: {e}")
        failures.append("SA")

    from solvers.qaoa import run_qaoa
    for mode, pre in QAOA_PREFIX.items():
        try:
            r = run_qaoa(viewsheds, weights, n_sensors, p_layers=QAOA_LAYERS, mode=mode,
                         max_iter=150, shots=QAOA_SHOTS, seed=seed)
        except Exception as e:
            print(f"    {pre} failed: {e}")
            failures.append(pre)
            continue
        # The enumerated optimum inside run_qaoa must agree with the ILP.
        if math.isfinite(ilp_obj) and ilp_obj > 1e-9 \
                and abs(r["optimum"] - ilp_obj) > 1e-6 * ilp_obj:
            mismatches += 1
            print(f"    WARNING: enumerated optimum {r['optimum']:.6f} != ILP {ilp_obj:.6f} ({mode})")
        row[f"{pre}_mean"] = r["mean_ratio"]
        row[f"{pre}_p_optimal"] = r["p_optimal"]
        row[f"{pre}_p_feasible"] = r["p_feasible"]
        if mode == "wsxy":
            row["wsxy_start_mean"] = r["start_mean_ratio"]
            row["wsxy_start_p_optimal"] = r["start_p_optimal"]
        if mode == "plain":
            # Random baseline is identical across modes (same viewsheds and N).
            row["random_mean"] = r["mean_ratio_random"]
            row["random_p_optimal"] = r["p_optimal_random"]

    row["seconds"] = time.time() - t_start
    row["failures"] = failures
    row["mismatches"] = mismatches
    return row


def _fmt(v, nd: int = 3) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def progress_line(row: dict) -> str:
    win = f"win=({row['window_row']},{row['window_col']})" if row["window_row"] != "" else "win=-"
    return (f"|S|={row['n_sites']:<2} N={row['n_sensors']} seed={row['seed']:<2} {win:<12} "
            f"ILP={_fmt(row['ilp_objective'])}  "
            f"greedy={_fmt(row['greedy'])} sa={_fmt(row['sa'])} rand={_fmt(row['random_mean'])} "
            f"qaoa={_fmt(row['qaoa_mean'])} xy={_fmt(row['xy_qaoa_mean'])} "
            f"wsxy0={_fmt(row['wsxy_start_mean'])} wsxy={_fmt(row['wsxy_qaoa_mean'])}  "
            f"{row['seconds']:.1f}s")


# ---------------------------------------------------------------------------
# Aggregation (pure: rows in, summary rows out)
# ---------------------------------------------------------------------------

def size_label(n_sites: int, n_sensors: int) -> str:
    return f"|S|={n_sites}, N={n_sensors}"


def _finite(vals) -> list[float]:
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean(vals) -> float:
    f = _finite(vals)
    return sum(f) / len(f) if f else NAN


def _sample_std(vals) -> float:
    f = _finite(vals)
    if len(f) < 2:
        return NAN
    m = sum(f) / len(f)
    return math.sqrt(sum((x - m) ** 2 for x in f) / (len(f) - 1))


def summary_columns() -> list[str]:
    cols = ["instance", "terrain", "n_instances"]
    for name in RATIO_COLS:
        cols += [name, f"{name}_std"]
    return cols + ["greedy_optimal_share", "sa_optimal_share"] + P_OPT_COLS + P_FEAS_COLS


def aggregate_runs(rows: list[dict]) -> list[dict]:
    """
    One summary dict per (terrain, n_sites, n_sensors), in order of first appearance.
    Only instances with a valid ILP objective (finite, > 0) count. NaNs are ignored
    in every mean and std; std is the sample std (n - 1) and NaN with fewer than two
    values. optimal share = fraction of non-NaN ratios >= 1 - OPT_TOL.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["terrain"], int(r["n_sites"]), int(r["n_sensors"])), []).append(r)

    out = []
    for (terrain, ns, nk), grp in groups.items():
        valid = [r for r in grp if _finite([r.get("ilp_objective")])
                 and float(r["ilp_objective"]) > 1e-9]
        s: dict = {"instance": size_label(ns, nk), "terrain": terrain,
                   "n_instances": len(valid)}
        for name in RATIO_COLS:
            vals = [r.get(name) for r in valid]
            s[name] = _mean(vals)
            s[f"{name}_std"] = _sample_std(vals)
        for name in ("greedy", "sa"):
            f = _finite(r.get(name) for r in valid)
            s[f"{name}_optimal_share"] = (sum(v >= 1 - OPT_TOL for v in f) / len(f)) if f else NAN
        for name in P_OPT_COLS + P_FEAS_COLS:
            s[name] = _mean(r.get(name) for r in valid)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _out_path(prefix: str, name: str) -> str:
    full = prefix + name
    return full if os.path.isabs(full) else os.path.join(OUT, full)


def _cell(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return "nan" if v is None or math.isnan(v) else f"{v:.6f}"


def write_runs_csv(rows: list[dict], path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RUN_COLS)
        for r in rows:
            w.writerow([_cell(r[c]) for c in RUN_COLS])
    print(f"Saved {path}")


def write_summary_csv(summary: list[dict], path: str):
    cols = summary_columns()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in summary:
            w.writerow([_cell(s[c]) if c != "n_instances" else str(s[c]) for c in cols])
    print(f"Saved {path}")


def plot_benchmark(summary: list[dict], path: str):
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    n_grp, n_meth = len(summary), len(BAR_METHODS)
    width = 0.115
    x = np.arange(n_grp) * (n_meth * width + 0.14)

    top = 1.0
    for mi, (method, color, col) in enumerate(zip(BAR_METHODS, BAR_COLORS, RATIO_COLS)):
        means = np.array([s[col] for s in summary], dtype=float)
        stds = np.array([s[f"{col}_std"] for s in summary], dtype=float)
        means_p = np.nan_to_num(means, nan=0.0)
        stds_p = np.nan_to_num(stds, nan=0.0)
        xs = x + mi * width
        ax.bar(xs, means_p, width, label=method, color=color, edgecolor="white", linewidth=0.5)
        ax.errorbar(xs, means_p, yerr=stds_p, fmt="none", ecolor="#222", elinewidth=1.0,
                    capsize=2.5, capthick=1.0)
        for xv, m, sd in zip(xs, means, stds_p):
            if math.isnan(m):
                continue
            y = m + sd + 0.012          # above the error bar cap
            top = max(top, y + 0.05)
            ax.text(xv, y, f"{m:.2f}", ha="center", va="bottom", fontsize=7, color="#333")

    ax.axhline(1.0, color="#222", linestyle="--", linewidth=1.2, label="Exact optimum (ILP)")

    terrain = summary[0]["terrain"] if summary else ""
    counts = sorted({s["n_instances"] for s in summary})
    n_txt = (str(counts[0]) if len(counts) == 1
             else f"{counts[0]} to {counts[-1]}") if counts else "0"
    ax.set_xticks(x + width * (n_meth - 1) / 2)
    ax.set_xticklabels([s["instance"] for s in summary], fontsize=10)
    ax.set_ylabel("Covered priority / exact optimum", fontsize=10)
    ax.set_xlabel("Greedy, SA: the placement returned.  Random and the QAOA variants: "
                  "expected value of one shot, no repair.",
                  fontsize=8, color="#555")
    ax.set_ylim(0, top)
    ax.set_title(f"Khao Yai sensor placement, {terrain} terrain: mean and standard deviation "
                 f"over {n_txt} instances per size", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"Saved {path}")


def print_summary(summary: list[dict]):
    print("\n" + "=" * 100)
    print("Mean +- sample std of (covered priority / exact optimum); QAOA family: one shot, no repair")
    for s in summary:
        print(f"\n{s['instance']}  ({s['terrain']}, n={s['n_instances']})  "
              f"optimal share: greedy={_fmt(s['greedy_optimal_share'], 2)} "
              f"sa={_fmt(s['sa_optimal_share'], 2)}")
        for method, col in zip(BAR_METHODS, RATIO_COLS):
            print(f"  {method:<20} {_fmt(s[col])} +- {_fmt(s[col + '_std'])}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_sizes(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        a, b = part.strip().split(":")
        out.append((int(a), int(b)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--terrain", choices=["real", "synthetic"], default="real")
    ap.add_argument("--seeds", type=int, default=20, help="instances per size (seeds 0..N-1)")
    ap.add_argument("--sizes", type=parse_sizes, default=DEFAULT_SIZES,
                    help="comma list of n_sites:n_sensors, e.g. 8:2,12:3")
    ap.add_argument("--large", type=int, default=0,
                    help="extra instances of size 20:5 (about 10 minutes each)")
    ap.add_argument("--out-prefix", default="",
                    help="prepended to output file names; relative prefixes resolve in the repo root")
    args = ap.parse_args(argv)

    plan = [(ns, nk, args.seeds) for ns, nk in args.sizes]
    if args.large > 0:
        plan.append((*LARGE_SIZE, args.large))

    rows: list[dict] = []
    n_fail = n_mismatch = 0
    size_seconds: dict[tuple, float] = {}
    t0 = time.time()
    for ns, nk, n_seeds in plan:
        print(f"\n--- size |S|={ns}, N={nk}: {n_seeds} instances, {args.terrain} terrain ---")
        for seed in range(n_seeds):
            try:
                row = run_instance(args.terrain, ns, nk, seed)
            except Exception as e:   # instance could not even be built
                print(f"    instance failed to build: {e}")
                row = {c: NAN for c in RUN_COLS}
                row.update(terrain=args.terrain, n_sites=ns, n_sensors=nk, seed=seed,
                           window_row="", window_col="", seconds=0.0,
                           failures=["instance"], mismatches=0)
            n_fail += len(row.pop("failures"))
            n_mismatch += row.pop("mismatches")
            size_seconds[(ns, nk)] = size_seconds.get((ns, nk), 0.0) + row["seconds"]
            rows.append(row)
            print(progress_line(row), flush=True)

    summary = aggregate_runs(rows)
    write_runs_csv(rows, _out_path(args.out_prefix, "benchmark_runs.csv"))
    write_summary_csv(summary, _out_path(args.out_prefix, "approx_ratio.csv"))
    plot_benchmark(summary, _out_path(args.out_prefix, "benchmark.png"))
    print_summary(summary)
    for (ns, nk), sec in size_seconds.items():
        print(f"Runtime |S|={ns}, N={nk}: {sec / 60:.1f} min")
    print(f"Total runtime {(time.time() - t0) / 60:.1f} min; "
          f"optimum mismatches (run_qaoa vs ILP): {n_mismatch}; method failures: {n_fail}")


if __name__ == "__main__":
    main()
