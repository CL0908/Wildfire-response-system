"""
Compare three ways of choosing detection sensor sites on held-out simulated fires.

SIMULATED, from a simplified and uncalibrated fire-spread model (see detection.fire_sim and
detection.objective). Every number written by this script compares placements under the stated
assumptions; none of it predicts real fires or real detection times.

    python -m detection.run                 # full run, rewrites detection/RESULTS.md and results.json
    python -m detection.run --quick         # small headline-only run, writes nothing
    python -m detection.run --quick --out DIR

The instance is the one in dashboards/public/results.json (20 x 20 sensor window, 16 candidate
sites, 4 sensors), placed inside the real 40 x 40 region. Ignitions fall only inside the sensor
window; fire spreads on the whole region. TRAIN scenarios (seed 101) choose placements and TEST
scenarios (seed 202) produce every reported number.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

from . import fire_sim
from .objective import score_detection, site_detection_times
from .solvers import exact_detection, greedy_detection

ROOT = fire_sim.ROOT
RESULTS_JSON_IN = os.path.join(ROOT, "dashboards", "public", "results.json")
META_JSON_IN = os.path.join(ROOT, "data", "khao_yai_meta.json")
OUT_DIR = os.path.join(ROOT, "detection")

TRAIN_SEED, TEST_SEED = 101, 202
N_SCENARIOS = 4000
QUICK_SCENARIOS = 300
HEADLINE_R0, HEADLINE_BASELINE_H = 2.0, 6.0
R0_GRID = (0.5, 1.0, 2.0, 4.0, 6.0)
BASELINE_GRID_H = (3.0, 6.0, 9.0, 12.0)
N_BOOT, BOOT_SEED = 1000, 12345
EXACT_TIME_LIMIT_S = 120
ND = 4  # decimals kept in results.json

NOTE = ("SIMULATED from a simplified, uncalibrated fire-spread model; not a prediction of real "
        "fires or real detection times.")


# ------------------------------------------------------------------------------ instance

def load_instance(results_path: str = RESULTS_JSON_IN, meta_path: str = META_JSON_IN) -> dict:
    """
    Sensors from the dashboard results file, mapped into the 40 x 40 region. Returns a dict with
    sees (n_sites, 1600) bool, ignition_weights (40 x 40, zero outside the sensor window), fuel,
    elevation, wind pool, k, the current placement and the window geometry.
    """
    with open(results_path) as f:
        res = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    g = res["grid"]
    w_rows, w_cols = int(g["rows"]), int(g["cols"])
    off_r, off_c = (int(v) for v in meta["default_window"]["offset_in_region"])
    shape = tuple(int(v) for v in meta["grid_shape"])
    sites = res["sites"]
    if [s["id"] for s in sites] != list(range(len(sites))):
        raise ValueError("site ids are expected to be 0..n-1 in order")
    sees = np.zeros((len(sites), shape[0] * shape[1]), dtype=bool)
    for s in sites:
        for flat in res["viewsheds"][str(s["id"])]:
            r, c = divmod(int(flat), w_cols)
            sees[s["id"], (r + off_r) * shape[1] + (c + off_c)] = True
    method = next(m for m in res["methods"] if m["name"] == "Greedy")
    current = sorted(int(i) for i in method["chosen"])
    ilp = next((m for m in res["methods"] if m["name"] == "ILP"), None)

    real = fire_sim.load_real_window()
    in_window = np.zeros(shape, dtype=bool)
    in_window[off_r:off_r + w_rows, off_c:off_c + w_cols] = True
    weights = np.where(in_window, real["ignition_weights"], 0.0)
    speed, direction = fire_sim.daytime_wind_pool(fire_sim.load_wind())
    return dict(sites=sites, sees=sees, k=int(res["n_sensors"]), current=current,
                current_equals_ilp=bool(ilp is not None and sorted(ilp["chosen"]) == current),
                shape=shape, window=dict(row0=off_r, col0=off_c, n_rows=w_rows, n_cols=w_cols),
                in_window=in_window, ignition_weights=weights, fuel=real["fuel"],
                elevation=real["elevation"], wind_speed=speed, wind_dir=direction)


def simulate_T(inst: dict, r0: float, n: int, seed: int, horizon_hours: float) -> np.ndarray:
    """Site detection times T (n, n_sites) in minutes for n scenarios; the arrival grid is dropped."""
    sc = fire_sim.simulate_scenarios(inst["ignition_weights"], inst["fuel"], inst["elevation"],
                                     inst["wind_speed"], inst["wind_dir"], n, seed, r0=r0,
                                     horizon_hours=horizon_hours)
    return site_detection_times(sc.arrival_min, inst["sees"])


# ------------------------------------------------------------------------------ placements

def _combos(n_sites: int, k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(n_sites), k)), dtype=np.int64)


def fire_history_optimum(sees: np.ndarray, weights: np.ndarray, k: int) -> tuple[list, float]:
    """Brute force over all k-subsets: maximise the ignition weight of the cells they see."""
    w = weights.ravel()
    best, best_val = None, -1.0
    for combo in _combos(sees.shape[0], k):
        v = float(w[sees[combo].any(axis=0)].sum())
        if v > best_val + 1e-12:
            best, best_val = [int(i) for i in combo], v
    return best, best_val / float(w.sum())


def _timed_values(T: np.ndarray, k: int, baseline_min: float) -> tuple[np.ndarray, np.ndarray]:
    V = np.maximum(baseline_min - T, 0.0)
    combos = _combos(T.shape[1], k)
    vals = np.empty(len(combos))
    for a in range(0, len(combos), 200):
        blk = combos[a:a + 200]
        vals[a:a + 200] = V[:, blk].max(axis=2).mean(axis=0)
    return combos, vals


def timed_optimum(T: np.ndarray, k: int, baseline_min: float) -> tuple[list, float]:
    """Brute force over all k-subsets of the exact objective (mean minutes saved)."""
    combos, vals = _timed_values(T, k, baseline_min)
    i = int(np.argmax(vals))
    return [int(x) for x in combos[i]], float(vals[i])


def timed_runner_up(T: np.ndarray, k: int, baseline_min: float) -> tuple[list, float]:
    """Second-best k-subset and its gap to the best, in hours of mean saving."""
    combos, vals = _timed_values(T, k, baseline_min)
    order = np.argsort(-vals, kind="stable")
    return [int(x) for x in combos[order[1]]], float((vals[order[0]] - vals[order[1]]) / 60.0)


# ------------------------------------------------------------------------------ summaries

def summarise(T: np.ndarray, chosen: list, baseline_min: float) -> dict:
    sc = score_detection(T, chosen, baseline_min)
    d_h = sc["detection_min"] / 60.0
    p10, p50, p90 = np.percentile(d_h, [10, 50, 90])
    return dict(sites=sc["chosen"],
                mean_detection_h=float(sc["mean_detection_min"] / 60.0),
                saved_h=float(sc["expected_saved_min"] / 60.0),
                share_at_ignition=sc["share_detected_at_time_0"],
                share_after_spread=sc["share_detected_by_sensor_after_spread"],
                share_satellite=sc["share_left_to_satellite"],
                p10_h=float(p10), p50_h=float(p50), p90_h=float(p90))


def after_spread_share_of_saving(T: np.ndarray, chosen: list, baseline_min: float) -> float | None:
    sc = score_detection(T, chosen, baseline_min)
    total = float(sc["saved_min"].sum())
    if total <= 0:
        return None
    return float(sc["saved_min"][sc["detection_min"] > 0].sum() / total)


def paired_bootstrap(d_a: np.ndarray, d_b: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Mean of (d_a - d_b) in hours over TEST scenarios with a percentile 95% interval."""
    diff = (np.asarray(d_a) - np.asarray(d_b)) / 60.0
    idx = np.random.default_rng(seed).integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean_diff_h=float(diff.mean()), ci95_low_h=float(lo), ci95_high_h=float(hi))


def _r(x, nd=ND):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


# ------------------------------------------------------------------------------ analysis

def run_analysis(quick: bool = False, n_scenarios: int | None = None, inst: dict | None = None,
                 log=print) -> dict:
    """
    Returns {"results": json-ready dict (deterministic), "timings": solve times in seconds
    (stdout only)}.
    """
    inst = inst or load_instance()
    n = n_scenarios or (QUICK_SCENARIOS if quick else N_SCENARIOS)
    k, sees = inst["k"], inst["sees"]
    n_sites = sees.shape[0]
    r0s = (HEADLINE_R0,) if quick else R0_GRID
    baselines_h = (HEADLINE_BASELINE_H,) if quick else BASELINE_GRID_H
    max_h = max(baselines_h)
    timings = {}

    hist_sites, hist_share = fire_history_optimum(sees, inst["ignition_weights"], k)
    current = inst["current"]
    t_limit = 20 if quick else EXACT_TIME_LIMIT_S

    headline, sens = None, []
    for r0 in r0s:
        t0 = time.time()
        # One simulation to the longest baseline serves every baseline: a detection time is
        # min(baseline, arrival), so arrivals past a shorter baseline never change a score.
        T_train = simulate_T(inst, r0, n, TRAIN_SEED, max_h)
        T_test = simulate_T(inst, r0, n, TEST_SEED, max_h)
        log(f"r0 = {r0}: simulated 2 x {n} scenarios in {time.time() - t0:.1f} s")
        for bh in baselines_h:
            bmin = bh * 60.0
            opt_sites, _ = timed_optimum(T_train, k, bmin)
            row = dict(r0_m_min=r0, baseline_h=bh, timed_optimum_sites=opt_sites,
                       same_sites_as_fire_history=opt_sites == hist_sites)
            for key, sites in (("current", current), ("fire_history", hist_sites),
                               ("timed_optimum", opt_sites)):
                row[f"test_saved_h_{key}"] = summarise(T_test, sites, bmin)["saved_h"]
            row["timed_optimum_saving_share_after_spread"] = after_spread_share_of_saving(
                T_test, opt_sites, bmin)
            sens.append(row)

            if r0 == HEADLINE_R0 and bh == HEADLINE_BASELINE_H:
                t1 = time.time()
                g = greedy_detection(T_train, k, bmin)
                timings["greedy_s"] = time.time() - t1
                t1 = time.time()
                ex = exact_detection(T_train, k, bmin, exactly_k=True, time_limit_s=t_limit)
                timings["exact_s"] = time.time() - t1
                bf_sites, bf_val = timed_optimum(T_train, k, bmin)
                ru_sites, ru_gap_h = timed_runner_up(T_train, k, bmin)
                placements = {
                    "satellite_only": [],
                    "current": current,
                    "fire_history": hist_sites,
                    "timed_greedy": g.chosen,
                    "timed_exact": ex.chosen,
                }
                table = {}
                for name, sites in placements.items():
                    s = summarise(T_test, sites, bmin)
                    s["train_saved_h"] = score_detection(T_train, sites, bmin)["expected_saved_min"] / 60.0
                    table[name] = s
                dets = {name: score_detection(T_test, sites, bmin)["detection_min"]
                        for name, sites in placements.items()}
                headline = dict(
                    r0_m_min=r0, baseline_h=bh, placements=table,
                    bootstrap=dict(
                        n_resamples=N_BOOT, seed=BOOT_SEED,
                        note="paired over TEST scenarios; difference in mean detection time, hours; negative = first placement detects earlier",
                        fire_history_minus_current=paired_bootstrap(dets["fire_history"], dets["current"]),
                        timed_exact_minus_fire_history=paired_bootstrap(dets["timed_exact"], dets["fire_history"])),
                    exact_solver=dict(
                        proven_optimal=bool(ex.proven_optimal), status=ex.status, time_limit_s=t_limit,
                        n_distinct_rows=ex.details["n_distinct_rows"],
                        n_x_variables=ex.details["n_x_variables"],
                        matches_brute_force=bool(ex.chosen == bf_sites
                                                 or abs(ex.objective - bf_val) <= 1e-9 * max(1.0, abs(bf_val))),
                        same_sites_as_brute_force=ex.chosen == bf_sites,
                        brute_force_sites=bf_sites),
                    greedy_matches_brute_force=g.chosen == bf_sites,
                    train_runner_up=dict(sites=ru_sites, gap_to_best_h=ru_gap_h))
                log(f"  greedy {timings['greedy_s']:.1f} s, exact {timings['exact_s']:.1f} s "
                    f"(proven optimal: {ex.proven_optimal})")

    results = dict(
        simulated=True, note=NOTE,
        assumptions=[
            "A sensor detects a fire the moment it enters a cell the sensor sees; no detection delay, no failures, no false alarms.",
            "Satellite detection happens at a fixed delay after ignition (the baseline); this is an assumption, not a measurement.",
            "Spread rate r0 is an assumption inside the range of experimental burns in deciduous forest elsewhere in Thailand; Khao Yai is mostly evergreen and has no published rate.",
            "One wind value (speed and direction, drawn from daytime hours of the fire-season record) applies to the whole map for the whole scenario.",
            "Ignitions are drawn from smoothed 2012 to 2025 satellite fire detections, inside the 20 x 20 sensor window only; fire spreads over the full 40 x 40 region.",
        ],
        instance=dict(
            n_sites=n_sites, n_sensors=k, sensor_window=inst["window"], region_shape=list(inst["shape"]),
            current_placement=current, current_placement_equals_ilp_optimum=inst["current_equals_ilp"],
            site_rows_cols_in_window=[[s["row"], s["col"]] for s in inst["sites"]],
            fire_history_sites=hist_sites, fire_history_share_of_ignition_weight_seen=hist_share,
            n_subsets_brute_forced=int(len(_combos(n_sites, k))),
            ignition_weight_share_seen_by_current=float(
                inst["ignition_weights"].ravel()[sees[current].any(axis=0)].sum()
                / inst["ignition_weights"].sum())),
        scenarios=dict(n_train=n, n_test=n, train_seed=TRAIN_SEED, test_seed=TEST_SEED, quick=quick),
        headline=headline,
        sensitivity=sens,
    )
    return dict(results=_r(results), timings=timings)


# ------------------------------------------------------------------------------ writing

PLACEMENT_LABELS = [
    ("satellite_only", "Satellite only"),
    ("current", "Current placement (hazard x value)"),
    ("fire_history", "Fire-history coverage"),
    ("timed_greedy", "Time-to-detection, greedy"),
    ("timed_exact", "Time-to-detection, exact"),
]


def _sites(s):
    return ", ".join(str(i) for i in s) if s else "none"


def _pct(x):
    return f"{100 * x:.1f}%"


def _ci(b):
    return f"{b['mean_diff_h']:+.3f} h (95% interval {b['ci95_low_h']:+.3f} to {b['ci95_high_h']:+.3f})"


def render_markdown(res: dict) -> str:
    h, sens, inst, sc = res["headline"], res["sensitivity"], res["instance"], res["scenarios"]
    base = h["baseline_h"]
    L = []
    a = L.append
    a("# Detection placement comparison (simulated)")
    a("")
    a(f"**{NOTE}** Generated by `python -m detection.run`; the same numbers are in `results.json`.")
    a("")
    a("## Question")
    a("")
    a("Four sensor masts go on 16 candidate sites in a 20 x 20 cell (10 km x 10 km) window of Khao Yai National Park. "
      "Does choosing the sites by simulated time to detection give a different placement from choosing them by where fires have started, "
      "and how do both compare with the placement the project already uses?")
    a("")
    a("## Method")
    a("")
    a("- Fire grid: the real 40 x 40 region (500 m cells, land cover and elevation from the project data). The sensor window sits at row 15, column 3 of the region. "
      "Ignitions are drawn only inside the window, with probability proportional to smoothed 2012 to 2025 satellite fire counts; fire then spreads over the whole region, so it can leave and re-enter the window.")
    a("- Spread: minimum-travel-time model on an 8-neighbour grid with hand-set fuel, slope and wind multipliers (`detection/fire_sim.py`). One wind value per scenario, drawn from daytime hours of the fire-season record.")
    a(f"- Scenarios: {sc['n_train']:,} TRAIN scenarios (seed {sc['train_seed']}) choose placements and {sc['n_test']:,} independent TEST scenarios (seed {sc['test_seed']}) produce every number below.")
    a("- Detection: a sensor detects the fire the moment it enters any cell it sees. The satellite detects every fire `baseline` hours after ignition. A scenario's detection time is the earlier of the two.")
    a("- Placements compared, all with 4 sensors and all scored by the same function on TEST:")
    a("  - Satellite only: no sensors.")
    a("  - Current placement: the dashboard's placement, which maximises hazard x value coverage and equals the exact optimum of that objective.")
    a(f"  - Fire-history coverage: the 4 sites that see the largest total ignition weight, with no spread model. Solved by brute force over all {inst['n_subsets_brute_forced']:,} subsets.")
    a("  - Time-to-detection, greedy: greedy maximisation of expected hours saved on TRAIN.")
    a(f"  - Time-to-detection, exact: integer programme (CBC, {h['exact_solver']['time_limit_s']} s limit) on TRAIN. The true optimum was also found by brute force over all subsets as a check.")
    a("")
    a(f"## Headline setting: r0 = {h['r0_m_min']:g} m/min, satellite baseline {base:g} hours")
    a("")
    a("| Placement | Sites | Mean detection (h) | Hours saved vs satellite | At ignition | By sensor after spread | Left to satellite | Detection time p10 / p50 / p90 (h) |")
    a("|---|---|---|---|---|---|---|---|")
    for key, label in PLACEMENT_LABELS:
        p = h["placements"][key]
        a(f"| {label} | {_sites(p['sites'])} | {p['mean_detection_h']:.3f} | {p['saved_h']:.3f} | {_pct(p['share_at_ignition'])} | "
          f"{_pct(p['share_after_spread'])} | {_pct(p['share_satellite'])} | {p['p10_h']:.2f} / {p['p50_h']:.2f} / {p['p90_h']:.2f} |")
    a("")
    bs = h["bootstrap"]
    a(f"Paired bootstrap over TEST scenarios ({bs['n_resamples']:,} resamples, seed {bs['seed']}), difference in mean detection time (negative means the first placement detects earlier):")
    a("")
    a(f"- Fire-history coverage minus current placement: {_ci(bs['fire_history_minus_current'])}")
    a(f"- Time-to-detection exact minus fire-history coverage: {_ci(bs['timed_exact_minus_fire_history'])}")
    a("")
    ex = h["exact_solver"]
    a(f"Exact solver on TRAIN: proven optimal = {ex['proven_optimal']} (status {ex['status']}, within the {ex['time_limit_s']} s limit); "
      f"its value {'matches' if ex['matches_brute_force'] else 'does NOT match'} the brute-force optimum over all subsets"
      f"{'' if ex['same_sites_as_brute_force'] else ' (different sites with the same value)' if ex['matches_brute_force'] else ''}. "
      f"Greedy {'also found' if h['greedy_matches_brute_force'] else 'did not find'} that optimum.")
    a("")
    a("## Sensitivity")
    a("")
    a("The timed optimum is the best 4 sites on TRAIN by brute force at each setting. Hours saved are on TEST. "
      "The last column is the share of the timed optimum's saving that comes from fires detected after they spread (detection time above 0).")
    a("")
    a("| r0 (m/min) | Baseline (h) | Timed optimum | Same sites as fire-history | Saved: current | Saved: fire-history | Saved: timed optimum | Saving from fires detected after spread |")
    a("|---|---|---|---|---|---|---|---|")
    for r in sens:
        sh = r["timed_optimum_saving_share_after_spread"]
        a(f"| {r['r0_m_min']:g} | {r['baseline_h']:g} | {_sites(r['timed_optimum_sites'])} | {'yes' if r['same_sites_as_fire_history'] else 'no'} | "
          f"{r['test_saved_h_current']:.3f} | {r['test_saved_h_fire_history']:.3f} | {r['test_saved_h_timed_optimum']:.3f} | "
          f"{'n/a' if sh is None else _pct(sh)} |")
    a("")
    a("## Finding")
    a("")
    diffs = [r for r in sens if not r["same_sites_as_fire_history"]]
    shares = [r["timed_optimum_saving_share_after_spread"] for r in sens
              if r["timed_optimum_saving_share_after_spread"] is not None]
    ru = h["train_runner_up"]
    start_in_view = 1.0 - max(shares)
    if not diffs:
        a(f"In all {len(sens)} settings the time-to-detection optimum is the same four sites as the fire-history coverage optimum "
          f"(sites {_sites(inst['fire_history_sites'])}). Fires that start inside a seen cell supply {_pct(start_in_view)} or more of the optimum's saving at every setting "
          f"(fires detected after spreading supply {_pct(min(shares))} to {_pct(max(shares))}). So in this simulation the time-to-detection objective reduces to "
          f"coverage weighted by where fires have started, and the spread model changes nothing about which sites are chosen.")
    else:
        where = "; ".join(f"r0 = {r['r0_m_min']:g}, baseline {r['baseline_h']:g} h (sites {_sites(r['timed_optimum_sites'])})" for r in diffs)
        a(f"The time-to-detection optimum equals the fire-history coverage optimum (sites {_sites(inst['fire_history_sites'])}) in "
          f"{len(sens) - len(diffs)} of {len(sens)} settings. It differs at: {where}. "
          f"Fires detected after spreading supply {_pct(min(shares))} to {_pct(max(shares))} of its saving.")
    a("")
    a(f"The choice is not always clear-cut: at the headline setting the second-best set on TRAIN ({_sites(ru['sites'])}) is {ru['gap_to_best_h']:.3f} h behind the best. "
      f"A different random TRAIN draw can put a near-tied set on top, so the equality above holds for these {sc['n_train']:,} scenarios and is not a guarantee.")
    a("")
    a("## Caveats")
    a("")
    a("- The current placement was optimised for hazard x value, which includes what is worth protecting. The other three placements optimise detection of ignitions only. "
      "That the current placement saves fewer simulated hours here does not make it the worse placement, and this comparison declares no winner.")
    a("- The fire-history and time-to-detection placements are chosen and scored on fires drawn from the same fire-history distribution. That is circular in the same way "
      "as the hazard layer, whose fire-density term comes from the same detections. TRAIN and TEST separate only the random scenarios, not the ignition distribution or the model.")
    a("- Spread rates come from experimental burns in deciduous forest elsewhere in Thailand; Khao Yai is mostly evergreen and no rate was found for it. The 6 m/min end of the sensitivity range has no source checked.")
    a("- Instant detection on entering a visible cell makes the numbers more favourable to sensors than real detection would be. The fixed satellite delay, the single wind value for the whole map and the unfitted spread model have an unknown effect on the comparison.")
    a("- Viewsheds exist only for the 20 x 20 window, so a sensor near the window edge is not credited with cells it would see outside it. This works against sensors.")
    a("- The sensitivity settings reuse the same TRAIN and TEST seeds, so the ignition and wind draws are the same in every setting. The 20 settings are not 20 independent confirmations, and a near-tie between sites breaks the same way each time.")
    a("- Differences between placements that sit within the bootstrap interval are not distinguishable at this number of scenarios.")
    a("")
    return "\n".join(L)


def write_outputs(res: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write(render_markdown(res))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare detection sensor placements on held-out simulated fires.")
    ap.add_argument("--quick", action="store_true",
                    help=f"{QUICK_SCENARIOS} scenarios, headline setting only; writes nothing unless --out is given")
    ap.add_argument("--out", default=None, help="directory for RESULTS.md and results.json (default: detection/ on a full run)")
    ap.add_argument("--n-scenarios", type=int, default=None, help=f"scenarios per set (default {N_SCENARIOS})")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = run_analysis(quick=args.quick, n_scenarios=args.n_scenarios)
    out_dir = args.out or (None if args.quick else OUT_DIR)
    if out_dir:
        write_outputs(out["results"], out_dir)
        print(f"wrote RESULTS.md and results.json to {out_dir}")
    print({k: round(v, 1) for k, v in out["timings"].items()}, f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
