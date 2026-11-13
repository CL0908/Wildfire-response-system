"""
Whole-park sensor placement for Khao Yai National Park: three objectives, one scoring path.

MODELLED AND SIMULATED. The priority surface (hazard x value) and the line-of-sight visibility are
modelled. Every fire scenario and every detection time is simulated by a simplified, uncalibrated
spread model (see detection.fire_sim and detection.objective). Nothing written here predicts real
fires or real detection times.

    python -m park.run                # full run, rewrites park/RESULTS.md, results.json, placements.png
    python -m park.run --quick        # a few hundred sites, few scenarios, k = 5; writes nothing
    python -m park.run --quick --out DIR

Objectives, all on the whole-park instance (every cell with inside_park > 0.5 is a candidate site):
  coverage       greedy maximum coverage of the modelled priority weights (the project's objective)
  fire_history   greedy maximum coverage of the ignition weights (smoothed 2012 to 2025 detections)
  timed_greedy   greedy expected hours saved against the satellite, on TRAIN scenarios (seed 101)
Every placement is scored by `summarise_placement`: coverage shares in the park, and detection
times on TEST scenarios (seed 202) through detection.objective.score_detection.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import dataclass

import numpy as np

from detection import fire_sim
from detection.objective import score_detection
from detection.run import paired_bootstrap
from detection.solvers import greedy_detection
from park.instance import PARK_NPZ, ParkInstance, build_park_instance, load_park_instance
from park.solvers import (coverage_bound, coverage_value, greedy_cover,
                          site_detection_times_sparse)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SEED, TEST_SEED = 101, 202
N_SCENARIOS = 2000
QUICK_SCENARIOS = 60
KS = (10, 20, 40)
QUICK_KS = (5,)
CURVE_MAX_K = 60
HEADLINE_R0, HEADLINE_BASELINE_H = 2.0, 6.0
E_R0S = (0.5, 2.0, 6.0)
E_BASELINES_H = (3.0, 6.0, 12.0)
E_K = 20
N_RANDOM, RANDOM_SEED = 200, 20261113
MIP_TIME_LIMIT_S = 300
QUICK_SITES = 300
MAP_K = 20

NOTE = ("MODELLED priority surface and line-of-sight visibility; SIMULATED fires and detection times "
        "from a simplified, uncalibrated spread model. Not a prediction of real fires or real "
        "detection times, and not a siting plan.")

PLACEMENT_LABELS = [
    ("coverage", "Priority coverage (hazard x value)"),
    ("fire_history", "Fire-history coverage"),
    ("timed_greedy", "Time-to-detection, greedy"),
]


# ------------------------------------------------------------------------------ inputs

@dataclass
class Inputs:
    park: ParkInstance
    inside: np.ndarray        # (n_cells,) bool, inside_park > 0.5
    fire_w: np.ndarray        # (n_cells,) ignition weights, zero outside the park and on non-fuel
    fuel: np.ndarray          # (rows, cols)
    wind_speed: np.ndarray
    wind_dir: np.ndarray


def load_inputs(park: ParkInstance | None = None) -> Inputs:
    """The park instance plus the spread-model inputs on the same grid."""
    park = park or load_park_instance()
    real = fire_sim.load_real_window((0, 0), park.shape, str(PARK_NPZ))
    if not np.array_equal(real["elevation"], park.elevation):
        raise ValueError("fire_sim elevation differs from the park instance elevation")
    inside = (park.inside_park > 0.5).ravel()
    fire_w = np.where(inside, real["ignition_weights"].ravel(), 0.0)
    speed, direction = fire_sim.daytime_wind_pool(fire_sim.load_wind())
    return Inputs(park, inside, fire_w, real["fuel"], speed, direction)


def quick_instance(n_sites: int = QUICK_SITES) -> ParkInstance:
    """A park instance on an evenly spaced subset of the candidate cells (no visibility cache)."""
    full = load_park_instance()
    cells = full.site_cells
    step = max(1, len(cells) // n_sites)
    return build_park_instance(site_cells=cells[::step][:n_sites], use_cache=False)


# ------------------------------------------------------------------------------ scoring

def covered_mask(sees, chosen) -> np.ndarray:
    covered = np.zeros(sees.shape[1], dtype=bool)
    for i in chosen:
        covered[sees.indices[sees.indptr[i]:sees.indptr[i + 1]]] = True
    return covered


def coverage_shares(sees, weights, inside, fire_count, chosen) -> dict:
    """
    Shares of the park covered by `chosen`: priority weight, area and 2012 to 2025 fire detections,
    each over the cells with `inside` True only (the denominators are the in-park totals).
    """
    cov = covered_mask(sees, chosen) & inside
    w = np.asarray(weights, dtype=np.float64)
    fc = np.asarray(fire_count, dtype=np.float64)
    return dict(priority_share=float(w[cov].sum() / w[inside].sum()),
                area_share=float(cov.sum() / inside.sum()),
                fire_share=float(fc[cov].sum() / fc[inside].sum()))


def _permille(shares) -> list[float]:
    """Round shares that sum to 1 to 3 decimals so that the rounded values still sum to 1."""
    v = np.asarray(shares, dtype=np.float64) * 1000.0
    fl = np.floor(v + 1e-9)
    short = int(round(1000.0 - fl.sum()))
    order = np.argsort(-(v - fl), kind="stable")
    fl[order[:max(short, 0)]] += 1.0
    return [float(x) / 1000.0 for x in fl]


def summarise_placement(inp: Inputs, chosen, T_test: np.ndarray, baseline_min: float) -> dict:
    """
    The one scoring path for every placement (and, with chosen = [], satellite only). Coverage
    shares are in-park; detection numbers are on the TEST scenarios in T_test. Rounded once:
    mean detection to 0.001 h and hours saved as baseline minus that rounded mean; the three
    detection shares to 0.001 with a sum of exactly 1; the other shares to 0.001.
    """
    park = inp.park
    chosen = sorted(int(i) for i in chosen)
    cov = coverage_shares(park.sees, park.weights, inp.inside, park.fire_count, chosen)
    sc = score_detection(T_test, chosen, baseline_min)
    base_h = baseline_min / 60.0
    mean_h = round(sc["mean_detection_min"] / 60.0, 3)
    at0, mid, sat = _permille([sc["share_detected_at_time_0"],
                               sc["share_detected_by_sensor_after_spread"],
                               sc["share_left_to_satellite"]])
    cols = park.shape[1]
    return dict(n_sensors=len(chosen),
                sites=[[int(park.site_cells[i] // cols), int(park.site_cells[i] % cols)] for i in chosen],
                priority_share=round(cov["priority_share"], 3), area_share=round(cov["area_share"], 3),
                fire_share=round(cov["fire_share"], 3),
                mean_detection_h=mean_h, saved_h=round(base_h - mean_h, 3),
                share_at_ignition=at0, share_after_spread=mid, share_satellite=sat)


def simulate_T(inp: Inputs, r0: float, n: int, seed: int, horizon_h: float) -> np.ndarray:
    """Site detection times (n, n_sites), minutes, for n scenarios; the arrival grid is dropped."""
    park = inp.park
    sc = fire_sim.simulate_scenarios(inp.fire_w.reshape(park.shape), inp.fuel, park.elevation,
                                     inp.wind_speed, inp.wind_dir, n, seed, r0=r0,
                                     horizon_hours=horizon_h)
    return site_detection_times_sparse(sc.arrival_min, park.sees)


# ------------------------------------------------------------------------------ analysis

def _site_rc(park: ParkInstance, idx) -> list:
    cols = park.shape[1]
    return [[int(park.site_cells[i] // cols), int(park.site_cells[i] % cols)] for i in sorted(idx)]


def random_baseline(sees, weights, k: int, n: int = N_RANDOM, seed: int = RANDOM_SEED) -> dict:
    """Share of the total weight covered by n random k-site placements: mean, min, max."""
    rng = np.random.default_rng(seed)
    total = float(np.sum(weights))
    vals = np.array([coverage_value(sees, weights, rng.choice(sees.shape[0], size=k, replace=False))
                     for _ in range(n)]) / total
    return dict(n_placements=n, seed=seed, mean=round(float(vals.mean()), 3),
                min=round(float(vals.min()), 3), max=round(float(vals.max()), 3))


def run_analysis(quick: bool = False, n_scenarios: int | None = None, inp: Inputs | None = None,
                 ks: tuple | None = None, log=print) -> dict:
    """
    Returns {"results": json-ready dict (deterministic), "timings": seconds (stdout only),
    "placements": site indices per objective and k (for the figure)}.
    """
    t_all = time.time()
    inp = inp or load_inputs(quick_instance() if quick else None)
    park = inp.park
    sees, weights = park.sees, park.weights
    n_sites = park.n_sites
    ks = tuple(k for k in (ks or (QUICK_KS if quick else KS)) if k <= n_sites)
    n = n_scenarios or (QUICK_SCENARIOS if quick else N_SCENARIOS)
    curve_max = min(10 if quick else CURVE_MAX_K, n_sites)
    kmax = max(max(ks), curve_max)
    e_k = E_K if not quick else ks[0]
    r0s = (HEADLINE_R0,) if quick else E_R0S
    baselines_h = (HEADLINE_BASELINE_H,) if quick else E_BASELINES_H
    horizon_h = max(baselines_h)
    mip_limit = 20 if quick else MIP_TIME_LIMIT_S
    timings: dict = {}
    total_w = float(weights.sum())

    # A. priority coverage ---------------------------------------------------------------
    t0 = time.time()
    a_order, a_cum = greedy_cover(sees, weights, kmax)
    timings["greedy_priority_s"] = time.time() - t0
    curve = []
    for k in range(1, curve_max + 1):
        sh = coverage_shares(sees, weights, inp.inside, park.fire_count, a_order[:k])
        prev = a_cum[k - 2] if k > 1 else 0.0
        curve.append(dict(k=k, objective_share=round(a_cum[k - 1] / total_w, 3),
                          added_by_last_sensor=round((a_cum[k - 1] - prev) / total_w, 4),
                          priority_share=round(sh["priority_share"], 3),
                          area_share=round(sh["area_share"], 3), fire_share=round(sh["fire_share"], 3)))
    bounds = {}
    for k in ks:
        b = coverage_bound(sees, weights, k, integer=True, time_limit_s=mip_limit)
        timings[f"lp_k{k}_s"] = b["lp_seconds"]
        timings[f"mip_k{k}_s"] = b["integer_seconds"] or 0.0
        greedy_v = a_cum[k - 1]
        row = dict(k=k, greedy_share=round(greedy_v / total_w, 3),
                   lp_bound_share=round(b["lp_value"] / total_w, 3),
                   greedy_over_lp_bound=round(greedy_v / b["lp_value"], 3),
                   integer_proven=bool(b["integer_proven"]),
                   integer_status=b["integer_status"], integer_time_limit_s=mip_limit,
                   lp_rows=b["n_rows"],
                   random=random_baseline(sees, weights, k))
        if b["integer_proven"]:
            row["integer_optimum_share"] = round(b["integer_value"] / total_w, 3)
            row["greedy_over_integer_optimum"] = round(greedy_v / b["integer_value"], 3)
        bounds[str(k)] = row
        log(f"k = {k}: greedy {greedy_v / total_w:.4f}, LP bound {b['lp_value'] / total_w:.4f}, "
            f"integer {b['integer_status']} ({b['integer_seconds'] or 0:.0f} s)")

    # B. fire-history coverage ------------------------------------------------------------
    b_order, b_cum = greedy_cover(sees, inp.fire_w, kmax)
    fire_total = float(inp.fire_w.sum())
    fh_own = {str(k): round(b_cum[k - 1] / fire_total, 3) for k in ks}

    # C, D, E. simulated time to detection ---------------------------------------------------
    headline_rows: dict = {}
    effect_rows: list = []
    timed: dict = {}
    for r0 in r0s:
        t0 = time.time()
        T_train = simulate_T(inp, r0, n, TRAIN_SEED, horizon_h)
        T_test = simulate_T(inp, r0, n, TEST_SEED, horizon_h)
        timings[f"simulate_r0_{r0:g}_s"] = time.time() - t0
        log(f"r0 = {r0:g}: simulated 2 x {n} scenarios, site times in {time.time() - t0:.0f} s")
        for bh in baselines_h:
            bmin = bh * 60.0
            t0 = time.time()
            if r0 == HEADLINE_R0 and bh == HEADLINE_BASELINE_H:
                for k in ks:
                    timed[k] = sorted(greedy_detection(T_train, k, bmin).chosen)
                timed_e = timed[e_k] if e_k in timed else sorted(greedy_detection(T_train, e_k, bmin).chosen)
                timings["greedy_timed_headline_s"] = time.time() - t0
                headline_rows["satellite_only"] = summarise_placement(inp, [], T_test, bmin)
                for k in ks:
                    headline_rows[str(k)] = {
                        "coverage": summarise_placement(inp, a_order[:k], T_test, bmin),
                        "fire_history": summarise_placement(inp, b_order[:k], T_test, bmin),
                        "timed_greedy": summarise_placement(inp, timed[k], T_test, bmin)}
            else:
                timed_e = sorted(greedy_detection(T_train, e_k, bmin).chosen)
            fh = sorted(b_order[:e_k])
            d_t = score_detection(T_test, timed_e, bmin)["detection_min"]
            d_f = score_detection(T_test, fh, bmin)["detection_min"]
            bs = paired_bootstrap(d_t, d_f)
            effect_rows.append(dict(
                r0_m_min=r0, baseline_h=bh, k=e_k,
                shared_sites=len(set(timed_e) & set(fh)),
                mean_detection_h_timed=round(float(d_t.mean()) / 60.0, 3),
                mean_detection_h_fire_history=round(float(d_f.mean()) / 60.0, 3),
                timed_minus_fire_history_h=round(bs["mean_diff_h"], 3),
                ci95_low_h=round(bs["ci95_low_h"], 3), ci95_high_h=round(bs["ci95_high_h"], 3)))
        del T_train, T_test

    ratio = float(park.fire_count[~inp.inside].sum() / max(park.fire_count[inp.inside].sum(), 1.0))
    results = dict(
        modelled_and_simulated=True, note=NOTE,
        instance=dict(
            grid_shape=list(park.shape), cell_m=park.cell_m, n_cells=park.n_cells,
            n_candidate_sites=n_sites, n_in_park_cells=int(inp.inside.sum()),
            n_sites_with_positive_ignition_weight=int((inp.fire_w[park.site_cells] > 0).sum()),
            total_priority_weight_in_park=round(float(weights[inp.inside].sum()), 1),
            total_priority_weight_on_grid=round(total_w, 1),
            in_park_fire_detections=int(park.fire_count[inp.inside].sum()),
            margin_fire_detections=int(park.fire_count[~inp.inside].sum()),
            margin_to_park_detection_ratio=round(ratio, 1),
            visible_site_cell_pairs=int(sees.nnz), quick=quick),
        scenarios=dict(n_train=n, n_test=n, train_seed=TRAIN_SEED, test_seed=TEST_SEED,
                       headline_r0_m_min=HEADLINE_R0, headline_baseline_h=HEADLINE_BASELINE_H,
                       horizon_h=horizon_h),
        k_values=list(ks),
        coverage_curve=curve,
        coverage_bounds=bounds,
        fire_history_own_objective_share={**fh_own},
        headline=headline_rows,
        spread_model_effect=effect_rows,
    )
    timings["total_s"] = time.time() - t_all
    timings["peak_memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    placements = {"coverage": {k: a_order[:k] for k in ks}, "fire_history": {k: b_order[:k] for k in ks},
                  "timed_greedy": dict(timed)}
    return dict(results=results, timings=timings, placements=placements)


# ------------------------------------------------------------------------------ markdown

def _pct(x) -> str:
    return f"{100 * x:.1f}%"


def _fmt_ci(r) -> str:
    return f"{r['timed_minus_fire_history_h']:+.3f} ({r['ci95_low_h']:+.3f} to {r['ci95_high_h']:+.3f})"


def render_markdown(res: dict) -> str:
    inst, sc, bnd = res["instance"], res["scenarios"], res["coverage_bounds"]
    ks, head, eff, curve = res["k_values"], res["headline"], res["spread_model_effect"], res["coverage_curve"]
    cby = {c["k"]: c for c in curve}
    e_k = eff[0]["k"]
    L = []
    a = L.append
    a("# Whole-park sensor placement (modelled and simulated)")
    a("")
    a(f"**{NOTE}** Generated by `python -m park.run`; the same numbers are in `results.json`.")
    a("")
    a("## Question")
    a("")
    a(f"Sensor masts go on any of {inst['n_candidate_sites']:,} candidate cells of Khao Yai National Park "
      f"({inst['grid_shape'][0]} x {inst['grid_shape'][1]} cells of {inst['cell_m']:g} m, the park plus a 5 km margin). "
      f"For {', '.join(str(k) for k in ks[:-1])} and {ks[-1]} sensors, how much of the park do three different placement rules cover, "
      "how close is the greedy coverage placement to the best possible, and does the simulated spread of fire change which sites are chosen?")
    a("")
    a("## Method")
    a("")
    a("- Candidates and visibility (modelled): every cell with more than half its area inside the park, a 30 m mast, 4 km range, line of sight over the elevation grid (`park/instance.py`). A site also sees cells in the margin.")
    a("- Priority (modelled): hazard x value from the project's hand formula, rescaled over the whole grid. Margin cells carry a value of 0.1 instead of 0.5 to 1.0, so they still have weight.")
    a("- Placements, all by greedy selection on the same visibility matrix (`park/solvers.py`):")
    a("  - Priority coverage: maximise the priority weight of the cells seen, over the whole grid (the project's existing objective at park scale).")
    a("  - Fire-history coverage: maximise the ignition weight of the cells seen. Ignition weights are the smoothed 2012 to 2025 satellite fire counts (log, Gaussian, sigma = 1 cell), restricted to cells inside the park.")
    a(f"  - Time-to-detection, greedy: maximise the expected hours saved against the satellite on {sc['n_train']:,} TRAIN scenarios (seed {sc['train_seed']}) with `detection.solvers.greedy_detection`. No exact solver is run at this scale.")
    a("- Scoring (the same function for every placement): the share of in-park priority weight, in-park area and 2012 to 2025 in-park fire detections that fall in cells the sensors see; and, on "
      f"{sc['n_test']:,} independent TEST scenarios (seed {sc['test_seed']}), the mean detection time and the hours saved against satellite only.")
    a(f"- Simulated fires (`detection/fire_sim.py`): ignitions are drawn only in cells inside the park, in proportion to the ignition weights; fire spreads over the whole grid. About {inst['margin_to_park_detection_ratio']:.0f} times more detections lie in the surrounding margin than in the park "
      "({:,} against {:,}); fires that start there and spread in are not modelled as ignitions. One daytime wind value per scenario, drawn from the fire-season record.".format(inst['margin_fire_detections'], inst['in_park_fire_detections']))
    a(f"- Detection: a sensor detects a fire the moment it enters a cell it sees; the satellite detects every fire {sc['headline_baseline_h']:g} hours after ignition (assumption). Headline spread rate r0 = {sc['headline_r0_m_min']:g} m/min. "
      "A scenario's detection time is the earlier of the two.")
    a("")
    a(f"## Results at r0 = {sc['headline_r0_m_min']:g} m/min, satellite baseline {sc['headline_baseline_h']:g} hours")
    a("")
    a("Coverage columns are shares of the park (in-park cells only). Detection columns are simulated, on TEST scenarios.")
    a("")
    a("| Sensors | Placement | Priority covered | Area covered | Fire detections covered | Mean detection (h) | Hours saved vs satellite | At ignition | By sensor after spread | Left to satellite |")
    a("|---|---|---|---|---|---|---|---|---|---|")

    def row(k, label, p):
        a(f"| {k} | {label} | {_pct(p['priority_share'])} | {_pct(p['area_share'])} | {_pct(p['fire_share'])} | {p['mean_detection_h']:.3f} | "
          f"{p['saved_h']:.3f} | {_pct(p['share_at_ignition'])} | {_pct(p['share_after_spread'])} | {_pct(p['share_satellite'])} |")
    row(0, "Satellite only", head["satellite_only"])
    for k in ks:
        for key, label in PLACEMENT_LABELS:
            row(k, label, head[str(k)][key])
    a("")
    a("Site coordinates (row, column on the park grid) for every placement are in `results.json`.")
    a("")
    if len(ks) > 1:
        k0, k1 = (e_k, ks[-1]) if e_k in ks and ks[-1] > e_k else (ks[0], ks[-1])
        a(f"Going from {k0} to {k1} sensors adds, in priority covered and in simulated hours saved:")
        a("")
        for key, label in PLACEMENT_LABELS:
            p0, p1 = head[str(k0)][key], head[str(k1)][key]
            a(f"- {label}: {100 * (p1['priority_share'] - p0['priority_share']):+.1f} percentage points priority, "
              f"{p1['saved_h'] - p0['saved_h']:+.3f} h saved ({p0['saved_h']:.3f} to {p1['saved_h']:.3f}).")
        a("")
    a("## Priority coverage against an upper bound")
    a("")
    a("The objective here is priority weight over the whole grid, margin included, so these shares differ from the in-park shares above. "
      "The upper bound is the LP relaxation of the maximum-coverage integer programme (HiGHS). The integer programme was given "
      f"{bnd[str(ks[0])]['integer_time_limit_s']} s per k.")
    a("")
    a("| Sensors | Greedy | LP upper bound | Greedy / bound | Integer optimum | Greedy / integer optimum | 200 random placements: mean (range) |")
    a("|---|---|---|---|---|---|---|")
    for k in ks:
        b = bnd[str(k)]
        if b["integer_proven"]:
            io, ir = _pct(b["integer_optimum_share"]), _pct(b["greedy_over_integer_optimum"])
        else:
            io, ir = f"not proved in {b['integer_time_limit_s']} s", "n/a"
        rb = b["random"]
        a(f"| {k} | {_pct(b['greedy_share'])} | {_pct(b['lp_bound_share'])} | {_pct(b['greedy_over_lp_bound'])} | {io} | {ir} | "
          f"{_pct(rb['mean'])} ({_pct(rb['min'])} to {_pct(rb['max'])}) |")
    a("")
    proven = [k for k in ks if bnd[str(k)]["integer_proven"]]
    if len(proven) == len(ks):
        a("The integer optimum was proved at every k, so the table states greedy against the true optimum of this objective.")
    elif proven:
        a(f"The integer optimum was proved only for k = {', '.join(str(k) for k in proven)}; where it was not, only the LP bound is a valid reference and the optimum lies between greedy and that bound.")
    else:
        a("The integer programme did not finish within the time limit at any k, so no optimum is claimed: the true optimum lies between greedy and the LP bound, and greedy is not shown to be optimal.")
    a("")
    a("## Coverage against sensor count")
    a("")
    a("Greedy priority-coverage placement. Objective is the whole-grid share; the other columns are in-park shares.")
    a("")
    a("| Sensors | Objective | Priority covered | Area covered | Fire detections covered | Added by the last sensor (objective) |")
    a("|---|---|---|---|---|---|")
    shown = [k for k in (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60) if k in cby]
    for k in shown:
        c = cby[k]
        a(f"| {k} | {_pct(c['objective_share'])} | {_pct(c['priority_share'])} | {_pct(c['area_share'])} | {_pct(c['fire_share'])} | "
          f"{100 * c['added_by_last_sensor']:.2f} pp |")
    a("")
    a(f"## Does the spread model change the answer? (k = {e_k})")
    a("")
    a(f"For each setting the timed-greedy placement is chosen on TRAIN and compared with the fire-history coverage placement. Shared sites is the number of the {e_k} sites that both pick. "
      "The difference is timed minus fire-history in mean detection time on TEST, in hours (negative means the timed placement detects earlier), with a paired bootstrap 95% interval over TEST scenarios (1,000 resamples, seed 12345).")
    a("")
    a("| r0 (m/min) | Baseline (h) | Shared sites | Mean detection: timed (h) | Mean detection: fire-history (h) | Difference (h) |")
    a("|---|---|---|---|---|---|")
    for r in eff:
        a(f"| {r['r0_m_min']:g} | {r['baseline_h']:g} | {r['shared_sites']} of {r['k']} | {r['mean_detection_h_timed']:.3f} | "
          f"{r['mean_detection_h_fire_history']:.3f} | {_fmt_ci(r)} |")
    a("")
    shared = [r["shared_sites"] for r in eff]
    n_same = sum(r["shared_sites"] == r["k"] for r in eff)
    n_fh = sum(r["ci95_low_h"] > 0 for r in eff)        # fire-history detects earlier
    n_timed = sum(r["ci95_high_h"] < 0 for r in eff)    # timed detects earlier
    n_unclear = len(eff) - n_fh - n_timed
    diffs = [r["timed_minus_fire_history_h"] for r in eff]
    a("## Finding")
    a("")
    if n_same == len(eff):
        a(f"In all {len(eff)} settings the timed-greedy placement is the same {e_k} sites as the fire-history placement. In this simulation the time-to-detection objective reduces to coverage weighted by where fires have started, and the spread model changes nothing about which sites are chosen.")
    else:
        a(f"The two placements share {min(shared)} to {max(shared)} of {e_k} sites ({n_same} of {len(eff)} settings are identical), so at this scale the spread model does change which sites are chosen. "
          f"It does not clearly improve detection: on TEST the fire-history placement detects earlier in {n_fh} of {len(eff)} settings, the timed placement in {n_timed}, and {n_unclear} are inside the bootstrap interval. "
          f"The difference in mean detection time (timed minus fire-history) runs from {min(diffs):+.3f} to {max(diffs):+.3f} h.")
        a("")
        a("The timed placement is chosen on a sample of simulated ignitions, while fire-history coverage uses the full smoothed ignition distribution, so part of the timed placement's shortfall may be sampling noise in that choice. This was not tested.")
    a("")
    a("The settings reuse the same TRAIN and TEST seeds, so they are not independent confirmations.")
    a("")
    a("## Limitations")
    a("")
    a("- Every in-park cell is a candidate site, with no account of road access, land tenure, power or maintenance. The result is a geometric comparison, not a siting plan.")
    a("- Visibility is line of sight over 500 m cells from a 30 m mast. It includes no tree-canopy height, smoke or weather. The rule rounds ray positions to cells in a way that depends on a site's absolute row and column parity, so about 6% of viewshed cells differ from the 20 x 20 window's results for the same sites.")
    a("- Priority terms are rescaled over the whole park, so cell values differ slightly from the small window's (correlation 0.996).")
    a("- The spread rate comes from experimental burns in deciduous forest elsewhere in Thailand. The land-cover data shows the park as 98% tree cover without separating evergreen from deciduous forest. The spread model is uncalibrated.")
    a("- Fires spread slowly in this model (a tree-cover cell burns at 1.2 m/min before slope and wind at the headline rate), so detection at ignition inside a seen cell supplies most of the sensor detections in the table. Instant detection on entering a seen cell favours sensors more than real detection would.")
    a(f"- Ignitions are drawn inside the park only. About {inst['margin_to_park_detection_ratio']:.0f} times more detections lie in the margin, and fires spreading in from outside are not simulated.")
    a("- The fire-history and time-to-detection placements are chosen and scored on the same fire-history distribution, which is circular; TRAIN and TEST separate only the random scenarios. The hazard layer's fire term comes from the same detections.")
    a("- The priority-coverage placement also weighs what is worth protecting, while the other two optimise ignitions only. That the placements score differently on each other's measures does not rank them, and no winner is declared between objectives.")
    a("- The satellite baseline is an assumption, not a measurement. Differences between placements inside a bootstrap interval are not distinguishable at this number of scenarios.")
    a("")
    return "\n".join(L)


# ------------------------------------------------------------------------------ figure

def make_figure(inp: Inputs, a_sites, b_sites, path: str) -> None:
    """Map of the park with two placements and the cells each one sees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    park = inp.park
    rows, cols = park.shape
    km = park.cell_m / 1000.0
    extent = [0, cols * km, rows * km, 0]
    ins = inp.inside.reshape(park.shape)
    cov_a = covered_mask(park.sees, a_sites).reshape(park.shape)
    cov_b = covered_mask(park.sees, b_sites).reshape(park.shape)

    OR, BL, GR = "#E69F00", "#56B4E9", "#009E73"       # Okabe-Ito
    tri_c, sq_c = "#B34700", "#0A4F82"
    fig = plt.figure(figsize=(16, 11.4), dpi=100)
    ax = fig.add_axes([0.06, 0.205, 0.92, 0.70])
    land = LinearSegmentedColormap.from_list("land", ["#f1eee6", "#aaa595"])
    ax.imshow(park.elevation, cmap=land, extent=extent, interpolation="nearest", aspect="equal")
    dim = np.zeros(park.shape + (4,))
    dim[~ins] = (1, 1, 1, 0.55)
    ax.imshow(dim, extent=extent, interpolation="nearest", aspect="equal")
    ov = np.zeros(park.shape + (4,))
    only_a, only_b, both = cov_a & ~cov_b, cov_b & ~cov_a, cov_a & cov_b
    ov[only_a] = (*matplotlib.colors.to_rgb(OR), 0.42)
    ov[only_b] = (*matplotlib.colors.to_rgb(BL), 0.42)
    ov[both] = (*matplotlib.colors.to_rgb(GR), 0.42)
    ax.imshow(ov, extent=extent, interpolation="nearest", aspect="equal")
    xs = (np.arange(cols) + 0.5) * km
    ys = (np.arange(rows) + 0.5) * km
    ax.contour(xs, ys, park.inside_park, levels=[0.5], colors="#222222", linewidths=1.6)

    def xy(sites):
        c = park.site_cells[sorted(sites)]
        return (c % cols + 0.5) * km, (c // cols + 0.5) * km
    bx, by = xy(b_sites)
    ax.scatter(bx, by, marker="s", s=150, c=sq_c, edgecolors="white", linewidths=1.4, zorder=4)
    ax_, ay_ = xy(a_sites)
    ax.scatter(ax_, ay_, marker="^", s=150, c=tri_c, edgecolors="white", linewidths=1.4, zorder=5)
    ax.set_xlim(0, cols * km)
    ax.set_ylim(rows * km, 0)
    ax.set_xlabel("km east of the grid's west edge", fontsize=13)
    ax.set_ylabel("km south of the grid's north edge (north is up)", fontsize=13)
    ax.tick_params(labelsize=12)
    k = len(a_sites)
    n_shared = len(set(a_sites) & set(b_sites))
    fig.suptitle(f"Where {k} sensors go in Khao Yai National Park under two placement rules",
                 x=0.06, y=0.965, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.06, 0.925, "Park boundary in black. Shading marks the cells each set of sensors can see (modelled line of sight).",
             ha="left", fontsize=13.5, color="#333333")
    handles = [
        Line2D([], [], marker="^", ls="", ms=13, mfc=tri_c, mec="white", label=f"Priority-coverage sites ({k})"),
        Line2D([], [], marker="s", ls="", ms=12, mfc=sq_c, mec="white", label=f"Fire-history sites ({k}); {n_shared} sites in both"),
        Patch(fc=OR, alpha=0.5, label="Seen only by the priority-coverage sites"),
        Patch(fc=BL, alpha=0.5, label="Seen only by the fire-history sites"),
        Patch(fc=GR, alpha=0.5, label="Seen by both sets"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.075), ncol=3, fontsize=12.5,
               frameon=False, columnspacing=2.0)
    el = park.elevation[ins]
    fig.text(0.06, 0.028,
             "The priority surface (hazard x value) behind the priority-coverage sites is modelled, not observed, and so is the visibility. "
             "The fire-history sites use satellite fire\n"
             f"detections from 2012 to 2025. Every park cell is treated as a possible site; access, land tenure and power are ignored, so this is not a siting plan. "
             f"Background: elevation,\ndarker is higher ({el.min():.0f} to {el.max():.0f} m in the park); the margin outside the park is paler.",
             ha="left", va="center", fontsize=11, color="#333333", linespacing=1.4)
    fig.savefig(path, dpi=100, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------------------ writing

def write_outputs(out: dict, inp: Inputs, out_dir: str) -> None:
    res = out["results"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write(render_markdown(res))
    kmap = MAP_K if MAP_K in res["k_values"] else res["k_values"][-1]
    make_figure(inp, out["placements"]["coverage"][kmap], out["placements"]["fire_history"][kmap],
                os.path.join(out_dir, "placements.png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Whole-park sensor placement: coverage, fire history and simulated time to detection.")
    ap.add_argument("--quick", action="store_true",
                    help=f"about {QUICK_SITES} candidate sites, {QUICK_SCENARIOS} scenarios, k = 5; writes nothing unless --out is given")
    ap.add_argument("--out", default=None, help="output directory (default: park/ on a full run)")
    ap.add_argument("--n-scenarios", type=int, default=None, help=f"scenarios per set (default {N_SCENARIOS})")
    args = ap.parse_args(argv)
    t0 = time.time()
    inp = load_inputs(quick_instance() if args.quick else None)
    out = run_analysis(quick=args.quick, n_scenarios=args.n_scenarios, inp=inp)
    out_dir = args.out or (None if args.quick else HERE)
    if out_dir:
        write_outputs(out, inp, out_dir)
        print(f"wrote RESULTS.md, results.json and placements.png to {out_dir}")
    print({k: round(v, 1) for k, v in out["timings"].items()}, f"wall {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
