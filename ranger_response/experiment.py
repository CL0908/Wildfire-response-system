"""Model-v2 experiment: greedy vs lookahead vs exact on seeded real-terrain scenarios.

    python -m ranger_response.experiment [--n 20] [--exact-limit 30] [--sens-n 20] [--sens-limit 20]
                                         [--workers 2] [--out ranger_response/experiment_results.json]

The protocol (scenario rule, parameters, sensitivity variants, metrics) is written in
ranger_response/README.md before the numbers were produced. Results are written in full, nothing is
filtered.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import time
from datetime import date

import numpy as np

from detection.fire_sim import (DEFAULT_R0_M_MIN, daytime_wind_pool, load_real_window, load_wind,
                                simulate_scenarios)

from .model import Plan
from .model2 import DEFAULT_HORIZON_H, DEFAULT_WALK_KMH, Instance2, make_instance2, simulate2
from .real import build_real_surface
from .run import GRID, default_station
from .solvers2 import solve_exact2, solve_greedy2, solve_lookahead2

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "experiment_results.json")

SITE_WINDOW_IN_200 = (80 + 15, 80 + 3)     # the site's 20 x 20 window inside the 200 x 200 hazard-training grid

# Crew line-cutting. Source: "Fireline production rate of handcrews in wildfires", International
# Journal of Wildland Fire 32(11):1503 (field data from southern Spain, 2014 to 2019): mean 0.16 m per
# minute per firefighter for indirect attack in timber fuel types (0.44 for direct attack in timber
# understory, 1.06 in timber litter). A firebreak cut ahead of the fire is indirect line, so the default
# uses 0.16. The crew size of 10 is an assumption. 10 * 0.16 * 60 = 96 m/h, i.e. 500 m takes 5.2 h.
CUT_RATE_M_MIN_PER_FF = 0.16
CREW_SIZE = 10
BASE_CONFIG = dict(
    n_scenarios=20, seed=42, horizon_h=DEFAULT_HORIZON_H, r0_m_min=DEFAULT_R0_M_MIN, walk_kmh=DEFAULT_WALK_KMH,
    cut_rate_m_min_per_ff=CUT_RATE_M_MIN_PER_FF, crew_size=CREW_SIZE, delay_h=0.0,
)
# name -> overrides applied to BASE_CONFIG (the sensitivity protocol)
SENSITIVITY = {
    "crew rate x0.5": dict(rate_mult=0.5),
    "crew rate x2": dict(rate_mult=2.0),
    "crew rate x2.75 (direct-attack timber-understory rate, 0.44 m/min/person)": dict(rate_mult=0.44 / 0.16),
    "r0 = 1 m/min": dict(r0_m_min=1.0),
}


def cut_hours(cfg: dict, rate_mult: float = 1.0) -> float:
    rate_m_h = cfg["cut_rate_m_min_per_ff"] * rate_mult * cfg["crew_size"] * 60.0
    return 500.0 / rate_m_h


_SITE = {}


def site_inputs() -> dict:
    """Real window inputs shared by every scenario (cached)."""
    if not _SITE:
        surf = build_real_surface()
        w = load_real_window(window=SITE_WINDOW_IN_200, shape=(GRID, GRID))
        assert np.array_equal(w["elevation"], surf["dem_m"]), "window mismatch with the site surface"
        _SITE.update(surf=surf, window=w, value=surf["priority"], dem=surf["dem_m"], fuel_factor=w["fuel"])
    return _SITE


def make_scenarios(n: int = 20, seed: int = 42) -> list:
    """Ignition cells ~ smoothed FIRMS history inside the window; wind ~ the daytime fire-season pool.
    Uses fire_sim.simulate_scenarios (same sampler as the sensor analysis), so (n, seed) fixes everything."""
    s = site_inputs()
    w = s["window"]
    sp, dr = daytime_wind_pool(load_wind())
    ss = simulate_scenarios(w["ignition_weights"], w["fuel"], w["elevation"], sp, dr, n, seed,
                            horizon_hours=BASE_CONFIG["horizon_h"])
    out = []
    for k in range(n):
        cell = divmod(int(ss.ignition_cell[k]), GRID)
        out.append(dict(id=k, ignition=[int(cell[0]), int(cell[1])], wind_speed_ms=float(ss.wind_speed_ms[k]),
                        wind_from_deg=float(ss.wind_dir_deg[k]),
                        fire_count=int(s["surf"]["fire_count"][cell]),
                        ignition_weight=float(w["ignition_weights"][cell])))
    return out


def build_instance(scen: dict, crews: int, cfg: dict | None = None, rate_mult: float = 1.0,
                   r0_m_min: float | None = None, horizon_h: float | None = None) -> Instance2:
    cfg = dict(BASE_CONFIG if cfg is None else cfg)
    if horizon_h is not None:
        cfg["horizon_h"] = horizon_h
    s = site_inputs()
    ig = tuple(scen["ignition"])
    station = default_station(GRID, GRID, ig)
    inst = make_instance2(s["fuel_factor"], s["value"], s["dem"], [ig], [station] * crews,
                          scen["wind_speed_ms"], scen["wind_from_deg"],
                          r0_m_min=cfg["r0_m_min"] if r0_m_min is None else r0_m_min,
                          horizon_h=cfg["horizon_h"], walk_kmh=cfg["walk_kmh"],
                          cut_h=cut_hours(cfg, rate_mult), delay_h=cfg["delay_h"])
    inst.params.update(terrain="real", scenario=scen["id"])
    return inst


def _method(sr, base_burned_value: float) -> dict:
    r = sr.result
    return dict(value_saved=float(r.value_saved), pct_saved=100.0 * float(r.value_saved) / base_burned_value
                if base_burned_value > 0 else 0.0,
                cells_burned=int(r.cells_burned), proven_optimal=sr.proven_optimal,
                stop_reason=sr.stop_reason or None, elapsed_s=float(sr.elapsed), nodes=int(sr.nodes),
                plan=[[list(map(int, c)) for c in route] for route in sr.plan.routes])


def run_task(task: dict) -> dict:
    """One (scenario, crews, variant) instance: baseline, greedy, lookahead, exact (warm-started from the
    better of greedy and lookahead)."""
    scen, crews = task["scenario"], task["crews"]
    inst = build_instance(scen, crews, rate_mult=task.get("rate_mult", 1.0), r0_m_min=task.get("r0_m_min"),
                          horizon_h=task.get("horizon_h"))
    base = simulate2(inst, Plan.empty(crews))
    total = float(inst.value[inst.burnable].sum())
    burned_value = total - base.baseline_objective
    g = solve_greedy2(inst)
    la = solve_lookahead2(inst)
    start = la if la.objective >= g.objective else g
    ex = solve_exact2(inst, time_limit=task["exact_limit"], warm_start=start.plan)
    return dict(scenario=scen["id"], crews=crews, variant=task.get("variant", "base"),
                station=list(inst.stations[0]), cut_h=inst.cut_h, r0_m_min=inst.r0_m_min,
                total_value=total, baseline_cells_burned=int(base.cells_burned),
                baseline_value_burned=burned_value,
                methods=dict(greedy=_method(g, burned_value), lookahead=_method(la, burned_value),
                             exact=_method(ex, burned_value)))


def run_tasks(tasks: list, workers: int) -> list:
    if workers <= 1:
        return [run_task(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(run_task, tasks, chunksize=1)


def summarize(results: list) -> dict:
    """Per-method and per-gap statistics over a list of run_task outputs (one crew count)."""
    out = {"n": len(results), "methods": {}, "gaps": {}}
    for m in ("greedy", "lookahead", "exact"):
        pct = [r["methods"][m]["pct_saved"] for r in results]
        val = [r["methods"][m]["value_saved"] for r in results]
        out["methods"][m] = dict(
            mean_pct_saved=statistics.fmean(pct), median_pct_saved=statistics.median(pct),
            mean_value_saved=statistics.fmean(val),
            mean_cells_burned=statistics.fmean(r["methods"][m]["cells_burned"] for r in results),
            proven_optimal=(sum(bool(r["methods"][m]["proven_optimal"]) for r in results) if m == "exact" else None),
            mean_runtime_s=statistics.fmean(r["methods"][m]["elapsed_s"] for r in results),
            max_runtime_s=max(r["methods"][m]["elapsed_s"] for r in results))
    out["baseline"] = dict(
        mean_cells_burned=statistics.fmean(r["baseline_cells_burned"] for r in results),
        mean_value_burned=statistics.fmean(r["baseline_value_burned"] for r in results),
        mean_total_value=statistics.fmean(r["total_value"] for r in results))
    for a, b, name in (("exact", "greedy", "exact - greedy"), ("exact", "lookahead", "exact - lookahead"),
                       ("lookahead", "greedy", "lookahead - greedy")):
        d_val = [r["methods"][a]["value_saved"] - r["methods"][b]["value_saved"] for r in results]
        d_pct = [r["methods"][a]["pct_saved"] - r["methods"][b]["pct_saved"] for r in results]
        srt = sorted(d_pct)
        out["gaps"][name] = dict(
            n_positive=sum(v > 1e-6 for v in d_val), n_zero=sum(abs(v) <= 1e-6 for v in d_val),
            n_negative=sum(v < -1e-6 for v in d_val),
            mean_value=statistics.fmean(d_val), mean_pct_points=statistics.fmean(d_pct),
            median_pct_points=statistics.median(d_pct), min_pct_points=srt[0], max_pct_points=srt[-1],
            p25_pct_points=float(np.percentile(d_pct, 25)), p75_pct_points=float(np.percentile(d_pct, 75)))
    return out


def run_experiment(n_scenarios: int = 20, seed: int = 42, exact_limit: float = 30.0, sens_n: int = 20,
                   sens_limit: float = 20.0, workers: int = 2, crews_list=(1, 2), log=print) -> dict:
    t0 = time.perf_counter()
    scens = make_scenarios(n_scenarios, seed)
    cfg = dict(BASE_CONFIG, n_scenarios=n_scenarios, seed=seed)
    tasks = [dict(scenario=s, crews=c, exact_limit=exact_limit) for c in crews_list for s in scens]
    res = run_tasks(tasks, workers)
    log(f"main experiment done in {time.perf_counter() - t0:.0f} s")
    sens = {}
    if sens_n > 0:
        for name, ov in SENSITIVITY.items():
            ts = [dict(scenario=s, crews=c, exact_limit=sens_limit, variant=name, **ov)
                  for c in crews_list for s in scens[:sens_n]]
            sens[name] = run_tasks(ts, workers)
            log(f"sensitivity '{name}' done at {time.perf_counter() - t0:.0f} s")
    summary = {str(c): summarize([r for r in res if r["crews"] == c]) for c in crews_list}
    # baseline comparison on the same scenarios for the sensitivity subset
    sens_summary = {name: {str(c): summarize([r for r in rs if r["crews"] == c]) for c in crews_list}
                    for name, rs in sens.items()}
    sens_summary["default (same subset)"] = {
        str(c): summarize([r for r in res if r["crews"] == c and r["scenario"] < sens_n]) for c in crews_list} \
        if sens_n > 0 else {}
    return dict(generated=date.today().isoformat(), config=dict(cfg, cut_h=cut_hours(cfg), exact_limit_s=exact_limit,
                                                                 sensitivity_exact_limit_s=sens_limit,
                                                                 sensitivity_scenarios=sens_n, workers=workers),
                scenarios=scens, results=res, summary=summary, sensitivity=sens,
                sensitivity_summary=sens_summary, total_runtime_s=time.perf_counter() - t0)


def markdown_tables(data: dict) -> str:
    """The tables in ranger_response/README.md, generated from the results file (python -m ranger_response.experiment --tables)."""
    L = []
    crews_list = sorted({r["crews"] for r in data["results"]})
    by = {(r["scenario"], r["crews"]): r for r in data["results"]}

    def sumtab(summ, title):
        L.append(f"**{title}**\n")
        L.append("| crews | method | mean % saved | median % saved | mean value saved | mean cells burned | proven optimal | mean / max runtime (s) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in crews_list:
            sm = summ[str(c)]
            b = sm["baseline"]
            L.append(f"| {c} | no response | 0 | 0 | 0 | {b['mean_cells_burned']:.1f} | n/a | n/a |")
            for m, v in sm["methods"].items():
                pr = "n/a" if v["proven_optimal"] is None else f"{v['proven_optimal']} / {sm['n']}"
                L.append(f"| {c} | {m} | {v['mean_pct_saved']:.1f} | {v['median_pct_saved']:.1f} | {v['mean_value_saved']:.2f} | "
                         f"{v['mean_cells_burned']:.1f} | {pr} | {v['mean_runtime_s']:.2f} / {v['max_runtime_s']:.1f} |")
        L.append("")

    def gaptab(summ, title):
        L.append(f"**{title}**\n")
        L.append("| crews | gap | scenarios better / equal / worse | mean (pp) | median (pp) | 25th-75th pct (pp) | max (pp) | mean (value units) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in crews_list:
            for g, v in summ[str(c)]["gaps"].items():
                L.append(f"| {c} | {g} | {v['n_positive']} / {v['n_zero']} / {v['n_negative']} | {v['mean_pct_points']:.2f} | "
                         f"{v['median_pct_points']:.2f} | {v['p25_pct_points']:.2f} to {v['p75_pct_points']:.2f} | "
                         f"{v['max_pct_points']:.2f} | {v['mean_value']:.3f} |")
        L.append("")

    sumtab(data["summary"], "Main experiment: 20 scenarios, value saved as % of the value that burns within 24 h with no response")
    gaptab(data["summary"], "Main experiment: differences in % points of the no-response burned value (pp)")
    L.append("**Sensitivity (same 20 scenarios, exact limit 20 s; 'default' is the main experiment, exact limit 30 s)**\n")
    L.append("| variant | crews | mean % saved: greedy / lookahead / exact | exact - greedy: mean / median pp (better, equal, worse) | exact - lookahead: mean / median pp (better, equal, worse) | exact proven | no-response cells burned |")
    L.append("|---|---|---|---|---|---|---|")
    for name, by_c in [("default", data["summary"])] + [(k, v) for k, v in data["sensitivity_summary"].items() if k != "default (same subset)"]:
        for c in crews_list:
            sm = by_c[str(c)]
            m = sm["methods"]
            g1, g2 = sm["gaps"]["exact - greedy"], sm["gaps"]["exact - lookahead"]
            L.append(f"| {name} | {c} | {m['greedy']['mean_pct_saved']:.1f} / {m['lookahead']['mean_pct_saved']:.1f} / {m['exact']['mean_pct_saved']:.1f} | "
                     f"{g1['mean_pct_points']:.1f} / {g1['median_pct_points']:.1f} ({g1['n_positive']}, {g1['n_zero']}, {g1['n_negative']}) | "
                     f"{g2['mean_pct_points']:.1f} / {g2['median_pct_points']:.1f} ({g2['n_positive']}, {g2['n_zero']}, {g2['n_negative']}) | "
                     f"{m['exact']['proven_optimal']} / {sm['n']} | {sm['baseline']['mean_cells_burned']:.1f} |")
    L.append("")
    for k, v in data.get("posthoc", {}).items():
        sumtab(v["summary"], f"Post-hoc (added after reading the main results): horizon {v['horizon_h']:g} h, exact limit {v['exact_limit_s']:g} s")
        gaptab(v["summary"], f"Post-hoc horizon {v['horizon_h']:g} h: differences (pp)")
    L.append("**Per scenario (% of no-response burned value saved; * = exact not proven optimal)**\n")
    L.append("| id | ignition | FIRMS count | wind m/s from deg | no-response cells / value burned | 1 crew: greedy / lookahead / exact | 2 crews: greedy / lookahead / exact |")
    L.append("|---|---|---|---|---|---|---|")
    for sc in data["scenarios"]:
        r1, r2 = by[(sc["id"], 1)], by[(sc["id"], 2)]
        def cell(r):
            ms = r["methods"]
            star = "" if ms["exact"]["proven_optimal"] else "*"
            return f"{ms['greedy']['pct_saved']:.1f} / {ms['lookahead']['pct_saved']:.1f} / {ms['exact']['pct_saved']:.1f}{star}"
        L.append(f"| {sc['id']} | ({sc['ignition'][0]}, {sc['ignition'][1]}) | {sc['fire_count']} | {sc['wind_speed_ms']:.1f} / {sc['wind_from_deg']:.0f} | "
                 f"{r1['baseline_cells_burned']} / {r1['baseline_value_burned']:.2f} | {cell(r1)} | {cell(r2)} |")
    L.append("")
    L.append(plan_diagnostics(data))
    return "\n".join(L)


def plan_diagnostics(data: dict) -> str:
    """Size and timing slack of the stored plans (slack = fire arrival - firebreak finish, hours, over held firebreaks
    on cells the fire reaches). A plan with a small minimum slack only works if the fire arrives on schedule."""
    rows = ["**Plan diagnostics (main experiment, 24 h)**\n",
            "| crews | method | mean firebreaks | median of per-plan minimum slack (h) | plans with a firebreak finished < 0.25 h before the fire | median slack over all firebreaks (h) |",
            "|---|---|---|---|---|---|"]
    for crews in sorted({r["crews"] for r in data["results"]}):
        for m in ("greedy", "lookahead", "exact"):
            nd, mins, allsl = [], [], []
            for r in data["results"]:
                if r["crews"] != crews:
                    continue
                inst = build_instance(data["scenarios"][r["scenario"]], crews)
                res = simulate2(inst, Plan.of([[tuple(c) for c in rt] for rt in r["methods"][m]["plan"]]))
                sl = [d.fire_arrival - d.finish for d in res.defences if np.isfinite(d.fire_arrival)]
                nd.append(len(res.defences))
                allsl += sl
                if sl:
                    mins.append(min(sl))
            rows.append(f"| {crews} | {m} | {statistics.fmean(nd):.2f} | {statistics.median(mins):.2f} | "
                        f"{sum(v < 0.25 for v in mins)} / {len(mins)} | {statistics.median(allsl):.2f} |")
    return "\n".join(rows) + "\n"


def run_posthoc_horizon(path: str, horizon_h: float, exact_limit: float = 20.0, workers: int = 2, log=print) -> dict:
    """Added AFTER the main results were read (not part of the pre-set protocol): the same 20 scenarios with a
    longer horizon, to see how much of the 24 h 'saved' value is a truncation effect. Stored under
    results["posthoc"]["horizon_<h>"] in the JSON at `path`."""
    t0 = time.perf_counter()
    with open(path) as f:
        data = json.load(f)
    scens = data["scenarios"]
    crews_list = sorted({r["crews"] for r in data["results"]})
    tasks = [dict(scenario=s, crews=c, exact_limit=exact_limit, variant=f"horizon {horizon_h:g} h", horizon_h=horizon_h)
             for c in crews_list for s in scens]
    rs = run_tasks(tasks, workers)
    data.setdefault("posthoc", {})[f"horizon_{horizon_h:g}"] = dict(
        horizon_h=horizon_h, exact_limit_s=exact_limit, results=rs,
        summary={str(c): summarize([r for r in rs if r["crews"] == c]) for c in crews_list})
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    log(f"post-hoc horizon {horizon_h:g} h done in {time.perf_counter() - t0:.0f} s")
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exact-limit", type=float, default=30.0)
    ap.add_argument("--sens-n", type=int, default=20)
    ap.add_argument("--sens-limit", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--tables", action="store_true", help="print the README markdown tables from --out and exit")
    ap.add_argument("--posthoc-horizon", type=float, default=None,
                    help="add a longer-horizon rerun of the same scenarios to the existing --out file and exit")
    a = ap.parse_args(argv)
    if a.tables:
        with open(a.out) as f:
            print(markdown_tables(json.load(f)))
        return
    if a.posthoc_horizon:
        run_posthoc_horizon(a.out, a.posthoc_horizon, a.sens_limit, a.workers)
        return
    data = run_experiment(a.n, a.seed, a.exact_limit, a.sens_n, a.sens_limit, a.workers)
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KiB), total {data['total_runtime_s']:.0f} s")
    for c, s in data["summary"].items():
        print(f"crews={c}: " + ", ".join(f"{m} mean {v['mean_pct_saved']:.1f}%" for m, v in s["methods"].items()))


if __name__ == "__main__":
    main()
