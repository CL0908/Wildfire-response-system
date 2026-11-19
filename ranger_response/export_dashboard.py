#!/usr/bin/env python
"""Write dashboards/public/ranger.json (model v2) from the saved experiment results.

    python -m ranger_response.experiment          # writes ranger_response/experiment_results.json (about 15 min)
    python -m ranger_response.export_dashboard    # writes dashboards/public/ranger.json (seconds)

The exporter runs no solver. It takes the plans stored in experiment_results.json, replays them through
model2.simulate2 on the real 20 x 20 window (a check that the stored numbers are reproduced), and writes
per-cell arrival times for the featured scenarios plus the statistics over all scenarios.

Featured scenarios are chosen by these rules, fixed before the results were looked at:
  most-fire-prone  the sampled scenario whose ignition cell has the highest smoothed FIRMS fire-history
                   weight (ties: lowest scenario id);
  median-gap       the scenario whose (exact - greedy) gap, one crew, as a share of the value that burns
                   with no response, is the median of the sampled scenarios (lower median; ties: lowest id);
  largest-gap      the scenario with the largest such gap (ties: lowest id). Shown so the upper end of the
                   distribution is visible and labelled as such; it is not typical.
One scenario can hold several roles. Each is exported for one and for two crews.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np

from priority_surface import real_window_bounds

from .experiment import BASE_CONFIG, CREW_SIZE, CUT_RATE_M_MIN_PER_FF, RESULTS_PATH, SENSITIVITY, build_instance
from .model import Plan
from .model2 import simulate2
from .run import GRID

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "dashboards", "public", "ranger.json")
METHODS = ("greedy", "lookahead", "exact")

CAVEATS = [
    "This is a simplified model on real terrain. It has not been calibrated or validated against any observed fire and is not a tool rangers could use as it is.",
    "Fire spread is the sensor-analysis spread model (minimum travel time; land-cover fuel factors, slope factor, one constant wind per scenario). The base rate r0 = 2 m/min is an assumption within published Thai deciduous-forest burns; no rate for evergreen forest in Khao Yai was found. Spread times scale directly with r0.",
    "All 400 cells of this window are land-cover burnable, so the earlier arbitrary 'fuel = hazard above a cut-off' rule is gone; the gain from firebreaks no longer depends on it.",
    "Only the first 24 hours after ignition are modelled. Value 'saved' means not reached within 24 h; a held line that the fire would reach on day two is counted as saved.",
    "Crews start at the middle of the window edge nearest the ignition at the moment of ignition (no detection or dispatch delay, no roads), and walk in straight lines at 2 km/h (an assumption). Two crews start from the same station.",
    "Cutting a 500 m line takes 5.2 h by default: 10 people at 0.16 m per minute each (indirect attack in timber, southern Spain, 2014 to 2019 field data). The crew size is an assumption and rates vary by more than a factor of 6 between fuel types and attack modes; see the sensitivity table.",
    "A firebreak is one 500 m cell and holds only if it is finished before the fire arrives. The fire may go around it through neighbouring cells; a fire cannot slip diagonally between two firebreak cells that touch at a corner.",
    "No spotting, no fuel moisture, no fire suppression by the fire itself, no crew rest, no helicopters or engines, and no limit on the number of cells cut other than crew time.",
    "Value is the project's priority score, a weighted model and not a measured loss.",
    "The exact solver is time limited; a result with proven_optimal false is the best plan found, not a proven optimum.",
    "Twenty sampled scenarios are a small sample: the percentages have wide uncertainty and the ranking of methods could change with more scenarios.",
]


def _arrival(res) -> list:
    return [round(float(v), 2) if np.isfinite(v) else None for v in res.burn_time.ravel()]


def _plan_defences(res) -> list:
    return [dict(crew=d.crew, order=d.order, cell=[int(d.cell[0]), int(d.cell[1])], arrive_h=round(d.arrive, 2),
                 finish_h=round(d.finish, 2),
                 fire_arrival_h=round(d.fire_arrival, 2) if np.isfinite(d.fire_arrival) else None,
                 held=bool(d.held)) for d in res.defences]


def pick_featured(scens: list, results: list) -> dict:
    """Role -> scenario id under the rules in the module docstring (needs the 1-crew results)."""
    one = {r["scenario"]: r for r in results if r["crews"] == 1}
    gaps = {i: r["methods"]["exact"]["pct_saved"] - r["methods"]["greedy"]["pct_saved"] for i, r in one.items()}
    ids = sorted(gaps)
    fire_prone = min(scens, key=lambda s: (-s["ignition_weight"], s["id"]))["id"]
    ordered = sorted(ids, key=lambda i: (gaps[i], i))
    median_gap = ordered[(len(ordered) - 1) // 2]
    largest = min(ids, key=lambda i: (-gaps[i], i))
    return {"most-fire-prone": fire_prone, "median-gap": median_gap, "largest-gap": largest}


def _scenario_export(scen: dict, rec: dict, crews: int, cfg: dict) -> dict:
    inst = build_instance(scen, crews, cfg)
    base = simulate2(inst, Plan.empty(crews))
    out = dict(crews=crews, stations=[list(c) for c in inst.stations],
               baseline=dict(cells_burned=int(base.cells_burned), value_burned=round(rec["baseline_value_burned"], 4),
                             total_value=round(rec["total_value"], 4), arrival_h=_arrival(base)),
               methods={})
    best, best_obj = None, -np.inf
    for m in METHODS:
        mr = rec["methods"][m]
        res = simulate2(inst, Plan.of([[tuple(c) for c in route] for route in mr["plan"]]))
        if abs(res.value_saved - mr["value_saved"]) > 1e-6 or res.cells_burned != mr["cells_burned"]:
            raise RuntimeError(f"stored result for scenario {scen['id']} {m} {crews} crews does not replay "
                               f"({res.value_saved} vs {mr['value_saved']})")
        out["methods"][m] = dict(value_saved=round(res.value_saved, 4), pct_saved=round(mr["pct_saved"], 2),
                                 cells_burned=int(res.cells_burned), proven_optimal=mr["proven_optimal"],
                                 stop_reason=mr["stop_reason"], elapsed_s=round(mr["elapsed_s"], 2),
                                 defences=_plan_defences(res), arrival_h=_arrival(res))
        if res.objective > best_obj + 1e-9:        # first (simplest) method reaching the maximum wins ties
            best, best_obj = m, res.objective
    out["best"] = best
    return out


def _round_summary(s: dict) -> dict:
    def rec(x):
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        if isinstance(x, float):
            return round(x, 3)
        return x
    return rec(s)


def build(results_path: str = RESULTS_PATH) -> dict:
    with open(results_path) as f:
        exp = json.load(f)
    cfg = dict(BASE_CONFIG, n_scenarios=exp["config"]["n_scenarios"], seed=exp["config"]["seed"])
    scens, results = exp["scenarios"], exp["results"]
    by_key = {(r["scenario"], r["crews"]): r for r in results}
    crews_list = sorted({r["crews"] for r in results})
    roles = pick_featured(scens, results)
    ids = sorted(set(roles.values()))

    from .experiment import site_inputs
    surf = site_inputs()["surf"]

    per_scenario = []
    for s in scens:
        row = dict(id=s["id"], ignition=s["ignition"], wind_speed_ms=round(s["wind_speed_ms"], 2),
                   wind_from_deg=round(s["wind_from_deg"], 1), fire_count=s["fire_count"], crews={})
        for c in crews_list:
            r = by_key[(s["id"], c)]
            row["crews"][str(c)] = dict(
                baseline_cells_burned=r["baseline_cells_burned"], baseline_value_burned=round(r["baseline_value_burned"], 3),
                **{m: dict(pct_saved=round(r["methods"][m]["pct_saved"], 2), value_saved=round(r["methods"][m]["value_saved"], 3),
                           cells_burned=r["methods"][m]["cells_burned"], proven_optimal=r["methods"][m]["proven_optimal"],
                           elapsed_s=round(r["methods"][m]["elapsed_s"], 2)) for m in METHODS})
        per_scenario.append(row)

    featured = []
    for sid in ids:
        scen = scens[sid]
        featured.append(dict(
            scenario=sid, roles=[k for k, v in roles.items() if v == sid], ignition=scen["ignition"],
            wind_speed_ms=round(scen["wind_speed_ms"], 2), wind_from_deg=round(scen["wind_from_deg"], 1),
            fire_count=scen["fire_count"],
            runs=[_scenario_export(scen, by_key[(sid, c)], c, cfg) for c in crews_list]))

    c = exp["config"]
    params = {
        "horizon_h": dict(value=cfg["horizon_h"], unit="h", source="fire_sim.DEFAULT_HORIZON_HOURS; design choice"),
        "r0_m_min": dict(value=cfg["r0_m_min"], unit="m/min at fuel factor 1",
                         source="assumption inside 0.44-2.55 m/min from Thai deciduous-forest burns (Junpen et al. 2013, ScienceAsia 39:535; "
                                "Wiriya and Kaitpraneet 2009; Himmapan and Kaitpraneet 2006), as cited in detection/fire_sim.py; not calibrated for Khao Yai"),
        "fuel_factors": dict(value=dict(grass=1.0, shrub=0.9, tree=0.6, crop=0.5, bare=0.1), unit="relative spread multiplier",
                             source="detection/fire_sim.FUEL_FACTORS (assumption)"),
        "slope": dict(value="exp(3.93 * slope), clipped to 0.5-4", unit="factor",
                      source="detection/fire_sim: forward spread doubles per 10 degrees uphill (McArthur rule of thumb)"),
        "wind": dict(value="exp(0.12 * min(U, 15 m/s) * cos(angle))", unit="factor",
                     source="detection/fire_sim (assumption); one constant wind per scenario drawn from the daytime (10:00 to 18:00 UTC+7) fire-season hourly record in data/wind_fire_season.npz"),
        "walk_kmh": dict(value=cfg["walk_kmh"], unit="km/h", source="assumption (slow, loaded walk through forest); straight lines between cell centres"),
        "crew_size": dict(value=CREW_SIZE, unit="people", source="assumption"),
        "cut_rate_m_min_per_person": dict(value=CUT_RATE_M_MIN_PER_FF, unit="m/min/person",
                                          source="mean indirect-attack handline rate in timber fuels, Int. J. Wildland Fire 32(11):1503, 'Fireline production rate of handcrews in wildfires' (southern Spain wildfires 2014-2019). Direct attack in timber understory 0.44, timber litter 1.06"),
        "cut_h_per_cell": dict(value=round(c["cut_h"], 3), unit="h per 500 m cell", source="500 m / (crew_size * rate * 60)"),
        "dispatch_delay_h": dict(value=cfg["delay_h"], unit="h", source="assumption: crews leave at the moment of ignition"),
        "station": dict(value="middle of the window edge nearest the ignition", unit="cell", source="rule carried over from model v1; no road or ranger-station data in the project"),
        "exact_time_limit_s": dict(value=c["exact_limit_s"], unit="s per instance", source="experiment setting"),
        "n_scenarios": dict(value=c["n_scenarios"], unit="scenarios", source=f"ignition cells ~ smoothed FIRMS 2012-2025 history; seed {c['seed']}"),
    }

    return dict(
        model="v2", generated=date.today().isoformat(), terrain="real",
        grid=dict(rows=GRID, cols=GRID, cell_size_m=500, window=list(surf["window"]), bounds=real_window_bounds(surf)),
        value=[round(float(v), 3) for v in surf["priority"].ravel()],
        note="Same 20 x 20 window as results.json; flat arrays are row-major (index = row * 20 + col), row 0 is the north edge. "
             "arrival_h is the hour the fire reaches a cell within the 24 h horizon, null if it does not. value is the priority score per cell. "
             "pct_saved is value saved as a percentage of the value that burns within 24 h with no response.",
        parameters=params,
        experiment=dict(
            protocol="Twenty scenarios, ignition cells sampled in proportion to the smoothed FIRMS fire history inside the window, one wind per scenario drawn from the daytime fire-season record, fixed seed. "
                     "Each scenario is solved with one and with two crews by greedy, greedy with 2-step lookahead, and exact branch and bound (warm-started from the better of the two heuristics).",
            n_scenarios=c["n_scenarios"], seed=c["seed"],
            summary={k: _round_summary(v) for k, v in exp["summary"].items()},
            per_scenario=per_scenario,
            sensitivity_scenarios=c.get("sensitivity_scenarios"),
            sensitivity={name: {k: _round_summary(v) for k, v in by.items()} for name, by in exp["sensitivity_summary"].items()},
            sensitivity_note="Same 20 scenarios; exact time limit 20 s instead of 30 s. 'default (same subset)' is the main experiment.",
            posthoc_horizon={k: dict(horizon_h=v["horizon_h"], exact_limit_s=v["exact_limit_s"],
                                     summary={c: _round_summary(sm) for c, sm in v["summary"].items()})
                             for k, v in exp.get("posthoc", {}).items()},
            posthoc_note="Added after the main results had been read, not part of the protocol: the same scenarios with a longer horizon, to show how much of the 24 h 'saved' value is a truncation effect.",
        ),
        featured_rule="most-fire-prone: highest smoothed fire-history weight at the ignition cell among the sampled scenarios; "
                      "median-gap: median (exact - greedy) gap with one crew; largest-gap: largest such gap (not typical). Fixed before the results were seen.",
        featured=featured,
        caveats=CAVEATS,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    data = build(a.results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KiB)")
    for fs in data["featured"]:
        for r in fs["runs"]:
            print(f"scenario {fs['scenario']} {fs['roles']} crews {r['crews']}: best {r['best']}, " +
                  ", ".join(f"{m} {v['pct_saved']}%" for m, v in r["methods"].items()))


if __name__ == "__main__":
    main()
