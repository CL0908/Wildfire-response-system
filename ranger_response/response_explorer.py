"""Click-to-respond precompute for the site's "Fire response" mode (python side only).

For every cell of the 20 x 20 window as an ignition, with one representative wind, it works out
  1. which sensor of the site's optimal placement sees the fire first and when (no-response arrival
     times, detection rule of detection/objective.py: a sensor detects the moment the fire enters a
     cell it sees; the satellite detects 6 h after ignition; detection is the earlier of the two),
  2. crews dispatched DISPATCH_DELAY_H after detection (model v2 `start_h`) and the lookahead plan,
  3. for detection by sensors and by satellite, with one and two crews.
Everything is written to dashboards/public/response.json and response_arrival.bin (see export()).

    python -m ranger_response.response_explorer --measure          # time a few ignitions, no output
    python -m ranger_response.response_explorer                    # full precompute + export
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from datetime import date

import numpy as np

from detection.fire_sim import daytime_wind_pool, load_wind
from detection.objective import DEFAULT_BASELINE_MIN

from .experiment import BASE_CONFIG, CREW_SIZE, CUT_RATE_M_MIN_PER_FF, cut_hours, site_inputs
from .model import Plan
from .model2 import make_instance2, simulate2
from .run import GRID, default_station
from .solvers2 import solve_exact2, solve_greedy2, solve_lookahead2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUBLIC = os.path.join(ROOT, "dashboards", "public")
RESULTS_JSON = os.path.join(PUBLIC, "results.json")
OUT_JSON = os.path.join(PUBLIC, "response.json")
OUT_BIN = os.path.join(PUBLIC, "response_arrival.bin")

SATELLITE_H = DEFAULT_BASELINE_MIN / 60.0     # 6 h, detection/objective.DEFAULT_BASELINE_MIN (assumption there)
DISPATCH_DELAY_H = 0.5                        # assumption of this task: time from detection to crews leaving
MODES = ("sensors", "satellite")
CREWS = (1, 2)
# order of the five grids stored per ignition in response_arrival.bin
PLANS = (("no_response", None, None), ("sensors", "sensors", 1), ("sensors", "sensors", 2),
         ("satellite", "satellite", 1), ("satellite", "satellite", 2))
PLAN_KEYS = ("no_response", "sensors_1", "sensors_2", "satellite_1", "satellite_2")
STEP_H = 0.1
NOT_REACHED = 255
N = GRID * GRID


# ------------------------------------------------------------------------------ inputs

def load_site(results_path: str = RESULTS_JSON) -> dict:
    """Sensor placement shown on the site: the exact (ILP) optimum, its sites and viewsheds (window cells)."""
    with open(results_path) as f:
        res = json.load(f)
    ilp = next(m for m in res["methods"] if m["name"] == "ILP")
    chosen = sorted(int(i) for i in ilp["chosen"])
    if abs(ilp["objective"] - res["optimum"]) > 1e-9:
        raise ValueError("the ILP placement is not the optimum recorded in results.json")
    sites = {int(s["id"]): (int(s["row"]), int(s["col"])) for s in res["sites"]}
    sees = {j: np.array(sorted(int(c) for c in res["viewsheds"][str(j)]), dtype=int) for j in chosen}
    return dict(chosen=chosen, sites=sites, sees=sees, objective=float(ilp["objective"]))


def representative_wind() -> dict:
    """Median speed and circular-mean direction of the daytime (10:00-18:00 UTC+7) fire-season hourly
    wind pool that detection/fire_sim draws scenario winds from (data/wind_fire_season.npz)."""
    sp, dr = daytime_wind_pool(load_wind())
    a = np.radians(dr)
    s_, c_ = np.sin(a).mean(), np.cos(a).mean()
    mean_dir = float(np.degrees(np.arctan2(s_, c_)) % 360.0)
    sectors, _ = np.histogram(dr, bins=np.arange(0, 361, 45))
    k = int(np.argmax(sectors))
    return dict(speed_ms=round(float(np.median(sp)), 2), from_deg=round(mean_dir, 1),
                n_hours=int(sp.size), mean_resultant_length=round(float(np.hypot(s_, c_)), 3),
                modal_45deg_sector_from=[45 * k, 45 * (k + 1)], modal_sector_share=round(float(sectors[k] / sp.size), 3),
                mean_speed_ms=round(float(sp.mean()), 2))


def base_instance(wind: dict):
    s = site_inputs()
    cfg = BASE_CONFIG
    ig = (GRID // 2, GRID // 2)
    inst = make_instance2(s["fuel_factor"], s["value"], s["dem"], [ig], [default_station(GRID, GRID, ig)],
                          wind["speed_ms"], wind["from_deg"], r0_m_min=cfg["r0_m_min"], horizon_h=cfg["horizon_h"],
                          walk_kmh=cfg["walk_kmh"], cut_h=cut_hours(cfg), delay_h=0.0)
    inst.params.update(terrain="real")
    return inst


def instance_for(base, ignition, crews: int, start_h: float):
    st = default_station(GRID, GRID, ignition)
    return base.with_params(ignitions=[ignition], stations=[st] * crews, start_h=start_h)


def detection(base, site: dict, ignition, arrival_flat=None) -> dict:
    """Detection of one ignition by the placement, from the no-response arrival times.

    sensor_h  = earliest arrival over cells seen by a chosen site, per site and overall (inf if none reached).
    detection_h (sensors mode) = min(SATELLITE_H, sensor_h), as detection/objective.py; the detecting
    site is the one with the smallest time (tie: lowest site id), or "satellite" if sensor_h >= SATELLITE_H.
    """
    if arrival_flat is None:
        arrival_flat = simulate2(instance_for(base, ignition, 1, 0.0), Plan.empty(1)).burn_flat
    arr = np.asarray(arrival_flat, dtype=float)
    per_site = {j: float(arr[cells].min()) for j, cells in site["sees"].items()}
    best_t, best_j = min((t, j) for j, t in per_site.items())
    out = dict(per_site_h=per_site, sensor_h=best_t if np.isfinite(best_t) else None)
    if best_t < SATELLITE_H:
        out.update(detection_h=best_t, detecting_site=int(best_j))
    else:
        out.update(detection_h=SATELLITE_H, detecting_site="satellite")
    return out


# ------------------------------------------------------------------------------ one ignition

def solve_ignition(task: dict) -> dict:
    """All four plans (mode x crews) plus the no-response run for one ignition cell."""
    ignition = tuple(task["ignition"])
    solver = {"lookahead": solve_lookahead2, "greedy": solve_greedy2}[task.get("solver", "lookahead")]
    base, site = _worker_state(task)
    inst0 = instance_for(base, ignition, 1, 0.0)
    nores = simulate2(inst0, Plan.empty(1))
    det = detection(base, site, ignition, nores.burn_flat)
    total = float(inst0.value[inst0.burnable].sum())
    base_val_burned = total - nores.objective
    modes = {"sensors": dict(detection_h=det["detection_h"], detecting_site=det["detecting_site"]),
             "satellite": dict(detection_h=SATELLITE_H, detecting_site="satellite")}
    out = dict(ignition=list(ignition), station=list(inst0.stations[0]), sensor_h=det["sensor_h"],
               baseline=dict(cells_burned=int(nores.cells_burned), value_burned=base_val_burned),
               arrival={"no_response": nores.burn_time.ravel().copy()}, modes={}, runs={})
    sat_plans = {}
    for mode in ("satellite", "sensors"):        # satellite first: its plans are reused as candidates for sensors
        start = modes[mode]["detection_h"] + DISPATCH_DELAY_H
        modes[mode]["start_h"] = start
        out["modes"][mode] = dict(modes[mode])
        for crews in CREWS:
            inst = instance_for(base, ignition, crews, start)
            t0 = time.perf_counter()
            sr = solver(inst)
            if mode == "satellite":
                sat_plans[crews] = sr.plan
            else:
                # Crews that leave earlier can follow the later-start plan unchanged (every firebreak then
                # finishes earlier, which can only keep it holding and only delay the fire), so the sensors
                # result is never worse than the satellite result. The heuristic does not guarantee that
                # by itself, hence this extra candidate.
                alt = simulate2(inst, sat_plans[crews])
                if alt.objective > sr.result.objective + 1e-9:
                    sr = type(sr)(sr.method, sat_plans[crews], alt, None, sr.nodes, sr.elapsed, "later-start plan")
            el = time.perf_counter() - t0
            r = sr.result
            out["arrival"][f"{mode}_{crews}"] = r.burn_time.ravel().copy()
            out["runs"][f"{mode}_{crews}"] = dict(
                value_saved=float(r.value_saved), cells_burned=int(r.cells_burned), elapsed_s=el,
                solver=sr.method, reused_later_start_plan=sr.stop_reason == "later-start plan", plan=[[list(map(int, c)) for c in route] for route in sr.plan.routes],
                defences=[dict(crew=d.crew, order=d.order, cell=[int(d.cell[0]), int(d.cell[1])],
                               arrive_h=float(d.arrive), finish_h=float(d.finish),
                               fire_arrival_h=float(d.fire_arrival) if np.isfinite(d.fire_arrival) else None,
                               held=bool(d.held)) for d in r.defences])
    return out


_STATE = {}


def _worker_state(task):
    key = "s"
    if key not in _STATE:
        wind = task["wind"]
        _STATE[key] = (base_instance(wind), load_site())
    return _STATE[key]


def sanity_exact(wind, n=10, seed=2024, time_limit=20.0):
    """Lookahead vs exact (one crew, sensors mode) on `n` random ignitions."""
    base, site = base_instance(wind), load_site()
    rng = random.Random(seed)
    cells = [divmod(i, GRID) for i in rng.sample(range(N), n)]
    rows = []
    for ig in cells:
        det = detection(base, site, ig)
        inst = instance_for(base, ig, 1, det["detection_h"] + DISPATCH_DELAY_H)
        base_burned = float(inst.value[inst.burnable].sum()) - simulate2(inst, Plan.empty(1)).baseline_objective
        la = solve_lookahead2(inst)
        ex = solve_exact2(inst, time_limit=time_limit, warm_start=la.plan)
        rows.append(dict(ignition=list(ig), detection_h=det["detection_h"], start_h=inst.start_h,
                         baseline_value_burned=base_burned, lookahead_saved=la.result.value_saved,
                         exact_saved=ex.result.value_saved, proven_optimal=ex.proven_optimal,
                         lookahead_s=la.elapsed, exact_s=ex.elapsed, stop_reason=ex.stop_reason or None))
    return rows


CAVEATS = [
    "SIMULATED from a simplified, uncalibrated fire-spread model on real terrain; not a prediction of a real fire or a tool rangers could use as it is. Nothing here is validated against an observed fire.",
    "One representative wind for every ignition: the median speed and the circular-mean direction of the daytime fire-season hourly record. Real fires see changing wind; a different wind changes every number.",
    "Detection follows detection/objective.py: a sensor detects the moment the fire enters a cell it sees (no delay, no failures, no smoke threshold) and the satellite detects 6 h after ignition, an assumption in that module. Detection is the earlier of the two.",
    "Sensor viewsheds cover only the 20 x 20 window and fire spreads only inside it here, so a fire that would leave and re-enter the window, or be seen from outside it, is not modelled.",
    "Dispatch 0.5 h after detection is an assumption (alert handling, mustering, leaving the station); no data was found. Crews start from the middle of the window edge nearest the ignition and walk straight lines at 2 km/h.",
    "Crews cut a 500 m cell in 5.2 h (10 people at 0.16 m/min each, indirect attack in timber, southern Spain field data); a satellite-detected fire starts its response at 6.5 h, by which time most of the fire is already out of reach of a 5 h cut.",
    "The plans are from a heuristic (greedy with a 2-step lookahead), not proven optimal; on a random check the exact solver found more value in some ignitions (see sanity_check).",
    "Value is the project's priority score, not a measured loss. Value saved is value not reached within 24 h. Percentages are of the value that burns in 24 h with no response.",
    "Plans are deterministic and some rely on firebreaks finished minutes before the modelled fire; real fire arrival times are uncertain by far more than that.",
]


def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return dict(mean=round(float(x.mean()), 4), median=round(float(np.median(x)), 4),
                p10=round(float(np.percentile(x, 10)), 4), p90=round(float(np.percentile(x, 90)), 4))


def quantise(arrival: np.ndarray) -> np.ndarray:
    """Hours -> uint8 steps of STEP_H, NOT_REACHED for inf. Rounded to nearest, so the error is at most 0.05 h."""
    q = np.full(arrival.shape, NOT_REACHED, dtype=np.uint8)
    ok = np.isfinite(arrival)
    q[ok] = np.rint(arrival[ok] / STEP_H).astype(np.uint8)
    return q


def decode_arrival(buf: bytes, ignition_index: int, plan_key: str) -> np.ndarray:
    """Inverse of the bin layout: float hours (nan = not reached) as a (20, 20) array."""
    k = PLAN_KEYS.index(plan_key)
    off = (ignition_index * len(PLAN_KEYS) + k) * N
    a = np.frombuffer(buf, dtype=np.uint8, count=N, offset=off).astype(float) * STEP_H
    a[np.frombuffer(buf, dtype=np.uint8, count=N, offset=off) == NOT_REACHED] = np.nan
    return a.reshape(GRID, GRID)


def build_export(results: list, wind: dict, site: dict, solver: str, check: list, weights: np.ndarray):
    results = sorted(results, key=lambda r: r["ignition"][0] * GRID + r["ignition"][1])
    assert [r["ignition"][0] * GRID + r["ignition"][1] for r in results] == list(range(N))
    cfg = BASE_CONFIG
    buf = bytearray()
    ignitions = []
    for i, r in enumerate(results):
        for key in PLAN_KEYS:
            buf += quantise(r["arrival"][key]).tobytes()
        vb = r["baseline"]["value_burned"]
        plans = {}
        for key, run in r["runs"].items():
            plans[key] = dict(value_saved=round(run["value_saved"], 4), cells_burned=run["cells_burned"],
                              pct_saved=round(100.0 * run["value_saved"] / vb, 2) if vb > 1e-12 else 0.0,
                              defences=[dict(crew=d["crew"], order=d["order"], cell=d["cell"], arrive_h=round(d["arrive_h"], 2),
                                             finish_h=round(d["finish_h"], 2),
                                             fire_arrival_h=None if d["fire_arrival_h"] is None else round(d["fire_arrival_h"], 2),
                                             held=d["held"]) for d in run["defences"]])
        modes = {m: dict(detection_h=round(v["detection_h"], 3), detecting_site=v["detecting_site"], start_h=round(v["start_h"], 3))
                 for m, v in r["modes"].items()}
        ignitions.append(dict(id=i, cell=r["ignition"], station=r["station"],
                              sensor_first_h=None if r["sensor_h"] is None else round(r["sensor_h"], 3),
                              baseline=dict(cells_burned=r["baseline"]["cells_burned"], value_burned=round(vb, 4)),
                              modes=modes, plans=plans))

    def col(mode, crews, field):
        return np.array([r["runs"][f"{mode}_{crews}"][field] for r in results], dtype=float)

    w = weights / weights.sum()
    agg = dict(n_ignitions=N, weighting="uniform over the 400 cells unless named fire_history_weighted "
                                       "(weights = smoothed FIRMS 2012-2025 history, as detection/)")
    sens = np.array([r["modes"]["sensors"]["detection_h"] for r in results])
    site_ids = [r["modes"]["sensors"]["detecting_site"] for r in results]
    by_sensor = sorted({s for s in site_ids if s != "satellite"})
    by_sensor_a = np.array([s != "satellite" for s in site_ids])
    vb = np.array([r["baseline"]["value_burned"] for r in results])
    agg["baseline"] = dict(cells_burned=_stats([r["baseline"]["cells_burned"] for r in results]),
                           value_burned=_stats(vb), total_value=round(float(sum(0 for _ in [])) or 0, 1))
    agg["detection"] = dict(
        sensors_detect_share=round(float(by_sensor_a.mean()), 4),
        sensors_detect_share_fire_history_weighted=round(float(w @ by_sensor_a), 4),
        detected_at_ignition_share=round(float((sens <= 0).mean()), 4),
        detected_by_sensor_after_spread_share=round(float(((sens > 0) & by_sensor_a).mean()), 4),
        left_to_satellite_share=round(float((~by_sensor_a).mean()), 4),
        detection_h_sensors=_stats(sens), detection_h_sensors_fire_history_weighted_mean=round(float(w @ sens), 4),
        detection_h_sensors_given_sensor_detects=_stats(sens[by_sensor_a]),
        detection_h_satellite=SATELLITE_H,
        detecting_site_counts={str(s): int(sum(1 for x in site_ids if x == s)) for s in by_sensor})
    for crews in CREWS:
        d = {}
        for mode in MODES:
            vs, cb = col(mode, crews, "value_saved"), col(mode, crews, "cells_burned")
            pct = np.where(vb > 1e-12, 100.0 * vs / np.maximum(vb, 1e-12), 0.0)
            d[mode] = dict(value_saved=_stats(vs), cells_burned=_stats(cb), pct_of_burning_value_saved=_stats(pct),
                           value_saved_mean_fire_history_weighted=round(float(w @ vs), 4),
                           cells_burned_mean_fire_history_weighted=round(float(w @ cb), 4),
                           ignitions_with_any_value_saved=int((vs > 1e-9).sum()))
        vs_s, vs_t = col("sensors", crews, "value_saved"), col("satellite", crews, "value_saved")
        cs, ct = col("sensors", crews, "cells_burned"), col("satellite", crews, "cells_burned")
        d["sensors_minus_satellite"] = dict(
            value_saved_mean=round(float((vs_s - vs_t).mean()), 4), cells_burned_mean=round(float((cs - ct).mean()), 4),
            ignitions_sensors_better=int((vs_s > vs_t + 1e-9).sum()), equal=int((abs(vs_s - vs_t) <= 1e-9).sum()),
            ignitions_sensors_worse=int((vs_s < vs_t - 1e-9).sum()))
        agg[f"crews_{crews}"] = d
    agg["baseline"].pop("total_value")
    agg["baseline"]["total_value"] = round(float(site_inputs()["value"][site_inputs()["fuel_factor"] > 0].sum()), 4)

    parameters = {
        "horizon_h": dict(value=cfg["horizon_h"], unit="h", source="fire_sim.DEFAULT_HORIZON_HOURS"),
        "wind": dict(value=wind, unit="m/s and degrees the wind blows FROM (0 = north, 90 = east)",
                     source="detection/fire_sim.daytime_wind_pool(load_wind()): hourly fire-season record data/wind_fire_season.npz, local 10:00 up to 18:00 (UTC+7). "
                            "speed = median of the pool; direction = circular mean of the pool (the pool is not strongly directional: mean resultant length "
                            f"{wind['mean_resultant_length']}); the most frequent 45-degree sector is {wind['modal_45deg_sector_from'][0]}-{wind['modal_45deg_sector_from'][1]} degrees, which contains the circular mean"
                            if wind["modal_45deg_sector_from"][0] <= wind["from_deg"] < wind["modal_45deg_sector_from"][1] else
                            "speed = median of the pool; direction = circular mean of the pool"),
        "r0_m_min": dict(value=cfg["r0_m_min"], unit="m/min at fuel factor 1", source="detection/fire_sim.DEFAULT_R0_M_MIN (assumption within published Thai deciduous-forest burns; see ranger_response/README.md)"),
        "detection_rule": dict(value="a sensor detects the moment the fire enters any cell its viewshed contains; detection_h = min(6 h, earliest sensor time); the detecting site has the smallest time (tie: lowest site id), 'satellite' if no sensor time is below 6 h",
                               source="detection/objective.py (assumption 1 to 3) and detection/RESULTS.md"),
        "satellite_detection_h": dict(value=SATELLITE_H, unit="h after ignition", source="detection/objective.DEFAULT_BASELINE_MIN = 360 min; an assumption there (FIRMS overpass gap plus up to 3 h data latency)"),
        "dispatch_delay_h": dict(value=DISPATCH_DELAY_H, unit="h after detection", source="assumption of this analysis; crews leave their station at start_h = detection_h + 0.5"),
        "walk_kmh": dict(value=cfg["walk_kmh"], unit="km/h", source="ranger_response model v2 (assumption)"),
        "crew_size": dict(value=CREW_SIZE, unit="people", source="ranger_response model v2 (assumption)"),
        "cut_rate_m_min_per_person": dict(value=CUT_RATE_M_MIN_PER_FF, unit="m/min/person", source="Int. J. Wildland Fire 32(11):1503, mean indirect-attack handline rate in timber fuels (southern Spain, 2014-2019); see ranger_response/README.md"),
        "cut_h_per_cell": dict(value=round(cut_hours(cfg), 3), unit="h per 500 m cell", source="500 m / (crew_size * rate * 60)"),
        "station": dict(value="middle of the window edge nearest the ignition (ranger_response.run.default_station); two crews start from the same station",
                        source="ranger model v2 rule; no road or station data in the project"),
        "solver": dict(value=solver, source="ranger_response.solvers2; heuristic plan, not proven optimal"),
        "sensor_placement": dict(value=dict(sites=site["chosen"], cells={str(j): list(site["sites"][j]) for j in site["chosen"]}, objective=round(site["objective"], 4)),
                                 source="dashboards/public/results.json: the ILP method (exact optimum of hazard x value coverage, equal to the optimum shown on the site); viewsheds are results.json['viewsheds']"),
    }
    meta = dict(
        model="v2+start_h", generated=date.today().isoformat(), terrain="real",
        note="SIMULATED, simplified and uncalibrated. Heuristic plans, not proven optimal. Flat arrays are row-major (index = row * 20 + col), row 0 is the north edge; ignition id = row * 20 + col.",
        grid=dict(rows=GRID, cols=GRID, cell_size_m=500, horizon_h=cfg["horizon_h"]),
        arrival_bin=dict(
            file="response_arrival.bin", dtype="uint8", bytes=len(buf), step_h=STEP_H, not_reached=NOT_REACHED,
            layout="uint8[ignition 400][plan 5][cell 400], ignition-major, no header; offset = ((ignition_id * 5 + plan_index) * 400 + cell_index); "
                   "hours = value * 0.1, 255 = not reached within the 24 h horizon; round-to-nearest, error at most 0.05 h",
            plans=list(PLAN_KEYS), plan_notes="no_response: no firebreaks; <mode>_<crews>: arrival times with that mode's plan and number of crews"),
        parameters=parameters,
        sensors=dict(site_ids=site["chosen"], cells={str(j): list(site["sites"][j]) for j in site["chosen"]},
                     seen_cells={str(j): [int(c) for c in site["sees"][j]] for j in site["chosen"]}),
        assumptions=[
            "Detection by sensors: first arrival of the no-response fire in any cell seen by the placement (detection/objective.py rule); satellite: 6 h after ignition; sensors mode uses the earlier of the two.",
            "Crews leave 0.5 h after detection (assumption): start_h = detection_h + 0.5, so 0.5 h with sensors detecting at ignition and 6.5 h by satellite.",
            "One constant representative wind (median daytime fire-season speed, circular-mean direction).",
            "Crews walk at 2 km/h and cut a 500 m cell in 5.2 h (10 people at 0.16 m/min); one or two crews from the edge midpoint nearest the ignition.",
            "Plans from ranger_response.solvers2.solve_lookahead2: heuristic plan, not proven optimal."],
        ignition_fields="id, cell [row, col], station [row, col], sensor_first_h (earliest sensor arrival, null if never), baseline {cells_burned, value_burned} (no response), "
                        "modes {sensors|satellite: detection_h, detecting_site (site id or 'satellite'), start_h}, plans {sensors_1, sensors_2, satellite_1, satellite_2: "
                        "value_saved, cells_burned, pct_saved (of baseline.value_burned), defences [{crew, order, cell, arrive_h, finish_h, fire_arrival_h (null: never reached), held}]}",
        ignitions=ignitions, aggregate=agg,
        sanity_check=dict(
            description="Lookahead against the exact branch and bound (warm-started from the lookahead plan) on random ignitions, sensors mode, one crew",
            rows=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("lookahead_s", "exact_s")} for row in check]),
        caveats=CAVEATS)
    return meta, bytes(buf)


def export(meta: dict, buf: bytes, out_json: str = OUT_JSON, out_bin: str = OUT_BIN):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(buf)
    with open(out_json, "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    print(f"wrote {out_json} ({os.path.getsize(out_json) / 1024:.0f} KiB) and {out_bin} ({os.path.getsize(out_bin) / 1024:.0f} KiB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", type=int, nargs="?", const=8, default=None, help="time N random ignitions and exit")
    ap.add_argument("--solver", default="lookahead", choices=["lookahead", "greedy"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check-n", type=int, default=10)
    ap.add_argument("--check-limit", type=float, default=20.0, help="exact time limit (s) per sanity-check instance")
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-bin", default=OUT_BIN)
    a = ap.parse_args(argv)
    wind = representative_wind()
    print("wind", wind)
    if a.measure is not None:
        rng = random.Random(1)
        cells = [divmod(i, GRID) for i in rng.sample(range(N), a.measure)]
        t0 = time.perf_counter()
        for ig in cells:
            t1 = time.perf_counter()
            r = solve_ignition(dict(ignition=ig, wind=wind, solver=a.solver))
            print(ig, f"{time.perf_counter() - t1:.1f}s", {k: round(v['value_saved'], 2) for k, v in r["runs"].items()},
                  r["modes"]["sensors"]["detection_h"], r["baseline"]["cells_burned"], flush=True)
        print("total", time.perf_counter() - t0)
        return
    site = load_site()
    tasks = [dict(ignition=divmod(i, GRID), wind=wind, solver=a.solver) for i in range(N)]
    t0 = time.perf_counter()
    with mp.Pool(a.workers) as pool:
        results = pool.map(solve_ignition, tasks, chunksize=10)
    el = time.perf_counter() - t0
    run_s = [run["elapsed_s"] for r in results for run in r["runs"].values()]
    print(f"{len(run_s)} solver runs in {el:.1f} s wall ({a.workers} workers); per run mean {np.mean(run_s):.3f} s, max {np.max(run_s):.2f} s")
    t0 = time.perf_counter()
    check = sanity_exact(wind, a.check_n, time_limit=a.check_limit)
    print(f"sanity check {time.perf_counter() - t0:.1f} s")
    for row in check:
        print(row)
    weights = site_inputs()["window"]["ignition_weights"].ravel()
    meta, buf = build_export(results, wind, site, a.solver, check, weights)
    export(meta, buf, a.out_json, a.out_bin)


if __name__ == "__main__":
    main()
