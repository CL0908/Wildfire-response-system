#!/usr/bin/env python
"""Write dashboards/public/ranger.json: ranger-response scenarios on the real Khao Yai 20 x 20 window.

Runs no response / greedy / exact for each scenario in SCENARIOS (same code as `python -m
ranger_response.run`) and writes flat, rounded arrays the dashboard can draw. Cells are flat indices
r * cols + c on the same window as results.json (row 0 is the north edge). Run from the repo root:

    python -m ranger_response.export_dashboard [--time-limit 60] [--out path]

The exact solver is time limited, so a run that is not proven optimal can give a slightly different
best-found plan on a slower or faster machine; `proven_optimal` and `stop_reason` say which case it is.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np

from priority_surface import real_window_bounds

from .model import Plan, simulate
from .run import GRID, make_instance_for
from .solvers import solve_exact, solve_greedy

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "dashboards", "public", "ranger.json")

# Scenario definitions; `ignition` None = default (real cell with the most FIRMS detections).
SCENARIOS = [
    dict(id="hazard-1crew", label="One crew, default ignition (most FIRMS detections), fuel = hazard >= 0.55",
         fuel_rule="hazard", crews=1),
    dict(id="hazard-2crew", label="Two crews, default ignition, fuel = hazard >= 0.55", fuel_rule="hazard", crews=2),
    dict(id="hazard-1crew-ig2", label="One crew, 2nd most-detected ignition, fuel = hazard >= 0.55",
         fuel_rule="hazard", crews=1, ignition=[(12, 4)]),
    dict(id="hazard-1crew-ig3", label="One crew, 3rd ignition, fuel = hazard >= 0.55",
         fuel_rule="hazard", crews=1, ignition=[(18, 6)]),
    dict(id="hazard-1crew-ig4", label="One crew, 4th ignition, fuel = hazard >= 0.55",
         fuel_rule="hazard", crews=1, ignition=[(16, 12)]),
    dict(id="landcover-1crew", label="One crew, default ignition, fuel = every forest or crop cell (land cover)",
         fuel_rule="landcover", crews=1),
]

CAVEATS = [
    "This is a simplified model on real terrain, not a prediction of a real fire or a tool rangers could use as is.",
    "Fire spreads deterministically between the four neighbouring fuel cells: no wind, no spotting, no fuel moisture, no randomness.",
    "A step is a unitless time step, not minutes. Spread takes 1 step per cell, or 2 steps downhill across a drop of more than 50 m.",
    "Crews walk in a straight line over the grid at a fixed speed, ignoring roads, trails, slope and the fire itself.",
    "A firebreak is a single 500 m cell that holds only if it is finished before the fire reaches it.",
    "A held firebreak blocks spread through its own cell, but the fire can still go around it through any neighbouring fuel cell.",
    "Which cells count as fuel is an assumption: 'hazard' treats cells below a hazard cut-off as unable to carry fire, "
    "'landcover' treats every forest or crop cell as fuel. The size of the gain from planning depends mostly on this choice.",
    "The value saved is the sum of the project's priority score over fuel cells that never burn; the score is a weighted model, not a measured loss.",
    "The ignition is the fuel cell with the most NASA FIRMS detections in 2012 to 2025; real fires start elsewhere too.",
    "The station is the middle of the window edge nearest the ignition, not a real ranger station.",
]


def _steps(arr) -> list:
    """Per-cell step as a flat list of ints, null for never."""
    return [int(v) if np.isfinite(v) else None for v in np.asarray(arr, dtype=float).ravel()]


def _plan_json(res) -> list:
    return [dict(crew=d.crew, order=d.order, cell=[int(d.cell[0]), int(d.cell[1])], arrive=int(d.arrive),
                 finish=int(d.finish),
                 fire_arrival=int(d.fire_arrival) if np.isfinite(d.fire_arrival) else None, held=bool(d.held))
            for d in res.defences]


def _method(name, solved, res, total, proven, stop, elapsed, nodes):
    return dict(name=name, value_saved=round(float(res.value_saved), 4), cells_burned=int(res.cells_burned),
                total_value=round(float(total), 4), proven_optimal=proven, stop_reason=stop or None,
                elapsed_s=round(float(elapsed), 2), nodes=nodes, burn_step=_steps(res.burn_time),
                defences=_plan_json(res) if solved else [])


def run_scenario(spec: dict, time_limit: float, node_limit: int = 300_000) -> dict:
    surf, inst = make_instance_for("real", spec.get("fuel_rule", "hazard"), spec.get("fuel_threshold", 0.55),
                                   ignitions=spec.get("ignition"), stations=spec.get("station"),
                                   crews=spec.get("crews", 1), speed=spec.get("speed", 3.0),
                                   work_time=spec.get("work_time", 1))
    total = float(inst.value[inst.fuel].sum())
    base = simulate(inst, Plan.empty(inst.n_crews))
    greedy = solve_greedy(inst)
    exact = solve_exact(inst, node_limit=node_limit, time_limit=time_limit)
    methods = [_method("no response", False, base, total, None, "", 0.0, 0),
               _method("greedy", True, greedy.result, total, None, greedy.stop_reason, greedy.elapsed, greedy.nodes),
               _method("exact", True, exact.result, total, exact.proven_optimal, exact.stop_reason,
                       exact.elapsed, exact.nodes)]
    best = "exact" if exact.objective >= greedy.objective else "greedy"
    return dict(
        id=spec["id"], label=spec["label"],
        fuel_rule=inst.params["fuel_rule"],
        fuel_threshold=inst.params["fuel_threshold"] if inst.params["fuel_rule"] == "hazard" else None,
        n_fuel=inst.n_fuel,
        fuel=[int(v) for v in inst.fuel.ravel()],
        ignitions=[list(c) for c in inst.ignitions],
        stations=[list(c) for c in inst.stations],
        params=dict(crews=inst.n_crews, speed=inst.speed, work_time=inst.work_time,
                    downhill_drop_m=inst.params["downhill_drop_m"], downhill_time=inst.params["downhill_time"],
                    exact_time_limit_s=time_limit, exact_node_limit=node_limit),
        best=best,
        methods=methods,
    )


def build(time_limit: float, scenarios=None) -> dict:
    scenarios = SCENARIOS if scenarios is None else scenarios
    surf, inst0 = make_instance_for("real")
    return dict(
        generated=date.today().isoformat(),
        terrain="real",
        grid=dict(rows=GRID, cols=GRID, cell_size_m=500, window=list(surf["window"]),
                  bounds=real_window_bounds(surf)),
        value=[round(float(v), 3) for v in inst0.value.ravel()],
        note="Same 20 x 20 window as results.json; flat arrays are row-major, row 0 is the north edge. "
             "value is the priority score per cell (the same for every scenario). burn_step is the time step a cell ignites, null if it never burns.",
        caveats=CAVEATS,
        scenarios=[run_scenario(s, time_limit) for s in scenarios],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds for each exact solve")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    data = build(a.time_limit)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KiB)")
    for s in data["scenarios"]:
        for m in s["methods"]:
            print(f"{s['id']:<18}{m['name']:<12}saved {m['value_saved']:>7.2f} burned {m['cells_burned']:>4} "
                  f"proven {m['proven_optimal']}")


if __name__ == "__main__":
    main()
