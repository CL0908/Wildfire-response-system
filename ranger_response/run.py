"""Build the default 20x20 instance, run baseline / greedy / exact, print a table and save a figure.

    MPLBACKEND=Agg python -m ranger_response.run [--terrain real|synthetic] [--fuel-rule hazard|landcover]
                                                 [--seed 42] [--ignition r,c] [--station r,c]
                                                 [--crews 1] [--speed 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from priority_surface import build_priority_surface

from .model import DEFAULT_FUEL_THRESHOLD, Plan, build_instance, simulate
from .real import build_real_surface, ignitions_by_fire_history, real_fuel_mask
from .solvers import solve_exact, solve_greedy

GRID = 20
CELL_KM = 0.5


def parse_cell(text: str):
    r, c = text.split(",")
    return int(r), int(c)


def default_ignition(surf, fuel_threshold):
    """Highest-hazard fuel cell."""
    h = np.where(surf["hazard"] >= fuel_threshold, surf["hazard"], -np.inf)
    r, c = np.unravel_index(int(np.argmax(h)), h.shape)
    return int(r), int(c)


def default_station(rows, cols, ignition):
    """Midpoint of the grid edge that is closest to the ignition."""
    mids = [(0, cols // 2), (rows - 1, cols // 2), (rows // 2, 0), (rows // 2, cols - 1)]
    return min(mids, key=lambda m: (m[0] - ignition[0]) ** 2 + (m[1] - ignition[1]) ** 2)


def make_instance_for(terrain="real", fuel_rule="hazard", fuel_threshold=DEFAULT_FUEL_THRESHOLD, seed=42,
                      ignitions=None, stations=None, crews=1, speed=3.0, work_time=1, value="priority"):
    """Build (surface, instance) for the CLI and the exporter.

    terrain "real": the 20 x 20 Khao Yai window main.py uses; fuel from `fuel_rule` ("hazard" or
    "landcover"); default ignition = fuel cell with the most real FIRMS detections (ties: hazard).
    terrain "synthetic": seeded synthetic surface, fuel = hazard >= threshold, default ignition =
    highest-hazard fuel cell. Default station (both): midpoint of the grid edge nearest the ignition.
    """
    if terrain == "real":
        surf = build_real_surface(GRID, seed=42)
        fuel = real_fuel_mask(surf, fuel_rule, fuel_threshold)
        ig = list(ignitions or [ignitions_by_fire_history(surf, fuel, 1)[0]])
    elif terrain == "synthetic":
        surf = build_priority_surface(GRID, GRID, seed)
        fuel = surf["hazard"] >= fuel_threshold
        ig = list(ignitions or [default_ignition(surf, fuel_threshold)])
    else:
        raise ValueError(f"terrain must be 'real' or 'synthetic', got {terrain!r}")
    st = default_station(GRID, GRID, ig[0])
    sts = list(stations or [])
    sts += [sts[-1] if sts else st] * (crews - len(sts))
    sts = sts[:crews]
    inst = build_instance(surf, ig, sts, fuel_threshold, value, speed, work_time, fuel=fuel)
    inst.params.update(terrain=terrain, fuel_rule=fuel_rule if terrain == "real" else "hazard")
    return surf, inst


def _outline_segments(mask):
    rows, cols = mask.shape
    segs = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            for dr, dc, seg in ((-1, 0, ((c, r), (c + 1, r))), (1, 0, ((c, r + 1), (c + 1, r + 1))),
                                (0, -1, ((c, r), (c, r + 1))), (0, 1, ((c + 1, r), (c + 1, r + 1)))):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols) or not mask[rr, cc]:
                    segs.append([(x * CELL_KM, y * CELL_KM) for x, y in seg])
    return segs


def plot(inst, base, best, label, path):
    import matplotlib
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows, cols = inst.rows, inst.cols
    extent = (0, cols * CELL_KM, rows * CELL_KM, 0)
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    ax.imshow(inst.fuel.astype(int), cmap=ListedColormap(["#ecebe6", "#bcd9a8"]), extent=extent,
              interpolation="nearest", vmin=0, vmax=1)
    saved = base.burned & ~best.burned
    ax.imshow(np.ma.masked_where(~saved, np.ones((rows, cols))), cmap=ListedColormap(["#3f8f3a"]),
              extent=extent, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~best.burned, np.ones((rows, cols))), cmap=ListedColormap(["#e8743b"]),
              extent=extent, interpolation="nearest", alpha=0.9)
    ax.add_collection(LineCollection(_outline_segments(base.burned), colors="#7f1d1d", linewidths=1.8,
                                     linestyles="--"))

    def xy(cell):
        return (cell[1] + 0.5) * CELL_KM, (cell[0] + 0.5) * CELL_KM

    crew_colours = ["#1d4ed8", "#7e22ce", "#0f766e"]
    for ci in range(inst.n_crews):
        route = [d for d in best.defences if d.crew == ci]
        col = crew_colours[ci % len(crew_colours)]
        pts = [xy(inst.stations[ci])] + [xy(d.cell) for d in route]
        if len(pts) > 1:
            ax.plot(*zip(*pts), "--", color=col, lw=1.4, zorder=3)
        sx, sy = xy(inst.stations[ci])
        ax.plot(sx, sy, "s", ms=11, mfc="white", mec=col, mew=2.5, zorder=5)
        for d in route:
            x, y = xy(d.cell)
            ax.add_patch(plt.Rectangle((x - CELL_KM / 2, y - CELL_KM / 2), CELL_KM, CELL_KM,
                                       fc=col, ec="white", lw=0.8, zorder=4))
            tag = f"{d.order + 1}" if inst.n_crews == 1 else f"{'ABC'[ci]}{d.order + 1}"
            ax.text(x, y, tag, color="white", ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6)
    for ig in inst.ignitions:
        ax.plot(*xy(ig), marker="*", ms=17, mfc="#facc15", mec="black", mew=1.2, zorder=7, ls="")

    pad = 0.15   # km of margin so the outline stays visible along the plot border
    ax.set_xlim(-pad, cols * CELL_KM + pad)
    ax.set_ylim(rows * CELL_KM + pad, -pad)
    ax.set_xticks(np.arange(0, cols * CELL_KM + 0.01, 1.0))
    ax.set_yticks(np.arange(0, rows * CELL_KM + 0.01, 1.0))
    ax.set_xlabel("East (km)")
    ax.set_ylabel("South (km)")
    terrain = ("real Khao Yai 10 km window (modelled fire spread)" if inst.params.get("terrain") == "real"
               else "synthetic terrain (not real survey data)")
    ax.set_title(f"Ranger response plan on {terrain}\n{label}: "
                 f"burned {best.cells_burned} of {inst.n_fuel} fuel cells "
                 f"(baseline {base.cells_burned}), value saved {best.value_saved:.1f}", fontsize=11)
    handles = [Patch(fc="#bcd9a8", label="fuel, never reached by fire"), Patch(fc="#ecebe6", label="no fuel"),
               Patch(fc="#e8743b", label="burned despite the plan"),
               Patch(fc="#3f8f3a", label="saved: burns with no response, kept by the plan"),
               Line2D([], [], color="#7f1d1d", lw=1.8, ls="--", label="edge of the no-response burned area"),
               Line2D([], [], marker="*", color="none", mfc="#facc15", mec="black", ms=14, label="ignition"),
               Line2D([], [], marker="s", color="none", mfc="white", mec="#1d4ed8", mew=2.5, ms=9, label="station"),
               Patch(fc="#1d4ed8", label="firebreak (number = order); dashed = crew path")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", choices=("real", "synthetic"), default="real",
                    help="real: the 20x20 Khao Yai window used by main.py (default); synthetic: seeded random terrain")
    ap.add_argument("--fuel-rule", choices=("hazard", "landcover"), default="hazard",
                    help="real terrain only: hazard >= --fuel-threshold, or land-cover (tree/shrub/grass/crop >= 50%%)")
    ap.add_argument("--seed", type=int, default=42, help="synthetic terrain only")
    ap.add_argument("--ignition", type=parse_cell, action="append", help="r,c (repeatable); default: real = fuel cell with most FIRMS detections, synthetic = highest-hazard fuel cell")
    ap.add_argument("--station", type=parse_cell, action="append", help="r,c per crew (repeatable); default: edge midpoint nearest the ignition")
    ap.add_argument("--crews", type=int, default=1)
    ap.add_argument("--speed", type=float, default=3.0, help="cells per step")
    ap.add_argument("--work-time", type=int, default=1)
    ap.add_argument("--fuel-threshold", type=float, default=DEFAULT_FUEL_THRESHOLD)
    ap.add_argument("--value", default="priority", help="surface layer used as value")
    ap.add_argument("--time-limit", type=float, default=20.0, help="seconds for the exact solver")
    ap.add_argument("--node-limit", type=int, default=300_000)
    ap.add_argument("--out", default=None, help="figure path; default ranger_response.png (real) or "
                                                "ranger_response_synthetic.png (synthetic), next to this file")
    a = ap.parse_args(argv)
    if a.out is None:
        name = "ranger_response.png" if a.terrain == "real" else "ranger_response_synthetic.png"
        a.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    try:
        surf, inst = make_instance_for(a.terrain, a.fuel_rule, a.fuel_threshold, a.seed, a.ignition,
                                       a.station, a.crews, a.speed, a.work_time, a.value)
    except (ValueError, KeyError) as e:
        sys.exit(f"error: {e.args[0] if e.args else e}")
    ignitions, stations = list(inst.ignitions), list(inst.stations)

    if a.terrain == "real":
        rule = (f"hazard >= {a.fuel_threshold}" if a.fuel_rule == "hazard"
                else "land cover (tree+shrub+grass+crop >= 50%)")
        print(f"real Khao Yai window (row0, col0) = {surf['window']}, grid {GRID}x{GRID}; fuel = {rule}: "
              f"{inst.n_fuel} fuel cells ({100 * inst.n_fuel / GRID**2:.0f}%)")
    else:
        print(f"synthetic grid {GRID}x{GRID}, seed {a.seed}; fuel = hazard >= {a.fuel_threshold}: "
              f"{inst.n_fuel} fuel cells ({100 * inst.n_fuel / GRID**2:.0f}%)")
    print(f"ignition(s) (row,col): {ignitions}; station(s): {stations}; crews {a.crews}, "
          f"speed {a.speed} cells/step, work {a.work_time} step(s); value layer '{a.value}'")

    base = simulate(inst, Plan.empty(inst.n_crews))
    greedy = solve_greedy(inst)
    exact = solve_exact(inst, node_limit=a.node_limit, time_limit=a.time_limit)

    rows = [("no response", base, "n/a", 0.0),
            ("greedy", greedy.result, "n/a", greedy.elapsed),
            ("exact B&B", exact.result, "yes" if exact.proven_optimal else f"no ({exact.stop_reason})", exact.elapsed)]
    print(f"\n{'method':<12}{'value saved':>12}{'cells burned':>14}{'proven optimal':>24}{'time (s)':>10}")
    for name, r, proven, el in rows:
        print(f"{name:<12}{r.value_saved:>12.2f}{r.cells_burned:>14d}{proven:>24}{el:>10.1f}")
    print(f"(total value of fuel cells {inst.value[inst.fuel].sum():.1f}; "
          f"baseline burns {base.cells_burned} cells and keeps value {base.objective:.1f})")
    if not exact.proven_optimal:
        print(f"exact search stopped at the {exact.stop_reason} after {exact.nodes} nodes: "
              "its plan is the best found, not proven optimal.")

    best, label = (exact, "exact") if exact.objective >= greedy.objective else (greedy, "greedy")
    print(f"\nplan ({label}), steps are time steps:")
    for ci in range(inst.n_crews):
        print(f"  crew {ci + 1}, start {inst.stations[ci]}")
        recs = [d for d in best.result.defences if d.crew == ci]
        if not recs:
            print("    (no defence helps)")
        for d in recs:
            fa = "never" if d.fire_arrival == float("inf") else int(d.fire_arrival)
            print(f"    {d.order + 1:>2}. cell {d.cell}  arrive {d.arrive:>3}  firebreak done {d.finish:>3}  "
                  f"fire arrival {fa}  {d.reason}")

    plot(inst, base, best.result, label, a.out)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
