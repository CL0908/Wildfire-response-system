"""
Write dashboards/public/hardware.json: a per-shot summary of every real-hardware run
that has raw counts in this folder. Run from the repo root:

    python hardware_runs/export_dashboard.py
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "dashboards", "public", "hardware.json")

# Each run: folder, facts that are not in the result file, and how to read its counts.
RUNS = [
    dict(
        folder="iqm_garnet_2026-06-01",
        device="IQM Garnet (20 qubits)",
        access="AWS Braket through qBraid",
        date="2026-06-01",
        circuit="Plain QAOA, 2 layers, original cost function",
        program="program.qasm",
        counts=lambda d: d["resultData"]["measurementCounts"],
        note="Run with the original cost function, which double-counts overlapping coverage "
             "and uses gate angles of up to 344 full turns.",
    ),
    dict(
        folder="iqm_garnet_2026-11-12",
        device="IQM Garnet (20 qubits)",
        access="AWS Braket through qBraid",
        date="2026-11-12",
        circuit="Plain QAOA, 1 layer, corrected cost function",
        program="program.qasm",
        counts=lambda d: d["resultData"]["measurementCounts"],
        note="Same instance with the cost function fixed and every gate angle under one full turn. "
             "A larger share of shots place 2 sensors than in June, but the average shot is still well below "
             "a random 2-sensor placement.",
    ),
]


def load_rescore():
    spec = importlib.util.spec_from_file_location(
        "rescore", os.path.join(HERE, "iqm_garnet_2026-06-01", "rescore.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    rs = load_rescore()
    viewsheds, weights = rs.instance()
    cov = rs.coverage_table(viewsheds, weights)
    optimum = float(cov.max())
    n = rs.N_SITES
    feasible = np.array([bin(i).count("1") == rs.N_SENSORS for i in range(2 ** n)])
    optimal = cov >= optimum - 1e-9

    def stats(p):
        return dict(p_feasible=round(float(p[feasible].sum()), 4),
                    p_optimal=round(float(p[optimal].sum()), 4),
                    mean_ratio=round(float(p @ cov) / optimum, 4))

    runs = []
    for run in RUNS:
        folder = os.path.join(HERE, run["folder"])
        counts = run["counts"](json.load(open(os.path.join(folder, "result.json"))))
        shots = sum(counts.values())
        measured = np.zeros(2 ** n)
        for bits, c in counts.items():
            measured[int(bits, 2)] += c          # rightmost character = qubit 0
        measured /= shots
        runs.append(dict(
            id=run["folder"], device=run["device"], access=run["access"], date=run["date"],
            circuit=run["circuit"], note=run["note"], shots=shots,
            optimal_shots=int(round(float(measured[optimal].sum()) * shots)),
            measured=stats(measured),
            noiseless=stats(rs.simulate_program(os.path.join(folder, run["program"]))),
        ))

    data = dict(
        instance=dict(n_sites=n, n_sensors=rs.N_SENSORS, seed=rs.SEED, terrain="synthetic",
                      n_placements=int(feasible.sum())),
        baselines=dict(
            uniform_bitstrings=stats(np.full(2 ** n, 1 / 2 ** n)),
            random_placement=stats(feasible / feasible.sum()),
        ),
        runs=runs,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(runs)} run(s)")


if __name__ == "__main__":
    main()
