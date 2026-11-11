"""
Score a finished job against the instance, per shot, with no classical repair.

    python hardware_runs/proposed_garnet_run/score.py path/to/result.json [--qubit0-leftmost]

Accepts the JSON that submit.py saves, or a result file downloaded from the qBraid
job page. Run from the repo root. Needs only numpy and scipy.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JUNE = os.path.join(os.path.dirname(HERE), "iqm_garnet_2026-06-01")
sys.path.insert(0, JUNE)
import rescore  # noqa: E402  (instance, coverage table and QASM simulator)

N = rescore.N_SITES


def load_counts(path):
    data = json.load(open(path))
    for key in ("counts", "measurementCounts"):
        if key in data:
            return data[key]
    return data["resultData"]["measurementCounts"]


def probs(counts, qubit0_rightmost=True):
    p = np.zeros(2 ** N)
    for bits, c in counts.items():
        bits = bits.replace(" ", "")
        p[int(bits if qubit0_rightmost else bits[::-1], 2)] += c
    return p / p.sum()


def main(path):
    viewsheds, weights = rescore.instance()
    cov = rescore.coverage_table(viewsheds, weights)
    optimum = cov.max()
    idx = np.arange(2 ** N)
    feasible = np.array([bin(i).count("1") == rescore.N_SENSORS for i in idx])
    optimal = cov >= optimum - 1e-9
    marginals = lambda p: np.array([p[(idx >> k) & 1 == 1].sum() for k in range(N)])

    counts = load_counts(path)
    shots = sum(counts.values())
    ideal = rescore.simulate_program(os.path.join(HERE, "job.qasm"))

    # Qubit 0 is the rightmost character (Qiskit's order), as established for the June job.
    # This circuit's qubits behave too alike for the data to confirm that by itself, so the
    # check below only warns if the data clearly points the other way.
    order = "--qubit0-leftmost" not in sys.argv
    corr = {o: float(np.corrcoef(marginals(probs(counts, o)), marginals(ideal))[0, 1])
            for o in (True, False)}
    print(f"{shots} shots, {len(counts)} distinct bitstrings")
    print(f"reading qubit 0 as the {'rightmost' if order else 'leftmost'} character "
          f"(per-qubit correlation with the noiseless circuit: rightmost {corr[True]:+.2f}, "
          f"leftmost {corr[False]:+.2f})")
    if corr[not order] - corr[order] > 0.3:
        print("WARNING: the data fits the other bit order better. P(2 sensors) is unaffected; "
              "P(optimal) and mean ratio are not. Compare with --qubit0-leftmost.")

    measured = probs(counts, order)
    june = probs(json.load(open(os.path.join(JUNE, "result.json")))["resultData"]["measurementCounts"])
    uniform = np.full(2 ** N, 1 / 2 ** N)
    tvd = lambda a, b: 0.5 * np.abs(a - b).sum()

    print(f"\n{'distribution':<36}{'P(2 sensors)':>13}{'P(optimal)':>12}{'mean ratio':>12}")
    for name, p in [("this job, measured", measured),
                    ("this job, noiseless simulation", ideal),
                    ("June job, measured", june),
                    ("uniform over all 256 bitstrings", uniform),
                    ("uniform over 2-sensor placements", feasible / feasible.sum())]:
        print(f"{name:<36}{p[feasible].sum():>13.3f}{p[optimal].sum():>12.4f}{(p @ cov) / optimum:>12.3f}")

    se = np.sqrt(measured[feasible].sum() * (1 - measured[feasible].sum()) / shots)
    print(f"\nstandard error on P(2 sensors): {se:.3f}")
    print(f"total variation distance, measured vs noiseless {tvd(measured, ideal):.3f}; "
          f"measured vs uniform {tvd(measured, uniform):.3f}")
    print(f"mean P(qubit reads 1): measured {marginals(measured).mean():.3f}, "
          f"noiseless {marginals(ideal).mean():.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
