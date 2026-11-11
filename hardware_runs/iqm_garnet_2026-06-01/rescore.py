"""
Re-score the IQM Garnet run of 1 June 2026 from its raw counts, with no classical repair.

Run from the repo root:  python hardware_runs/iqm_garnet_2026-06-01/rescore.py

The job ran plain QAOA (2 layers) on the smallest benchmark instance: 8 candidate
sites, 2 sensors, seed 42, synthetic terrain. Qubit k is candidate site k.
Bitstrings in result.json are read with qubit 0 as the RIGHTMOST character; see
README.md for how that was established.
"""
import itertools
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from priority_surface import build_priority_surface
from viewshed import generate_candidate_sites, compute_viewsheds

N_SITES, N_SENSORS, SEED, GRID = 8, 2, 42, 20


def instance():
    # The run used the synthetic terrain, so ask for it explicitly where supported.
    try:
        surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=SEED, source="synthetic")
    except TypeError:
        surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=SEED)
    sites = generate_candidate_sites((GRID, GRID), n_candidates=N_SITES,
                                     priority=surf["priority"], dem_norm=surf["dem"], seed=SEED)
    viewsheds = compute_viewsheds(surf["dem_m"], sites)
    weights = {r * GRID + c: float(surf["priority"][r, c])
               for r in range(GRID) for c in range(GRID)}
    return viewsheds, weights


def coverage_table(viewsheds, weights):
    """Covered priority of every 8-bit state (bit k = site k); 0 unless exactly 2 sensors."""
    cov = np.zeros(2 ** N_SITES)
    for pair in itertools.combinations(range(N_SITES), N_SENSORS):
        cells = set().union(*(viewsheds[s] for s in pair))
        cov[sum(1 << s for s in pair)] = sum(weights[c] for c in cells)
    return cov


def simulate_program(path):
    """Noiseless statevector of program.qasm (gates h, rzz, rz, rx only). Returns probabilities."""
    n, dim = N_SITES, 2 ** N_SITES
    idx = np.arange(dim)
    z = [1 - 2 * ((idx >> k) & 1) for k in range(n)]
    sv = np.full(dim, 1 / np.sqrt(dim), dtype=complex)       # the eight H gates
    text = open(path).read()
    for m in re.finditer(r"^(rzz|rz|rx)\(([-\d.e]+)\) q\[(\d)\](?:, q\[(\d)\])?;", text, re.M):
        gate, theta, a = m.group(1), float(m.group(2)), int(m.group(3))
        if gate == "rzz":
            sv = sv * np.exp(-0.5j * theta * z[a] * z[int(m.group(4))])
        elif gate == "rz":
            sv = sv * np.exp(-0.5j * theta * z[a])
        else:
            c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
            lo = idx[(idx >> a) & 1 == 0]
            hi = lo | (1 << a)
            sv_lo, sv_hi = sv[lo].copy(), sv[hi].copy()
            sv[lo], sv[hi] = c * sv_lo + s * sv_hi, s * sv_lo + c * sv_hi
    return np.abs(sv) ** 2


def main():
    viewsheds, weights = instance()
    cov = coverage_table(viewsheds, weights)
    optimum = cov.max()
    feasible = np.array([bin(i).count("1") == N_SENSORS for i in range(2 ** N_SITES)])
    optimal = cov >= optimum - 1e-9

    counts = json.load(open(os.path.join(HERE, "result.json")))["resultData"]["measurementCounts"]
    shots = sum(counts.values())
    hardware = np.zeros(2 ** N_SITES)
    for bits, c in counts.items():
        hardware[int(bits, 2)] += c          # rightmost character = qubit 0
    hardware /= shots

    ideal = simulate_program(os.path.join(HERE, "program.qasm"))
    uniform = np.full(2 ** N_SITES, 1 / 2 ** N_SITES)
    random_feasible = feasible / feasible.sum()

    best = int(np.argmax(cov))
    print(f"instance: {N_SITES} sites, {N_SENSORS} sensors, seed {SEED}; optimum {optimum:.4f} "
          f"at sites {[k for k in range(N_SITES) if best >> k & 1]}; {shots} shots, "
          f"{len(counts)} distinct bitstrings")
    print(f"\n{'distribution':<34}{'P(2 sensors)':>13}{'P(optimal)':>12}{'mean ratio':>12}")
    for name, p in [("IQM Garnet (measured)", hardware),
                    ("same circuit, noiseless simulation", ideal),
                    ("uniform over all 256 bitstrings", uniform),
                    ("uniform over 2-sensor placements", random_feasible)]:
        print(f"{name:<34}{p[feasible].sum():>13.3f}{p[optimal].sum():>12.4f}"
              f"{(p @ cov) / optimum:>12.3f}")
    tvd = lambda a, b: 0.5 * np.abs(a - b).sum()
    print(f"\ntotal variation distance, measured vs noiseless: {tvd(hardware, ideal):.3f}; "
          f"measured vs uniform: {tvd(hardware, uniform):.3f}")
    print(f"shots that were the optimal placement: {int(round(hardware[optimal].sum() * shots))}")


if __name__ == "__main__":
    main()
