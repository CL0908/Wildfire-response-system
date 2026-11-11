"""
Build and compare candidate circuits for one more IQM Garnet job, before spending credits.

Needs IQM's Qiskit tools, which are kept out of requirements.txt:
    pip install "iqm-client[qiskit]" qiskit-aer scipy
Run from the repo root:  python hardware_runs/proposed_garnet_run/prepare.py

Everything here is simulation. The noise numbers come from IQM's published model of
Garnet (IQMFakeGarnet), checked against the real job of 1 June 2026.
"""
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
from iqm.qiskit_iqm import IQMFakeGarnet

from priority_surface import build_priority_surface
from viewshed import generate_candidate_sites, compute_viewsheds
from solvers import qaoa

N, K, SEED, GRID = 8, 2, 42, 20
JUNE = os.path.join(ROOT, "hardware_runs", "iqm_garnet_2026-06-01")
NOISY_SHOTS = 40_000


# ── instance ────────────────────────────────────────────────────────────────
def instance():
    try:
        surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=SEED, source="synthetic")
    except TypeError:
        surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=SEED)
    sites = generate_candidate_sites((GRID, GRID), n_candidates=N,
                                     priority=surf["priority"], dem_norm=surf["dem"], seed=SEED)
    vs = compute_viewsheds(surf["dem_m"], sites)
    w = {r * GRID + c: float(surf["priority"][r, c]) for r in range(GRID) for c in range(GRID)}
    return vs, w


VS, W = instance()
SITES = sorted(VS)
IDX = np.arange(2 ** N)
HAMMING = np.array([bin(i).count("1") for i in IDX])
FEASIBLE = HAMMING == K
COV = np.zeros(2 ** N)                       # index bit k = site k (Qiskit order)
for pair in itertools.combinations(range(N), K):
    cells = set().union(*(VS[SITES[s]] for s in pair))
    COV[sum(1 << s for s in pair)] = sum(W[c] for c in cells)
OPT = COV.max()
OPTIMAL = COV >= OPT - 1e-9


def stats(p):
    return dict(p_feasible=float(p[FEASIBLE].sum()), p_optimal=float(p[OPTIMAL].sum()),
                mean_ratio=float(p @ COV) / OPT)


def counts_to_probs(counts):
    p = np.zeros(2 ** N)
    for bits, c in counts.items():
        p[int(bits.replace(" ", ""), 2)] += c
    return p / p.sum()


def tvd(a, b):
    return 0.5 * float(np.abs(a - b).sum())


# ── circuits ────────────────────────────────────────────────────────────────
def ising(mode):
    """(z_terms, zz_terms) of the 2-local cost, scaled so the largest coefficient is 1."""
    H = (qaoa.build_penalty_cost_hamiltonian(VS, W, SITES, K) if mode == "plain"
         else qaoa.build_cost_hamiltonian(VS, W, SITES, K))
    z, zz, _ = qaoa._extract_ising(H, N)
    scale = max(abs(c) for *_, c in z + zz)
    return [(q, c / scale) for q, c in z], [(a, b, c / scale) for a, b, c in zz]


def dicke(n, k):
    """Dicke state |D^k_n> by the Bärtschi-Eidenbenz construction (verified below)."""
    qc = QuantumCircuit(n)
    qc.x(range(n - k, n))

    def scs(m, kk):                      # acts on qubits m-kk-1 .. m-1 (0-indexed)
        last = m - 1
        qc.cx(last - 1, last)
        qc.cry(2 * np.arccos(np.sqrt(1 / m)), last, last - 1)
        qc.cx(last - 1, last)
        for l in range(2, kk + 1):
            t = last - l
            qc.cx(t, last)
            qc.mcry(2 * np.arccos(np.sqrt(l / m)), [last, t + 1], t, None, mode="noancilla")
            qc.cx(t, last)

    for m in range(n, k, -1):
        scs(m, k)
    for m in range(k, 1, -1):
        scs(m, m - 1)
    return qc


def build(mode, params, measure=True):
    gammas, betas = params[: len(params) // 2], params[len(params) // 2:]
    z, zz = ising(mode)
    qc = QuantumCircuit(N, N)
    if mode == "plain":
        qc.h(range(N))
    else:
        qc.compose(dicke(N, K), inplace=True)
    for g, b in zip(gammas, betas):
        for a, c, coef in zz:
            qc.rzz(2 * coef * g, a, c)
        for q, coef in z:
            qc.rz(2 * coef * g, q)
        if mode == "plain":
            qc.rx(2 * b, range(N))
        else:
            for a, c in qaoa._xy_pairs(N):
                qc.append(XXPlusYYGate(2 * b), [a, c])
    if measure:
        qc.measure(range(N), range(N))
    return qc


def ideal_probs(qc):
    bare = qc.remove_final_measurements(inplace=False)
    return Statevector(bare).probabilities()


def tune(mode, layers=1, seed=0):
    """Angles that maximise the expected ratio of one noiseless shot."""
    rng = np.random.default_rng(seed)
    f = lambda x: -stats(ideal_probs(build(mode, x, measure=False)))["mean_ratio"]
    best = None
    for _ in range(12):
        res = minimize(f, rng.uniform(0, np.pi, 2 * layers), method="COBYLA",
                       options={"maxiter": 200, "rhobeg": 0.4})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


# ── device model ────────────────────────────────────────────────────────────
BACKEND = IQMFakeGarnet()


def on_device(qc, tries=20):
    """Route onto Garnet's layout; keep the routing with the fewest CZ gates."""
    best = None
    for s in range(tries):
        t = transpile(qc, BACKEND, optimization_level=3, seed_transpiler=s)
        n_cz = t.count_ops().get("cz", 0)
        if best is None or n_cz < best[0]:
            best = (n_cz, t)
    return best[1]


def noisy_probs(tqc, shots=NOISY_SHOTS):
    return counts_to_probs(BACKEND.run(tqc, shots=shots).result().get_counts())


def june_circuit():
    qc = QuantumCircuit(N, N)
    for line in open(os.path.join(JUNE, "program.qasm")):
        m = re.match(r"(h|rzz|rz|rx)(?:\(([-\d.e]+)\))? q\[(\d)\](?:, q\[(\d)\])?;", line.strip())
        if not m:
            continue
        g, th, a, b = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        if g == "h":
            qc.h(a)
        elif g == "rzz":
            qc.rzz(float(th), a, int(b))
        elif g == "rz":
            qc.rz(float(th), a)
        else:
            qc.rx(float(th), a)
    qc.measure(range(N), range(N))
    return qc


def row(name, p, tqc=None):
    s = stats(p)
    gates = f"{tqc.count_ops().get('cz', 0):>5}{tqc.depth():>7}" if tqc is not None else f"{'':>12}"
    print(f"{name:<40}{gates}{s['p_feasible']:>10.3f}{s['p_optimal']:>10.4f}{s['mean_ratio']:>10.3f}")
    return s


def main():
    target = Statevector(dicke(N, K)).probabilities()
    assert np.allclose(target[FEASIBLE], 1 / FEASIBLE.sum()) and abs(target.sum() - 1) < 1e-9, \
        "Dicke construction is wrong"

    uniform = np.full(2 ** N, 1 / 2 ** N)
    random_feasible = FEASIBLE / FEASIBLE.sum()
    print(f"instance: {N} sites, {K} sensors, seed {SEED}; optimum {OPT:.4f}\n")
    print(f"{'':<40}{'CZ':>5}{'depth':>7}{'P(2 sens)':>10}{'P(opt)':>10}{'mean':>10}")
    row("uniform over all 256 bitstrings", uniform)
    row("uniform over 2-sensor placements", random_feasible)

    # 1. Does the device model reproduce the real June job?
    measured = counts_to_probs(json.load(open(os.path.join(JUNE, "result.json")))
                               ["resultData"]["measurementCounts"])
    jq = june_circuit()
    jt = on_device(jq)
    j_ideal, j_noisy = ideal_probs(jq), noisy_probs(jt)
    print()
    row("June job: measured on Garnet", measured)
    row("June job: device model", j_noisy, jt)
    row("June job: noiseless", j_ideal)
    rng = np.random.default_rng(0)
    floor = np.mean([tvd(rng.multinomial(2000, j_noisy) / 2000, j_noisy) for _ in range(200)])
    marg = lambda p: np.array([p[(IDX >> k) & 1 == 1].sum() for k in range(N)])
    print(f"  TVD measured vs device model {tvd(measured, j_noisy):.3f} "
          f"(2,000-shot sampling alone gives {floor:.3f}); vs noiseless {tvd(measured, j_ideal):.3f}; "
          f"vs uniform {tvd(measured, uniform):.3f}")
    print(f"  mean P(qubit reads 1): measured {marg(measured).mean():.3f}, "
          f"model {marg(j_noisy).mean():.3f}, noiseless {marg(j_ideal).mean():.3f}; "
          f"per-qubit correlation measured vs model {np.corrcoef(marg(measured), marg(j_noisy))[0, 1]:.2f}")

    # 2. Candidates
    out = {}
    for name, mode, layers in [("A: plain QAOA, fixed cost, 1 layer", "plain", 1),
                               ("B: XY-QAOA (Dicke start), 1 layer", "xy", 1),
                               ("C: Dicke state only, no QAOA layer", "xy", 0)]:
        params = tune(mode, layers) if layers else np.array([])
        qc = build(mode, params)
        tqc = on_device(qc)
        print()
        row(f"{name}: noiseless", ideal_probs(qc))
        s = row(f"{name}: device model", noisy_probs(tqc), tqc)
        out[name] = dict(mode=mode, layers=layers, params=[float(x) for x in params], **s,
                         cz=int(tqc.count_ops().get("cz", 0)), depth=int(tqc.depth()))
    json.dump(out, open(os.path.join(HERE, "candidates.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
