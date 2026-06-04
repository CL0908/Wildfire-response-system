"""
qaoa_hardware.py  —  QAOA on IBM Marrakesh (156-qubit Heron r2).

Workflow
--------
1. Classical optimisation  — NumPy simulator finds (γ, β) parameters.
                             Fast, free, no queue.
2. Circuit build           — Concrete Qiskit QuantumCircuit with the
                             optimised angles.
3. Transpile               — Qiskit maps logical qubits to Marrakesh's
                             heavy-hex physical layout and decomposes
                             everything to the CZ + RZ + SX + X native set.
4. Hardware execution      — SamplerV2 submits to ibm_marrakesh, decodes
                             bitstrings with the same per-shot repair and
                             scoring logic as the local simulator.

Why this split?
  COBYLA needs ~150 circuit evaluations to converge.  On hardware each
  evaluation costs queue time and credits.  Optimising on the NumPy
  statevector simulator first and running the hardware only once for the
  final shot sample is the standard NISQ practice.

Install
-------
    pip install qiskit qiskit-ibm-runtime

Usage
-----
    # from another module
    from solvers.qaoa_hardware import run_qaoa_hardware
    result = run_qaoa_hardware(viewsheds, weights, n_sensors=2,
                               ibm_token="<your_token>")

    # as a script (runs the smallest benchmark instance)
    cd qprefire
    python solvers/qaoa_hardware.py --token <your_token>
    python solvers/qaoa_hardware.py --token <your_token> --p 1 --mode plain

Notes on Marrakesh
------------------
* Heavy-hex topology  — not all qubit pairs are neighbours; Qiskit inserts
  SWAP gates for non-adjacent RZZ terms.  Use optimization_level=3 for
  the best routing (fewest SWAPs).
* CZ native gate      — RZZ(θ) is decomposed as  H·CNOT·RZ(θ)·CNOT·H
  → CNOT → H·CZ·H, so each RZZ costs ~2 CZ gates + single-qubit rotations.
* Depth budget        — Heron r2 has improved coherence but QAOA depth
  still grows as  p × (3·|ZZ| + |Z| + n).  Recommend p=1 or p=2 for
  hardware; p=4 is fine for the local simulator.
* WS-QAOA caution     — The Dicke warm-start adds O(2^n) state-prep depth
  (via StatePreparation decomposition).  For hardware, mode='plain' is
  the safer choice unless n is very small.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.optimize import minimize
from itertools import combinations

from solvers.qaoa import (
    build_penalty_cost_hamiltonian,
    build_cost_hamiltonian,
    _extract_ising,
    _init_sv,
    _simulate,
    _expectation,
    _best_from_probs,
    _best_from_counts,
    _circuit_depth,
)


# ---------------------------------------------------------------------------
# Qiskit circuit builder
# ---------------------------------------------------------------------------

def _build_circuit(n, n_sensors, z_terms, zz_terms, params, p, mode):
    """
    Build a concrete Qiskit QuantumCircuit for one (γ, β) assignment.

    Gate choices:
      rzz  — decomposes to 2 CZ + RZ on Marrakesh (via transpiler)
      rz   — native on Marrakesh
      rx   — decomposes to RZ + SX + RZ (3 gates, all native)
      h    — decomposes to RZ + SX + RZ

    The transpiler handles all decompositions; we just write readable gates.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n, n)

    # ── initial state ──────────────────────────────────────────────────────
    if mode == "plain":
        qc.h(range(n))
    else:
        # Dicke state |D^k_n⟩: uniform superposition over all weight-k strings.
        from qiskit.circuit.library import StatePreparation
        k   = n_sensors
        dim = 2 ** n
        sv  = np.zeros(dim, dtype=float)
        idxs = [sum(1 << b for b in bits) for bits in combinations(range(n), k)]
        sv[idxs] = 1.0 / np.sqrt(len(idxs))
        qc.append(StatePreparation(sv), range(n))

    # ── p QAOA layers ──────────────────────────────────────────────────────
    gammas, betas = params[:p], params[p:]
    for layer in range(p):
        g = float(gammas[layer])
        b = float(betas[layer])

        # Cost unitary: encode the coverage objective
        for q0, q1, c in zz_terms:
            qc.rzz(2 * c * g, q0, q1)
        for q, c in z_terms:
            qc.rz(2 * c * g, q)

        # X mixer: RX(2β) on every qubit
        for q in range(n):
            qc.rx(2 * b, q)

    qc.measure(range(n), range(n))
    return qc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_qaoa_hardware(
    viewsheds: dict,
    weights:   dict,
    n_sensors: int,
    ibm_token: str | None = None,
    backend_name: str = "ibm_marrakesh",
    p_layers:  int  = 2,
    mode:      str  = "plain",
    max_iter:  int  = 150,
    shots:     int  = 2000,
    seed:      int  = 0,
    optimization_level: int = 3,
) -> dict:
    """
    Run QAOA on IBM Marrakesh.

    Parameters
    ----------
    ibm_token          IBM Quantum API token (or set env QISKIT_IBM_TOKEN).
    backend_name       Target backend; default 'ibm_marrakesh'.
    p_layers           QAOA depth.  Use 1–2 for hardware; 4 for simulator.
    mode               'plain' (X-mixer, |+⟩^n) or 'ws' (X-mixer, Dicke warm-start).
    max_iter           COBYLA iterations for classical optimisation.
    shots              Measurement shots to request from hardware.
    optimization_level Qiskit transpiler level 1–3 (3 = best SWAP routing).

    Returns
    -------
    dict — identical keys to run_qaoa() in qaoa.py, plus:
      transpiled_depth    int   gate depth after Qiskit transpilation
      transpiled_cz_count int   number of CZ gates in the transpiled circuit
      job_id              str   IBM Quantum job identifier
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

    sites = sorted(viewsheds.keys())
    n     = len(sites)
    rng   = np.random.default_rng(seed)

    site_vs_weight = {s: sum(weights.get(c, 0.0) for c in vs)
                      for s, vs in viewsheds.items()}

    # ── 1. Build Ising Hamiltonian ────────────────────────────────────────
    H_C = (build_penalty_cost_hamiltonian(viewsheds, weights, sites, n_sensors)
           if mode == "plain"
           else build_cost_hamiltonian(viewsheds, weights, sites))
    z_terms, zz_terms, const = _extract_ising(H_C, n)
    init_sv = _init_sv(n, n_sensors, mode)

    print(f"  [{mode}] n_qubits={n}  |ZZ|={len(zz_terms)}  |Z|={len(z_terms)}")

    # ── 2. Classical parameter optimisation (NumPy, no queue) ────────────
    eval_count = [0]

    def _obj(params):
        eval_count[0] += 1
        sv = _simulate(n, init_sv, params, p_layers, mode, z_terms, zz_terms)
        return _expectation(sv, z_terms, zz_terms, const, n)

    x0     = rng.uniform(0, np.pi, 2 * p_layers)
    res    = minimize(_obj, x0, method="COBYLA",
                      options={"maxiter": max_iter, "rhobeg": 0.5})
    opt_params   = res.x
    params_moved = bool(np.any(np.abs(opt_params - x0) > 1e-3))
    print(f"  [classical] iters={eval_count[0]}  "
          f"params_moved={'YES' if params_moved else 'NO ← COBYLA did not move'}")

    # ── 3. Build circuit ──────────────────────────────────────────────────
    use_qbraid = backend_name.startswith("aws:")

    qc = _build_circuit(n, n_sensors, z_terms, zz_terms, opt_params, p_layers, mode)

    if use_qbraid:
        # ── qBraid / IQM path ─────────────────────────────────────────────
        # qBraid handles transpilation internally; skip Qiskit pass manager.
        try:
            from qbraid.runtime import QbraidProvider
        except ImportError as e:
            raise ImportError("Install qbraid: pip install qbraid") from e

        qb_api_key = os.environ.get("QBRAID_API_KEY")
        if not qb_api_key:
            raise ValueError("Set env var QBRAID_API_KEY before running")
        provider = QbraidProvider(api_key=qb_api_key)
        device   = provider.get_device(backend_name)

        # Report logical circuit stats (no Qiskit transpile available).
        t_depth = qc.depth()
        t_cz    = qc.count_ops().get("cz", 0)
        print(f"  [logical]  depth={t_depth}  CZ={t_cz}  "
              f"qubits={qc.num_qubits}  (qBraid transpiles on-device)")

        # ── 4a. Submit via qBraid ─────────────────────────────────────────
        # device.run() takes a list and returns a list of jobs.
        jobs = device.run([qc], shots=shots)
        job  = jobs[0]
        print(f"  [hardware] job submitted: {job.id}  (waiting for results...)")

        result = job.result()
        # result.data.get_counts() → {bitstring: count}, same as Qiskit convention.
        counts = result.data.get_counts()

    else:
        # ── IBM Quantum path ──────────────────────────────────────────────
        token = ibm_token or os.environ.get("QISKIT_IBM_TOKEN", "")
        if not token:
            raise ValueError("Provide ibm_token= or set env var QISKIT_IBM_TOKEN")

        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
        backend = service.backend(backend_name)

        pm   = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
        )
        qc_t = pm.run(qc)

        t_depth = qc_t.depth()
        t_cz    = qc_t.count_ops().get("cz", 0)
        print(f"  [transpile] depth={t_depth}  CZ={t_cz}  "
              f"physical_qubits={qc_t.num_qubits}")
        if t_depth > 500:
            print(f"  [transpile] WARNING: depth={t_depth} is high — "
                  f"consider reducing p_layers or switching to mode='plain'")

        # ── 4b. Submit via Qiskit Runtime ─────────────────────────────────
        sampler = Sampler(backend)
        job     = sampler.run([qc_t], shots=shots)
        print(f"  [hardware] job submitted: {job.job_id()}  (waiting for results...)")

        pub_result = job.result()[0]
        # BitArray → counts dict.  Qiskit: rightmost char = qubit 0 (LSB).
        counts = pub_result.data.c.get_counts()

    total = sum(counts.values())
    print(f"  [hardware] received {int(total)} shots  "
          f"unique_bitstrings={len(counts)}")

    # ── 5. Decode real hardware counts directly (no RNG re-sampling) ─────
    best_chosen, best_obj, best_bs, n_violations = _best_from_counts(
        counts, n, n_sensors, sites, viewsheds, weights, site_vs_weight
    )

    # Rebuild probs for sanity-flag comparison against initial state ──────
    probs = np.zeros(2 ** n)
    for bs, cnt in counts.items():
        idx = int(bs.replace(" ", ""), 2)
        if idx < len(probs):
            probs[idx] += cnt
    if total > 0:
        probs /= total

    # Sanity flag 2: is best result better than just sampling the initial state?
    seed_probs = np.abs(init_sv) ** 2
    seed_probs /= seed_probs.sum()
    seed_rng   = np.random.default_rng(seed + 99_999)
    _, _, seed_bs, _ = _best_from_probs(
        seed_probs, n, n_sensors, sites, viewsheds, weights, site_vs_weight, shots, seed_rng
    )
    best_ne_seed = (best_bs != seed_bs)

    if not best_chosen:
        from solvers.greedy import greedy_coverage
        best_chosen, best_obj = greedy_coverage(viewsheds, weights, n_sensors)
        best_ne_seed = False

    job_id = job.id if use_qbraid else job.job_id()

    return dict(
        chosen=best_chosen,
        objective=best_obj,
        iterations=eval_count[0],
        circuit_depth=_circuit_depth(n, p_layers, len(zz_terms), len(z_terms)),
        transpiled_depth=t_depth,
        transpiled_cz_count=t_cz,
        mode=mode,
        n_qubits=n,
        constraint_violations=n_violations,
        params_moved=params_moved,
        best_ne_seed=best_ne_seed,
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# Script entry point — runs the smallest benchmark instance
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Run QAOA on IBM Marrakesh for the smallest QPreFire instance "
                    "(|S|=8, N=2, seed=42)."
    )
    ap.add_argument("--token",   default=os.environ.get("QISKIT_IBM_TOKEN", ""),
                    help="IBM Quantum API token (or set env QISKIT_IBM_TOKEN)")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--p",       type=int, default=2,
                    help="QAOA depth layers (recommend 1 or 2 for hardware)")
    ap.add_argument("--shots",   type=int, default=2000)
    ap.add_argument("--mode",    default="plain", choices=["plain", "ws"],
                    help="plain = X-mixer + |+>^n  |  ws = X-mixer + Dicke warm-start")
    ap.add_argument("--opt-level", type=int, default=3, dest="opt_level",
                    help="Qiskit transpiler optimisation level (1–3)")
    args = ap.parse_args()

    from priority_surface import build_priority_surface
    from viewshed import generate_candidate_sites, compute_viewsheds, MAX_RANGE_CELLS
    from solvers.ilp import ilp_coverage

    GRID = 20
    surf = build_priority_surface(grid_rows=GRID, grid_cols=GRID, seed=42)
    sites = generate_candidate_sites(
        (GRID, GRID), n_candidates=8,
        priority=surf["priority"], dem_norm=surf["dem"], seed=42,
    )
    vs = compute_viewsheds(surf["dem_m"], sites, max_range_cells=MAX_RANGE_CELLS)
    weights = {r * GRID + c: float(surf["priority"][r, c])
               for r in range(GRID) for c in range(GRID)}

    print("=" * 60)
    print(f"Instance: |S|=8, N=2, seed=42  (20×20, 500 m, 4 km range)")
    print(f"Backend:  {args.backend}  p={args.p}  mode={args.mode}")
    print("=" * 60)

    _, ilp_obj, ilp_status = ilp_coverage(vs, weights, n_sensors=2)
    print(f"ILP optimal: {ilp_obj:.4f} ({ilp_status})\n")

    result = run_qaoa_hardware(
        vs, weights, n_sensors=2,
        ibm_token=args.token,
        backend_name=args.backend,
        p_layers=args.p,
        mode=args.mode,
        shots=args.shots,
        optimization_level=args.opt_level,
    )

    ratio = result["objective"] / ilp_obj if ilp_obj > 1e-9 else 0.0
    pm = "YES" if result["params_moved"]  else "NO — COBYLA did not move"
    bn = "YES" if result["best_ne_seed"] else "NO — echoing initial state"

    print("\n" + "=" * 60)
    print("RESULT")
    print("-" * 60)
    print(f"  chosen sites      : {result['chosen']}")
    print(f"  objective         : {result['objective']:.4f}")
    print(f"  ratio vs ILP      : {ratio:.3f}")
    print(f"  violations        : {result['constraint_violations']}/{args.shots} shots")
    print(f"  [sanity] params_moved  : {pm}")
    print(f"  [sanity] best_ne_seed  : {bn}")
    print(f"  logical depth     : {result['circuit_depth']}")
    print(f"  transpiled depth  : {result['transpiled_depth']}")
    print(f"  transpiled CZ     : {result['transpiled_cz_count']}")
    print(f"  job_id            : {result['job_id']}")
    print("=" * 60)
