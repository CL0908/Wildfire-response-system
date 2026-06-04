"""
QAOA family: Plain QAOA, WS-QAOA.

Pure-NumPy statevector simulator (no Qiskit in the hot loop).
Qiskit used only for SparsePauliOp Hamiltonian construction.

Shot decode: sample ≥2000 bitstrings from the final state, repair each to
exactly N sensors (greedy trim/extend), keep the best placement found.
Two sanity flags are returned per run:
  params_moved  – did COBYLA actually move (γ,β) from their start?
  best_ne_seed  – is the best shot different from just sampling the initial state?
"""

import numpy as np
from scipy.optimize import minimize
from itertools import combinations

try:
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_OK = True
except ImportError:
    QISKIT_OK = False


# ---------------------------------------------------------------------------
# NumPy statevector gate primitives
# ---------------------------------------------------------------------------

def _apply_rz(sv, q, angle, n):
    bit = 1 << (n - 1 - q)
    idxs = np.arange(len(sv))
    out = sv.copy()
    out[idxs & bit == 0] *= np.exp(-0.5j * angle)
    out[idxs & bit != 0] *= np.exp(0.5j * angle)
    return out


def _apply_rzz(sv, q0, q1, angle, n):
    b0 = 1 << (n - 1 - q0)
    b1 = 1 << (n - 1 - q1)
    idxs = np.arange(len(sv))
    same = ((idxs & b0) != 0) == ((idxs & b1) != 0)
    out = sv.copy()
    out[same] *= np.exp(-0.5j * angle)
    out[~same] *= np.exp(0.5j * angle)
    return out


def _apply_rx(sv, q, angle, n):
    c, s = np.cos(angle / 2), -1j * np.sin(angle / 2)
    bit = 1 << (n - 1 - q)
    idxs = np.arange(len(sv))
    i0 = idxs[idxs & bit == 0]
    i1 = idxs[idxs & bit != 0]
    a0, a1 = sv[i0].copy(), sv[i1].copy()
    out = sv.copy()
    out[i0] = c * a0 + s * a1
    out[i1] = s * a0 + c * a1
    return out


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------

def _pauli_dict_from(n, add_fn):
    """Helper: return add_fn that accumulates into a pauli_dict, then return it."""
    pd: dict[str, float] = {}
    I = "I" * n

    def add(label, coeff):
        pd[label] = pd.get(label, 0.0) + coeff

    add_fn(add, I, n)
    return [(lbl, c) for lbl, c in pd.items() if abs(c) > 1e-10] or [(I, 0.0)]


def build_cost_hamiltonian(viewsheds, weights, sites):
    """Compact H_C on y_i variables only (no penalty, uses OR approximation)."""
    n = len(sites)
    s_to_q = {s: q for q, s in enumerate(sites)}
    cell_to_sites: dict = {}
    for s, vs in viewsheds.items():
        for c in vs:
            cell_to_sites.setdefault(c, []).append(s)

    pd: dict[str, float] = {}
    I = "I" * n

    def add(lbl, v):
        pd[lbl] = pd.get(lbl, 0.0) + v

    for c, w in weights.items():
        seeing = [s for s in cell_to_sites.get(c, []) if s in s_to_q]
        if not seeing:
            continue
        if len(seeing) == 1:
            a = s_to_q[seeing[0]]
            z = list(I); z[n - 1 - a] = "Z"
            add(I, -w / 2)
            add("".join(z), w / 2)
        elif len(seeing) == 2:
            a, b = s_to_q[seeing[0]], s_to_q[seeing[1]]
            add(I, -w * 3 / 4)
            za = list(I); za[n - 1 - a] = "Z"
            zb = list(I); zb[n - 1 - b] = "Z"
            zab = list(I); zab[n - 1 - a] = "Z"; zab[n - 1 - b] = "Z"
            add("".join(za), w / 4)
            add("".join(zb), w / 4)
            add("".join(zab), -w / 4)
        else:
            for s in seeing:
                a = s_to_q[s]
                z = list(I); z[n - 1 - a] = "Z"
                add(I, -w / len(seeing))
                add("".join(z), w / len(seeing))

    terms = [(lbl, c) for lbl, c in pd.items() if abs(c) > 1e-10] or [(I, 0.0)]
    return SparsePauliOp.from_list(terms)


def build_penalty_cost_hamiltonian(viewsheds, weights, sites, n_sensors, P1=None):
    """H_C with explicit cardinality penalty (used by plain QAOA)."""
    if P1 is None:
        P1 = sum(weights.values()) * 5
    n = len(sites)
    s_to_q = {s: q for q, s in enumerate(sites)}
    I = "I" * n
    pd: dict[str, float] = {}

    def add(lbl, v):
        pd[lbl] = pd.get(lbl, 0.0) + v

    cell_to_sites: dict = {}
    for s, vs in viewsheds.items():
        for c in vs:
            cell_to_sites.setdefault(c, []).append(s)

    for c, w in weights.items():
        seeing = [s for s in cell_to_sites.get(c, []) if s in s_to_q]
        for s in seeing:
            a = s_to_q[s]
            z = list(I); z[n - 1 - a] = "Z"
            add(I, -w / 2)
            add("".join(z), w / 2)

    # (sum y_i - N)^2 in Z basis, y_i = (1-Z_i)/2
    offset = n / 2 - n_sensors
    add(I, P1 * offset ** 2)
    for i in range(n):
        z = list(I); z[n - 1 - i] = "Z"
        add("".join(z), P1 * (-offset))
        add(I, P1 * 0.25)
    for i in range(n):
        for j in range(i + 1, n):
            zz = list(I); zz[n - 1 - i] = "Z"; zz[n - 1 - j] = "Z"
            add("".join(zz), P1 * 0.5)

    terms = [(lbl, c) for lbl, c in pd.items() if abs(c) > 1e-10] or [(I, 0.0)]
    return SparsePauliOp.from_list(terms)


def _extract_ising(H, n):
    z_terms, zz_terms = [], []
    const = 0.0
    for pauli, coeff in zip(H.paulis, H.coeffs):
        label = pauli.to_label()
        zq = [n - 1 - i for i, c in enumerate(label) if c == "Z"]
        cr = float(coeff.real)
        if len(zq) == 0:
            const += cr
        elif len(zq) == 1:
            z_terms.append((zq[0], cr))
        elif len(zq) == 2:
            zz_terms.append((zq[0], zq[1], cr))
    return z_terms, zz_terms, const


# ---------------------------------------------------------------------------
# Statevector evolution and expectation
# ---------------------------------------------------------------------------

def _simulate(n, init_sv, params, p, mode, z_terms, zz_terms):
    sv = init_sv.copy()
    gammas, betas = params[:p], params[p:]
    for l in range(p):
        g = gammas[l]
        for q0, q1, c in zz_terms:
            sv = _apply_rzz(sv, q0, q1, 2 * c * g, n)
        for q, c in z_terms:
            sv = _apply_rz(sv, q, 2 * c * g, n)
        b = float(betas[l])
        for q in range(n):
            sv = _apply_rx(sv, q, 2 * b, n)
    return sv


def _expectation(sv, z_terms, zz_terms, const, n):
    idxs = np.arange(len(sv))
    probs = np.abs(sv) ** 2
    exp = const * probs.sum()
    for q, c in z_terms:
        bit = 1 << (n - 1 - q)
        sign = np.where(idxs & bit != 0, -1.0, 1.0)
        exp += c * (probs * sign).sum()
    for q0, q1, c in zz_terms:
        b0 = 1 << (n - 1 - q0)
        b1 = 1 << (n - 1 - q1)
        same = ((idxs & b0) != 0) == ((idxs & b1) != 0)
        sign = np.where(same, 1.0, -1.0)
        exp += c * (probs * sign).sum()
    return float(exp)


def _init_sv(n, n_sensors, mode):
    sv = np.zeros(2 ** n, dtype=complex)
    if mode == "plain":
        sv[:] = 1.0 / np.sqrt(2 ** n)
    else:  # ws: Dicke state
        idxs = [sum(1 << b for b in bits) for bits in combinations(range(n), n_sensors)]
        sv[idxs] = 1.0 / np.sqrt(len(idxs))
    return sv


# ---------------------------------------------------------------------------
# Per-shot repair and decode
# ---------------------------------------------------------------------------

def _repair(bs: str, n_sensors: int, sites: list, viewsheds: dict,
            site_vs_weight: dict[int, float]) -> list[int]:
    """
    Project a raw bitstring to exactly n_sensors selected sites.
    Over-budget: keep N with highest individual viewshed coverage weight.
    Under-budget: greedily extend with highest-marginal-gain remaining sites.
    """
    n = len(bs)
    raw = [sites[q] for q, b in enumerate(reversed(bs)) if b == "1"]
    hw = len(raw)

    if hw == n_sensors:
        return raw

    if hw > n_sensors:
        raw.sort(key=lambda s: site_vs_weight[s], reverse=True)
        return raw[:n_sensors]

    # Under-budget: greedy extension
    in_set = set(raw)
    covered: set[int] = set()
    for s in raw:
        covered |= viewsheds[s]
    chosen = list(raw)
    remaining = sorted(
        [s for s in sites if s not in in_set],
        key=lambda s: site_vs_weight[s], reverse=True
    )
    while len(chosen) < n_sensors and remaining:
        best_s = max(remaining,
                     key=lambda s: sum(1 for c in viewsheds[s] if c not in covered))
        chosen.append(best_s)
        covered |= viewsheds[best_s]
        remaining.remove(best_s)
    return chosen


def _best_from_probs(
    probs: np.ndarray,
    n: int,
    n_sensors: int,
    sites: list,
    viewsheds: dict,
    weights: dict,
    site_vs_weight: dict,
    shots: int,
    rng: np.random.Generator,
) -> tuple[list[int], float, str, int]:
    """
    Sample `shots` bitstrings, repair each to feasible, return best.
    Returns (chosen, objective, best_bs_str, n_hw_violations).
    """
    sample_idxs = rng.choice(len(probs), size=shots, p=probs)
    violations = 0
    best_obj = -1.0
    best_chosen: list[int] = []
    best_bs = "0" * n

    for idx in sample_idxs:
        bs = format(int(idx), f"0{n}b")
        if bs.count("1") != n_sensors:
            violations += 1
        chosen = _repair(bs, n_sensors, sites, viewsheds, site_vs_weight)
        covered: set[int] = set()
        for s in chosen:
            covered |= viewsheds[s]
        obj = sum(weights.get(c, 0.0) for c in covered)
        if obj > best_obj:
            best_obj, best_chosen, best_bs = obj, chosen, bs

    return best_chosen, best_obj, best_bs, violations


def _best_from_counts(
    counts: dict,
    n: int,
    n_sensors: int,
    sites: list,
    viewsheds: dict,
    weights: dict,
    site_vs_weight: dict,
) -> tuple[list[int], float, str, int]:
    """
    Decode directly from real hardware counts dict — no RNG re-sampling.
    Each unique bitstring is evaluated once, weighted by its shot count.
    Returns (chosen, objective, best_bs_str, n_hw_violations).
    """
    violations = 0
    best_obj = -1.0
    best_chosen: list[int] = []
    best_bs = "0" * n
    total_shots = 0

    for bs, cnt in counts.items():
        bs = bs.replace(" ", "").zfill(n)
        total_shots += cnt
        if bs.count("1") != n_sensors:
            violations += cnt
        chosen = _repair(bs, n_sensors, sites, viewsheds, site_vs_weight)
        covered: set[int] = set()
        for s in chosen:
            covered |= viewsheds[s]
        obj = sum(weights.get(c, 0.0) for c in covered)
        if obj > best_obj:
            best_obj, best_chosen, best_bs = obj, chosen, bs

    return best_chosen, best_obj, best_bs, violations


def _circuit_depth(n, p, n_zz, n_z):
    cost = n_zz * 3 + n_z
    mixer = n
    return p * (cost + mixer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_qaoa(
    viewsheds: dict[int, set[int]],
    weights: dict[int, float],
    n_sensors: int,
    p_layers: int = 4,
    mode: str = "plain",
    max_iter: int = 150,
    shots: int = 2000,
    seed: int = 0,
) -> dict:
    """
    mode ∈ {'plain', 'ws'}.

    Returns dict with keys:
      chosen, objective, iterations, circuit_depth, mode, n_qubits,
      constraint_violations  – total shots with Hamming weight ≠ n_sensors,
      params_moved           – bool: did COBYLA move (γ,β) from x0?
      best_ne_seed           – bool: best shot ≠ best shot from initial (unoptimised) state?
    """
    if not QISKIT_OK:
        raise ImportError("Install qiskit: pip install qiskit")
    if mode not in ("plain", "ws"):
        raise ValueError(f"mode must be 'plain' or 'ws', got {mode!r}")

    sites = sorted(viewsheds.keys())
    n = len(sites)
    rng = np.random.default_rng(seed)

    site_vs_weight = {s: sum(weights.get(c, 0.0) for c in vs)
                      for s, vs in viewsheds.items()}

    H_C = (build_penalty_cost_hamiltonian(viewsheds, weights, sites, n_sensors)
           if mode == "plain"
           else build_cost_hamiltonian(viewsheds, weights, sites))

    z_terms, zz_terms, const = _extract_ising(H_C, n)
    init_sv = _init_sv(n, n_sensors, mode)

    # ── Optimise ──────────────────────────────────────────────────────────────
    eval_count = [0]

    def objective(params):
        eval_count[0] += 1
        sv = _simulate(n, init_sv, params, p_layers, mode, z_terms, zz_terms)
        return _expectation(sv, z_terms, zz_terms, const, n)

    x0 = rng.uniform(0, np.pi, 2 * p_layers)
    result = minimize(objective, x0, method="COBYLA",
                      options={"maxiter": max_iter, "rhobeg": 0.5})

    # ── Decode from optimised state ───────────────────────────────────────────
    sv_opt = _simulate(n, init_sv, result.x, p_layers, mode, z_terms, zz_terms)
    probs_opt = np.abs(sv_opt) ** 2
    probs_opt /= probs_opt.sum()

    best_chosen, best_obj, best_bs, n_violations = _best_from_probs(
        probs_opt, n, n_sensors, sites, viewsheds, weights, site_vs_weight, shots, rng
    )

    # ── Sanity flag 1: did the optimiser actually move the parameters? ────────
    params_moved = bool(np.any(np.abs(result.x - x0) > 1e-3))

    # ── Sanity flag 2: is best shot ≠ best shot from the INITIAL (p=0) state? ─
    seed_probs = np.abs(init_sv) ** 2
    seed_probs /= seed_probs.sum()
    seed_rng = np.random.default_rng(seed + 99_999)
    _, _, seed_bs, _ = _best_from_probs(
        seed_probs, n, n_sensors, sites, viewsheds, weights, site_vs_weight, shots, seed_rng
    )
    best_ne_seed = (best_bs != seed_bs)

    # ── Fallback if no shot yielded a placement (shouldn't happen) ────────────
    if not best_chosen:
        from solvers.greedy import greedy_coverage
        best_chosen, best_obj = greedy_coverage(viewsheds, weights, n_sensors)
        best_ne_seed = False

    depth = _circuit_depth(n, p_layers, len(zz_terms), len(z_terms))

    return dict(
        chosen=best_chosen,
        objective=best_obj,
        iterations=eval_count[0],
        circuit_depth=depth,
        mode=mode,
        n_qubits=n,
        constraint_violations=n_violations,
        params_moved=params_moved,
        best_ne_seed=best_ne_seed,
    )
