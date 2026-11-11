"""
QAOA family on a pure-NumPy statevector simulator.

  plain : uniform start, X mixer, cardinality enforced by a penalty term.
  xy    : Dicke start (uniform over all N-sensor placements) with a ring XY
          mixer, which keeps every amplitude at exactly N sensors.
  wsxy  : warm start with the XY mixer. Solves the LP relaxation of the
          placement problem classically, clips each site's value c_i to
          [eps, 1-eps], and starts every qubit at R_Y(2*arcsin(sqrt(c_i)))|0>.
          The XY mixer and the cost both conserve the number of selected
          sensors, so P(exactly N sensors) is fixed by this starting state and
          the circuit cannot improve it. Its baseline is the starting state
          itself (all angles zero), not random placement.

run_qaoa applies the exact coverage objective as a diagonal phase. That is an
idealised cost oracle: the true objective has terms up to k-body (k = number
of sites seeing a cell), so it is not a shallow hardware circuit. The 2-local
Hamiltonians below are an approximation of it, used by qaoa_hardware.py.

Results are reported without classical repair, next to the baseline of
drawing N-sensor placements uniformly at random.
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
    # Qubit q is bit n-1-q, so it is axis 1 of a (2^q, 2, rest) view.
    v = sv.reshape(1 << q, 2, -1)
    out = np.empty_like(v)
    out[:, 0] = c * v[:, 0] + s * v[:, 1]
    out[:, 1] = s * v[:, 0] + c * v[:, 1]
    return out.reshape(-1)


def _apply_xy(sv, q0, q1, beta, n):
    """exp(-i*beta*(XX+YY)/2) on (q0, q1): rotates |01> <-> |10>, so the number of 1s is kept."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    # Qubit q is axis q of the (2,)*n tensor; bring the pair to the front.
    t = np.moveaxis(sv.reshape((2,) * n), (q0, q1), (0, 1)).copy()
    a10, a01 = t[1, 0].copy(), t[0, 1].copy()
    t[1, 0] = c * a10 + s * a01
    t[0, 1] = s * a10 + c * a01
    return np.ascontiguousarray(np.moveaxis(t, (0, 1), (q0, q1))).reshape(-1)


def _xy_pairs(n):
    """Ring of neighbouring pairs: even pairs, then odd pairs, then the closing pair."""
    pairs = [(q, q + 1) for q in range(0, n - 1, 2)] + [(q, q + 1) for q in range(1, n - 1, 2)]
    if n > 2:
        pairs.append((n - 1, 0))
    return pairs


def _apply_mixer(sv, beta, n, mode):
    if mode == "plain":
        for q in range(n):
            sv = _apply_rx(sv, q, 2 * beta, n)
    else:
        for q0, q1 in _xy_pairs(n):
            sv = _apply_xy(sv, q0, q1, beta, n)
    return sv


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


def _or_fit(m_max: int) -> tuple[float, float]:
    """
    Coefficients (a, b) of the quadratic a*m + b*m*(m-1)/2 that best fits
    OR(m) = min(1, m) for m = 1..m_max chosen sensors seeing a cell.
    Exact for m_max <= 2 (a=1, b=-1); least squares beyond that.
    """
    if m_max <= 1:
        return 1.0, 0.0
    m = np.arange(1, m_max + 1, dtype=float)
    A = np.stack([m, m * (m - 1) / 2], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, np.ones_like(m), rcond=None)
    return float(a), float(b)


def _add_coverage_terms(add, viewsheds, weights, sites, n_sensors=None):
    """Accumulate the 2-local approximation of -sum_c w_c * OR_c, y_i = (1-Z_i)/2."""
    n = len(sites)
    s_to_q = {s: q for q, s in enumerate(sites)}
    I = "I" * n

    def z_label(*qs):
        lbl = list(I)
        for q in qs:
            lbl[n - 1 - q] = "Z"
        return "".join(lbl)

    cell_to_sites: dict = {}
    for s in sites:
        for c in viewsheds[s]:
            cell_to_sites.setdefault(c, []).append(s)

    for c, w in weights.items():
        seeing = [s_to_q[s] for s in cell_to_sites.get(c, [])]
        if not seeing:
            continue
        m_max = len(seeing) if n_sensors is None else min(len(seeing), n_sensors)
        a, b = _or_fit(m_max)
        for q in seeing:                       # -w*a*y_q
            add(I, -w * a / 2)
            add(z_label(q), w * a / 2)
        for i, q0 in enumerate(seeing):        # -w*b*y_q0*y_q1
            for q1 in seeing[i + 1:]:
                add(I, -w * b / 4)
                add(z_label(q0), w * b / 4)
                add(z_label(q1), w * b / 4)
                add(z_label(q0, q1), -w * b / 4)


def _to_pauli_op(pd, n):
    terms = [(lbl, c) for lbl, c in pd.items() if abs(c) > 1e-10] or [("I" * n, 0.0)]
    return SparsePauliOp.from_list(terms)


def build_cost_hamiltonian(viewsheds, weights, sites, n_sensors=None):
    """
    2-local approximation of H_C = -sum_c w_c * OR(y_i : i sees c).
    Exact while at most two chosen sensors see any one cell; a least-squares
    quadratic fit otherwise. Pass n_sensors to fit only up to N chosen sensors.
    """
    pd: dict[str, float] = {}

    def add(lbl, v):
        pd[lbl] = pd.get(lbl, 0.0) + v

    _add_coverage_terms(add, viewsheds, weights, sites, n_sensors)
    return _to_pauli_op(pd, len(sites))


def _default_penalty(viewsheds, weights):
    """Just above the most a single extra sensor can add to the objective."""
    return 1.5 * max(sum(weights.get(c, 0.0) for c in vs) for vs in viewsheds.values())


def build_penalty_cost_hamiltonian(viewsheds, weights, sites, n_sensors, P1=None):
    """build_cost_hamiltonian plus the cardinality penalty P1*(sum y_i - N)^2."""
    if P1 is None:
        P1 = _default_penalty(viewsheds, weights)
    n = len(sites)
    I = "I" * n
    pd: dict[str, float] = {}

    def add(lbl, v):
        pd[lbl] = pd.get(lbl, 0.0) + v

    _add_coverage_terms(add, viewsheds, weights, sites, n_sensors)

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

    return _to_pauli_op(pd, n)


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
        sv = _apply_mixer(sv, float(betas[l]), n, mode)
    return sv


def coverage_diagonal(viewsheds, weights, sites):
    """
    Exact covered priority of every one of the 2^n placements.
    Site sites[q] is selected in basis state idx iff bit (n-1-q) of idx is set.
    Returns (coverage, hamming_weight) arrays of length 2^n.
    """
    n = len(sites)
    idxs = np.arange(2 ** n)
    mask_weight: dict[int, float] = {}
    cell_mask: dict[int, int] = {}
    for q, s in enumerate(sites):
        for c in viewsheds[s]:
            cell_mask[c] = cell_mask.get(c, 0) | (1 << (n - 1 - q))
    for c, mask in cell_mask.items():
        mask_weight[mask] = mask_weight.get(mask, 0.0) + weights.get(c, 0.0)

    coverage = np.zeros(2 ** n)
    for mask, w in mask_weight.items():
        coverage[idxs & mask != 0] += w
    hamming = np.zeros(2 ** n, dtype=int)
    for q in range(n):
        hamming += (idxs >> q) & 1
    return coverage, hamming


def _simulate_diag(n, init_sv, params, p, mode, phase_diag):
    """QAOA with the cost applied as an exact diagonal phase."""
    sv = init_sv.copy()
    gammas, betas = params[:p], params[p:]
    for l in range(p):
        sv = sv * np.exp(-1j * float(gammas[l]) * phase_diag)
        sv = _apply_mixer(sv, float(betas[l]), n, mode)
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


def lp_relaxation(viewsheds, weights, sites, n_sensors):
    """Site values c_i in [0, 1] from the LP relaxation of weighted maximum coverage."""
    from scipy.optimize import linprog

    n = len(sites)
    cell_to_sites: dict = {}
    for q, s in enumerate(sites):
        for c in viewsheds[s]:
            if weights.get(c, 0.0) > 0:
                cell_to_sites.setdefault(c, []).append(q)
    cells = sorted(cell_to_sites)

    # Variables: y_0..y_{n-1}, then one z per cell. Maximise sum w_c z_c.
    cost = np.concatenate([np.zeros(n), [-weights[c] for c in cells]])
    A_ub = np.zeros((len(cells), n + len(cells)))
    for j, c in enumerate(cells):                      # z_c - sum y_i <= 0
        A_ub[j, n + j] = 1.0
        A_ub[j, cell_to_sites[c]] = -1.0
    A_eq = np.concatenate([np.ones(n), np.zeros(len(cells))])[None, :]
    res = linprog(cost, A_ub=A_ub, b_ub=np.zeros(len(cells)), A_eq=A_eq,
                  b_eq=[n_sensors], bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"LP relaxation failed: {res.message}")
    return np.clip(res.x[:n], 0.0, 1.0)


def _product_state(c):
    """Tensor product of R_Y(2*arcsin(sqrt(c_q)))|0>; qubit q is bit n-1-q of the index."""
    n = len(c)
    idxs = np.arange(2 ** n)
    sv = np.ones(2 ** n, dtype=complex)
    for q in range(n):
        on = idxs & (1 << (n - 1 - q)) != 0
        sv *= np.where(on, np.sqrt(c[q]), np.sqrt(1.0 - c[q]))
    return sv


def _init_sv(n, n_sensors, mode, warm_c=None):
    sv = np.zeros(2 ** n, dtype=complex)
    if mode == "plain":
        sv[:] = 1.0 / np.sqrt(2 ** n)
    elif mode == "wsxy":
        sv = _product_state(warm_c)
    else:  # Dicke state
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
        # Reverse so qubit q sits at bs[n-1-q], the order _repair expects (Qiskit's).
        bs = format(int(idx), f"0{n}b")[::-1]
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
    restarts: int = 5,
    warm_start: list[float] | None = None,
    warm_eps: float = 0.25,
) -> dict:
    """
    mode in {'plain', 'xy', 'wsxy'} (see module docstring). No classical repair
    is applied. COBYLA is run from `restarts` random starting points; the run
    with the lowest expected cost is kept. In 'wsxy' mode the first starting
    point is all zeros (the warm-start state itself). `warm_start` gives c_i
    per site in sorted-site order (LP relaxation if omitted); values are
    clipped to [warm_eps, 1 - warm_eps].

    Returns dict with keys:
      chosen, objective      - best feasible sampled placement ([] and 0.0 if none)
      optimum                - best N-sensor objective, by enumeration
      p_feasible             - probability of measuring exactly N sensors
      p_optimal              - probability of measuring an optimal placement
      mean_ratio             - expected objective / optimum, infeasible shots count 0
      p_optimal_random, mean_ratio_random
                             - the same two numbers for a uniformly random
                               N-sensor placement (the no-circuit baseline)
      constraint_violations  - sampled shots with Hamming weight != N
      params_moved           - did COBYLA move the parameters from their start?
      warm_start, start_mean_ratio, start_p_optimal
                             - 'wsxy' only: the clipped c_i, and mean_ratio and
                               p_optimal of the starting state before any layer
      iterations             - cost evaluations summed over all restarts
      shots, mode, n_qubits
    """
    if mode not in ("plain", "xy", "wsxy"):
        raise ValueError(f"mode must be 'plain', 'xy' or 'wsxy', got {mode!r}")

    sites = sorted(viewsheds.keys())
    n = len(sites)
    if not 0 < n_sensors <= n:
        raise ValueError(f"n_sensors must be between 1 and {n} (the number of sites), got {n_sensors}")
    rng = np.random.default_rng(seed)

    coverage, hamming = coverage_diagonal(viewsheds, weights, sites)
    feasible = hamming == n_sensors
    optimum = float(coverage[feasible].max())
    is_optimal = feasible & (coverage >= optimum - 1e-9)

    if mode == "plain":
        cost = -coverage + _default_penalty(viewsheds, weights) * (hamming - n_sensors) ** 2
        support = np.ones(2 ** n, dtype=bool)
    else:
        cost = -coverage
        support = feasible
    warm_c = None
    if mode == "wsxy":
        if warm_start is None:
            warm_start = lp_relaxation(viewsheds, weights, sites, n_sensors)
        if len(warm_start) != n:
            raise ValueError(f"warm_start needs one value per site ({n}), got {len(warm_start)}")
        warm_c = np.clip(np.asarray(warm_start, dtype=float), warm_eps, 1.0 - warm_eps)
        # Score only N-sensor outcomes: the other sectors are frozen by the XY mixer.
        cost = np.where(feasible, cost, 0.0)
        support = np.ones(2 ** n, dtype=bool)
    init_sv = _init_sv(n, n_sensors, mode, warm_c)

    # Rescale the phase so gamma in [0, pi] spans the cost range once.
    spread = float(cost[support].max() - cost[support].min()) or 1.0
    phase_diag = cost / spread

    eval_count = [0]

    def objective(params):
        eval_count[0] += 1
        sv = _simulate_diag(n, init_sv, params, p_layers, mode, phase_diag)
        return float((np.abs(sv) ** 2) @ cost)

    result = x0 = None
    for attempt in range(max(1, restarts)):
        start = rng.uniform(0, np.pi, 2 * p_layers)
        if mode == "wsxy" and attempt == 0:
            start = np.zeros(2 * p_layers)
        res = minimize(objective, start, method="COBYLA",
                       options={"maxiter": max_iter, "rhobeg": 0.5})
        if result is None or res.fun < result.fun:
            result, x0 = res, start

    sv_opt = _simulate_diag(n, init_sv, result.x, p_layers, mode, phase_diag)
    probs = np.abs(sv_opt) ** 2
    probs /= probs.sum()

    feasible_cov = np.where(feasible, coverage, 0.0)
    samples = rng.choice(len(probs), size=shots, p=probs)
    sampled_cov = feasible_cov[samples]
    best_idx = int(samples[int(np.argmax(sampled_cov))])
    if feasible[best_idx]:
        chosen = [sites[q] for q in range(n) if best_idx & (1 << (n - 1 - q))]
        best_obj = float(coverage[best_idx])
    else:
        chosen, best_obj = [], 0.0

    extra = {}
    if mode == "wsxy":
        start_probs = np.abs(init_sv) ** 2
        extra = dict(
            warm_start=[float(c) for c in warm_c],
            start_mean_ratio=float(start_probs @ feasible_cov) / optimum if optimum > 0 else 0.0,
            start_p_optimal=float(start_probs[is_optimal].sum()),
        )

    return dict(
        **extra,
        chosen=chosen,
        objective=best_obj,
        optimum=optimum,
        p_feasible=float(probs[feasible].sum()),
        p_optimal=float(probs[is_optimal].sum()),
        mean_ratio=float(probs @ feasible_cov) / optimum if optimum > 0 else 0.0,
        p_optimal_random=float(is_optimal.sum() / feasible.sum()),
        mean_ratio_random=float(coverage[feasible].mean()) / optimum if optimum > 0 else 0.0,
        constraint_violations=int((~feasible[samples]).sum()),
        params_moved=bool(np.any(np.abs(result.x - x0) > 1e-3)),
        iterations=eval_count[0],
        shots=shots,
        mode=mode,
        n_qubits=n,
    )
