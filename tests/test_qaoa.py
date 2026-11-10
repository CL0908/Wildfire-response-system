"""Checks on the QAOA building blocks: cost, mixer, decoding."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers import qaoa

VIEWSHEDS = {0: {0, 1, 2}, 1: {0, 1, 3}, 2: {0, 4}, 3: {1}}
WEIGHTS = {0: 3.0, 1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 4.0}
SITES = sorted(VIEWSHEDS)
N = len(SITES)


def coverage(chosen):
    covered = set().union(*(VIEWSHEDS[s] for s in chosen)) if chosen else set()
    return sum(WEIGHTS[c] for c in covered)


def sites_of(idx):
    return [SITES[q] for q in range(N) if idx & (1 << (N - 1 - q))]


def test_coverage_diagonal_matches_set_union():
    cov, hamming = qaoa.coverage_diagonal(VIEWSHEDS, WEIGHTS, SITES)
    for idx in range(2 ** N):
        assert abs(cov[idx] - coverage(sites_of(idx))) < 1e-12
        assert hamming[idx] == len(sites_of(idx))


def _ising_energy(H, idx):
    z_terms, zz_terms, const = qaoa._extract_ising(H, N)
    z = lambda q: -1.0 if idx & (1 << (N - 1 - q)) else 1.0
    return const + sum(c * z(q) for q, c in z_terms) + sum(c * z(a) * z(b) for a, b, c in zz_terms)


def test_two_local_cost_is_exact_up_to_two_sensors_per_cell():
    # With N=2 at most two chosen sensors see any cell, so the fit is exact.
    H = qaoa.build_cost_hamiltonian(VIEWSHEDS, WEIGHTS, SITES, n_sensors=2)
    for idx in range(2 ** N):
        chosen = sites_of(idx)
        if len(chosen) <= 2:
            assert abs(_ising_energy(H, idx) + coverage(chosen)) < 1e-9, chosen


def test_penalty_hamiltonian_ground_state_is_the_optimum():
    H = qaoa.build_penalty_cost_hamiltonian(VIEWSHEDS, WEIGHTS, SITES, n_sensors=2)
    best = min(range(2 ** N), key=lambda i: _ising_energy(H, i))
    optimum = max(coverage(c) for c in itertools.combinations(SITES, 2))
    assert len(sites_of(best)) == 2
    assert abs(coverage(sites_of(best)) - optimum) < 1e-9


def test_xy_gate_matches_qiskit_and_keeps_hamming_weight():
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import Operator, Statevector
    from qiskit import QuantumCircuit

    rng = np.random.default_rng(1)
    sv = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    sv /= np.linalg.norm(sv)
    beta, q0, q1 = 0.37, 0, 2
    out = qaoa._apply_xy(sv, q0, q1, beta, N)

    # Our index puts qubit q at bit N-1-q; Qiskit puts it at bit q.
    rev = np.array([int(format(i, f"0{N}b")[::-1], 2) for i in range(2 ** N)])
    qc = QuantumCircuit(N)
    qc.append(XXPlusYYGate(2 * beta), [q0, q1])
    expected = Statevector(sv[rev]).evolve(Operator(qc)).data[rev]
    assert np.allclose(out, expected)

    hamming = np.array([bin(i).count("1") for i in range(2 ** N)])
    for k in range(N + 1):
        assert abs(np.sum(np.abs(sv[hamming == k]) ** 2)
                   - np.sum(np.abs(out[hamming == k]) ** 2)) < 1e-12


def test_xy_mode_never_leaves_the_feasible_subspace():
    r = qaoa.run_qaoa(VIEWSHEDS, WEIGHTS, 2, p_layers=2, mode="xy", max_iter=40, shots=200)
    assert abs(r["p_feasible"] - 1.0) < 1e-9
    assert r["constraint_violations"] == 0
    assert len(r["chosen"]) == 2
    assert abs(r["objective"] - coverage(r["chosen"])) < 1e-9
    assert r["objective"] <= r["optimum"] + 1e-9


def test_simulator_decode_maps_bits_to_the_right_sites():
    # All probability on "sites 0 and 1 selected": not symmetric under bit reversal.
    idx = (1 << (N - 1 - 0)) | (1 << (N - 1 - 1))
    probs = np.zeros(2 ** N)
    probs[idx] = 1.0
    site_w = {s: sum(WEIGHTS[c] for c in vs) for s, vs in VIEWSHEDS.items()}
    chosen, obj, _, violations = qaoa._best_from_probs(
        probs, N, 2, SITES, VIEWSHEDS, WEIGHTS, site_w, 10, np.random.default_rng(0))
    assert sorted(chosen) == [0, 1]
    assert violations == 0


def test_hardware_counts_decode_uses_qiskit_bit_order():
    # Qiskit prints qubit 0 as the rightmost character.
    site_w = {s: sum(WEIGHTS[c] for c in vs) for s, vs in VIEWSHEDS.items()}
    chosen, _, _, violations = qaoa._best_from_counts(
        {"0011": 7}, N, 2, SITES, VIEWSHEDS, WEIGHTS, site_w)
    assert sorted(chosen) == [0, 1]
    assert violations == 0


def test_run_qaoa_rejects_more_sensors_than_sites():
    import pytest
    with pytest.raises(ValueError):
        qaoa.run_qaoa(VIEWSHEDS, WEIGHTS, N + 1, mode="xy")


def test_hardware_circuit_matches_the_simulator():
    """The Qiskit circuit sent to hardware must be the circuit the angles were tuned on."""
    from qiskit.quantum_info import Statevector
    from solvers.qaoa_hardware import _build_circuit

    n_sensors, p = 2, 2
    params = np.array([0.4, 1.1, 0.3, 0.8])
    rev = np.array([int(format(i, f"0{N}b")[::-1], 2) for i in range(2 ** N)])
    for mode in ("plain", "xy"):
        H = (qaoa.build_penalty_cost_hamiltonian(VIEWSHEDS, WEIGHTS, SITES, n_sensors)
             if mode == "plain"
             else qaoa.build_cost_hamiltonian(VIEWSHEDS, WEIGHTS, SITES, n_sensors))
        z_terms, zz_terms, _ = qaoa._extract_ising(H, N)
        sim = qaoa._simulate(N, qaoa._init_sv(N, n_sensors, mode), params, p, mode,
                             z_terms, zz_terms)
        qc = _build_circuit(N, n_sensors, z_terms, zz_terms, params, p, mode)
        qc.remove_final_measurements()
        circuit_probs = Statevector(qc).probabilities()[rev]
        assert np.allclose(np.abs(sim) ** 2, circuit_probs, atol=1e-9), mode


def test_lp_relaxation_is_a_valid_fractional_placement():
    c = qaoa.lp_relaxation(VIEWSHEDS, WEIGHTS, SITES, 2)
    assert len(c) == N and np.all(c >= 0) and np.all(c <= 1)
    assert abs(c.sum() - 2) < 1e-6


def test_warm_start_state_and_conserved_sensor_count():
    c = np.array([0.75, 0.75, 0.25, 0.25])
    sv = qaoa._product_state(c)
    assert abs(np.sum(np.abs(sv) ** 2) - 1) < 1e-12
    # P(site q selected) must equal c_q, with qubit q at bit N-1-q.
    for q in range(N):
        on = (np.arange(2 ** N) & (1 << (N - 1 - q))) != 0
        assert abs(np.sum(np.abs(sv[on]) ** 2) - c[q]) < 1e-12

    r = qaoa.run_qaoa(VIEWSHEDS, WEIGHTS, 2, p_layers=2, mode="wsxy", max_iter=60,
                      shots=200, warm_start=list(c))
    # The XY mixer cannot change how many sensors are selected.
    _, hamming = qaoa.coverage_diagonal(VIEWSHEDS, WEIGHTS, SITES)
    assert abs(r["p_feasible"] - np.sum(np.abs(sv[hamming == 2]) ** 2)) < 1e-9
    # The zero-angle start reproduces the warm start, so the result cannot be worse than it.
    assert r["mean_ratio"] >= r["start_mean_ratio"] - 1e-9
