"""TEST-state builders checked on small instances against independent references."""
import numpy as np

from shadowopt.states import expectations_statevector, mps_to_statevector, tfim_hamiltonian
from shadowopt.states_test import (heisenberg_mps, mps_two_point, cluster_stabilizers, random_clifford_stabilizers,
                                   stabilizer_simulator, stabilizer_expectation, molecule_state)
from shadowopt.store import terms_from_pairs


def _heis_dense(n):
    X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)
    H = np.zeros((2 ** n, 2 ** n), complex)
    for i in range(n - 1):
        for P in (X, Y, Z):
            ops = [np.eye(2)] * n
            ops[i], ops[i + 1] = P, P
            M = ops[n - 1]
            for q in range(n - 2, -1, -1):  # little-endian: qubit 0 is the least significant factor
                M = np.kron(M, ops[q])
            H += M
    return H


def test_heisenberg_small_exact():
    n = 8
    B, info = heisenberg_mps(n, chi_max=16, cutoff=1e-12)
    psi = mps_to_statevector(B)
    H = _heis_dense(n)
    w, v = np.linalg.eigh(H)
    assert abs(np.vdot(psi, H @ psi).real - w[0]) < 1e-8
    assert abs(np.linalg.norm(psi) - 1) < 1e-10
    pairs = [[(i, a), (j, a)] for a in range(3) for i in range(n) for j in range(i + 1, n)]
    U = terms_from_pairs(n, pairs)
    ev = expectations_statevector(psi, U)
    k = 0
    for a in range(3):
        C = mps_two_point(B, a)
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(C[i, j] - ev[k]) < 1e-9
                k += 1


def test_stabilizer_states():
    for stabs in (cluster_stabilizers(12), random_clifford_stabilizers(10, seed=100)):
        n = len(stabs)
        sim = stabilizer_simulator(stabs)
        for s in stabs:
            assert sim.peek_observable_expectation(s) == 1
    sim = stabilizer_simulator(cluster_stabilizers(5))
    assert stabilizer_expectation(sim, 5, [0, 1], [0, 2]) == 1.0  # X0 Z1
    assert stabilizer_expectation(sim, 5, [0], [2]) == 0.0


def test_beh2_energy_from_terms():
    n, psi, terms, const, E, info = molecule_state("BeH2")
    assert n == 14 and abs(np.linalg.norm(psi) - 1) < 1e-9
    ev = expectations_statevector(psi, terms)
    assert abs(const + float(np.dot(terms.coeff, ev)) - E) < 1e-8
