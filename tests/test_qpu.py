"""HW1 circuits: the basis rotations measure the intended Pauli (exact statevector check, no QPU)."""
import numpy as np
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import transpile

from shadowopt.qpu import state_circuit, state_only, binding_values, ANGLES, grid_prediction, GRID, N_QUBITS


def _pauli_label(n, qs, bs):
    lab = ["I"] * n
    for q, b in zip(qs, bs):
        lab[n - 1 - q] = "XYZ"[b]
    return "".join(lab)


def test_basis_rotations_measure_intended_pauli():
    n = 4
    qc, a, b = state_circuit(n=n, layers=2, seed=7)
    psi = Statevector(state_only(n=n, layers=2, seed=7))
    rng = np.random.default_rng(0)
    for _ in range(20):
        setting = rng.integers(0, 3, size=(1, n))
        vals = binding_values(setting, list(qc.parameters))
        bound = qc.assign_parameters(dict(zip(qc.parameters, vals[0]))).remove_final_measurements(inplace=False)
        probs = Statevector(bound).probabilities_dict()
        # <Z...Z on a subset> after rotation = <P> on the state with P = setting on that subset
        for qs in ([0], [1, 2], [0, 1, 3], [0, 1, 2, 3]):
            ez = sum(p * (-1) ** sum(int(k[n - 1 - q]) for q in qs) for k, p in probs.items())
            ep = psi.expectation_value(SparsePauliOp(_pauli_label(n, qs, setting[0, qs]))).real
            assert abs(ez - ep) < 1e-9, (setting, qs, ez, ep)


def test_transpiled_parameters_survive():
    from qiskit.providers.fake_provider import GenericBackendV2
    be = GenericBackendV2(num_qubits=N_QUBITS + 4, basis_gates=["cz", "id", "rz", "sx", "x"], seed=1)
    qc, _, _ = state_circuit()
    t = transpile(qc, be, optimization_level=1, seed_transpiler=1)
    assert len(t.parameters) == 2 * N_QUBITS


def test_grid_matches_estimate():
    g = grid_prediction()
    assert abs(g["one_pass_s"] - 64.5) < 0.01 and abs(g["all_reps_s"] - 129.0) < 0.02
    assert len(GRID) == 13
