"""HW1 (guide §0e, §P6.5): cost-model fit on ibm_cleveland. Circuits, the approved job grid, and the usage model.

Nothing here submits a job; scripts/hw1_run.py does, and only with `--approved HW1 --cap <s>` and the pinned estimate.

State: a fixed 12-qubit hardware-efficient state (2 layers of seeded RY angles + a CZ chain). Measurement basis per qubit
via a parameterized rotation RX(alpha_i) RY(beta_i) applied before measuring Z:
  Z: (0, 0);  X: (0, -pi/2) [RY(-pi/2) maps X to Z];  Y: (pi/2, 0) [RX(pi/2) maps Y to Z]
(verified in tests/test_qpu.py against Pauli expectations of the state). Basis codes 0 = X, 1 = Y, 2 = Z (store.py).
"""
import hashlib
import math

import numpy as np

N_QUBITS = 12
HW1_ESTIMATE_SHA256 = "1839c3ce161b4072af4323b4d4d21f578b655bf6c391b265507e90306f9054c8"  # approved 2026-11-07 (DD #11)
HW1_APPROVED_CAP_S = 200.0
ANGLES = {0: (0.0, -math.pi / 2), 1: (math.pi / 2, 0.0), 2: (0.0, 0.0)}

# The approved grid (qpu/HW1_estimate.md): name, settings S, shots s, mode, role.
# mode "param" = one parameterized PUB with S parameter bindings; "circ" = S separately bound circuits (S PUBs);
# "single" = one fixed circuit (Z basis everywhere is not special: a seeded random setting, bound locally).
# "random" in the estimate = random bases, i.e. the APPEND form; settings are always seeded random here, which does not
# change the cost structure of any job (the cost depends on S, s and mode only).
GRID = [
    ("A", 1, 1000, "single", "train"),
    ("B", 1, 10000, "single", "train"),
    ("C", 10, 1000, "param", "train"),
    ("D1", 100, 100, "param", "train"),
    ("D2", 100, 100, "circ", "train"),
    ("E1", 1000, 10, "param", "train"),
    ("E2", 1000, 10, "circ", "train"),
    ("F1", 1000, 1, "param", "train"),
    ("F2", 1000, 1, "circ", "train"),
    ("G", 5000, 1, "param", "train"),
    ("H", 300, 30, "param", "heldout"),
    ("I", 3000, 1, "param", "heldout"),
    ("J", 10, 3000, "param", "heldout"),
]
REPETITIONS = 2


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def predicted_usage(S, s, t_sub=2.0, t_set=0.0, t_shot=350e-6, exec_per_subjob=100_000):
    """Documented / planning model of the estimate: U = n_sub t_sub + S t_set + S s t_shot."""
    n_exec = S * s
    n_sub = math.ceil(n_exec / exec_per_subjob)
    return n_sub * t_sub + S * t_set + n_exec * t_shot


def grid_prediction():
    one = sum(predicted_usage(S, s) for _, S, s, _, _ in GRID)
    return {"one_pass_s": one, "all_reps_s": one * REPETITIONS, "with_margin_s": one * REPETITIONS * 1.3}


def state_circuit(n=N_QUBITS, layers=2, seed=20261107):
    """Logical circuit: seeded hardware-efficient state + parameterized basis layer + measurement of all qubits."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n, n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    a = ParameterVector("a", n)
    b = ParameterVector("b", n)
    for q in range(n):
        qc.ry(b[q], q)
        qc.rx(a[q], q)
    qc.measure(range(n), range(n))
    return qc, list(a), list(b)


def state_only(n=N_QUBITS, layers=2, seed=20261107):
    """The prepared state without basis layer or measurement (for exact Pauli expectations in tests)."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(float(rng.uniform(0, 2 * np.pi)), q)
        for q in range(n - 1):
            qc.cz(q, q + 1)
    return qc


def binding_values(settings, params_order):
    """settings (S, n) basis codes -> parameter value array (S, 2n) in the order of `params_order` (a list of Parameter
    objects of the transpiled circuit; names a[i] / b[i])."""
    S, n = settings.shape
    out = np.zeros((S, len(params_order)))
    for j, p in enumerate(params_order):
        name = p.name  # e.g. "a[3]"
        kind, idx = name[0], int(name[name.index("[") + 1:-1])
        for r in range(S):
            al, be = ANGLES[int(settings[r, idx])]
            out[r, j] = al if kind == "a" else be
    return out


def job_settings(name, rep, S, n=N_QUBITS):
    """Seeded random settings for one grid job (seed = hash of name and repetition; recorded in the ledger)."""
    seed = int.from_bytes(hashlib.sha256(f"HW1|{name}|{rep}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed).integers(0, 3, size=(S, n), dtype=np.uint8), seed
