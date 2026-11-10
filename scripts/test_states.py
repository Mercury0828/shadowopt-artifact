"""Build the TEST states and their exact truth (prereg/M1.md §2, M1-A3). Run in the pinned .venv, one light process
(NUMBA/OMP/BLAS threads = 1, BelowNormal). Outputs stores/states/<name>/{state.npz, truth_<family>[_s<seed>].npz,
manifest.json}. Resumable per state (a state with a manifest is skipped). Usage: test_states.py [names...]"""
import json
import os
import sys
import time

for v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from shadowopt.states import expectations_statevector, mps_to_statevector
from shadowopt.states_test import (molecule_state, heisenberg_mps, mps_two_point, cluster_stabilizers,
                                   random_clifford_stabilizers, stabilizer_simulator, stabilizer_expectation)
from shadowopt.store import sha256_file
from shadowopt.workloads import w2_correlators, w3_cluster_fidelities, w4_zipf, w5_drift, w6_heavytail

OUT = os.path.join(ROOT, "stores", "states")
TEST_SEEDS = (100, 101, 102, 103, 104)
CLIFFORD_SEED = 20261108  # the seeded random Clifford state n = 50 (fixed here, before any TEST number)


def save(name, arrays, man):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    files = {}
    for fn, arrs in arrays.items():
        p = os.path.join(d, fn)
        np.savez(p, **arrs)
        files[fn] = sha256_file(p)
    man.update({"files": files, "split": "TEST", "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(name, json.dumps({k: v for k, v in man.items() if k != "files"})[:500], flush=True)


def terms_arrays(U, truth, coeff=None):
    a = {"ptr": U.ptr, "q": U.q, "b": U.b, "truth": np.asarray(truth, np.float64)}
    if coeff is not None:
        a["coeff"] = coeff
    return a


def build_molecule(name):
    t0 = time.time()
    n, psi, terms, const, E, info = molecule_state(name)
    ev = expectations_statevector(psi, terms)
    E_check = const + float(np.dot(terms.coeff, ev))
    assert abs(E_check - E) < 1e-8, (name, E_check, E)
    info.update({"n": n, "const": const, "E_from_terms": E_check, "E_check_abs_err": abs(E_check - E),
                 "build_s": time.time() - t0})
    save(name, {"state.npz": {"psi": psi}, "truth_W1.npz": terms_arrays(terms, ev, terms.coeff)}, info)


def build_heisenberg(n):
    t0 = time.time()
    B, info = heisenberg_mps(n, chi_max=128, cutoff=1e-10)
    U = w2_correlators(n).universe
    C = [mps_two_point(B, a) for a in range(3)]
    ev = np.empty(U.T)
    for t in range(U.T):
        qq, bb = U.term(t)
        ev[t] = C[int(bb[0])][int(qq[0]), int(qq[1])]
    info.update({"state": f"Heis{n}", "J": 1.0, "boundary": "open", "build_s": time.time() - t0,
                 "definition": "the stored state IS the MPS (right-canonical, renormalized); truth from the MPS",
                 "su2_check_max_abs_diff_xx_zz": float(np.max(np.abs(C[0] - C[2])))})
    save(f"Heis{n}", {"state.npz": {f"B{q}": b for q, b in enumerate(B)}, "truth_W2.npz": terms_arrays(U, ev)}, info)


def stab_truth(sim, n, U):
    return np.array([stabilizer_expectation(sim, n, *U.term(t)) for t in range(U.T)])


def build_cluster():
    n = 100
    stabs = cluster_stabilizers(n)
    sim = stabilizer_simulator(stabs)
    arrays = {"state.npz": {"stabilizers": np.array([str(s) for s in stabs])}}
    U = w3_cluster_fidelities(n).universe
    arrays["truth_W3.npz"] = terms_arrays(U, stab_truth(sim, n, U))
    for s in TEST_SEEDS:
        U6 = w6_heavytail(n, seed=s).universe
        arrays[f"truth_W6_s{s}.npz"] = terms_arrays(U6, stab_truth(sim, n, U6))
    save("Cluster100", arrays, {"state": "Cluster100", "n": n, "definition": "H on all qubits, CZ on (i, i+1)"})


def build_clifford():
    n = 50
    stabs = random_clifford_stabilizers(n, CLIFFORD_SEED)
    sim = stabilizer_simulator(stabs)
    arrays = {"state.npz": {"stabilizers": np.array([str(s) for s in stabs])}}
    for s in TEST_SEEDS:
        for fam, gen in (("W4", w4_zipf), ("W5", w5_drift)):
            U = gen(n, s).universe
            arrays[f"truth_{fam}_s{s}.npz"] = terms_arrays(U, stab_truth(sim, n, U))
    save("Clifford50", arrays, {"state": "Clifford50", "n": n, "clifford_seed": CLIFFORD_SEED,
                                "definition": "qiskit.quantum_info.random_clifford(50, seed) applied to |0...0>"})


BUILDERS = {"BeH2": lambda: build_molecule("BeH2"), "H2O": lambda: build_molecule("H2O"),
            "N2": lambda: build_molecule("N2"), "Heis50": lambda: build_heisenberg(50),
            "Heis100": lambda: build_heisenberg(100), "Cluster100": build_cluster, "Clifford50": build_clifford}


def main():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if os.environ.get("SHADOWOPT_CPU"):
            psutil.Process().cpu_affinity([int(os.environ["SHADOWOPT_CPU"])])
    except Exception as e:
        print("priority not lowered:", e)
    names = sys.argv[1:] or list(BUILDERS)
    for name in names:
        if os.path.exists(os.path.join(OUT, name, "manifest.json")):
            print(name, "exists, skipped", flush=True)
            continue
        BUILDERS[name]()


if __name__ == "__main__":
    main()
