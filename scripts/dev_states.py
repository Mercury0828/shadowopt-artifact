"""Build the three DEV states (guide §P6.2) and their exact expectation values for the Phase-0 workloads.

Outputs stores/states/<name>/{state.npz, truth_<family>.npz, manifest.json}. Deterministic (no randomness except the
Lanczos start vector, seeded).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.states import (lih_state, tfim_ground_state, statevector_to_mps, mps_to_statevector,
                              expectations_statevector, ghz_expectation)
from shadowopt.store import sha256_file
from shadowopt.workloads import w1_energy, w2_correlators, w3_local_fidelities

OUT = os.path.join(os.path.dirname(__file__), "..", "stores", "states")


def save(name, arrays, man):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    files = {}
    for fn, arrs in arrays.items():
        p = os.path.join(d, fn)
        np.savez(p, **arrs)
        files[fn] = sha256_file(p)
    man["files"] = files
    man["built_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(name, json.dumps({k: v for k, v in man.items() if k != "files"})[:600])


def main():
    # S1 LiH (DEV)
    n, psi, terms, const, E0, info = lih_state()
    ev = expectations_statevector(psi, terms)
    E_check = const + float(np.dot(terms.coeff, ev))
    info.update({"n": n, "const": const, "E_from_terms": E_check, "E_check_abs_err": abs(E_check - E0),
                 "state": "LiH", "split": "DEV"})
    assert abs(E_check - E0) < 1e-9, (E_check, E0)
    fam = w1_energy(terms, const)
    save("LiH", {"state.npz": {"psi": psi},
                 "truth_W1.npz": {"ptr": terms.ptr, "q": terms.q, "b": terms.b, "coeff": terms.coeff, "truth": ev}},
         info)

    # S2 TFIM n = 20, J = 1, h = 1 (critical), open boundary
    n = 20
    E0, psi_exact, H = tfim_ground_state(n, 1.0, 1.0, seed=0)
    tens, disc = statevector_to_mps(psi_exact, n, chi_max=64, cutoff=1e-8)
    psi_mps = mps_to_statevector(tens)
    nrm = np.linalg.norm(psi_mps)
    psi_mps = psi_mps / nrm
    fid = abs(np.vdot(psi_mps, psi_exact)) ** 2
    E_mps = float(np.real(np.vdot(psi_mps, H @ psi_mps)))
    chis = [t.shape[2] for t in tens]
    fam = w2_correlators(n)
    ev = expectations_statevector(psi_mps, fam.universe)
    info = {"state": "TFIM", "n": n, "J": 1.0, "h": 1.0, "boundary": "open", "E0_exact_lanczos": E0, "E_mps": E_mps,
            "fidelity_mps_vs_exact": fid, "chi_max_used": int(max(chis)), "chi_cap": 64, "svd_cutoff_rel": 1e-8,
            "discarded_weight": disc, "mps_norm_before_renorm": float(nrm), "split": "DEV",
            "definition": "the stored state is the MPS (renormalized); truth computed from its statevector"}
    U = fam.universe
    save("TFIM20", {"state.npz": {**{f"B{q}": t for q, t in enumerate(tens)}, "psi": psi_mps},
                    "truth_W2.npz": {"ptr": U.ptr, "q": U.q, "b": U.b, "truth": ev}}, info)

    # S3 GHZ n = 50
    n = 50
    fam = w3_local_fidelities(n)
    U = fam.universe
    ev = np.array([ghz_expectation(n, *U.term(t)) for t in range(U.T)])
    # cross-check with stim (exact stabilizer expectation)
    import stim
    sim = stim.TableauSimulator()
    sim.h(0)
    for q in range(n - 1):
        sim.cnot(q, q + 1)
    mism = 0
    for t in range(U.T):
        s = ["_"] * n
        qq, bb = U.term(t)
        for qi, bi in zip(qq, bb):
            s[qi] = "XYZ"[bi]
        val = sim.peek_observable_expectation(stim.PauliString("".join(s)))
        if abs(val - ev[t]) > 1e-12:
            mism += 1
    rng = np.random.default_rng(123)
    mism_rand = 0
    for _ in range(5000):
        k = int(rng.integers(1, n + 1))
        qq = sorted(rng.choice(n, size=k, replace=False).tolist())
        bb = rng.integers(0, 3, size=k).tolist()
        s = ["_"] * n
        for qi, bi in zip(qq, bb):
            s[qi] = "XYZ"[bi]
        if abs(sim.peek_observable_expectation(stim.PauliString("".join(s))) - ghz_expectation(n, qq, bb)) > 1e-12:
            mism_rand += 1
    info = {"state": "GHZ50", "n": n, "split": "DEV", "W3_terms": U.T, "W3_queries": len(fam.batches[0]),
            "stim_mismatch_W3_terms": mism, "stim_mismatch_5000_random_terms": mism_rand,
            "definition": "analytic GHZ sampler and truth; truth cross-checked with stim peek_observable_expectation"}
    assert mism == 0 and mism_rand == 0
    save("GHZ50", {"truth_W3.npz": {"ptr": U.ptr, "q": U.q, "b": U.b, "truth": ev}}, info)


if __name__ == "__main__":
    main()
