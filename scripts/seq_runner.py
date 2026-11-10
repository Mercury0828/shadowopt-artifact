"""Integrated DEV sequence runner (December 2026 checkpoint; orientation check: "run the actual persistent engine").

One persistent store per state: starts with the first N0 rows of the DEV random store (stores/<state>/1e6), then for every
batch: (1) check (counts-only cache on the bitmap path, compiled optimized-split half-widths); (2) if some queries are not
certified, the feasible completion plan is ACQUIRED by sampling the actual state for the planned settings (simulated QPU;
usage charged with the HW1-fitted cost model, one job per batch); (3) rows appended (src "target"; pooled — route R3,
C-BOUND pending), index refreshed; (4) re-check (every query must now be certified — asserted); (5) answer every query from
the full (m, s) term cache; error vs exact truth recorded (coverage is descriptive here; H-c uses independent replicates).
Classical times per stage, QPU usage and the store size are logged per batch. Decisions never read outcomes (B2).
Output: derisk/results/SEQ/<state>_<family>_N0<N0>_seed<seed>.json
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gen_stores import load_state
from shadowopt.store import Store, TermBatch
from shadowopt.engine import AppendStore, IncrementalIndex, TermCache
from shadowopt.certify import BatchCSR, CountSource, check_and_plan, certify
from shadowopt.workloads import w1_energy, w2_correlators, w3_local_fidelities
from shadowopt.cost import hw1_fitted

OUT = os.path.join(ROOT, "derisk", "results", "SEQ")


def family_and_truth(state):
    d = np.load(os.path.join(ROOT, "stores", "states", state, {"LiH": "truth_W1.npz", "TFIM20": "truth_W2.npz",
                                                               "GHZ50": "truth_W3.npz"}[state]))
    if state == "LiH":
        man = json.load(open(os.path.join(ROOT, "stores", "states", "LiH", "manifest.json")))
        U = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
        fam = w1_energy(U, man["const"], eps_seq=(0.5, 0.3, 0.2, 0.1))
    elif state == "TFIM20":
        fam = w2_correlators(20)
    else:
        fam = w3_local_fidelities(50)
    U = fam.universe
    assert np.array_equal(U.ptr, d["ptr"]) and np.array_equal(U.q, d["q"]) and np.array_equal(U.b, d["b"]), \
        "truth file does not match the family's term universe"
    return fam, d["truth"]


def run(state, N0=100_000, seed=0, log=print):
    fam, truth = family_and_truth(state)
    U = fam.universe
    cost = hw1_fitted()
    n, sampler = load_state(state)
    base = Store.load(os.path.join(ROOT, "stores", state, "1e6"))
    app = AppendStore(n, cap=max(N0, 1 << 16))
    app.append(base.bases[:N0], base.outcomes[:N0])
    t0 = time.perf_counter()
    ix = IncrementalIndex(app, cap_rows=max(N0, 1 << 16))
    out = {"state": state, "family": fam.name, "N0": N0, "seed": seed, "cost": cost.as_dict(),
           "build_s": time.perf_counter() - t0, "batches": []}
    src = CountSource(ix, "bitmap", cached=True)
    full = TermCache(ix, "bitmap")
    rng = np.random.default_rng([seed, 9091])
    usage = 0.0
    for tau, batch in enumerate(fam.batches):
        rec = {"tau": tau, "N_before": int(app.N)}
        csr = BatchCSR(batch)
        t = time.perf_counter()
        r = check_and_plan(U, csr, src, cost)
        rec["check_s"] = time.perf_counter() - t
        rec["certified_before"] = int(r["ok"].sum())
        rec["queries"] = int(len(r["ok"]))
        plan = r["plan"]
        if plan:
            shots = [k for _, k in plan]
            u = cost.target(shots)
            usage += u
            t = time.perf_counter()
            B = np.concatenate([np.tile(np.asarray(s, np.uint8)[None, :], (k, 1)) for s, k in plan if k > 0])
            O = sampler(B, rng)
            rec["sample_s"] = time.perf_counter() - t
            t = time.perf_counter()
            app.append(B, O, src="target")
            rec["append_s"] = time.perf_counter() - t
            rec.update({"settings": len(plan), "shots": int(sum(shots)), "U_s": u})
            t = time.perf_counter()
            r2 = check_and_plan(U, csr, src, cost)
            rec["recheck_s"] = time.perf_counter() - t
            assert np.all(r2["ok"]), "completion did not certify every query"
            hw = r2["hw"]
        else:
            rec.update({"settings": 0, "shots": 0, "U_s": 0.0})
            hw = r["hw"]
        t = time.perf_counter()
        sub = U.subset(csr.terms)
        M, S = full.answer(sub)
        rec["answer_s"] = time.perf_counter() - t
        mu = S / np.maximum(M, 1)
        est = np.array([sum(c * mu[j] for j, c in zip(csr.qterm[a:b], csr.qcoef[a:b])) + q.const
                        for q, a, b in zip(batch, csr.qptr[:-1], csr.qptr[1:])])
        tru = np.array([sum(c * truth[csr.terms[j]] for j, c in zip(csr.qterm[a:b], csr.qcoef[a:b])) + q.const
                        for q, a, b in zip(batch, csr.qptr[:-1], csr.qptr[1:])])
        err = np.abs(est - tru)
        rec.update({"N_after": int(app.N), "max_err_over_hw": float(np.max(err / hw)),
                    "misses": int(np.sum(err > hw)), "max_hw_over_eps": float(np.max(hw / csr.qeps))})
        out["batches"].append(rec)
        log(state, tau, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
    out["usage_s"] = usage
    return out


def main():
    try:  # machine sharing: BelowNormal, optional single-core pinning (D:/_shared/compute_reservations.md)
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if os.environ.get("SHADOWOPT_CPU"):
            psutil.Process().cpu_affinity([int(os.environ["SHADOWOPT_CPU"])])
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    states = [a for a in sys.argv[1:] if not a.startswith("--")] or ["TFIM20", "LiH", "GHZ50"]
    for state in states:
        o = run(state)
        json.dump(o, open(os.path.join(OUT, f"{state}_{o['family']}_N0{o['N0']}_seed{o['seed']}.json"), "w"),
                  indent=1, default=float)
        print(state, "total usage", round(o["usage_s"], 2), flush=True)


if __name__ == "__main__":
    main()
