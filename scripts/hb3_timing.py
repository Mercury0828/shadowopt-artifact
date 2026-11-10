"""PREREG A8: DEV timing screen for H-b′3. Refuses to run unless the lock file ($SHADOWOPT_LOCK_FILE, default CPU_LOCK.txt)'s HOLDER line names
shadowopt (timings only under the lock).

Post-append rows = the last 1e4 rows of each N = 1e6 DEV store (independent samples of the same state's sampler): the
index is built on the first 990 000 rows, then those 1e4 rows are appended before the post-append check.
Output: derisk/results/HB3/results.json (+ per-repetition times and system load).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from shadowopt.store import Store, TermBatch
from shadowopt.engine import AppendStore, IncrementalIndex, _bitmap_range
from shadowopt.certify import BatchCSR, CountSource, check_and_plan, certify, complete
from shadowopt.alloc import complete_alloc
from shadowopt.sharedplan import Trie, _shared_batch
from shadowopt.cost import hw1_fitted
from shadowopt.workloads import w1_energy, w2_correlators, w3_local_fidelities

LOCK = os.environ.get("SHADOWOPT_LOCK_FILE", "CPU_LOCK.txt")
OUT = os.path.join(ROOT, "derisk", "results", "HB3")
REPS = 5
HOLD = 10_000


def holds_lock():
    first = open(LOCK, encoding="utf-8-sig").readline()
    return first.startswith("HOLDER:") and "shadowopt" in first.split("(")[0]


def load_cases():
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    U1 = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
    return [("LiH", w1_energy(U1, 0.0, eps_seq=(0.1,))),
            ("TFIM20", w2_correlators(20)),
            ("GHZ50", w3_local_fidelities(50))]


def load_load():
    try:
        import psutil
        return psutil.cpu_percent(interval=0.2)
    except Exception:
        return None


def timed(fn, reps=REPS):
    fn()  # warm-up
    ts, loads = [], []
    for _ in range(reps):
        loads.append(load_load())
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(ts)), "iqr_s": [float(np.percentile(ts, 25)), float(np.percentile(ts, 75))],
            "times_s": ts, "load_pct_before": loads}


def main():
    if not holds_lock() and "--dry" not in sys.argv:
        sys.exit("refused: shadowopt does not hold the CPU lock (timings only under the lock)")
    os.makedirs(OUT, exist_ok=True)
    res = {"threads": int(os.environ.get("NUMBA_NUM_THREADS", "0")), "reps": REPS, "cases": {}}
    for state, fam in load_cases():
        st = Store.load(os.path.join(ROOT, "stores", state, "1e6"))
        U = fam.universe
        csr = BatchCSR(fam.batches[0])
        sub = U.subset(csr.terms)
        app = AppendStore(U.n, cap=st.N)
        app.append(st.bases[:-HOLD], st.outcomes[:-HOLD])
        t0 = time.perf_counter()
        ix = IncrementalIndex(app, cap_rows=st.N)
        build = time.perf_counter() - t0
        row = {"N": int(st.N), "terms": int(len(csr.terms)), "queries": int(len(csr.qptr) - 1), "build_s": build}
        srcs = {k: CountSource(ix, kind, cached) for k, (kind, cached) in
                {"B": ("bitmap", False), "S": ("sstar", False), "Bc": ("bitmap", True), "Sc": ("sstar", True),
                 "T": ("shared", False)}.items()}  # A16: T = shared-prefix trie (plan rebuilt per call)
        outs = {k: check_and_plan(U, csr, s) for k, s in srcs.items()}  # correctness: identical decisions
        ref = outs["B"]
        row["identical_decisions"] = all(np.array_equal(o["m"], ref["m"]) and len(o["plan"]) == len(ref["plan"])
                                         for o in outs.values())
        row["certified"] = int(ref["ok"].sum())
        cost = hw1_fitted()
        chk = lambda src: certify(csr, src.counts(U, csr))  # check only: counts + compiled half-widths
        full_ = lambda src: check_and_plan(U, csr, src, cost)  # the engine's full pipeline (A13 cost-aware completion)
        for case, keys in (("cold", ("B", "S", "T")), ("reused", ("Bc", "Sc"))):
            for k in keys:
                row[f"{case}_check_{k}"] = timed(lambda k=k: chk(srcs[k]))
                row[f"{case}_full_{k}"] = timed(lambda k=k: full_(srcs[k]))
        # completion planning alone (M1-A1: reported separately; identical for every source): A14 uniform and A15 alloc
        m_ref, (hw_ref, ok_ref) = ref["m"], certify(csr, ref["m"])
        row["plan_uniform"] = timed(lambda: complete(U, csr, m_ref, hw_ref, ok_ref, cost=cost))
        row["plan_alloc"] = timed(lambda: complete_alloc(U, csr, m_ref, hw_ref, ok_ref, cost), reps=1)
        # indexed answer time (denominator of the <= 10 % target): full (m, s) for the batch; the FASTER indexed path
        # (bitmap or shared trie incl. its plan) is the denominator (A16: the stricter reading)
        row["answer_B"] = timed(lambda: _bitmap_range(ix.Z, ix.O, sub.ptr, sub.q, sub.b, 0, ix.N_rep, False))

        def ans_T():
            tr = Trie(sub)
            return _shared_batch(ix.Z, ix.O, tr.node_q, tr.node_b, tr.node_depth, tr.term_ptr, tr.term_ids, tr.T,
                                 tr.max_depth, 2048, False)
        row["answer_T"] = timed(ans_T)
        # post-append: append the held-out rows once, then one check per source (refresh charged to the check)
        app.append(st.bases[-HOLD:], st.outcomes[-HOLD:])
        # shared maintenance timed once and charged equally to every source; then each source's
        # check runs on the refreshed index (its own delta work only: cache deltas for Bc/Sc, full passes for B/S)
        t0 = time.perf_counter()
        ix.refresh()
        row["post_append_refresh_s"] = time.perf_counter() - t0
        pa, pf = {}, {}
        for k in ("Bc", "Sc", "B", "S", "T"):
            t0 = time.perf_counter()
            chk(srcs[k])  # M1-A1 claimed quantity: the check (cached sources evaluate the appended delta here)
            pa[k] = time.perf_counter() - t0
            t0 = time.perf_counter()
            check_and_plan(U, csr, srcs[k], cost)  # + completion planning (reported separately)
            pf[k] = time.perf_counter() - t0
        row["post_append_check_once_s"] = pa
        row["post_append_check_then_plan_s"] = pf
        res["cases"][state] = row
        print(state, json.dumps({k: (v["median_s"] if isinstance(v, dict) and "median_s" in v else v)
                                 for k, v in row.items()}), flush=True)
        json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
