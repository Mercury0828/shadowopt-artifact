"""D1 timing campaign per derisk/PREREG.md (CPU lock must be held). Resumable: every measurement is appended as one JSON
line to derisk/results/D1/results.jsonl; a (key) already present is skipped on restart (--fresh to override).
"""
import argparse
import gc
import json
import os
import sys
import time

import numba
import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.store import Store, pack_rows, build_bitmaps, packed_bytes, bitmap_bytes
from shadowopt.scan import (SStar, load_hkp, to_hkp_measurements, hkp_batch, pennylane_batch, pennylane_inputs,
                            naive_reference)
from shadowopt.bitmaps import BitmapPath
from shadowopt.sharedplan import SharedPlanPath
from shadowopt.views import ViewSet, ViewPath, select_views
from scripts.d0_anchors import load_truth

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "derisk", "results", "D1")
RES = os.path.join(OUT, "results.jsonl")
LOCK = os.environ.get("SHADOWOPT_LOCK_FILE", "CPU_LOCK.txt")
STATES = [("LiH", "W1"), ("TFIM20", "W2"), ("GHZ50", "W3")]


def lock_holder():
    try:
        with open(LOCK, encoding="utf-8") as f:
            return f.readline().strip()[:90]
    except Exception as e:  # noqa
        return f"unreadable: {e}"


def cpu_load():
    try:
        import psutil
        return psutil.cpu_percent(interval=0.5)
    except Exception:
        return None


def done_keys():
    if not os.path.exists(RES):
        return set()
    return {json.loads(l)["key"] for l in open(RES, encoding="utf-8") if l.strip()}


def emit(rec):
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["lock_holder"] = lock_holder()
    rec["label"] = "provisional — Phase 0, internal, DEV split"
    tmp = json.dumps(rec)
    with open(RES, "a", encoding="utf-8") as f:
        f.write(tmp + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(rec["key"], {k: rec[k] for k in ("median_s", "q25_s", "q75_s") if k in rec}, flush=True)


def timeit(fn, reps, warm=True):
    if warm:
        fn()
    ts = []
    loads = []
    for _ in range(reps):
        loads.append(cpu_load())
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"times_s": ts.tolist(), "median_s": float(np.median(ts)), "q25_s": float(np.quantile(ts, 0.25)),
            "q75_s": float(np.quantile(ts, 0.75)), "reps": int(reps), "cpu_load_before_pct": loads}


# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _triad(a, b, c, s):
    for i in prange(a.shape[0]):
        a[i] = b[i] + s * c[i]


def stream(done):
    n = 50_000_000  # 3 arrays x 400 MB
    a = np.zeros(n)
    b = np.ones(n)
    c = np.ones(n)
    for T in (1, 16):
        key = f"stream|T{T}"
        if key in done:
            continue
        numba.set_num_threads(T)
        r = timeit(lambda: _triad(a, b, c, 3.0), 5)
        r.update({"key": key, "kind": "stream_triad", "threads": T, "bytes": 3 * 8 * n,
                  "GBps_median": 3 * 8 * n / r["median_s"] / 1e9})
        emit(r)
    del a, b, c


def tune(done):
    big = Store.load(os.path.join(ROOT, "stores", "LiH", "1e6"))
    st = big.head(100_000)
    terms, _ = load_truth("LiH")
    numba.set_num_threads(16)
    rows = pack_rows(st)
    Z, O = build_bitmaps(st)
    best = {}
    for blk in (512, 2048, 8192, 32768):
        key = f"tune|S*|block{blk}"
        if key not in done:
            p = SStar(rows, st.n, block=blk)
            r = timeit(lambda: p.run(terms), 3)
            r.update({"key": key, "kind": "tune", "method": "S*", "param": blk, "threads": 16})
            emit(r)
    for ch in (256, 1024, 4096, 16384):
        for name, cls in (("bitmaps", BitmapPath), ("shared", SharedPlanPath)):
            key = f"tune|{name}|chunk{ch}"
            if key not in done:
                p = cls(Z, O, st.N, chunk=ch)
                r = timeit(lambda: p.run(terms), 3)
                r.update({"key": key, "kind": "tune", "method": name, "param": ch, "threads": 16})
                emit(r)


def tuned_params():
    rec = [json.loads(l) for l in open(RES, encoding="utf-8") if l.strip()]
    out = {}
    for m in ("S*", "bitmaps", "shared"):
        cand = [r for r in rec if r.get("kind") == "tune" and r["method"] == m]
        out[m] = min(cand, key=lambda r: r["median_s"])["param"]
    return out


def query_frequencies(state, T):
    """PREREG: view-selection frequency of a term = number of queries of the workload containing it."""
    from shadowopt.workloads import w2_correlators, w3_local_fidelities
    if state == "GHZ50":
        fam = w3_local_fidelities(50)
        f = np.zeros(T)
        for q in fam.batches[0]:
            f[q.terms] += 1
        return f.tolist()
    return [1] * T  # W1: one query containing every term; W2: one query per term


def run_workload(state, wl, N, done, params, hkp_mod):
    big = Store.load(os.path.join(ROOT, "stores", state, "1e6"))
    st = big if N == big.N else big.head(N)
    man = big.meta
    terms, _ = load_truth(state)
    base = {"state": state, "workload": wl, "N": N, "n": st.n, "n_terms": terms.T, "store_sha256_1e6": man["sha256"],
            "store_seed": man["seed_bases"], "prefix_rows": N}
    M0, S0 = naive_reference(st, terms)

    def check(M, S, name):
        if not (np.array_equal(M, M0) and np.array_equal(S, S0)):
            raise RuntimeError(f"bit-identical check failed for {name} at {state} N={N}")

    # ---- builds (separate, not in batch time) ----
    numba.set_num_threads(16)
    t0 = time.perf_counter(); rows = pack_rows(st); t_pack = time.perf_counter() - t0
    t0 = time.perf_counter(); Z, O = build_bitmaps(st); t_bm = time.perf_counter() - t0
    sup = [frozenset(terms.term(t)[0].tolist()) for t in range(terms.T)]
    freq = query_frequencies(state, terms.T)
    cap = 5 / 3 * packed_bytes(st.n, st.N)
    t0 = time.perf_counter()
    chosen, used = select_views(sup, freq, (st.N + 63) // 64, cap, s_max=6)
    t_sel = time.perf_counter() - t0
    t0 = time.perf_counter(); vs = ViewSet(chosen).build(st); t_vb = time.perf_counter() - t0
    t0 = time.perf_counter()
    for v in vs.views:
        v.prepare()
    t_prep = time.perf_counter() - t0
    key = f"viewprep|{state}|{N}"
    if key not in done:
        emit({**base, "key": key, "kind": "build", "threads": 16, "view_prepare_s": t_prep,
              "view_build_plus_prepare_s": t_vb + t_prep, "view_freq_rule": "number of queries containing the term"})
    key = f"build|{state}|{N}"
    if key not in done:
        emit({**base, "key": key, "kind": "build", "threads": 16, "pack_s": t_pack, "bitmap_build_s": t_bm,
              "view_select_s": t_sel, "view_build_s": t_vb, "n_views": len(chosen), "view_bytes": used,
              "view_cap_bytes": cap, "packed_bytes": packed_bytes(st.n, st.N), "bitmap_bytes": bitmap_bytes(st.n, st.N),
              "views_covered_terms": int((vs.assign(terms) >= 0).sum()),
              "view_sizes": np.bincount([len(c) for c in chosen], minlength=7).tolist()})

    for T in (16, 1):
        numba.set_num_threads(T)
        paths = {"S*": SStar(rows, st.n, block=params["S*"]),
                 "bitmaps": BitmapPath(Z, O, st.N, chunk=params["bitmaps"]),
                 "shared": SharedPlanPath(Z, O, st.N, chunk=params["shared"])}
        paths["views"] = ViewPath(vs, SharedPlanPath(Z, O, st.N, chunk=params["shared"]))
        for name, p in paths.items():
            M, S = p.run(terms)
            check(M, S, name)
            key = f"batch|{state}|{N}|{name}|T{T}"
            if key not in done:
                r = timeit(lambda: p.run(terms), 5)
                r.update({**base, "key": key, "kind": "batch", "method": name, "threads": T, "plan": "included",
                          "param": params.get(name, params["shared"])})
                emit(r)
            key = f"batchcached|{state}|{N}|{name}|T{T}"
            if key not in done:
                plan = p.plan(terms)
                r = timeit(lambda: p.execute(plan), 5)
                r.update({**base, "key": key, "kind": "batch", "method": name, "threads": T, "plan": "cached"})
                emit(r)
        key = f"select|{state}|{N}|S*|T{T}"
        if key not in done:
            Ms, _ = paths["S*"].run(terms, select_only=True)
            assert np.array_equal(Ms, M0)
            r = timeit(lambda: paths["S*"].run(terms, select_only=True), 5)
            r.update({**base, "key": key, "kind": "select", "method": "S*", "threads": T})
            emit(r)
        for name in ("bitmaps",):
            key = f"select|{state}|{N}|{name}|T{T}"
            if key not in done:
                r = timeit(lambda: paths[name].run(terms, select_only=True), 5)
                r.update({**base, "key": key, "kind": "select", "method": name, "threads": T})
                emit(r)

    # ---- published baselines (single process as published) ----
    numba.set_num_threads(1)
    prior = {json.loads(l)["key"]: json.loads(l) for l in open(RES, encoding="utf-8") if l.strip()}
    # PennyLane
    plin = pennylane_inputs(st)
    for sel in (False, True):
        tag = "select" if sel else "batch"
        key = f"{tag}|{state}|{N}|pennylane|T1"
        if key in done:
            continue
        reps = 3 if sel else 5
        if N == 1_000_000:
            m5 = prior.get(f"{tag}|{state}|100000|pennylane|T1", {}).get("median_s")
            if sel:
                continue  # selection share is measured at N = 1e5 (PREREG)
            if m5 is not None and 10 * m5 > 7200:
                emit({**base, "key": key, "kind": tag, "method": "pennylane", "skipped": True,
                      "reason": f"estimated {10 * m5:.0f} s > 2 h"})
                continue
            reps = 5 if (m5 is not None and 10 * m5 <= 600) else 1
        if not sel:
            tot, nnz = pennylane_batch(st, terms, inputs=plin)
            assert np.array_equal(nnz, M0)
        r = timeit(lambda: pennylane_batch(st, terms, select_only=sel, inputs=plin), reps, warm=(reps > 1))
        r.update({**base, "key": key, "kind": tag, "method": "pennylane", "threads": 1,
                  "note": "chunked over terms, 2 GB per call; single run" if reps == 1 else "chunked over terms, 2 GB per call"})
        emit(r)
    del plin
    gc.collect()
    # HKP
    meas = None
    for sel in (False, True):
        tag = "select" if sel else "batch"
        key = f"{tag}|{state}|{N}|hkp|T1"
        if key in done:
            continue
        reps = 3 if sel else 5
        if N == 1_000_000:
            if sel:
                continue
            m5 = prior.get(f"{tag}|{state}|100000|hkp|T1", {}).get("median_s")
            if m5 is not None and 10 * m5 > 7200:
                emit({**base, "key": key, "kind": tag, "method": "hkp", "skipped": True,
                      "reason": f"estimated {10 * m5:.0f} s > 2 h"})
                continue
            reps = 5 if (m5 is not None and 10 * m5 <= 600) else 1
        if meas is None:
            t0 = time.perf_counter()
            meas = to_hkp_measurements(st)
            t_conv = time.perf_counter() - t0
        if not sel:  # one untimed, validated (and warming) call regardless of the repetition count
            Mh, Sh = hkp_batch(hkp_mod, meas, terms)
            check(Mh, Sh, "hkp")
        r = timeit(lambda: hkp_batch(hkp_mod, meas, terms, select_only=sel), reps, warm=False)
        r.update({**base, "key": key, "kind": tag, "method": "hkp", "threads": 1, "store_conversion_s": t_conv,
                  "note": "single run" if reps == 1 else ""})
        emit(r)
    del meas
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.fresh and os.path.exists(RES):
        os.replace(RES, RES + f".superseded_{int(time.time())}")
    if "TFIM_SKIP" in os.environ:
        pass
    done = done_keys()
    stream(done)
    tune(done_keys())
    params = tuned_params()
    print("tuned", params, flush=True)
    hkp_mod = load_hkp()
    for state, wl in STATES:
        if a.only and state not in a.only.split(","):
            continue
        for N in (100_000, 1_000_000):
            run_workload(state, wl, N, done_keys(), params, hkp_mod)
            gc.collect()
    print("D1 campaign complete", flush=True)


if __name__ == "__main__":
    main()
