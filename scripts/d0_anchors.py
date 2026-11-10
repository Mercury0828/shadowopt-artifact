"""D0 (iv) H0a correctness and (v) H0b estimator anchor, per derisk/PREREG.md. Light job (4 threads)."""
import json
import os
import sys
import time

import numba
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.store import Store, pack_rows, build_bitmaps, terms_from_pairs, TermBatch, packed_bytes
from shadowopt.scan import naive_reference, SStar, load_hkp, to_hkp_measurements, hkp_batch, pennylane_batch
from shadowopt.bitmaps import BitmapPath
from shadowopt.sharedplan import SharedPlanPath
from shadowopt.views import ViewSet, ViewPath, select_views
from scripts.gen_stores import load_state, STATE_ID
from shadowopt.sampling import uniform_bases

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "derisk", "results", "D0")
WL = {"LiH": "W1", "TFIM20": "W2", "GHZ50": "W3"}


def load_truth(name):
    d = np.load(os.path.join(ROOT, "stores", "states", name, f"truth_{WL[name]}.npz"))
    n = {"LiH": 12, "TFIM20": 20, "GHZ50": 50}[name]
    return TermBatch(n, d["ptr"], d["q"], d["b"]), d["truth"]


def random_terms(n, T, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(T):
        k = int(rng.integers(1, min(n, 10) + 1))
        qs = rng.choice(n, size=k, replace=False)
        pairs.append([(int(q), int(rng.integers(0, 3))) for q in qs])
    return terms_from_pairs(n, pairs)


def workload_views(terms, st):
    sup = [frozenset(terms.term(t)[0].tolist()) for t in range(terms.T)]
    cap = 5 / 3 * packed_bytes(st.n, st.N)
    chosen, used = select_views(sup, [1] * len(sup), (st.N + 63) // 64, cap, s_max=6)
    return chosen, used, cap


def check_paths(st, terms, views):
    M0, S0 = naive_reference(st, terms)
    rows = pack_rows(st)
    Z, O = build_bitmaps(st)
    res = {}
    paths = {"S*": SStar(rows, st.n), "bitmaps": BitmapPath(Z, O, st.N), "shared": SharedPlanPath(Z, O, st.N)}
    vs = ViewSet(views).build(st)
    paths["views"] = ViewPath(vs, SharedPlanPath(Z, O, st.N))
    for k, p in paths.items():
        M, S = p.run(terms)
        res[k] = {"identical": bool(np.array_equal(M, M0) and np.array_equal(S, S0)),
                  "n_mismatch_terms": int(np.sum((M != M0) | (S != S0)))}
    res["views_covered_terms"] = int((vs.assign(terms) >= 0).sum())
    return res


def h0a():
    out = {}
    for name in ["LiH", "TFIM20", "GHZ50"]:
        big = Store.load(os.path.join(ROOT, "stores", name, "1e6"))
        wterms, _ = load_truth(name)
        for N in (100_000, 1_000_000):
            st = big if N == big.N else big.head(N)
            views, used, cap = workload_views(wterms, st)
            r = {"n_views": len(views), "view_bytes": used, "view_cap_bytes": cap}
            if N == 100_000:
                r["random_10k"] = check_paths(st, random_terms(st.n, 10_000, 77), views)
            r["workload"] = check_paths(st, wterms, views)
            out[f"{name}@{N}"] = r
            print(name, N, json.dumps(r)[:300], flush=True)
    ok = all(v2["identical"] for r in out.values() for key in ("random_10k", "workload") if key in r
             for k2, v2 in r[key].items() if isinstance(v2, dict))
    out["H0a_pass"] = bool(ok)
    return out


def h0b():
    out = {}
    hkp = load_hkp()
    for name in ["LiH", "TFIM20", "GHZ50"]:
        big = Store.load(os.path.join(ROOT, "stores", name, "1e6"))
        st = big.head(100_000)
        wterms, truth = load_truth(name)
        sub = wterms.subset(np.arange(100))
        M0, S0 = naive_reference(st, sub)
        meas = to_hkp_measurements(st)
        Mh, Sh = hkp_batch(hkp, meas, sub)
        del meas
        hkp_ok = bool(np.array_equal(Mh, M0) and np.array_equal(Sh, S0))
        with np.errstate(invalid="ignore", divide="ignore"):
            mm = np.where(M0 > 0, S0 / np.maximum(M0, 1), 0.0)
            mh = np.where(Mh > 0, Sh / np.maximum(Mh, 1), 0.0)
        tot, nnz = pennylane_batch(st, sub)
        k = sub.weights()
        pl_mean = tot / st.N
        ours = (3.0 ** k) * S0 / st.N
        pl_ok = bool(np.max(np.abs(pl_mean - ours)) <= 1e-12 and np.array_equal(nnz, M0))
        out[name] = {"hkp_identical_mS": hkp_ok, "hkp_ratio_maxdiff": float(np.max(np.abs(mm - mh))),
                     "pennylane_formula_maxdiff": float(np.max(np.abs(pl_mean - ours))),
                     "pennylane_nnz_equals_m": bool(np.array_equal(nnz, M0)), "pennylane_ok": pl_ok}
        # 200 independent replicates, 5 fixed terms with k <= 4 (first five of the workload list)
        n, sampler = load_state(name)
        w = wterms.weights()
        tids = [int(t) for t in np.nonzero(w <= 4)[0][:5]]
        sub5 = wterms.subset(tids)
        misses = np.zeros(5, np.int64)
        for r in range(200):
            seed = 10 ** 6 + 10 ** 4 * STATE_ID[name] + 100 * 0 + r
            rb = np.random.default_rng(seed)
            ro = np.random.default_rng(seed + 500_000)
            Nrep = 20_000
            B = uniform_bases(rb, Nrep, n)
            O = sampler(B, ro)
            s2 = Store(n)
            s2.append(B, O)
            M, S = naive_reference(s2, sub5)
            rad = np.sqrt(2 * np.log(2 / 0.05) / np.maximum(M, 1))
            est = S / np.maximum(M, 1)
            misses += (np.abs(est - truth[tids]) > rad) | (M == 0)
        pv = [binomtest(int(x), 200, 0.05, alternative="greater").pvalue for x in misses]
        out[name].update({"rep_terms": tids, "rep_misses_of_200": misses.tolist(), "binom_p": pv,
                          "rep_ok": bool(all(p >= 0.01 for p in pv))})
        print(name, json.dumps(out[name]), flush=True)
    out["H0b_pass"] = bool(all(v["hkp_identical_mS"] and v["pennylane_ok"] and v["rep_ok"] and
                               v["hkp_ratio_maxdiff"] <= 1e-12 for k, v in out.items() if isinstance(v, dict)))
    return out


if __name__ == "__main__":
    numba.set_num_threads(int(os.environ.get("D0_THREADS", "4")))
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    meta = {"label": "provisional — Phase 0, internal, DEV split", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threads": numba.get_num_threads()}
    if which in ("h0a", "both"):
        r = h0a()
        r.update(meta)
        json.dump(r, open(os.path.join(OUT, "H0a.json"), "w"), indent=1)
        print("H0a_pass", r["H0a_pass"])
    if which in ("h0b", "both"):
        r = h0b()
        r.update(meta)
        json.dump(r, open(os.path.join(OUT, "H0b.json"), "w"), indent=1)
        print("H0b_pass", r["H0b_pass"])
