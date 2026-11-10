"""Calibrate pathopt.Rates on INDEPENDENT synthetic data (uniform random stores, random term batches; never DEV/TEST).
Lock-gated like hb3_timing (timings only while the lock file ($SHADOWOPT_LOCK_FILE, default CPU_LOCK.txt) names shadowopt). PREREG A10.
For each path and grid point (n in {12, 24, 50, 100}, N in {1e5, 1e6}, batch of 200 random terms with k ~ U{1..6}):
median of 5 timed runs (16 threads), work counts from pathopt; rate = least-squares slope of time on work, tau = intercept
(non-negative). Output: derisk/results/PATHOPT/rates.json (frozen once written; held-out check on 20 % of grid points).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from hb3_timing import holds_lock
from shadowopt.store import terms_from_pairs, Store, pack_rows, build_bitmaps
from shadowopt.scan import SStar
from shadowopt.bitmaps import BitmapPath
from shadowopt.sharedplan import Trie, SharedPlanPath
from shadowopt.pathopt import sstar_words, bitmap_words, shared_words

OUT = os.path.join(ROOT, "derisk", "results", "PATHOPT")


def rand_terms(rng, n, T):
    pairs = []
    for _ in range(T):
        k = int(rng.integers(1, 7))
        qs = sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
        pairs.append([(q, int(rng.integers(0, 3))) for q in qs])
    return terms_from_pairs(n, pairs)


def med(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    if not holds_lock() and "--dry" not in sys.argv:
        sys.exit("refused: shadowopt does not hold the CPU lock")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261108)
    pts = []
    for n in (12, 24, 50, 100):
        for N in (100_000, 1_000_000):
            st = Store(n, rng.integers(0, 3, (N, n), dtype=np.uint8), rng.integers(0, 2, (N, n), dtype=np.uint8),
                       np.zeros(N, np.int32), np.zeros(N, np.uint8))
            rows = pack_rows(st)
            Z, O = build_bitmaps(st)
            terms = rand_terms(rng, n, 200)
            tr = Trie(terms)
            s, b, sh = SStar(rows, n), BitmapPath(Z, O, N), SharedPlanPath(Z, O, N)
            pts.append({"n": n, "N": N,
                        "sstar": (sstar_words(terms, n, N), med(lambda: s.run(terms))),
                        "bitmap": (bitmap_words(terms, N), med(lambda: b.run(terms))),
                        "shared": (shared_words(tr, N), med(lambda: sh.execute(tr)))})
            print(n, N, {k: pts[-1][k] for k in ("sstar", "bitmap", "shared")}, flush=True)
    rng2 = np.random.default_rng(1)
    hold = set(rng2.choice(len(pts), size=max(1, len(pts) // 5), replace=False).tolist())
    fit = {}
    for path in ("sstar", "bitmap", "shared"):
        tr_ = [p[path] for i, p in enumerate(pts) if i not in hold]
        X = np.array([[1.0, w] for w, _ in tr_])
        y = np.array([t for _, t in tr_])
        from scipy.optimize import nnls
        c, _ = nnls(X, y)
        ho = [p[path] for i, p in enumerate(pts) if i in hold]
        err = [abs(c[0] + c[1] * w - t) / t for w, t in ho]
        fit[path] = {"tau": c[0], "sec_per_word": c[1], "words_per_s": 1 / c[1] if c[1] > 0 else None,
                     "heldout_median_rel_err": float(np.median(err)) if err else None}
    json.dump({"points": pts, "heldout_idx": sorted(hold), "fit": fit,
               "threads": os.environ.get("NUMBA_NUM_THREADS")}, open(os.path.join(OUT, "rates.json"), "w"), indent=1)
    print(json.dumps(fit, indent=1))


if __name__ == "__main__":
    main()
