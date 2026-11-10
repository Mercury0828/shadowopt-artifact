"""D2 — coverage of the stated bound under the planner's decision classes (derisk/PREREG.md). Exact truth.

Replicate seeds: 10^6 + 10^4*state_id + 100*config + r (store stream), +500_000 for outcomes, +700_000 for targeted
shots. Configs: a=1, b=2, c=3, d=4, e=5, f=6 (f_alt1=7, f_alt2=8). Resumable: one JSON per (state, config).
"""
import json
import os
import sys
import time

import numba
import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.bounds import radius
from shadowopt.sampling import uniform_bases
from scripts.gen_stores import load_state, STATE_ID
from scripts.d0_anchors import load_truth

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "derisk", "results", "D2")
DELTA = 0.05
EPS = 0.25
T_NEED = int(np.ceil(2 * np.log(2 / DELTA) / EPS ** 2))  # 119
REPS = 1000


def checkpoint_terms(state):
    terms, truth = load_truth(state)
    w = terms.weights()
    if state == "TFIM20":
        want = [((0, 2), (1, 2)), ((3, 0), (4, 0)), ((5, 1), (9, 1))]
        ids = []
        for wt in want:
            for t in range(terms.T):
                qq, bb = terms.term(t)
                if tuple(zip(qq.tolist(), bb.tolist())) == wt:
                    ids.append(t)
                    break
    else:
        ids = [0, 1, int(np.nonzero(w == 4)[0][0])]
    return terms, truth, ids


def matched(B, O, qq, bb):
    mask = np.all(B[:, qq] == bb, axis=1)
    m = int(mask.sum())
    par = O[mask][:, qq].sum(axis=1) & 1
    return m, m - 2 * int(par.sum())


class Rep:
    """Lazy snapshot stream of one replicate."""

    def __init__(self, n, sampler, seed):
        self.n, self.sampler = n, sampler
        self.rb = np.random.default_rng(seed)
        self.ro = np.random.default_rng(seed + 500_000)
        self.B = np.zeros((0, n), np.uint8)
        self.O = np.zeros((0, n), np.uint8)

    def upto(self, N):
        if self.B.shape[0] < N:
            k = N - self.B.shape[0]
            B = uniform_bases(self.rb, k, self.n)
            O = self.sampler(B, self.ro)
            self.B = np.vstack([self.B, B])
            self.O = np.vstack([self.O, O])
        return self.B[:N], self.O[:N]


def targeted(n, sampler, qq, bb, s, seed):
    rng = np.random.default_rng(seed + 700_000)
    B = uniform_bases(rng, s, n)
    B[:, qq] = bb
    O = sampler(B, np.random.default_rng(seed + 900_000))
    return B, O


def run_config(state, cfg, ids, terms, truth):
    n, sampler = load_state(state)
    sid = STATE_ID[state]
    cid = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "f_alt1": 7, "f_alt2": 8}[cfg]
    res = {}
    for t in ids:
        qq, bb = terms.term(t)
        bb8 = bb.astype(np.uint8)
        res[t] = []
    for r in range(REPS):
        seed = 10 ** 6 + 10 ** 4 * sid + 100 * cid + r
        rep = Rep(n, sampler, seed)
        for t in ids:
            qq, bb = terms.term(t)
            bb8 = bb.astype(np.uint8)
            tr = truth[t]
            if cfg == "a":
                m, S = matched(*rep.upto(2000), qq, bb8)
                res[t].append((m, S))
            elif cfg == "b":
                N = 200
                while True:
                    m, S = matched(*rep.upto(N), qq, bb8)
                    if m >= T_NEED or N >= 200_000:
                        break
                    N += 100
                res[t].append((m, S))
            elif cfg == "c":
                row = []
                for N in (500, 1000, 2000, 4000, 8000):
                    row.append(matched(*rep.upto(N), qq, bb8))
                res[t].append(row)
            elif cfg in ("d", "e"):
                m, S = matched(*rep.upto(2000), qq, bb8)
                if cfg == "d":
                    if m >= T_NEED:
                        res[t].append((m, S, "store"))
                    else:
                        Bt, Ot = targeted(n, sampler, qq, bb8, T_NEED, seed)
                        mt, St = matched(Bt, Ot, qq, bb8)
                        assert mt == T_NEED
                        res[t].append((mt, St, "target"))
                else:
                    s = max(0, T_NEED - m)
                    if s:
                        Bt, Ot = targeted(n, sampler, qq, bb8, s, seed)
                        mt, St = matched(Bt, Ot, qq, bb8)
                    else:
                        mt, St = 0, 0
                    res[t].append((m + mt, S + St, "pooled"))
            elif cfg in ("f", "f_alt1", "f_alt2"):
                N = 200
                m, S = matched(*rep.upto(N), qq, bb8)
                old = S / m if m else None
                while N < 20_000:
                    N += 50
                    m, S = matched(*rep.upto(N), qq, bb8)
                    new = S / m if m else None
                    if new is not None:
                        if cfg == "f" and old is not None and abs(new - old) < 0.01:
                            break
                        if cfg == "f_alt1" and abs(new) <= EPS / 4:
                            break
                        if cfg == "f_alt2" and abs(new) > float(radius(m, DELTA)):
                            break
                    old = new
                res[t].append((m, S))
    # evaluate
    out = {}
    for t in ids:
        tr = float(truth[t])
        if cfg == "c":
            per = []
            for j, N in enumerate((500, 1000, 2000, 4000, 8000)):
                ms = [row[j] for row in res[t]]
                per.append(evaluate(ms, tr, f"N={N}"))
            out[int(t)] = per
        else:
            out[int(t)] = [evaluate([x[:2] for x in res[t]], tr, cfg)]
            if cfg == "d":
                out[int(t)][0]["n_target_answers"] = int(sum(1 for x in res[t] if x[2] == "target"))
    return out


def evaluate(ms, tr, label):
    misses = 0
    zero = 0
    for m, S in ms:
        if m == 0:
            zero += 1
            misses += 1  # no compatible snapshot: counted as a miss (no bounded answer exists)
            continue
        if abs(S / m - tr) > float(radius(m, DELTA)):
            misses += 1
    n = len(ms)
    p_le = binomtest(misses, n, DELTA, alternative="greater").pvalue
    return {"label": label, "truth": tr, "replicates": n, "misses": misses, "miss_rate": misses / n,
            "m_zero": zero, "binom_p_greater": p_le, "rejects_at_0.01": bool(p_le < 0.01),
            "median_m": float(np.median([m for m, _ in ms]))}


def main():
    numba.set_num_threads(int(os.environ.get("D2_THREADS", "16")))
    os.makedirs(OUT, exist_ok=True)
    cfgs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["a", "b", "c", "d", "e", "f"]
    for state in ("GHZ50", "TFIM20", "LiH"):
        terms, truth, ids = checkpoint_terms(state)
        if "f_alt2" in cfgs:
            pass
        for cfg in cfgs:
            fn = os.path.join(OUT, f"{state}_{cfg}.json")
            if os.path.exists(fn):
                continue
            use_ids = ids
            if cfg == "f_alt2":
                z = np.nonzero(np.abs(truth) < (1e-12 if state != "TFIM20" else 0.05))[0]
                use_ids = [int(z[0])] if len(z) else ids[:1]
            t0 = time.time()
            out = run_config(state, cfg, use_ids, terms, truth)
            rec = {"state": state, "config": cfg, "delta": DELTA, "eps": EPS, "t_need": T_NEED, "reps": REPS,
                   "checkpoint_terms": {int(t): terms.to_strings()[t] for t in use_ids}, "results": out,
                   "seconds": time.time() - t0, "label": "provisional — Phase 0, internal, DEV split",
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            with open(fn + ".tmp", "w") as f:
                json.dump(rec, f, indent=1)
            os.replace(fn + ".tmp", fn)
            print(state, cfg, json.dumps(out)[:400], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
