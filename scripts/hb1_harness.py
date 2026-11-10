"""PREREG A9: DEV screen of H-b′1. All methods run on the same sequence from the same initial store (first N0 rows of the
seeded uniform stream). Resumable: derisk/results/HB1/runs/<fam>_s<seed>_N0<N0>_<method>.json. One light process
(NUMBA_NUM_THREADS=1, BelowNormal, optional SHADOWOPT_CPU pinning; shared reservation file). `--summary` prints ratios.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from m1_repair1 import fam_of, lower_priority
from shadowopt.planner import run_strategy
from shadowopt.planner_v1 import run_v1
from shadowopt.planner_v3 import run_v3
from shadowopt.planner_v4 import run_v4
from shadowopt.cost import hw1_fitted

OUT = os.path.join(ROOT, "derisk", "results", "HB1")
RUNS = os.path.join(OUT, "runs")
FAMS = ["W2", "W4", "W6", "W3", "W1", "W5"]  # W5 added after A11 (second pass)
C1_SINGLE = {"W2", "W6"}  # C1 families whose queries are single terms (selective precover applies)
R = ["F1", "F2", "F3", "F4", "F5"]
S = ["F7", "F2qs", "F3qs", "F4qs"]
N0S = [0, 100_000]


def methods(fam_name):
    m = list(R) + list(S) + ["ENG"]
    if fam_name in C1_SINGLE:
        m.insert(len(R) + len(S), "Fcat")
    return m


A15 = "--a15" in sys.argv  # PREREG A15 (repair #1 of H-b'1): ENG = SO4 for W1/W3/W4/W5 (+ layer-S control F3qsA)
RUNS_A15 = os.path.join(ROOT, "derisk", "results", "HB1_A15", "runs")


def methods_a15(fam_name):
    """A15 re-runs only what changed: the engine on the SO1 families and the new layer-S control F3qsA. R/S runs and the
    SO3 engine (W2, W6; code unchanged) are read from the A9 screen."""
    return [] if fam_name in C1_SINGLE else ["ENG", "F3qsA"]


def run_one(fam_name, seed, N0, meth, cost):
    fn = os.path.join(RUNS_A15 if A15 else RUNS, f"{fam_name}_s{seed}_N0{N0}_{meth}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    fam = fam_of(fam_name, seed)
    t0 = time.time()
    try:
        if A15 and meth in ("ENG", "F3qsA"):
            r = run_v4("SO4" if meth == "ENG" else "F3qsA", fam, seed, cost, L0=N0)
        elif meth in R or meth == "F7":
            r = run_strategy(meth, fam, seed=seed, cost=cost, L0=N0, pooled=(meth != "F1"))  # F1: no reuse (published)
        elif meth in ("F2qs", "F3qs", "F4qs"):
            r = run_v1(meth, fam, seed, cost, L0=N0, pool_all=True)
        elif meth == "Fcat":
            r = run_v3("Fcat", fam, seed, cost, L0=N0)
        elif meth == "ENG":
            r = run_v3("SO3", fam, seed, cost, L0=N0) if fam_name in C1_SINGLE else run_v1("SO1", fam, seed, cost,
                                                                                            L0=N0)
        else:
            raise KeyError(meth)
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {e}"}
    r = {k: v for k, v in r.items() if k not in ("log", "settings_used")}
    r.update({"family": fam_name, "seed": seed, "N0": N0, "method": meth, "seconds": time.time() - t0})
    with open(fn + ".tmp", "w") as f:
        json.dump(r, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(fn + ".tmp", fn)
    print(fam_name, seed, N0, meth, "U=", r.get("usage", r.get("usage_lower_bound", r.get("error"))),
          "ok=", r.get("all_ok"), f"{r['seconds']:.0f}s", flush=True)
    return r


SEEDS = (1, 2)


def summary():
    """Ratios with seeds matched by ID (each method needs exactly SEEDS), zero costs handled explicitly (0/0 = tie = 1.0;
    a positive engine cost against a zero reference = inf), and the headline aggregate only if every family is complete
."""
    recs = {}

    def load(d, replace):
        for f in os.listdir(d):
            if not f.endswith(".json"):
                continue
            r = json.load(open(os.path.join(d, f)))
            strict = all(pb.get("worst", 2.0) <= 1.0 for pb in r.get("per_batch", []))  # exact HW <= eps (C-BOUND R1)
            if r.get("all_ok") and strict and "usage" in r:
                key = (r["family"], r["N0"], r["method"])
                if key in replace:
                    continue
                assert r["seed"] not in recs.setdefault(key, {}), f"duplicate seed {r['seed']} for {key}"
                recs[key][r["seed"]] = float(r["usage"])

    if A15:  # A15 records replace the A9 engine records of the SO1 families; every other record comes from A9
        load(RUNS_A15, set())
        load(RUNS, {(fam, N0, "ENG") for fam in FAMS if fam not in C1_SINGLE for N0 in N0S})
    else:
        load(RUNS, set())
    # seed-geometric mean; any zero-cost seed makes it 0 (its limit)
    gm = lambda xs: 0.0 if any(x == 0 for x in xs) else math.exp(np.mean(np.log(xs)))

    def val(fam, N0, m):
        d = recs.get((fam, N0, m), {})
        return gm([d[s] for s in SEEDS]) if set(d) == set(SEEDS) else None

    def ratio(a, b):
        if a is None or b is None:
            return None
        if b == 0:
            return 1.0 if a == 0 else float("inf")
        return a / b

    out = {}
    for N0 in N0S:
        fams = {}
        for fam in FAMS:
            eng = val(fam, N0, "ENG")
            rv = [(val(fam, N0, m), m) for m in R]
            rv = [x for x in rv if x[0] is not None]
            sv = [(val(fam, N0, m), m) for m in S + ["Fcat", "F3qsA"]]
            sv = [x for x in sv if x[0] is not None]
            rbest = min(rv) if rv else (None, None)
            sbest = min(sv) if sv else (None, None)
            fams[fam] = {"ENG": eng, "best_R": rbest, "best_S": sbest, "ratio_R": ratio(eng, rbest[0]),
                         "ratio_S": ratio(eng, sbest[0]), "complete": eng is not None and rbest[0] is not None}
        complete = all(v["complete"] for v in fams.values())
        rs = [v["ratio_R"] for v in fams.values() if v["complete"]]
        agg = math.exp(np.mean(np.log(rs))) if rs and all(0 < x < float("inf") for x in rs) else None
        out[N0] = {"families": fams, "all_families_complete": complete,
                   "geomean_ratio_R": agg if complete else None, "geomean_ratio_R_partial": agg}
    print(json.dumps(out, indent=1, default=str))
    out_dir = os.path.dirname(RUNS_A15) if A15 else OUT
    json.dump(out, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)


def main():
    if "--summary" in sys.argv:
        return summary()
    lower_priority()
    os.makedirs(RUNS_A15 if A15 else RUNS, exist_ok=True)
    cost = hw1_fitted()
    fams = FAMS
    for a in sys.argv:  # optional subset for running families in separate single-thread processes: --fams=W4,W5
        if a.startswith("--fams="):
            fams = [f for f in FAMS if f in a[7:].split(",")]
    for N0 in N0S:
        for fam_name in fams:
            for seed in (1, 2):
                for meth in (methods_a15(fam_name) if A15 else methods(fam_name)):
                    run_one(fam_name, seed, N0, meth, cost)
    summary()


if __name__ == "__main__":
    main()
