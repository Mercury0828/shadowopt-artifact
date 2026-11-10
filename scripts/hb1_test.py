"""H-b'1 / H-b'2 TEST campaign. Untimed, resumable, one JSON per
run in results/HB1_TEST/runs/<fam>_<inst>_s<seed>_N0<N0>_<method>.json. Run in the pinned .venv, single-thread,
BelowNormal; several processes may run disjoint --inst= shards. `--summary` computes the registered aggregate.

Instances (M1-A3): W1 BeH2/H2O/N2 (term sets from stores/states/<s>/truth_W1.npz), W2 Heis50/Heis100 (generator n),
W3 Cluster100, W4/W5 n = 50 (Clifford50), W6 n = 100 (Cluster100). Seeds 100-104. N0 in {0, 1e5} (M1-A5).
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
from shadowopt.store import TermBatch
from shadowopt.workloads import w1_energy, w2_correlators, w3_cluster_fidelities, w4_zipf, w5_drift, w6_heavytail
from shadowopt.planner import run_strategy
from shadowopt.planner_v1 import run_v1
from shadowopt.planner_v3 import run_v3
from shadowopt.planner_v4 import run_v4
from shadowopt.cost import hw1_fitted

OUT = os.path.join(ROOT, "results", "HB1_TEST")
RUNS = os.path.join(OUT, "runs")
SEEDS = (100, 101, 102, 103, 104)
N0S = (0, 100_000)
C1_SINGLE = {"W2", "W6"}
R = ["F1", "F2", "F3", "F4", "F5"]
S = ["F7", "F2qs", "F3qs", "F4qs", "F3qsA"]
INSTANCES = [("W2", "Heis50"), ("W2", "Heis100"), ("W4", "Clifford50"), ("W5", "Clifford50"), ("W6", "Cluster100"),
             ("W1", "BeH2"), ("W1", "H2O"), ("W1", "N2"), ("W3", "Cluster100")]


def methods(fam):
    m = R + S + ["ENG"]
    if fam in C1_SINGLE:
        m = [x for x in m if x != "F3qsA"] + ["Fcat"]  # F3qsA is the SO4 control (multi-term families), as on DEV
    return m


def family(fam, inst, seed):
    if fam == "W1":
        d = np.load(os.path.join(ROOT, "stores", "states", inst, "truth_W1.npz"))
        man = json.load(open(os.path.join(ROOT, "stores", "states", inst, "manifest.json")))
        U = TermBatch(int(man["n"]), d["ptr"], d["q"], d["b"], d["coeff"])
        return w1_energy(U, float(man["const"]), eps_seq=(0.5, 0.3, 0.2, 0.1))
    if fam == "W2":
        return w2_correlators(int(inst[4:]))
    if fam == "W3":
        return w3_cluster_fidelities(100)
    if fam == "W4":
        return w4_zipf(50, seed=seed)
    if fam == "W5":
        return w5_drift(50, seed)
    if fam == "W6":
        return w6_heavytail(100, seed=seed)
    raise KeyError(fam)


def run_one(fam, inst, seed, N0, meth, cost):
    fn = os.path.join(RUNS, f"{fam}_{inst}_s{seed}_N0{N0}_{meth}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    path = os.path.join(ROOT, "stores", "states", inst, "manifest.json")
    if fam == "W1" and not os.path.exists(path):  # state not built yet: skip (no record), a later pass runs it
        print("waiting for", inst, flush=True)
        return None
    F = family(fam, inst, seed)
    t0 = time.time()
    try:
        if meth == "ENG":  # M1-A4
            r = run_v3("SO3", F, seed, cost, L0=N0) if fam in C1_SINGLE else run_v4("SO4", F, seed, cost, L0=N0)
        elif meth == "F3qsA":
            r = run_v4("F3qsA", F, seed, cost, L0=N0)
        elif meth in R or meth == "F7":
            r = run_strategy(meth, F, seed=seed, cost=cost, L0=N0, pooled=(meth != "F1"))  # F1: no reuse (published)
        elif meth in ("F2qs", "F3qs", "F4qs"):
            r = run_v1(meth, F, seed, cost, L0=N0, pool_all=True)
        elif meth == "Fcat":
            r = run_v3("Fcat", F, seed, cost, L0=N0)
        else:
            raise KeyError(meth)
    except Exception as e:  # recorded, never silently dropped
        r = {"error": f"{type(e).__name__}: {e}"}
    r = {k: v for k, v in r.items() if k not in ("log", "settings_used")}
    r.update({"family": fam, "instance": inst, "seed": seed, "N0": N0, "method": meth, "seconds": time.time() - t0,
              "python": sys.version.split()[0]})
    with open(fn + ".tmp", "w") as f:
        json.dump(r, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(fn + ".tmp", fn)
    print(fam, inst, seed, N0, meth, "U=", r.get("usage", r.get("error")), "ok=", r.get("all_ok"),
          f"{r['seconds']:.0f}s", flush=True)
    return r


def summary(n_boot=2000):
    """prereg/M1.md §6 + M1-A3: per family, geometric mean of usage over its (instance, seed) cells (0 if any cell is 0);
    best layer-R reference = min of that mean; equal-family geometric mean of ENG/best-R; bootstrap 95 % CI resampling
    seeds within (family, instance), seed 0. Only feasible runs with strict HW <= eps count; a family is complete only
    if every (instance, seed) cell of every compared method is present."""
    recs = {}
    for f in os.listdir(RUNS):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(RUNS, f)))
        strict = all(pb.get("worst", 2.0) <= 1.0 for pb in r.get("per_batch", []))
        if r.get("all_ok") and strict and "usage" in r:
            recs.setdefault((r["family"], r["N0"], r["method"]), {})[(r["instance"], r["seed"])] = float(r["usage"])
    fams = sorted({f for f, _ in INSTANCES})
    cells = {f: [(i, s) for ff, i in INSTANCES if ff == f for s in SEEDS] for f in fams}

    def gm(xs):
        return 0.0 if any(x == 0 for x in xs) else math.exp(np.mean(np.log(xs)))

    def fam_val(f, N0, m, pick):
        d = recs.get((f, N0, m), {})
        return gm([d[c] for c in pick]) if all(c in d for c in pick) else None

    def ratio(a, b):
        if a is None or b is None:
            return None
        return (1.0 if a == 0 else float("inf")) if b == 0 else a / b

    def agg(N0, picks):
        rs, per = [], {}
        for f in fams:
            eng = fam_val(f, N0, "ENG", picks[f])
            rv = [(fam_val(f, N0, m, picks[f]), m) for m in R]
            rv = [x for x in rv if x[0] is not None]
            sv = [(fam_val(f, N0, m, picks[f]), m) for m in S + ["Fcat"]]
            sv = [x for x in sv if x[0] is not None]
            rb = min(rv) if len(rv) == len(R) else (None, None)
            sb = min(sv) if sv else (None, None)
            per[f] = {"ENG": eng, "best_R": rb, "best_S": sb, "ratio_R": ratio(eng, rb[0]), "ratio_S": ratio(eng, sb[0])}
            rs.append(per[f]["ratio_R"])
        ok = all(x is not None and 0 < x < float("inf") for x in rs)
        return (math.exp(np.mean(np.log(rs))) if ok else None), per

    out = {}
    rng = np.random.default_rng(0)
    for N0 in N0S:
        point, per = agg(N0, cells)
        boots = []
        if point is not None:
            for _ in range(n_boot):
                picks = {}
                for f in fams:
                    picks[f] = [(i, int(s)) for ff, i in INSTANCES if ff == f for s in rng.choice(SEEDS, len(SEEDS))]
                b, _ = agg(N0, picks)
                if b is not None:
                    boots.append(b)
        ci = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else None
        out[N0] = {"families": per, "geomean_ratio_R": point, "ci95": ci, "pass_le_0.70": None if point is None
                   else point <= 0.70}
    print(json.dumps(out, indent=1, default=str))
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)


def main():
    if "--summary" in sys.argv:
        return summary()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if os.environ.get("SHADOWOPT_CPU"):
            psutil.Process().cpu_affinity([int(os.environ["SHADOWOPT_CPU"])])
    except Exception as e:
        print("priority not lowered:", e)
    os.makedirs(RUNS, exist_ok=True)
    cost = hw1_fitted()
    insts = INSTANCES
    for a in sys.argv:  # disjoint shards for separate single-thread processes: --inst=W1:BeH2,W3:Cluster100
        if a.startswith("--inst="):
            want = [tuple(x.split(":")) for x in a[7:].split(",")]
            insts = [x for x in INSTANCES if x in want]
    seeds, n0s, only = SEEDS, N0S, None
    for a in sys.argv:  # optional restriction (order of work only; the summary rules are unchanged)
        if a.startswith("--seeds="):
            seeds = tuple(int(x) for x in a[8:].split(","))
        if a.startswith("--N0="):
            n0s = tuple(int(x) for x in a[5:].split(","))
        if a.startswith("--methods="):
            only = a[10:].split(",")
    for N0 in n0s:
        for fam, inst in insts:
            for seed in seeds:
                for meth in (only or methods(fam)):
                    if meth in methods(fam):
                        run_one(fam, inst, seed, N0, meth, cost)


if __name__ == "__main__":
    main()
