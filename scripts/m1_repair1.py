"""PREREG addendum A5: planner repair attempt #1 screen (DEV). Resumable: one JSON per run in
derisk/results/M1_L3/repair1/runs/. Phase 1: all strategies x families x seeds under the HW1-fitted cost model.
Phase 2 (`--sensitivity`): SO1 and the best baseline per family at the four (t_sub, t_bind) CI corners.
Runs as ONE light process: numba threads = 1, BelowNormal priority (no timing is taken; CPU lock held by others).
"""
import json
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from d4_planner import families
from shadowopt.workloads import w2_correlators, w3_local_fidelities
from shadowopt.planner import run_strategy
from shadowopt.planner_v1 import run_v1
from shadowopt.cost import hw1_fitted

OUT = os.path.join(ROOT, "derisk", "results", "M1_L3", "repair1")
RUNS = os.path.join(OUT, "runs")
OLD = ["F1", "F2", "F3", "F4", "F5", "F7"]
NEW = ["F2qs", "F3qs", "F4qs", "F4qsF", "SO1"]
FAMS = ["W4", "W6", "W2", "W3", "W1"]
CORNERS = [(1.67, 0.155e-3), (1.67, 1.52e-3), (3.52, 0.155e-3), (3.52, 1.52e-3)]


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        aff = os.environ.get("SHADOWOPT_CPU")  # one logical core
        if aff:
            psutil.Process().cpu_affinity([int(aff)])
    except Exception as e:
        print("priority not lowered:", e)


def fam_of(name, seed):
    if name in ("W1", "W4", "W6"):
        return families(seed)[name]
    if name == "W2":
        return w2_correlators(20)
    if name == "W3":
        return w3_local_fidelities(50)
    if name == "W5":
        from shadowopt.workloads import w5_drift
        return w5_drift(20, seed)
    raise KeyError(name)


def run(fam_name, seed, strat, cost, tag):
    fn = os.path.join(RUNS, f"{fam_name}_s{seed}_{strat}_{tag}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    fam = fam_of(fam_name, seed)
    t0 = time.time()
    try:
        r = run_strategy(strat, fam, seed=seed, cost=cost) if strat in OLD else run_v1(strat, fam, seed, cost)
    except Exception as e:  # recorded, never silently dropped
        r = {"name": strat, "error": f"{type(e).__name__}: {e}"}
    r["seconds"] = time.time() - t0
    r.update({"family": fam_name, "seed": seed, "cost": cost.as_dict(), "tag": tag})
    r.pop("settings_used", None)
    with open(fn + ".tmp", "w") as f:
        json.dump(r, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(fn + ".tmp", fn)
    print(fam_name, seed, strat, tag, "U=", r.get("usage", r.get("usage_lower_bound", r.get("error"))),
          "ok=", r.get("all_ok"), f"{r['seconds']:.0f}s", flush=True)
    return r


def main():
    lower_priority()
    os.makedirs(RUNS, exist_ok=True)
    cost = hw1_fitted()
    sens = "--sensitivity" in sys.argv
    for fam_name in FAMS:
        for seed in (1, 2):
            if not sens:
                for strat in OLD + NEW:
                    run(fam_name, seed, strat, cost, "fit")
            else:
                rs = {s: run(fam_name, seed, s, cost, "fit") for s in OLD + NEW}
                base = {s: r["usage"] for s, r in rs.items() if s != "SO1" and r.get("all_ok") and "usage" in r}
                best = min(base, key=base.get)
                for ts, tb in CORNERS:
                    c = hw1_fitted(t_sub=ts, t_set=tb)
                    tag = f"corner_{ts}_{tb}"
                    run(fam_name, seed, "SO1", c, tag)
                    run(fam_name, seed, best, c, tag)


if __name__ == "__main__":
    main()
