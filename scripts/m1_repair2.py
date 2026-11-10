"""PREREG A6: repair attempt #2 screen (W4, W6; seeds 1, 2; fitted cost model) -> derisk/results/M1_L3/repair2/"""
import json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from m1_repair1 import fam_of, lower_priority
from shadowopt.planner_v2 import run_v2
from shadowopt.cost import hw1_fitted
lower_priority()
OUT = os.path.join(ROOT, "derisk", "results", "M1_L3", "repair2"); os.makedirs(OUT, exist_ok=True)
cost = hw1_fitted()
for fn in ("W4", "W6"):
    for seed in (1, 2):
        for s in ("F3R", "SO2C0", "Fcat", "F3cat", "SO2C1"):
            f = os.path.join(OUT, f"{fn}_s{seed}_{s}.json")
            if os.path.exists(f):
                continue
            t = time.time(); r = run_v2(s, fam_of(fn, seed), seed, cost); r["seconds"] = time.time() - t
            json.dump(r, open(f, "w"), default=float)
            print(fn, seed, s, round(r["usage"], 3), "jobs", r["jobs"], "ok", r["all_ok"], f"{r['seconds']:.0f}s",
                  (r["decisions"][:3] if r["decisions"] else ""), flush=True)
