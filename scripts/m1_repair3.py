"""PREREG A7: repair attempt #3 screen (W4, W6; seeds 1, 2; fitted cost model) -> derisk/results/M1_L3/repair3/"""
import json, os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from m1_repair1 import fam_of, lower_priority
from shadowopt.planner_v3 import run_v3
from shadowopt.cost import hw1_fitted
lower_priority()
OUT = os.path.join(ROOT, "derisk", "results", "M1_L3", "repair3"); os.makedirs(OUT, exist_ok=True)
cost = hw1_fitted()
for fn in ("W4", "W6"):
    for seed in (1, 2):
        for s in ("Fcat", "Ffix0.25", "Ffix0.5", "Ffix0.75", "SO3"):
            f = os.path.join(OUT, f"{fn}_s{seed}_{s}.json")
            if os.path.exists(f):
                continue
            t = time.time(); r = run_v3(s, fam_of(fn, seed), seed, cost); r["seconds"] = time.time() - t
            json.dump(r, open(f, "w"), default=float)
            print(fn, seed, s, round(r["usage"], 3), "jobs", r["jobs"], "ok", r["all_ok"], f"{r['seconds']:.0f}s",
                  [(d["tau"], d["f"]) for d in r["decisions"]][:8], flush=True)
