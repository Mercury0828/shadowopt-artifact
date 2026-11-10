"""Report of the H-b'1 TEST campaign over the runs that are present. Same rules as
hb1_test.summary (feasible, strict HW <= eps; seed/instance geometric means; best layer-R = minimum), applied to the
(instance, seed) cells where ENG and every FEASIBLE layer-R method are present. A layer-R method whose runs hit the
planner's round cap without certifying (infeasible) is excluded from the minimum, as on DEV (A9), and its usage lower
bound is reported so a reader can see it is not the cheapest. Families and cells are reported with their counts; the
cross-family aggregate is labelled with the families it covers (it is NOT the registered six-family headline)."""
import json, math, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
RUNS = os.path.join(ROOT, "results", "HB1_TEST", "runs")
R = ["F1", "F2", "F3", "F4", "F5"]
S = ["F7", "F2qs", "F3qs", "F4qs", "F3qsA", "Fcat"]
ok, lb = {}, {}
for f in os.listdir(RUNS):
    if not f.endswith(".json"):
        continue
    r = json.load(open(os.path.join(RUNS, f)))
    key = (r["family"], int(r["N0"]), r["method"])
    cell = (r["instance"], int(r["seed"]))
    strict = all(pb.get("worst", 2.0) <= 1.0 for pb in r.get("per_batch", []))
    if r.get("all_ok") and strict and "usage" in r:
        ok.setdefault(key, {})[cell] = float(r["usage"])
    elif r.get("infeasible") in (True, "True"):
        lb.setdefault(key, {})[cell] = float(r.get("usage_lower_bound", r.get("usage", "nan")))
gm = lambda xs: 0.0 if any(x == 0 for x in xs) else math.exp(np.mean(np.log(xs)))
out = {}
for N0 in (0, 100000):
    fams = {}
    for fam in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        eng = ok.get((fam, N0, "ENG"), {})
        cells = [c for c in eng if all(c in ok.get((fam, N0, m), {}) or c in lb.get((fam, N0, m), {}) for m in R)]
        if not cells:
            continue
        feas = [m for m in R if all(c in ok.get((fam, N0, m), {}) for c in cells)]
        infe = {m: gm([lb[(fam, N0, m)][c] for c in cells]) for m in R if m not in feas}
        rv = sorted((gm([ok[(fam, N0, m)][c] for c in cells]), m) for m in feas)
        sv = sorted((gm([ok[(fam, N0, m)][c] for c in cells]), m) for m in S
                    if all(c in ok.get((fam, N0, m), {}) for c in cells))
        e = gm([eng[c] for c in cells])
        b = rv[0]
        fams[fam] = {"cells": sorted(cells), "n_cells": len(cells), "ENG": e, "best_R": b, "ratio_R": e / b[0] if b[0] else
                     (1.0 if e == 0 else float("inf")), "best_S": sv[0] if sv else None,
                     "ratio_S": (e / sv[0][0] if sv and sv[0][0] else None), "infeasible_R_lower_bounds": infe}
    rs = [v["ratio_R"] for v in fams.values()]
    out[N0] = {"families": fams, "families_covered": sorted(fams),
               "geomean_ratio_R_covered": math.exp(np.mean(np.log(rs))) if rs and all(0 < x < float("inf") for x in rs) else None}
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(ROOT, "results", "HB1_TEST", "partial_summary.json"), "w"), indent=1, default=str)
