"""D3 (H3, counts-known-before-submission check) and D4 (planner screen) per derisk/PREREG.md + addendum A2.
Resumable: one JSON per (family, seed, strategy) in derisk/results/D4/runs/.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.store import TermBatch
from shadowopt.workloads import w1_energy, w4_zipf, w6_heavytail
from shadowopt.planner import run_strategy, StreamCounter
from shadowopt.hindsight import solve_hindsight
from shadowopt.cost import CostModel
from shadowopt.sampling import uniform_bases, sample_statevector

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "derisk", "results", "D4")
RUNS = os.path.join(OUT, "runs")
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0]
FIXED = ["F1", "F2", "F3", "F4", "F5", "F7"]


def families(seed):
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    import json as _j
    const = _j.load(open(os.path.join(ROOT, "stores", "states", "LiH", "manifest.json")))["const"]
    U = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
    return {"W1": w1_energy(U, const, eps_seq=(0.5, 0.3, 0.2, 0.1)),
            "W4": w4_zipf(20, seed=seed), "W6": w6_heavytail(50, seed=seed)}


def run(fam_name, fam, seed, strat, lam=0.0):
    fn = os.path.join(RUNS, f"{fam_name}_s{seed}_{strat}_lam{lam}.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    r = run_strategy(strat, fam, seed=seed, lam=lam)  # stream seed [family seed, chunk] (A2)
    r["seconds"] = time.time() - t0
    r["family"], r["seed"] = fam_name, seed
    r["settings_used"] = [list(s) for s in r.get("settings_used", [])]
    with open(fn + ".tmp", "w") as f:
        json.dump(r, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    os.replace(fn + ".tmp", fn)
    print(fam_name, seed, strat, lam, "U=", r.get("usage", r.get("usage_lower_bound")), "ok=", r.get("all_ok"),
          f"{r['seconds']:.0f}s", flush=True)
    return r


def counts_known_check():
    """D3: post-append counts computed from drawn bases before 'submission' equal counts recomputed from the store after
    sampling outcomes (LiH state, 20 000 snapshots, the W1 terms)."""
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    U = TermBatch(12, d["ptr"], d["q"], d["b"])
    psi = np.load(os.path.join(ROOT, "stores", "states", "LiH", "state.npz"))["psi"]
    rng = np.random.default_rng(424242)
    B = uniform_bases(rng, 20_000, 12)
    pre = np.array([int(np.all(B[:, U.term(t)[0]] == U.term(t)[1], axis=1).sum()) for t in range(U.T)])
    O = sample_statevector(psi, 12, B, np.random.default_rng(1))
    from shadowopt.store import Store
    from shadowopt.scan import naive_reference
    st = Store(12)
    st.append(B, O)
    post, _ = naive_reference(st, U)
    return {"terms": int(U.T), "equal": bool(np.array_equal(pre, post))}


def main():
    os.makedirs(RUNS, exist_ok=True)
    res = {"counts_known_before_submission": counts_known_check(), "cost_model": CostModel().as_dict()}
    print("D3 counts check", res["counts_known_before_submission"], flush=True)
    # seed 0: lambda grid + all fixed strategies (tuning); seeds 1-2: evaluation
    tuned = {}
    for fam_name in ("W4", "W6", "W1"):
        fam = families(0)[fam_name]
        us = {lam: run(fam_name, fam, 0, "SHADOWOPT", lam).get("usage", np.inf) for lam in LAMS}
        tuned[fam_name] = min(LAMS, key=lambda l: (us[l], l))
        for s in FIXED + ["SHADOWOPT_APPENDONLY"]:
            run(fam_name, fam, 0, s)
    res["lambda_tuned_seed0"] = tuned
    for seed in (1, 2):
        fams = families(seed)
        for fam_name in ("W4", "W6", "W1"):
            fam = fams[fam_name]
            run(fam_name, fam, seed, "SHADOWOPT", tuned[fam_name])
            run(fam_name, fam, seed, "SHADOWOPT", 0.0)
            for s in FIXED + ["SHADOWOPT_APPENDONLY"]:
                run(fam_name, fam, seed, s)
    json.dump(res, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    # OPT_hind on W4 seed 1
    fn = os.path.join(OUT, "opt_hind_W4_s1.json")
    if not os.path.exists(fn):
        fam = families(1)["W4"]
        sets = []
        for s in ["SHADOWOPT", "F1", "F3", "F5", "F7"]:
            lam = tuned["W4"] if s == "SHADOWOPT" else 0.0
            sets += [np.array(x) for x in run("W4", fam, 1, s, lam).get("settings_used", [])]
        t0 = time.time()
        h = solve_hindsight(fam, StreamCounter(fam.universe, 1), CostModel(), sets, time_limit=600)
        h["seconds"] = time.time() - t0
        json.dump(h, open(fn, "w"), indent=1)
        print("OPT_hind", h, flush=True)
    # F6 time-boxed attempt on W1 (seed 0..2), 15 min per batch
    for seed in (0, 1, 2):
        fn = os.path.join(RUNS, f"W1_s{seed}_F6_lam0.0.json")
        if os.path.exists(fn):
            continue
        fam = families(seed)["W1"]
        fam.f6_timebox_s = 900
        t0 = time.time()
        try:
            r = run_strategy("F6", fam, seed=seed)
            r["seconds"] = time.time() - t0
        except TimeoutError as e:
            r = {"name": "F6", "timeout": str(e), "seconds": time.time() - t0}
        r["family"], r["seed"] = "W1", seed
        r.pop("settings_used", None)
        json.dump(r, open(fn, "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print("F6 W1", seed, r.get("usage"), r.get("timeout"), flush=True)
        if "timeout" in r:
            break  # obstacle recorded; the time box (one working day) is not spent on further seeds
    print("D4 done", flush=True)


if __name__ == "__main__":
    main()
