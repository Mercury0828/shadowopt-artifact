"""H-c replicate campaign.
Usage: hc_campaign.py <state> <n_replicates> [--summary]   (resumable by validated replicate IDs)
Checkpoints: W1 (LiH) the energy at every batch; W2 (TFIM20) the first 10 correlators at batches 0 and 5;
W3 (GHZ50) every window fidelity at batch 0. Replicate r uses seed [1_000_000 + r] (fresh store and outcomes). The same replicate id gives related random streams for
different states: states are tested separately (per-checkpoint marginal claim), so this cross-state
dependence does not affect any reported test; it is documented, not changed mid-campaign.
Output: derisk/results/HC/<state>.jsonl and a summary (misses per checkpoint, exact one-sided binomial p, unavailable).
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gen_stores import load_state
from seq_runner import family_and_truth
from shadowopt.hc import run_replicate

OUT = os.path.join(ROOT, "derisk", "results", "HC")
N0 = 100_000


def checkpoints(state, fam):
    if state == "LiH":
        return {(t, 0) for t in range(len(fam.batches))}
    if state == "TFIM20":
        return {(t, q) for t in (0, 5) for q in range(10)}
    if state == "GHZ50":
        return {(0, q) for q in range(len(fam.batches[0]))}
    raise KeyError(state)


def _valid_rows(fn):
    """Complete JSON records only. Only the LAST non-empty line may be a truncated record (it is dropped); a corrupt record
    anywhere else raises; duplicate ids raise; the file is rewritten so that it always ends with a newline before any
    append."""
    if not os.path.exists(fn):
        return []
    raw = open(fn).read()
    lines = [l for l in raw.split("\n") if l.strip()]
    rows = []
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise RuntimeError(f"corrupt record in the middle of {fn} (record {i})")
            lines = lines[:-1]
    clean = "".join(x + "\n" for x in lines)
    if clean != raw:
        open(fn, "w").write(clean)
    ids = [r["r"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate replicate ids in {fn}")
    return rows


def summarize(state, delta=0.05):
    from scipy.stats import binomtest
    rows = _valid_rows(os.path.join(OUT, f"{state}.jsonl"))
    cells = {}
    for r in rows:
        for tau, qi, err, hw, mmin in r["cp"]:
            c = cells.setdefault((tau, qi), {"n": 0, "miss": 0, "unavailable": 0})
            if mmin == 0:
                c["unavailable"] += 1
                continue
            c["n"] += 1
            c["miss"] += int(err > hw)
    out = {}
    for (tau, qi), c in sorted(cells.items()):
        p = binomtest(c["miss"], c["n"], delta, alternative="greater").pvalue if c["n"] else None
        out[f"{tau}:{qi}"] = {**c, "rate": c["miss"] / max(1, c["n"]), "p_one_sided": p, "reject_at_0.01": bool(p is not None and p < 0.01)}
    worst = max(out.values(), key=lambda v: v["rate"]) if out else None
    summ = {"state": state, "replicates": len(rows), "checkpoints": len(out), "rejections": sum(v["reject_at_0.01"] for v in out.values()),
            "max_rate": worst["rate"] if worst else None, "cells": out}
    json.dump(summ, open(os.path.join(OUT, f"{state}_summary.json"), "w"), indent=1)
    print({k: v for k, v in summ.items() if k != "cells"})


def main():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if os.environ.get("SHADOWOPT_CPU"):
            psutil.Process().cpu_affinity([int(os.environ["SHADOWOPT_CPU"])])
    except Exception:
        pass
    extra = [a for a in sys.argv[3:] if a != "--summary"]
    if extra:
        sys.exit(f"refused: unknown options {extra} (no --start: replicates always run as ids 0..R-1, resumed by id)")
    state, R = sys.argv[1], int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    if "--summary" in sys.argv:
        return summarize(state)
    fam, truth = family_and_truth(state)
    n, sampler = load_state(state)
    psi = None
    if state in ("LiH", "TFIM20"):
        psi = np.load(os.path.join(ROOT, "stores", "states", state, "state.npz"))["psi"]
    cps = checkpoints(state, fam)
    fn = os.path.join(OUT, f"{state}.jsonl")
    rows = _valid_rows(fn)  # drops a truncated trailing record (rewrites the file), rejects duplicate ids
    done = {r["r"] for r in rows}
    with open(fn, "a") as f:
        for r in range(R):
            if r in done:
                continue
            t0 = time.time()
            cp = run_replicate(fam, truth, n, sampler, psi, N0, 1_000_000 + r, cps)
            f.write(json.dumps({"r": r, "seconds": time.time() - t0, "cp": cp}) + "\n")
            f.flush()
            if r % 20 == 0:
                print(state, r, f"{time.time() - t0:.1f}s", flush=True)
    summarize(state)


if __name__ == "__main__":
    main()
