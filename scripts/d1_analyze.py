"""Analyse D1 results (derisk/results/D1/results.jsonl) against H1, H1', H1'' (PREREG + addendum A1)."""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "derisk", "results", "D1", "results.jsonl")
STATES = [("LiH", "W1"), ("TFIM20", "W2"), ("GHZ50", "W3")]


def load():
    return {json.loads(l)["key"]: json.loads(l) for l in open(RES, encoding="utf-8") if l.strip()}


def g(R, key):
    r = R.get(key)
    return None if r is None or r.get("skipped") else r


def main():
    R = load()
    out = {"H1": {}, "H1prime": {}, "H1dprime": {}, "tables": {}}
    for st, wl in STATES:
        # H1: selection share of HKP and PennyLane at N = 1e5
        sh = {}
        for m in ("hkp", "pennylane"):
            b, s = g(R, f"batch|{st}|100000|{m}|T1"), g(R, f"select|{st}|100000|{m}|T1")
            sh[m] = None if not (b and s) else s["median_s"] / b["median_s"]
        out["H1"][st] = sh
        # H1': selection share of S* (and S*fast) at N = 1e6, T = 16
        hp = {}
        for m in ("S*", "S*fast"):
            b, s = g(R, f"batch|{st}|1000000|{m}|T16"), g(R, f"select|{st}|1000000|{m}|T16")
            hp[m] = None if not (b and s) else s["median_s"] / b["median_s"]
        out["H1prime"][st] = hp
        # H1'': best planning-set path vs S* and vs the faster of S*/S*fast at N = 1e6, T = 16 (plan included)
        tab = {}
        for N in (100_000, 1_000_000):
            for T in (16, 1):
                for plan in ("batch", "batchcached"):
                    row = {}
                    for m in ("S*", "S*fast", "bitmaps", "shared", "views", "hkp", "pennylane"):
                        r = g(R, f"{plan}|{st}|{N}|{m}|T{T}") if m not in ("hkp", "pennylane") else \
                            (g(R, f"batch|{st}|{N}|{m}|T1") if plan == "batch" else None)
                        if r:
                            row[m] = {"median": r["median_s"], "q25": r["q25_s"], "q75": r["q75_s"], "reps": r["reps"]}
                    tab[f"{plan}|N{N}|T{T}"] = row
        out["tables"][st] = tab
        row = tab["batch|N1000000|T16"]
        need = ("S*", "S*fast", "bitmaps", "shared", "views")
        if not all(m in row for m in need):
            out["H1dprime"][st] = {"pending": [m for m in need if m not in row]}
            continue
        paths = {m: row[m] for m in ("bitmaps", "shared", "views")}
        best = min(paths, key=lambda m: paths[m]["median"])
        cands = {}
        for k in ("S*", "S*fast"):
            cands[f"frozen_vs_{k}"] = row[k]["median"] / paths[best]["median"]
            pr = g(R, f"paired|{st}|1000000|{k}|{best}|T16")
            cands[f"paired_vs_{k}"] = pr["median_ratio"] if pr else None
        missing = [c for c, v in cands.items() if v is None]
        out["H1dprime"][st] = {"best_path": best, "speedups": cands,
                               "interval_frozen_vs_S*": [row["S*"]["q25"] / paths[best]["q75"],
                                                         row["S*"]["q75"] / paths[best]["q25"]],
                               "decisive_speedup": None if missing else min(cands.values()),
                               "pending": missing}
    # verdicts
    h1_vals = [v for st in out["H1"] for v in out["H1"][st].values()]
    out["H1_pass"] = None if any(v is None for v in h1_vals) else all(v >= 0.5 for v in h1_vals)
    sp_frozen = [out["H1prime"][st].get("S*") for st in out["H1prime"]]
    out["H1prime_pass_frozen"] = None if any(v is None for v in sp_frozen) else sum(v >= 0.3 for v in sp_frozen) >= 2
    sp_fast = [out["H1prime"][st].get("S*fast") for st in out["H1prime"]]
    out["H1prime_pass_Sfast"] = None if any(v is None for v in sp_fast) else sum(v >= 0.3 for v in sp_fast) >= 2
    dec = [out["H1dprime"][st].get("decisive_speedup") for st in ("LiH", "TFIM20", "GHZ50")]
    frozen = [out["H1dprime"][st].get("speedups", {}).get("frozen_vs_S*") for st in ("LiH", "TFIM20", "GHZ50")]
    out["H1dprime_pass_frozen"] = "pending" if any(v is None for v in frozen) else sum(v >= 2 for v in frozen) >= 2
    out["H1dprime_pass_decisive"] = "pending" if any(v is None for v in dec) else sum(v >= 2 for v in dec) >= 2
    out["Ha_info_5x_decisive"] = {st: (None if d is None else d >= 5) for st, d in zip(("LiH", "TFIM20", "GHZ50"), dec)}
    if out["H1_pass"] is None:
        out["H1_pass"] = "pending"
    for k in ("H1prime_pass_frozen", "H1prime_pass_Sfast"):
        if out[k] is None:
            out[k] = "pending"
    json.dump(out, open(os.path.join(ROOT, "derisk", "results", "D1", "D1_summary.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k != "tables"}, indent=1))


if __name__ == "__main__":
    main()
