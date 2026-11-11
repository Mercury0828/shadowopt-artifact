"""Paper figures from frozen result files only (no new measurement). Output: paper/figs/*.pdf.

Every figure reads the files named in writing/exemplar_map.md section 3; DEV/TEST labels follow the source directory.
"""
import glob
import json
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
PLAIN = FuncFormatter(lambda v, _: ("%g" % v))


def plainlog(a, axis="y"):
    ax_ = a.yaxis if axis == "y" else a.xaxis
    ax_.set_major_formatter(PLAIN)
    ax_.set_minor_formatter(NullFormatter())

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "lines.linewidth": 1.2,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": False, "axes.formatter.use_mathtext": False, "grid.color": "#dddddd", "grid.linewidth": 0.4})
W1C = 3.45  # one IEEE column, inches
# fixed entity colors (validated reference palette, slots 1-6) plus neutral for scans; markers give B/W identity
C = {"S*": "#555555", "bitmaps": "#184f95", "shared": "#eb6834", "views": "#9ec5f4",
     "ENG": "#184f95", "R": "#555555", "S": "#eda100"}
M = {"S*": "s", "bitmaps": "o", "shared": "^", "views": "D"}


def J(*p):
    return json.load(open(os.path.join(ROOT, *p), encoding="utf-8"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig)


def d1_rows():
    return [json.loads(l) for l in open(os.path.join(ROOT, "derisk/results/D1/results.jsonl"), encoding="utf-8")
            if l.strip()]


def fig_motivation():
    """(a) batch time vs n on synthetic calibration stores (N=1e6, 16 threads); (b) TEST W1 BeH2 QPU usage per method."""
    r = J("derisk/results/PATHOPT/rates.json")
    pts = sorted([p for p in r["points"] if p["N"] == 1_000_000], key=lambda p: p["n"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.45), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.32})
    ns = [p["n"] for p in pts]
    for k, lab in [("sstar", "S* scan"), ("bitmap", "bitmaps"), ("shared", "shared plans")]:
        key = {"sstar": "S*", "bitmap": "bitmaps", "shared": "shared"}[k]
        ax[0].plot(ns, [p[k][1] * 1e3 for p in pts], marker=M[key], ms=3.5, color=C[key], label=lab)
    ax[0].set_xlabel("qubits n (random stores, 1M rows)")
    ax[0].set_ylabel("batch time (ms)")
    ax[0].set_xticks(ns)
    ax[0].set_ylim(0, None)
    ax[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax[0].set_title("(a) answering one batch", fontsize=8)
    names = [("F1", "grouping,\nno reuse"), ("F3", "grouping\n+ reuse"), ("F5", "break-\neven"),
             ("F4", "derand-\nomization"), ("ENG", "SHADOW-\nOPT")]
    base = "results/HB1_TEST/runs/W1_BeH2_s100_N00_{}.json"
    vals = [J(base.format(m))["usage"] / 3600 for m, _ in names]
    lb = J(base.format("F2"))
    xs = np.arange(len(names))
    ax[1].bar(xs, vals, width=0.62, color=[C["R"]] * 4 + [C["ENG"]], edgecolor="white", linewidth=0.5)
    for x, v in zip(xs, vals):
        ax[1].text(x, v * 1.04, f"{v:.1f} h", ha="center", va="bottom", fontsize=7.5)
    ax[1].set_xticks(xs, [n for _, n in names], fontsize=7.5)
    ax[1].set_ylabel("estimated QPU usage (h)")
    ax[1].set_ylim(0, max(vals) * 1.22)
    ax[1].grid(axis="x", visible=False)
    ax[1].set_title("(b) four BeH2 energy queries, cumulative (TEST)", fontsize=8)
    save(fig, "fig_motivation")
    return {"n": ns, "sstar_ms": [p["sstar"][1] * 1e3 for p in pts], "bitmap_ms": [p["bitmap"][1] * 1e3 for p in pts],
            "usage_h": dict(zip([m for m, _ in names], vals)), "F2_lower_bound_h": lb["usage_lower_bound"] / 3600}


def fig_select():
    s = J("derisk/results/D1/D1_summary.json")["H1prime"]
    lab = {"LiH": "LiH / W1", "TFIM20": "Ising / W2", "GHZ50": "GHZ / W3"}
    fig, ax = plt.subplots(figsize=(W1C, 1.25))
    ys = np.arange(3)
    v = [s[k]["S*"] * 100 for k in lab]
    ax.barh(ys, v, height=0.55, color=C["S*"], edgecolor="white", label="compatibility test")
    ax.barh(ys, [100 - x for x in v], left=v, height=0.55, color="#cccccc", edgecolor="white", label="parity aggregation and rest")
    for y, x in zip(ys, v):
        ax.text(x / 2, y, f"{x:.0f}%", ha="center", va="center", color="white", fontsize=7.5)
    ax.set_yticks(ys, list(lab.values()))
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of S* batch time (%), 1M rows, DEV")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_select")
    return v


def fig_decisions():
    """Max miss rate over checkpoints per decision class (D2), DEV, delta = 0.05."""
    cls = [("a", "static\nstore"), ("b", "count-sized\nappends"), ("c", "repeated\nanswers"),
           ("d", "count-chosen\nsource"), ("e", "pooled\ntargeted"),
           ("f_alt2", "stop when\nsignificant")]
    states = [("LiH", "LiH"), ("TFIM20", "Ising"), ("GHZ50", "GHZ")]
    fig, ax = plt.subplots(figsize=(3.05, 2.0))
    w = 0.26
    out = {}
    ys = np.arange(len(cls))[::-1]
    for i, (st_, sl) in enumerate(states):
        vals = []
        for c, _ in cls:
            d = J(f"derisk/results/D2/{st_}_{c}.json")
            vals.append(max(x["miss_rate"] for v in d["results"].values() for x in v))
        out[st_] = vals
        ax.barh(ys - (i - 1) * w, vals, height=w, color=["#184f95", "#eb6834", "#9ec5f4"][i],
                edgecolor="white", linewidth=0.4, label=sl)
    ax.axvline(0.05, color="black", lw=0.8, ls="--")
    ax.text(0.051, ys[0] + 0.45, "$\\delta=0.05$", fontsize=7.5, ha="left", va="center")
    ax.set_yticks(ys, [l.replace("\n", " ") for _, l in cls])
    ax.set_xlabel("worst miss rate")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(0, 0.085)
    save(fig, "fig_decisions")
    return out


def fig_costmodel():
    h = J("qpu/HW1_fit.json")
    jobs = {j["key"]: j for j in h["jobs"]}
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    mods = [("M-doc", "documented model", "s", "#555555"), ("M2" if "M2" in h["models"] else "M1", "fitted model", "o", "#2a78d6")]
    out = {}
    for key, lab, mk, col in mods:
        m = h["models"][key]
        hd = m["heldout"]
        ax.scatter([x["U"] for x in hd], [x["pred"] for x in hd], marker=mk, s=16, color=col, label=f"{lab} (held out)",
                   zorder=3, edgecolor="white", linewidth=0.4)
        out[key] = m.get("heldout_median_abs_rel_err")
    lim = [0, max(x["U"] for x in h["models"]["M-doc"]["heldout"]) * 1.3]
    ax.plot(lim, lim, color="black", lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured usage (s)")
    ax.set_ylabel("predicted usage (s)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    save(fig, "fig_costmodel")
    return out, list(h["models"].keys())


def _batch(rows, state, N, path, T):
    k = f"batch|{state}|{N}|{path}|T{T}"
    r = [x for x in rows if x.get("key") == k]
    return r[-1] if r else None


def fig_speedup():
    rows = d1_rows()
    states = [("LiH", "LiH / W1"), ("TFIM20", "Ising / W2"), ("GHZ50", "GHZ / W3")]
    paths = ["bitmaps", "shared", "views"]
    fig, ax = plt.subplots(figsize=(W1C, 1.55))
    w = 0.26
    out = {}
    for j, p in enumerate(paths):
        meds, lo, hi = [], [], []
        for s, _ in states:
            a, b = _batch(rows, s, 1000000, "S*", 16), _batch(rows, s, 1000000, p, 16)
            r = a["median_s"] / b["median_s"]
            meds.append(r)
            lo.append(r - a["q25_s"] / b["q75_s"])
            hi.append(a["q75_s"] / b["q25_s"] - r)
        out[p] = meds
        ax.bar(np.arange(3) + (j - 1) * w, meds, width=w, color=C[p], edgecolor="white", linewidth=0.4, label=p,
               yerr=[lo, hi], error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": "black"})
    ax.axhline(1, color="black", lw=0.6)
    ax.set_xticks(np.arange(3), [l for _, l in states])
    ax.set_ylabel("speed-up over S*")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, max(max(v) for v in out.values()) * 1.35)
    save(fig, "fig_speedup")
    return out


def fig_scaling():
    rows = d1_rows()
    states = [("LiH", "LiH / W1"), ("TFIM20", "Ising / W2"), ("GHZ50", "GHZ / W3")]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.3))
    out = {}
    for i, (s, lab) in enumerate(states):
        for p in ["S*", "bitmaps", "shared", "views"]:
            for T, ls in [(16, "-"), (1, ":")]:
                ys = []
                for N in (100000, 1000000):
                    b = _batch(rows, s, N, p, T)
                    ys.append(b["median_s"] * 1e3 if b else float("nan"))
                out[(s, p, T)] = ys
                ax[i].plot([1e5, 1e6], ys, ls=ls, marker=M[p], ms=3, color=C[p],
                           label=f"{p}" if T == 16 else None)
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        plainlog(ax[i]); ax[i].set_xticks([1e5, 1e6], ["100k", "1M"]); ax[i].xaxis.set_minor_formatter(NullFormatter())
        ax[i].set_title(lab, fontsize=8)
        ax[i].set_xlabel("rows N")
    ax[0].set_ylabel("batch time (ms)")
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               title="solid lines: 16 threads, dotted lines: 1 thread", title_fontsize=7.5)
    save(fig, "fig_scaling")
    return {f"{k[0]}|{k[1]}|T{k[2]}": v for k, v in out.items()}


def fig_pathmodel():
    r = J("derisk/results/PATHOPT/rates.json")
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    keymap = {"sstar": "S*", "bitmap": "bitmaps", "shared": "shared"}
    allv = []
    for k, name in keymap.items():
        f = r["fit"][k]
        meas, pred, held = [], [], []
        for i, p in enumerate(r["points"]):
            words, sec = p[k]
            meas.append(sec * 1e3)
            pred.append((f["tau"] + f["sec_per_word"] * words) * 1e3)
            held.append(i in r["heldout_idx"])
        allv += meas + pred
        ax.scatter([m for m, h in zip(meas, held) if not h], [q for q, h in zip(pred, held) if not h], marker=M[name],
                   s=12, color=C[name], label=name, edgecolor="white", linewidth=0.3)
        ax.scatter([m for m, h in zip(meas, held) if h], [q for q, h in zip(pred, held) if h], marker=M[name], s=26,
                   facecolor="none", edgecolor="black", linewidth=0.7)
    lim = [min(allv) * 0.7, max(allv) * 1.4]
    ax.plot(lim, lim, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plainlog(ax, "x"); plainlog(ax, "y")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured time (ms)")
    ax.set_ylabel("predicted time (ms)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_pathmodel")
    return {k: r["fit"][k]["heldout_median_rel_err"] for k in keymap}


def fig_perbatch():
    """Cumulative QPU usage over the W1 batches (TEST BeH2, seed 100, empty store)."""
    meths = [("F1", "grouping, no reuse", "#999999", "x"), ("F3", "grouping + reuse", "#555555", "s"),
             ("F4", "derandomization", "#000000", "v"), ("F3qsA", "grouping + bound-aware alloc.", C["S"], "^"),
             ("ENG", "SHADOWOPT", C["ENG"], "o")]
    eps = [0.5, 0.3, 0.2, 0.1]
    fig, ax = plt.subplots(figsize=(W1C, 1.75))
    out = {}
    for m, lab, col, mk in meths:
        r = J(f"results/HB1_TEST/runs/W1_BeH2_s100_N00_{m}.json")
        cum = list(np.cumsum([b["U"] for b in r["per_batch"]]) / 3600)
        out[m] = cum
        ax.plot(range(len(cum)), cum, marker=mk, ms=3.2, color=col, label=lab,
                ls="--" if m == "F3qsA" else "-")
    ax.set_xticks(range(4), [f"target {e}" for e in eps])
    ax.set_ylabel("cum. est. usage (h)")
    ax.set_yscale("log")
    plainlog(ax)
    ax.set_ylim(0.02, 30)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, "fig_perbatch")
    return out


def fig_attribution():
    dev = J("derisk/results/HB1_A15/summary.json")["0"]["families"]
    tst = J("results/HB1_TEST/partial_summary.json")["0"]["families"]
    fams = ["W1", "W2", "W3", "W4", "W5", "W6"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.35), sharey=True, gridspec_kw={"wspace": 0.4})
    out = {}
    for a, (src, lab) in zip(ax, [(dev, "DEV"), (tst, "TEST")]):
        xs = np.arange(len(fams))
        rS = [src[f]["best_S"][0] / src[f]["best_R"][0] if src[f].get("best_S") else float("nan") for f in fams]
        rE = [src[f]["ratio_R"] for f in fams]
        out[lab] = {"S": rS, "ENG": rE}
        a.bar(xs - 0.2, rS, width=0.38, color=C["S"], edgecolor="white",
              label="best strengthened control")
        a.bar(xs + 0.2, rE, width=0.38, color=C["ENG"], edgecolor="white", label="SHADOWOPT")
        a.axhline(1, color="black", lw=0.6)
        a.set_xticks(xs, fams)
        a.set_title(f"{lab}, empty initial store", fontsize=8)
        a.grid(axis="x", visible=False)
    ax[0].set_ylabel("usage ratio")
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax[0].set_ylim(0, 1.2)
    save(fig, "fig_attribution")
    return out


def fig_validity():
    rates, ratios = [], []
    for s in ["LiH", "TFIM20", "GHZ50"]:
        sm = J(f"derisk/results/HC/{s}_summary.json")
        rates += [c["rate"] for c in sm["cells"].values()]
        for l in open(os.path.join(ROOT, f"derisk/results/HC/{s}.jsonl"), encoding="utf-8"):
            if l.strip():
                for tau, qi, err, hw, mmin in json.loads(l)["cp"]:
                    if hw and hw > 0:
                        ratios.append(err / hw)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.25))
    ax[0].hist(rates, bins=np.arange(0, 0.0525, 0.0025), color=C["ENG"], edgecolor="white")
    ax[0].axvline(0.05, color="black", ls="--", lw=0.8)
    ax[0].text(0.049, ax[0].get_ylim()[1] * 0.85, "$\\delta=0.05$", ha="right", fontsize=7.5)
    ax[0].set_xlabel("miss rate per checkpoint (1000 replicates)")
    ax[0].set_ylabel("checkpoints")
    ratios = np.array(ratios)
    cnt, edges = np.histogram(ratios, bins=np.linspace(0, max(1.05, float(ratios.max()) + 0.025), 45))
    keep = cnt > 0
    ax[1].bar(edges[:-1][keep], cnt[keep] - 0.8, bottom=0.8, width=np.diff(edges)[keep], align="edge", color=C["ENG"],
              edgecolor="white")
    ax[1].set_ylim(0.8, cnt.max() * 2)
    ax[1].axvline(1.0, color="black", ls="--", lw=0.8)
    ax[1].set_yscale("log")
    plainlog(ax[1])
    ax[1].set_xlabel("|error| / stated half-width")
    ax[1].set_ylabel("answers")
    save(fig, "fig_validity")
    return {"n_cp": len(rates), "max_rate": max(rates), "n_answers": int(len(ratios)),
            "median_ratio": float(np.median(ratios)), "p99_ratio": float(np.quantile(ratios, 0.99)),
            "share_over_1": float(np.mean(ratios > 1))}


if __name__ == "__main__":
    res = {}
    for f in [fig_motivation, fig_select, fig_decisions, fig_costmodel, fig_speedup, fig_scaling, fig_pathmodel,
              fig_perbatch, fig_attribution, fig_validity]:
        res[f.__name__] = f()
        print(f.__name__, "ok")
    json.dump(res, open(os.path.join(OUT, "figure_numbers.json"), "w"), indent=1, default=str)
