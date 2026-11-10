"""HW1 fit per qpu/HW1_analysis_plan.md (pre-registered before any job). Reads qpu/ledger.csv; writes qpu/HW1_fit.json.

Models (non-negative least squares on the 20 training jobs A-G x 2 reps, response job.usage()):
  M-doc: U = 2 n_sub + 350e-6 n_exec (no fit)
  M1:    U = t_sub n_sub + t_set S + t_shot n_exec
  M2:    U = t_sub n_sub + t_bind S_param + t_circ S_circ + t_shot n_exec   (primary; "single" counts as one circuit)
H-d: median |relative error| of M2 on the 6 held-out jobs (H, I, J x 2) <= 15 %.
Also: LOOCV on training jobs; bootstrap 95 % CIs of constants (resample training jobs, 2000 draws, seed 0);
secondary response span_sum_s reported only.
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import nnls

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from shadowopt.qpu import GRID

LEDGER = os.path.join(ROOT, "qpu", "ledger.csv")


def rows():
    out = []
    for r in csv.DictReader(open(LEDGER, newline="", encoding="utf-8")):
        if r["tier"] != "HW1" or r["status"] != "ok":
            continue
        S, s = int(r["settings"]), int(r["shots"])
        mode = r["mode"]
        out.append({"key": r["purpose"], "role": r["role"], "mode": mode, "S": S, "s": s, "n_exec": S * s,
                    "n_sub": 1, "U": float(r["usage_quantum_s"]),
                    "span": float(r["span_sum_s"]) if r["span_sum_s"] not in ("", "None") else None})
    return out


def X(rs, model):
    if model == "M1":
        return np.array([[r["n_sub"], r["S"], r["n_exec"]] for r in rs], float)
    if model == "M2":
        return np.array([[r["n_sub"], r["S"] if r["mode"] == "param" else 0, r["S"] if r["mode"] != "param" else 0,
                          r["n_exec"]] for r in rs], float)
    raise KeyError(model)


NAMES = {"M1": ["t_sub", "t_set", "t_shot"], "M2": ["t_sub", "t_bind", "t_circ", "t_shot"]}


def fit(rs, model, resp="U"):
    A = X(rs, model)
    y = np.array([r[resp] for r in rs], float)
    coef, _ = nnls(A, y)
    return coef


def predict(rs, model, coef):
    if model == "M-doc":
        return np.array([2.0 * r["n_sub"] + 350e-6 * r["n_exec"] for r in rs])
    return X(rs, model) @ coef


def main():
    rs = rows()
    tr = [r for r in rs if r["role"] == "train"]
    ho = [r for r in rs if r["role"] == "heldout"]
    res = {"n_train": len(tr), "n_heldout": len(ho), "jobs": rs, "models": {}}
    rng = np.random.default_rng(0)
    for model in ("M-doc", "M1", "M2"):
        m = {}
        coef = None if model == "M-doc" else fit(tr, model)
        if coef is not None:
            m["constants"] = dict(zip(NAMES[model], coef.tolist()))
            boot = []
            for _ in range(2000):
                idx = rng.integers(0, len(tr), len(tr))
                boot.append(fit([tr[i] for i in idx], model))
            boot = np.array(boot)
            m["ci95"] = {n: [float(np.percentile(boot[:, k], 2.5)), float(np.percentile(boot[:, k], 97.5))]
                         for k, n in enumerate(NAMES[model])}
            loo = []
            for i in range(len(tr)):
                c = fit(tr[:i] + tr[i + 1:], model)
                p = float(predict([tr[i]], model, c)[0])
                loo.append(abs(p - tr[i]["U"]) / tr[i]["U"])
            m["loocv_median_abs_rel_err"] = float(np.median(loo))
            span_tr = [r for r in tr if r["span"] is not None]
            if len(span_tr) >= len(NAMES[model]):
                m["span_constants"] = dict(zip(NAMES[model], fit(span_tr, model, "span").tolist()))
        if ho:
            p = predict(ho, model, coef)
            err = np.abs(p - np.array([r["U"] for r in ho])) / np.array([r["U"] for r in ho])
            m["heldout"] = [{"key": r["key"], "U": r["U"], "pred": float(pp), "abs_rel_err": float(e)}
                            for r, pp, e in zip(ho, p, err)]
            m["heldout_median_abs_rel_err"] = float(np.median(err))
        res["models"][model] = m
    if "heldout_median_abs_rel_err" in res["models"]["M2"]:
        res["H_d"] = {"model": "M2", "median_abs_rel_err": res["models"]["M2"]["heldout_median_abs_rel_err"],
                      "threshold": 0.15, "pass": res["models"]["M2"]["heldout_median_abs_rel_err"] <= 0.15}
    ci = res["models"]["M2"].get("ci95", {}).get("t_bind")
    res["t_bind_positive_95"] = bool(ci and ci[0] > 0)
    json.dump(res, open(os.path.join(ROOT, "qpu", "HW1_fit.json"), "w"), indent=1)
    for model, m in res["models"].items():
        print(model, m.get("constants"), "CI", m.get("ci95"), "LOOCV", m.get("loocv_median_abs_rel_err"),
              "held-out", m.get("heldout_median_abs_rel_err"))
    print("H-d", res.get("H_d"), "t_bind > 0 (95 %):", res["t_bind_positive_95"])


if __name__ == "__main__":
    main()
