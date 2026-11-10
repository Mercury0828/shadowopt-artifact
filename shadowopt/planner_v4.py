"""PREREG A15 — §0★★★ ladder, H-b'1 repair #1 (after the A9 DEV screen).

SO4 = SO1 (planner_v1: pooled counts, query-level stopping, per-round minimum-usage choice) with a larger candidate set:
- GROUP*  : the engine's completion (certify.complete, A13 orders) with per-setting bound-aware shot allocation
            (shadowopt/alloc.py; never costlier than the uniform-alpha completion on the same input);
- F4      : the faithful HKP derandomized plan of the round (term thresholds, pooled counts as its starting point —
            the same plan the layer-R workflow F4 would buy from this state);
- all SO1 candidates unchanged (APPEND-qs, GROUP-qs, DERAND-qs, DERAND-qs+fill, MIX_w).
Every candidate's usage is computed from counts before any QPU second (B2). The per-round choice is the minimum usage
(first in the listed order on ties).

F3qsA (layer-S control for attribution, H-b'2): GROUP* every round (pooled counts), no portfolio.
"""
import numpy as np

from .certify import BatchCSR, certify
from .alloc import complete_alloc
from .derand import hkp_derandomize, consolidate
from .planner import StreamCounter, batch_requirements, setting_counts, INF
from .planner_v1 import PooledState, batch_feasible, append_qs, group_qs, derand_qs


def _group_star(st, fam, tau, csr):
    m = st.best_counts(csr.terms)
    hw, ok = certify(csr, m)
    plan, _, info = complete_alloc(fam.universe, csr, m, hw, ok, st.cost)
    return plan, info


def run_v4(name, fam, seed, cost, stream=None, max_rounds=20, mix_weights=(1, 2, 3), L0=0):
    """Strategies: SO4 (engine), F3qsA (layer-S control). Returns dict like run_v1."""
    stream = stream or StreamCounter(fam.universe, seed)
    st = PooledState(fam.universe, stream, cost, pooled=True)
    st.L = int(L0)
    U = fam.universe
    w = U.weights()
    per_batch = []
    for tau in range(len(fam.batches)):
        req = batch_requirements(fam, tau)
        csr = BatchCSR(fam.batches[tau])
        u0 = st.usage
        for rnd in range(max_rounds):
            ok, _ = batch_feasible(st, fam, tau)
            if ok:
                break
            us = np.array(sorted(req), np.int64)
            have = st.best_counts(us)
            deficit = [int(u) for u, h in zip(us, have) if h < req[int(u)]]
            if not deficit:
                deficit = [int(u) for u in us]
            if name == "F3qsA":
                plan, info = _group_star(st, fam, tau, csr)
                st.do_target(plan, tau, f"F3qsA:{info['chosen']}")
                continue
            if name != "SO4":
                raise KeyError(name)
            fseed = [seed, tau, rnd, 7]
            cands = []
            n_add = append_qs(st, fam, tau)
            if n_add < INF:
                cands.append({"kind": "APPEND", "n_add": n_add, "U": cost.append(n_add)})
            gp = group_qs(st, fam, tau, deficit, req)
            cands.append({"kind": "GROUP", "target": gp, "U": cost.target([k for _, k in gp])})
            gs, ginfo = _group_star(st, fam, tau, csr)
            cands.append({"kind": "GROUP*", "target": gs, "U": cost.target([k for _, k in gs]), "info": ginfo["chosen"]})
            ss = derand_qs(st, fam, tau, deficit, req)
            cands.append({"kind": "DERAND", "derand": ss, "U": cost.job(len(ss), sum(k for _, k in ss))})
            ssf = derand_qs(st, fam, tau, deficit, req, fill_seed=fseed)
            cands.append({"kind": "DERANDF", "derand": ssf, "U": cost.job(len(ssf), sum(k for _, k in ssf))})
            dd = np.array(deficit, np.int64)
            seq, _ = hkp_derandomize(U, dd, np.array([req[u] for u in deficit]), have=st.best_counts(dd))
            s4 = consolidate(seq)
            cands.append({"kind": "F4", "derand": s4, "U": cost.job(len(s4), sum(k for _, k in s4))})
            for wcut in mix_weights:
                A = [u for u in deficit if w[u] <= wcut]
                if not A or len(A) == len(deficit):
                    continue
                pos_ = max(st.stream.position(u, req[u] - int(st.best_counts([u])[0]) + int(st.mR([u])[0])) for u in A)
                if pos_ >= INF:
                    continue
                na = max(0, pos_ - st.L)
                if na == 0:
                    continue
                dR = st.stream.count(np.array(deficit), st.L + na) - st.mR(np.array(deficit))
                ss2 = derand_qs(st, fam, tau, deficit, req, extraR=dict(zip(deficit, dR.tolist())))
                cands.append({"kind": f"MIX{wcut}", "n_add": na, "derand": ss2,
                              "U": cost.job(na + len(ss2), na + sum(k for _, k in ss2))})
            best = min(cands, key=lambda c: c["U"])
            st.log.append({"tau": tau, "action": "CANDIDATES", "chosen": best["kind"],
                           "U": {c["kind"]: c["U"] for c in cands}, "group_star": ginfo["chosen"]})
            if best["kind"].startswith("MIX"):
                na, ss2 = best["n_add"], best["derand"]
                st.L += int(na)
                st.mD += setting_counts(U, ss2)
                st.usage += best["U"]
                st.log.append({"tau": tau, "action": "MIX", "n_append": int(na), "U": best["U"],
                               "tag": f"SO4:{best['kind']}"})
            elif best["kind"] == "APPEND":
                st.do_append(best["n_add"], tau, "SO4:APPEND")
            elif best["kind"] in ("GROUP", "GROUP*"):
                st.do_target(best["target"], tau, f"SO4:{best['kind']}")
            else:
                st.do_derand(best["derand"], tau, f"SO4:{best['kind']}")
        ok, worst = batch_feasible(st, fam, tau)
        per_batch.append({"tau": tau, "U": st.usage - u0, "ok": ok, "worst": worst})
    return {"name": name, "usage": st.usage, "per_batch": per_batch, "log": st.log,
            "all_ok": all(p["ok"] for p in per_batch), "L_final": st.L}
