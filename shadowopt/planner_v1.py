"""§0★★★ ladder, planner repair attempt #1 (PREREG addendum A5) and the strengthened baselines it is compared with.

Differences from the Phase-0 planner (shadowopt/planner.py, kept unchanged for the frozen Phase-0 numbers):
- query-level, count-aware stopping for every strategy that acquires incrementally ("qs"): a batch is done when every
  query's stated half-width (optimized delta split, bounds.query_halfwidth) is <= eps at the counts it is answered from;
  stopping depends on counts only (B2);
- SO1 answers every term from POOLED counts (random + targeted + derandomized samples; route R3 — every basis in every
  source is chosen without looking at outcomes; C-BOUND must confirm before any frozen claim);
- SO1 chooses per round among APPEND(uniform, exact minimal size), TARGET by grouping (scaled shots), TARGET by HKP
  derandomization with random fill of idle qubits, and APPEND-then-derandomize mixtures, minimizing usage under the
  shared cost model (which charges every distinct setting t_set, the HW1-fitted per-binding cost);
- baselines get the same stopping rule (F2qs, F3qs, F4qs) and the same fill (F4qsF), so the comparison isolates the
  plan choice and pooling.
"""
import math

import numpy as np

from .bounds import query_halfwidth
from .derand import hkp_derandomize_until_feasible, consolidate
from .planner import (StreamCounter, EngineState, batch_requirements, compat_setting, group_terms, INF,
                      setting_counts)
from .cost import CostModel


class PooledState(EngineState):
    """EngineState whose answer counts are pooled over sources (SO1) or the max over sources (baselines; each baseline
    uses one source, so max = its own counts)."""

    def __init__(self, *a, pooled=False, **k):
        super().__init__(*a, **k)
        self.pooled = pooled

    def best_counts(self, u_idx):
        u_idx = np.asarray(u_idx, np.int64)
        if self.pooled:
            return self.mR(u_idx) + self.mT[u_idx] + self.mD[u_idx]
        return np.maximum(np.maximum(self.mR(u_idx), self.mT[u_idx]), self.mD[u_idx])


def batch_feasible(st, fam, tau, extra=None):
    """All queries of batch tau meet eps with the state's counts (+ `extra`: dict u -> additional count, added to the
    pooled total, or max-ed for non-pooled states). Returns (ok, worst ratio)."""
    worst = 0.0
    for qy in fam.batches[tau]:
        m = st.best_counts(qy.terms).astype(np.float64)
        if extra:
            add = np.array([extra.get(int(u), 0) for u in qy.terms], np.float64)
            m = m + add if st.pooled else np.maximum(m, add)
        hw = query_halfwidth(qy.coeffs, m, qy.delta)
        worst = max(worst, hw / qy.eps)
    return worst <= 1.0, worst  # exact predicate HW <= eps


# ----------------------------------------------------------------------------------------------------------------------
# Candidate pieces (each returns a plan dict with exact post-action counts, computed before any QPU second)
# ----------------------------------------------------------------------------------------------------------------------
def append_qs(st, fam, tau, cap_rows=None):
    """Smallest N_add of uniform snapshots (from the seeded stream) making the batch feasible; INF if beyond the cap."""
    cap = cap_rows or st.stream.max_chunks * st.stream.chunk
    us = fam.batch_terms(tau)
    base_R = st.mR(us)

    def ok(N):
        cumR = st.stream.count(us, st.L + N)
        # pooled: pass the increment (added to the pooled total); non-pooled: pass the CUMULATIVE random count (max-ed
        # with the other sources)
        add = cumR - base_R if st.pooled else cumR
        return batch_feasible(st, fam, tau, dict(zip(us.tolist(), add.tolist())))[0]

    if ok(0):
        return 0
    lo, hi = 0, 1024  # lo is always a known-infeasible size (ok(0) is False here)
    while not ok(hi):
        lo = hi
        hi *= 2
        if st.L + hi > cap:
            return INF
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def group_qs(st, fam, tau, deficit, req):
    """Greedy QWC grouping of the deficit terms (Z on unassigned qubits); shots_g = ceil(alpha * max need in group of
    (threshold - current targeted count)), alpha in (0, 1] minimal by bisection for batch feasibility."""
    groups = group_terms(st.universe, list(deficit))
    U = st.universe
    comp = [compat_setting(U, s) for s, _ in groups]
    cur = (lambda u: int(st.best_counts([u])[0])) if st.pooled else (lambda u: int(st.mT[u]))
    base = [max(0, max(req[u] - cur(u) for u in mem)) for _, mem in groups]

    def counts(alpha):
        add = np.zeros(U.T, np.int64)
        for c, b in zip(comp, base):
            add += c * int(math.ceil(alpha * b))
        return add

    def ok(alpha):
        add = counts(alpha)
        us = fam.batch_terms(tau)
        if st.pooled:
            return batch_feasible(st, fam, tau, {int(u): int(add[u]) for u in us})[0]
        # non-pooled: targeted source total = mT + add
        return batch_feasible(st, fam, tau, {int(u): int(st.mT[u] + add[u]) for u in us})[0]

    lo, hi = 0.0, 1.0
    if not ok(hi):
        hi = 1.0  # thresholds imply feasibility; keep alpha = 1 (should not happen)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return [(s, int(math.ceil(hi * b))) for (s, _), b in zip(groups, base)]


def derand_qs(st, fam, tau, deficit, req, fill_seed=None, extraR=None):
    """HKP derandomization over the deficit terms (weights = thresholds), stopped at batch feasibility; idle qubits
    filled with seeded uniform bases if fill_seed is given. `extraR`: dict u -> counts added by an APPEND planned in the
    same round (mixtures). Returns list of (setting, shots)."""
    U = st.universe
    dd = np.array(deficit, np.int64)
    pos = {u: i for i, u in enumerate(deficit)}
    qs_ = [qy for qy in fam.batches[tau] if any(u in pos for u in qy.terms.tolist())]
    extraR = extraR or {}
    have = st.best_counts(dd).astype(np.float64)
    if st.pooled:
        have = have + np.array([extraR.get(u, 0) for u in deficit])
    else:
        have = st.mD[dd].astype(np.float64)

    def feas(mvec):
        for qy in qs_:
            mm = st.best_counts(qy.terms).astype(np.float64)
            for k_, u in enumerate(qy.terms.tolist()):
                if st.pooled:
                    mm[k_] += extraR.get(u, 0)
                    if u in pos:
                        mm[k_] = mvec[pos[u]] + 0.0  # mvec already includes the pooled base
                elif u in pos:
                    mm[k_] = max(mm[k_], mvec[pos[u]])
            if query_halfwidth(qy.coeffs, mm, qy.delta) > qy.eps:
                return False
        return True

    rng = np.random.default_rng(fill_seed) if fill_seed is not None else None
    seq, _ = hkp_derandomize_until_feasible(U, dd, np.array([max(req[u], 1) for u in deficit]), have, feas,
                                            fill_rng=rng)
    return consolidate(seq)


def _derand_counts(U, ss):
    add = np.zeros(U.T, np.int64)
    for s, k in ss:
        add += compat_setting(U, s) * k
    return add


# ----------------------------------------------------------------------------------------------------------------------
def run_v1(name, fam, seed, cost, stream=None, max_rounds=20, mix_weights=(1, 2, 3), L0=0, pool_all=False):
    """Strategies: F2qs, F3qs, F4qs, F4qsF (fill), SO1 (pooled, per-round choice). Returns dict like run_strategy."""
    stream = stream or StreamCounter(fam.universe, seed)
    st = PooledState(fam.universe, stream, cost, pooled=(name == "SO1") or pool_all)
    st.L = int(L0)
    U = fam.universe
    w = U.weights()
    per_batch = []
    for tau in range(len(fam.batches)):
        req = batch_requirements(fam, tau)
        u0 = st.usage
        for rnd in range(max_rounds):
            ok, _ = batch_feasible(st, fam, tau)
            if ok:
                break
            us = np.array(sorted(req), np.int64)
            have = st.best_counts(us)
            deficit = [int(u) for u, h in zip(us, have) if h < req[int(u)]]
            if not deficit:  # query-level infeasible although every term meets its threshold: cannot happen
                deficit = [int(u) for u in us]
            fseed = [seed, tau, rnd, 7]
            if name == "F2qs":
                n_add = append_qs(st, fam, tau)
                if n_add >= INF:
                    return {"name": name, "infeasible": True, "per_batch": per_batch, "log": st.log}
                st.do_append(n_add, tau, name)
            elif name == "F3qs":
                st.do_target(group_qs(st, fam, tau, deficit, req), tau, name)
            elif name in ("F4qs", "F4qsF"):
                ss = derand_qs(st, fam, tau, deficit, req, fill_seed=fseed if name == "F4qsF" else None)
                st.do_derand(ss, tau, name)
            elif name == "SO1":
                cands = []
                n_add = append_qs(st, fam, tau)
                if n_add < INF:
                    cands.append({"kind": "APPEND", "n_add": n_add, "U": cost.append(n_add)})
                gp = group_qs(st, fam, tau, deficit, req)
                cands.append({"kind": "GROUP", "target": gp, "U": cost.target([k for _, k in gp])})
                ss = derand_qs(st, fam, tau, deficit, req)
                cands.append({"kind": "DERAND", "derand": ss, "U": cost.job(len(ss), sum(k for _, k in ss))})
                ssf = derand_qs(st, fam, tau, deficit, req, fill_seed=fseed)
                cands.append({"kind": "DERANDF", "derand": ssf, "U": cost.job(len(ssf), sum(k for _, k in ssf))})
                # mixtures: append until every deficit term of weight <= wcut reaches its threshold, then derandomize
                for wcut in mix_weights:
                    A = [u for u in deficit if w[u] <= wcut]
                    if not A or len(A) == len(deficit):
                        continue
                    pos_ = max(st.stream.position(u, req[u] - int(st.best_counts([u])[0]) + int(st.mR([u])[0]))
                               for u in A)
                    if pos_ >= INF:
                        continue
                    na = max(0, pos_ - st.L)
                    if na == 0:
                        continue
                    dR = st.stream.count(np.array(deficit), st.L + na) - st.mR(np.array(deficit))
                    ss2 = derand_qs(st, fam, tau, deficit, req, extraR=dict(zip(deficit, dR.tolist())))
                    # one job: na one-shot random bindings + the derandomized settings (several PUBs, one job)
                    Umix = cost.job(na + len(ss2), na + sum(k for _, k in ss2))
                    cands.append({"kind": f"MIX{wcut}", "n_add": na, "derand": ss2, "U": Umix})
                best = min(cands, key=lambda c: c["U"])
                st.log.append({"tau": tau, "action": "CANDIDATES", "chosen": best["kind"],
                               "U": {c["kind"]: c["U"] for c in cands}})
                if best["kind"].startswith("MIX"):  # one combined job, charged once
                    na, ss2 = best["n_add"], best["derand"]
                    st.L += int(na)
                    st.mD += setting_counts(U, ss2)
                    st.usage += best["U"]
                    st.log.append({"tau": tau, "action": "MIX", "n_append": int(na),
                                   "n_derand": int(sum(k for _, k in ss2)), "settings": int(na + len(ss2)),
                                   "U": best["U"], "tag": f"SO1:{best['kind']}"})
                elif best["kind"] == "APPEND":
                    st.do_append(best["n_add"], tau, "SO1:APPEND")
                elif best["kind"] == "GROUP":
                    st.do_target(best["target"], tau, "SO1:GROUP")
                else:
                    st.do_derand(best["derand"], tau, "SO1:DERAND")
            else:
                raise KeyError(name)
        ok, worst = batch_feasible(st, fam, tau)
        per_batch.append({"tau": tau, "U": st.usage - u0, "ok": ok, "worst": worst})
    return {"name": name, "usage": st.usage, "per_batch": per_batch, "log": st.log,
            "all_ok": all(p["ok"] for p in per_batch), "L_final": st.L}
