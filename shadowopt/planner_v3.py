"""§0★★★ ladder, planner repair attempt #3 (PREREG addendum A7): selective precoverage under contract C1 for
single-term families. Unsatisfied never-asked catalogue terms are exchangeable given the history (registered generators
assign popularity independently of structure), so the policy chooses a fraction f of them (seeded uniform random
subset) and the timing; precovered terms are grouped JOINTLY with the current deficit (one grouping, one job).
"""
import math

import numpy as np

from .planner import StreamCounter
from .planner_v1 import PooledState, batch_feasible
from .planner_v2 import _need, _assign_groups, _settings

F_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def _job_cost(U, cost, terms, need_of, have_of):
    """Cost of one job acquiring `terms` (jointly grouped); shots per group = max remaining need."""
    if not terms:
        return 0.0, [], []
    groups = _assign_groups(U, terms)
    shots = [max(0, max(need_of(u) - have_of(u) for u in mem)) for _, mem in groups]
    return cost.target(shots), groups, shots


def run_v3(name, fam, seed, cost, n_rollouts=50, L0=0):
    """Strategies: SO3, Fcat, Ffix0.25, Ffix0.5, Ffix0.75."""
    U = fam.universe
    assert all(len(q.terms) == 1 for b in fam.batches for q in b)
    st = PooledState(U, StreamCounter(U, seed), cost, pooled=True)
    st.L = int(L0)
    Tb = len(fam.batches)
    nstd = _need(fam.batches[0][0])
    asked = np.zeros(U.T, bool)
    asks = np.zeros(U.T)
    dpb = []
    per_batch, decisions = [], []
    for tau in range(Tb):
        qs = fam.batches[tau]
        need = {int(q.terms[0]): _need(q) for q in qs}
        u0 = st.usage
        cnt = st.best_counts(np.arange(U.T))
        deficit = [u for u in sorted(need) if cnt[u] < need[u]]
        unsat = [int(v) for v in range(U.T) if cnt[v] < nstd and int(v) not in need and not asked[v]]
        rng = np.random.default_rng([seed, tau, 4242])
        order = rng.permutation(len(unsat))
        subset = lambda f: [unsat[i] for i in order[:int(round(f * len(unsat)))]]
        have_of = lambda u: int(cnt[u])
        need_of = lambda u: need.get(u, nstd)
        f_choice = None
        if name.startswith("Ffix") or name == "Fcat":
            f_fixed = 1.0 if name == "Fcat" else float(name[4:])
            f_choice = f_fixed if tau == 0 else 0.0
        elif name == "SO3" and deficit:
            R = Tb - tau - 1
            d = float(np.mean(dpb + [len(need)]))
            n_seen = int(asked.sum() + sum(1 for u in need if not asked[u]))
            best = None
            for f in F_GRID:
                pre = subset(f) if R > 0 else []
                cur, _, _ = _job_cost(U, cost, deficit + pre, need_of, have_of)
                covered = set(pre)
                fut = 0.0
                # rollouts: each future draw is a previously asked term with prob (past asks)/(past asks + pseudo) else a
                # uniformly random never-asked term (exchangeable); pseudo-count = number of distinct terms seen so far
                tot_asks = asks.sum() + len(need)
                p_new = n_seen / (tot_asks + n_seen) if tot_asks > 0 else 1.0
                pool_new = [v for v in unsat]
                for r in range(n_rollouts):
                    rr = np.random.default_rng([seed, tau, r, 77])
                    sat = set(covered)
                    for _ in range(R):
                        k_new = rr.binomial(int(round(d)), p_new)
                        new = set(rr.choice(pool_new, size=min(k_new, len(pool_new)), replace=False).tolist()) \
                            if pool_new and k_new > 0 else set()
                        new = [v for v in new if v not in sat]
                        if new:
                            c, _, _ = _job_cost(U, cost, new, lambda u: nstd, lambda u: 0)
                            fut += c
                            sat.update(new)
                fut /= n_rollouts
                if best is None or cur + fut < best[0]:
                    best = (cur + fut, f)
            f_choice = best[1]
            decisions.append({"tau": tau, "f": f_choice, "est_total": best[0]})
        pre = subset(f_choice) if f_choice else []
        terms = deficit + pre
        if terms:
            _, groups, shots = _job_cost(U, cost, terms, need_of, have_of)
            st.do_target(_settings(U, groups, shots), tau, name)
        for u in need:
            asks[u] += 1
            asked[u] = True
        dpb.append(len(need))
        ok, worst = batch_feasible(st, fam, tau)
        per_batch.append({"tau": tau, "U": st.usage - u0, "ok": ok, "worst": worst})
    return {"name": name, "usage": st.usage, "per_batch": per_batch, "decisions": decisions,
            "jobs": sum(1 for e in st.log if e.get("action") == "TARGET"),
            "all_ok": all(p["ok"] for p in per_batch)}
