"""§0★★★ ladder, planner repair attempt #2 (PREREG addendum A6): forecast-guided speculative coverage for single-term
families under information contracts C0 (past/current batches only) and C1 (+ catalogue of possible terms and the
horizon; no popularity, no future sequence). All decisions use counts, past batches, the catalogue and seeded randomness
only (B2). Pooled counts (route R3); every strategy here acquires targeted settings only, so pooling = its own counts.
"""
import math

import numpy as np

from .planner import StreamCounter, setting_counts
from .planner_v1 import PooledState, batch_feasible


def _need(q):
    """Single-term threshold incl. the coefficient: HW = |a| sqrt(2 ln(2/delta)/m) <= eps  <=>  m >= 2 ln(2/delta) a^2 / eps^2
."""
    a = float(abs(q.coeffs[0]))
    return int(math.ceil(2 * math.log(2 / q.delta) * a * a / q.eps ** 2 - 1e-9))


def _assign_groups(U, terms):
    """Greedy first-fit QWC grouping returning (assignment dict qubit->basis, members) — like planner.group_terms but
    keeps the assignment explicit (unassigned qubits are not Z)."""
    w = U.weights()
    groups = []
    for u in sorted(terms, key=lambda u: (-w[u], u)):
        qq, bb = U.term(u)
        for asg, mem in groups:
            if all(asg.get(int(q), int(b)) == int(b) for q, b in zip(qq, bb)):
                for q, b in zip(qq, bb):
                    asg[int(q)] = int(b)
                mem.append(u)
                break
        else:
            groups.append(({int(q): int(b) for q, b in zip(qq, bb)}, [u]))
    return groups


def _add_riders(U, groups, riders):
    """Pack rider terms into existing groups' idle qubits (first fit). Returns number packed."""
    k = 0
    for v in riders:
        qq, bb = U.term(v)
        for asg, mem in groups:
            if all(asg.get(int(q), int(b)) == int(b) for q, b in zip(qq, bb)):
                for q, b in zip(qq, bb):
                    asg[int(q)] = int(b)
                k += 1
                break
    return k


def _settings(U, groups, shots, rng=None):
    out = []
    for (asg, _), s in zip(groups, shots):
        x = np.full(U.n, 2, np.int64) if rng is None else rng.integers(0, 3, U.n).astype(np.int64)
        for q, b in asg.items():
            x[q] = b
        out.append((x, int(s)))
    return out


def run_v2(name, fam, seed, cost, n_rollouts=20, p_grid=(1.01, 0.5, 0.2, 0.1, 0.05, 0.02, 0.0)):
    """Strategies: F3R (C0 baseline), SO2C0, Fcat, F3cat (C1 baselines), SO2C1. Single-term queries only."""
    U = fam.universe
    assert all(len(q.terms) == 1 for b in fam.batches for q in b)
    st = PooledState(U, StreamCounter(U, seed), cost, pooled=True)
    rng = np.random.default_rng([seed, 2026, 11, 7])
    catalogue = np.arange(U.T)
    n_req = {}
    asks = np.zeros(U.T)
    seen_batches = 0
    distinct_per_batch = []
    per_batch, decisions = [], []
    Tb = len(fam.batches)
    for tau in range(Tb):
        qs = fam.batches[tau]
        need = {int(q.terms[0]): _need(q) for q in qs}
        nstd = max(need.values())  # all single-term queries of these families share eps/delta
        u0 = st.usage
        have = st.best_counts(np.array(sorted(need)))
        deficit = [u for u, h in zip(sorted(need), have) if h < need[u]]
        if name == "Fcat" and tau == 0:  # precover the whole catalogue at batch 1 (union grouping)
            groups = _assign_groups(U, list(catalogue))
            st.do_target(_settings(U, groups, [nstd] * len(groups)), tau, "Fcat")
            deficit = [u for u in deficit if st.best_counts([u])[0] < need[u]]
        if deficit:
            groups = _assign_groups(U, deficit)
            shots = [max(need[u] - int(st.best_counts([u])[0]) for u in mem) for _, mem in groups]
            choice = None
            if name in ("F3cat", "SO2C1"):
                cnt = st.best_counts(catalogue)
                unsat = [int(v) for v in catalogue if cnt[v] < nstd and int(v) not in need]
                pop = (asks + 1.0) / (seen_batches + U.T)
                riders = sorted(unsat, key=lambda v: (-pop[v], v))
                if name == "SO2C1":
                    R = Tb - tau - 1
                    d = np.mean(distinct_per_batch + [len(need)])
                    pi = (asks + 1.0) / (asks.sum() + len(need) + U.T)  # per-draw popularity forecast
                    p_ask = 1.0 - (1.0 - pi) ** (R * d) if R > 0 else np.zeros(U.T)
                    best = None
                    for pstar in p_grid:
                        pre = [v for v in unsat if p_ask[v] >= pstar] if R > 0 else []
                        g2 = [(dict(a), list(m)) for a, m in groups]
                        _add_riders(U, g2, sorted(set(riders) - set(pre), key=lambda v: (-pop[v], v)))
                        g2_pre = _assign_groups(U, pre) if pre else []
                        cur = cost.target(shots + [nstd] * len(g2_pre))
                        covered = set(pre)
                        for a, m in g2:  # riders get the group's shots (approximation: full coverage if shots >= nstd)
                            pass
                        # expected future cost by rollouts of the frequency forecast
                        fut = 0.0
                        gsize = max(1.0, len(deficit) / max(1, len(groups)))
                        for r in range(n_rollouts):
                            rr = np.random.default_rng([seed, tau, r, 31])
                            sat = set(covered)
                            for _ in range(R):
                                draws = rr.choice(U.T, size=int(round(d)), replace=True, p=pi / pi.sum())
                                new = [v for v in set(draws.tolist()) if v not in sat and cnt[v] < nstd
                                       and v not in need]
                                if new:
                                    ng = math.ceil(len(new) / gsize)
                                    fut += cost.t_sub + ng * (cost.t_set + nstd * cost.t_shot)
                                    sat.update(new)
                        fut /= n_rollouts
                        tot = cur + fut
                        if best is None or tot < best[0]:
                            best = (tot, pstar, g2, g2_pre)
                    _, pstar, groups, g2_pre = best
                    choice = {"tau": tau, "pstar": pstar, "n_precover": sum(len(m) for _, m in g2_pre)}
                    sets = _settings(U, groups, shots) + _settings(U, g2_pre, [nstd] * len(g2_pre))
                else:
                    _add_riders(U, groups, riders)
                    sets = _settings(U, groups, shots)
            elif name == "F3R":
                sets = _settings(U, groups, shots, rng=rng)
            elif name == "SO2C0":  # per batch: cheaper of GROUP + random completion and DERAND-qs (A6)
                from .planner_v1 import derand_qs
                g_sets = _settings(U, groups, shots, rng=rng)
                d_sets = derand_qs(st, fam, tau, deficit, need)
                Ug = cost.target([k for _, k in g_sets])
                Ud = cost.job(len(d_sets), sum(k for _, k in d_sets))
                sets = g_sets if Ug <= Ud else d_sets
                choice = {"tau": tau, "chosen": "GROUP+completion" if Ug <= Ud else "DERAND", "Ug": Ug, "Ud": Ud}
            elif name == "Fcat":
                sets = _settings(U, groups, shots)
            else:
                raise KeyError(name)
            st.do_target(sets, tau, name)
            if choice:
                decisions.append(choice)
        for u in need:
            asks[u] += 1
        seen_batches += 1
        distinct_per_batch.append(len(need))
        ok, worst = batch_feasible(st, fam, tau)
        per_batch.append({"tau": tau, "U": st.usage - u0, "ok": ok, "worst": worst})
    return {"name": name, "usage": st.usage, "per_batch": per_batch, "decisions": decisions,
            "jobs": sum(1 for e in st.log if e.get("action") == "TARGET"),
            "all_ok": all(p["ok"] for p in per_batch)}
