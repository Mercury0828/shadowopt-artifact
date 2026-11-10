"""PREREG A15 (H-b'1 repair #1): bound-aware per-setting shot allocation for the engine's completion plan.

`certify.complete` (A13/A14) scales every group's shots by ONE factor alpha (s_g = ceil(alpha * base_g)). The certified
condition is a sum over terms, HW_q = sum_j |c_j| sqrt(2 ln(2/delta_j) / m_j) with the optimized delta split, and a
setting usually serves many terms of many queries, so a uniform factor over-buys cheap, widely shared settings' partners
and under-uses them. Here the shots are chosen per setting:

    minimize   sum_g s_g        s.t.  HW_q(m0 + C s) <= eps_q  for every uncertified query q,   s_g >= 1,

solved as a sequence of convex programs: with the split delta_j FIXED (from the exact optimal split at the current
point), HW_q is a sum of convex decreasing functions (m0 + C s)_j^(-1/2) of an affine map of s, so each subproblem is
convex (SLSQP); re-splitting can only lower HW (the split is optimal for the counts), so a feasible point stays feasible.
The continuous solution is scaled by the smallest beta in [0.5, ...] whose rounded-up plan ceil(beta s) passes the EXACT
compiled checker (bisection, 30 steps). Counts and coefficients only (B2); outcomes are never read.

The result is never worse than `certify.complete` with the same order keys (`engine_orders`): for every grouping order
the allocated plan and the uniform plan are both priced and the cheapest verified plan is returned (first strictly smallest, uniform plan first).
"""
import hashlib

import numpy as np
from scipy.optimize import minimize

from .bounds import term_thresholds
from .certify import complete, fast_group_terms, _compat, _batch_hw_split, _batch_halfwidths

ROUNDS = 4  # fixed-split convex solves per order (registered in A15)
SQUEEZE_STEPS = 30


def _bad_csr(csr, ok):
    bad = np.nonzero(~ok)[0]
    bptr, bterm, bcoef = [0], [], []
    for i in bad:
        a, b = csr.qptr[i], csr.qptr[i + 1]
        bterm += list(csr.qterm[a:b])
        bcoef += list(csr.qcoef[a:b])
        bptr.append(len(bterm))
    return (bad, np.array(bptr, np.int64), np.array(bterm, np.int64), np.array(bcoef, np.float64), csr.qdelta[bad],
            csr.qeps[bad])


def allocate(m_rows, Cr, bptr, tl, bcoef, bdelta, beps, k0, t_eff=1.0, rounds=ROUNDS):
    """Per-setting shots (int64, aligned with the columns of Cr) such that every query of the CSR (bptr, tl over the
    rows of Cr) has exact HW <= eps; None if no verified plan was found. k0: a feasible starting point (shots)."""
    Q = len(bptr) - 1
    G = Cr.shape[1]
    m0 = np.asarray(m_rows, np.float64)
    Crf = Cr.astype(np.float64)
    Cj = Crf[tl]
    m0j = m0[tl]
    qid = np.repeat(np.arange(Q), np.diff(bptr))
    k = np.maximum(np.asarray(k0, np.float64), 1.0)
    sc = max(1.0, float(k.mean()))
    for _ in range(rounds):
        mc = m0 + Crf @ k
        d = np.empty(len(tl))
        _batch_hw_split(bptr, tl, bcoef, bdelta, mc, d)
        aj = np.abs(bcoef) * np.sqrt(2.0 * np.log(2.0 / d))

        def cons(x):
            mm = m0j + Cj @ (x * sc)
            return beps - np.bincount(qid, aj / np.sqrt(np.maximum(mm, 1e-300)), Q)

        def cjac(x):
            mm = m0j + Cj @ (x * sc)
            g = 0.5 * aj * np.maximum(mm, 1e-300) ** -1.5 * sc
            J = np.zeros((Q, G))
            np.add.at(J, qid, g[:, None] * Cj)
            return J

        res = minimize(lambda x: float(x.sum()), k / sc, jac=lambda x: np.ones_like(x), method="SLSQP",
                       bounds=[(1.0 / sc, None)] * G, constraints=[{"type": "ineq", "fun": cons, "jac": cjac}],
                       options={"maxiter": 500, "ftol": 1e-10})
        x = np.maximum(res.x, 1.0 / sc)
        if np.all(np.isfinite(x)) and np.all(cons(x) >= -1e-9 * beps.max()):
            k = x * sc  # accept only (numerically) feasible iterates; otherwise keep the previous point

    def feas(beta):
        s = np.ceil(beta * k)
        h = _batch_halfwidths(bptr, tl, bcoef, bdelta, m0 + Crf @ s)
        return bool(np.all(h <= beps))

    hi = 1.0
    for _ in range(40):
        if feas(hi):
            break
        hi *= 1.05
    else:
        return None
    lo = 0.5
    if feas(lo):
        hi = lo
    else:
        for _ in range(SQUEEZE_STEPS):
            mid = 0.5 * (lo + hi)
            if feas(mid):
                hi = mid
            else:
                lo = mid
    s = np.ceil(hi * k).astype(np.int64)
    return s


def engine_orders(universe, n_random_orders=4):
    """The A13 order set with CONTENT-based keys (CODE-alloc R1 FATAL: keys must not depend on the catalogue size or on
    term indices, which contract C0 does not disclose): weight descending, need descending, weight ascending (ties by the
    term's (support, bases)), and n_random_orders seeded pseudo-random orders = BLAKE2b of (seed, r, support, bases).
    Each key is a function of (term index, need) that reads only that term's own content."""
    w = universe.weights()
    memo = {}

    def content(u):
        c = memo.get(u)
        if c is None:
            qq, bb = universe.term(u)
            c = memo[u] = (tuple(int(x) for x in qq), tuple(int(x) for x in bb))
        return c

    def rnd(r):
        def key(u, need):
            h = hashlib.blake2b(repr((20261108, r, content(u))).encode(), digest_size=8).digest()
            return (int.from_bytes(h, "big"), content(u))
        return key

    orders = [lambda u, need: (-int(w[u]), content(u)), lambda u, need: (-int(need), content(u)),
              lambda u, need: (int(w[u]), content(u))]
    return orders + [rnd(r) for r in range(n_random_orders)]


def complete_alloc(universe, csr, m, hw, ok, cost, n_random_orders=4):
    """The engine's completion with per-setting allocation (A15). Returns (settings [(setting, shots)], projected counts,
    info). Never costlier than certify.complete(..., cost=cost) on the same input."""
    keys = engine_orders(universe, n_random_orders)
    base_plan, base_mp = complete(universe, csr, m, hw, ok, cost=cost, orders=keys)
    if not base_plan:
        return base_plan, base_mp, {"chosen": "none"}
    best = (cost.target([k for _, k in base_plan]), base_plan, base_mp, "uniform")
    bad, bptr, bterm, bcoef, bdelta, beps = _bad_csr(csr, ok)
    slot_need = np.zeros(len(csr.terms), np.int64)
    for i in bad:
        a, b = csr.qptr[i], csr.qptr[i + 1]
        th = term_thresholds(csr.qcoef[a:b], csr.qeps[i], csr.qdelta[i])
        for j, t in zip(range(a, b), th):
            slot_need[csr.qterm[j]] = max(slot_need[csr.qterm[j]], int(t))
    m64 = np.asarray(m, np.int64)
    deficit_slots = [s for s in np.nonzero(slot_need > 0)[0] if m64[s] < slot_need[s]]
    deficit_terms = [int(csr.terms[s]) for s in deficit_slots]
    slot_of = {int(u): i for i, u in enumerate(csr.terms)}
    need_of = {int(csr.terms[s]): int(slot_need[s] - m64[s]) for s in deficit_slots}
    orders = [(lambda u, f=f: f(u, need_of[u])) for f in keys]  # the same content-based keys as the uniform plan
    sub = universe.subset(csr.terms)
    rows = np.unique(bterm)
    tl = np.searchsorted(rows, bterm).astype(np.int64)
    t_eff = cost.t_shot
    for oi, key in enumerate(orders):
        groups = fast_group_terms(universe, deficit_terms, key)
        if not groups:
            continue
        base = np.array([max(1, max(int(slot_need[slot_of[u]] - m64[slot_of[u]]) for u in mem)) for _, mem in groups],
                        np.float64)  # thresholds => feasible starting point (alpha = 1)
        S = np.array([s for s, _ in groups], np.int64)
        C = _compat(S, sub.ptr, sub.q, sub.b)
        s = allocate(m64[rows], C[rows], bptr, tl, bcoef, bdelta, beps, base, t_eff)
        if s is None:
            continue
        mp = m64 + C @ s
        h = _batch_halfwidths(bptr, bterm, bcoef, bdelta, mp.astype(np.float64))  # exact verification
        if not np.all(h <= beps):
            continue
        plan = [(S[g].copy(), int(s[g])) for g in range(len(groups)) if s[g] > 0]
        u = cost.target([k for _, k in plan])
        if u < best[0]:
            best = (u, plan, mp, f"alloc:order{oi}")
    return best[1], best[2], {"chosen": best[3], "U": best[0], "U_uniform": cost.target([k for _, k in base_plan])}
