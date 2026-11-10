"""Certified bracket [L, U] for the unrestricted offline hindsight optimum OPT of a batch sequence (replaces the restricted
hindsight.py MIP as the H-b reference; C-PLAN R1, audited SOUND).

- Any feasible hindsight schedule can be moved to batch 1 without raising cost (counts only grow, setting charges do not
  increase, ceil is subadditive), so OPT is a static design problem over the sequence's final requirements.
- U: the cheapest of several feasible static designs acquired at batch 1 (QWC grouping of the requested-term union with
  query-level shot scaling; HKP derandomization with query-level stopping), each verified by the independent
  feasibility check for every query at its batch.
- L (single-term queries only): t_sub * ceil(sum n / E) + |C| t_set + t_shot * sum_{u in C} n_u for a set C of pairwise
  INCOMPATIBLE requested terms (share a qubit with different bases): a row hits at most one of them and each needs its
  own threshold n_u = ceil(2 ln(2/delta)/eps^2) hits and at least one setting. Any such C gives a valid bound; C is found
  greedily (largest threshold first). Multi-term queries: L is not computed here (reported as None).
"""
import math

import numpy as np

from .bounds import query_halfwidth
from .planner import group_terms, setting_counts, compat_setting
from .derand import hkp_derandomize_until_feasible, consolidate


def final_requirements(fam):
    """Per term, the largest threshold over the sequence (single-term queries: ceil(2 ln(2/delta)/eps^2))."""
    req = {}
    for b in fam.batches:
        for q in b:
            if len(q.terms) == 1:
                a = float(abs(q.coeffs[0]))  # coefficient-aware
                n = int(math.ceil(2 * math.log(2 / q.delta) * a * a / q.eps ** 2 - 1e-9))
                u = int(q.terms[0])
                req[u] = max(req.get(u, 0), n)
    return req


def all_feasible(fam, m):
    worst = 0.0
    for b in fam.batches:
        for q in b:
            worst = max(worst, query_halfwidth(q.coeffs, m[q.terms].astype(float), q.delta) / q.eps)
    return worst <= 1.0, worst  # exact predicate HW <= eps (C-BOUND R1)


def union_grouping(fam, cost):
    U = fam.universe
    union = sorted({int(u) for b in fam.batches for q in b for u in q.terms})
    single = all(len(q.terms) == 1 for b in fam.batches for q in b)
    req = final_requirements(fam) if single else None
    groups = group_terms(U, union)
    if single:
        ss = [(s, max(req[u] for u in mem)) for s, mem in groups]
    else:  # scale a common shot count until every query is met
        lo, hi = 0, 1
        while not all_feasible(fam, setting_counts(U, [(s, hi) for s, _ in groups]))[0]:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if all_feasible(fam, setting_counts(U, [(s, mid) for s, _ in groups]))[0]:
                hi = mid
            else:
                lo = mid
        ss = [(s, hi) for s, _ in groups]
    m = setting_counts(U, ss)
    ok, worst = all_feasible(fam, m)
    return {"design": "union_grouping", "settings": len(ss), "shots": int(sum(k for _, k in ss)),
            "U": cost.target([k for _, k in ss]), "feasible": ok, "worst": worst}


def union_derand(fam, cost):
    """HKP derandomization over the union with query-level stopping on ALL queries of the sequence."""
    U = fam.universe
    union = np.array(sorted({int(u) for b in fam.batches for q in b for u in q.terms}), np.int64)
    pos = {int(u): i for i, u in enumerate(union)}
    need = np.zeros(len(union), np.int64)
    from .bounds import term_thresholds
    for b in fam.batches:
        for q in b:
            th = term_thresholds(q.coeffs, q.eps, q.delta)
            for u, t in zip(q.terms.tolist(), th.tolist()):
                need[pos[u]] = max(need[pos[u]], int(t))
    qs = [q for b in fam.batches for q in b]

    def feas(mvec):
        for q in qs:
            if query_halfwidth(q.coeffs, np.array([mvec[pos[int(u)]] for u in q.terms]), q.delta) > q.eps:
                return False
        return True

    seq, _ = hkp_derandomize_until_feasible(U, union, np.maximum(need, 1), np.zeros(len(union)), feas)
    ss = consolidate(seq)
    m = setting_counts(U, ss)
    ok, worst = all_feasible(fam, m)
    return {"design": "union_derand_qs", "settings": len(ss), "shots": int(sum(k for _, k in ss)),
            "U": cost.job(len(ss), sum(k for _, k in ss)), "feasible": ok, "worst": worst}


def incompatible_clique_lower_bound(fam, cost):
    if not all(len(q.terms) == 1 for b in fam.batches for q in b):
        return None
    U = fam.universe
    req = final_requirements(fam)
    terms = sorted(req, key=lambda u: (-req[u], u))
    C = []
    for u in terms:
        qu, bu = U.term(u)
        du = dict(zip(qu.tolist(), bu.tolist()))
        ok = True
        for v in C:
            qv, bv = U.term(v)
            if not any(du.get(int(q), int(b)) != int(b) for q, b in zip(qv, bv)):
                ok = False  # compatible with a chosen term (no conflicting qubit): cannot add
                break
        if ok:
            C.append(u)
    shots = sum(req[u] for u in C)
    L = cost.t_sub * max(1, math.ceil(shots / cost.exec_per_subjob)) + len(C) * cost.t_set + shots * cost.t_shot
    return {"clique": len(C), "shots": int(shots), "L": L}


def opt_bracket(fam, cost, with_derand=True):
    ups = [union_grouping(fam, cost)]
    if with_derand:
        ups.append(union_derand(fam, cost))
    feas = [u for u in ups if u["feasible"]]
    best = min(feas, key=lambda u: u["U"]) if feas else None
    lb = incompatible_clique_lower_bound(fam, cost)
    return {"upper": best, "candidates": ups, "lower": lb}
