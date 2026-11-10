"""Faithful HKP derandomization (Huang, Kueng, Preskill, PRL 127, 030503 (2021); reference code G2:
github.com/hsinyuan-huang/predicting-quantum-properties @ dce17912, data_acquisition_shadow.py,
`derandomized_classical_shadow`). F4 as published (no ADAPTATION; cf. planner.derandomize_until = F4').

The reference chooses ONE measurement (a full basis assignment) at a time, qubit by qubit, minimizing
  cost = sum over unfinished observables i of exp(-V_i / w_i - shift),
  V_i = eta/2 * m_i - log(1 - nu / 3^r_i)   (r_i = support qubits still unassigned, observable still matchable)
  V_i = eta/2 * m_i                          (observable already mismatched in this round)
with eta = 0.9, nu = 1 - exp(-eta/2), m_i = measurements of i so far, w_i its weight, an observable finished once
m_i >= floor(w_i * M). Ties go to the first of X, Y, Z. After the round, every fully matched observable gets m_i += 1.

This port computes, for qubit j, only the contributions of observables whose support contains j (the other terms are
identical for the three candidates, and `shift` multiplies every term by the same factor), so it makes the same choices
as the reference up to floating-point near-ties; tests/test_derand.py checks sequence identity against the reference
code on the registered workloads. Non-uniform targets use the reference's own mechanism: M = max need, w_i = need_i / M
(plus 1e-9 so that floor(w_i M) = need_i exactly). `have` initializes m_i (reuse of earlier measurements; the
reference starts at 0).
"""
import math

import numpy as np
from numba import njit

ETA = 0.9


@njit(cache=True)
def _derand(n, ptr, q, b, target, have, weight, eta, max_meas):
    T = ptr.shape[0] - 1
    nu = 1.0 - math.exp(-eta / 2.0)
    # observables by qubit: for each qubit, list of (obs, pauli)
    cnt = np.zeros(n, np.int64)
    for t in range(T):
        for j in range(ptr[t], ptr[t + 1]):
            cnt[q[j]] += 1
    qptr = np.zeros(n + 1, np.int64)
    for i in range(n):
        qptr[i + 1] = qptr[i] + cnt[i]
    qobs = np.empty(qptr[n], np.int64)
    qpau = np.empty(qptr[n], np.int64)
    fill = qptr[:-1].copy()
    for t in range(T):
        for j in range(ptr[t], ptr[t + 1]):
            qobs[fill[q[j]]] = t
            qpau[fill[q[j]]] = b[j]
            fill[q[j]] += 1
    m = have.astype(np.float64).copy()
    kk = np.empty(T, np.int64)
    for t in range(T):
        kk[t] = ptr[t + 1] - ptr[t]
    # g[t, r] = exp(log(1 - nu/3^r) / w_t) for r = 0..k_t  (multiplicative factor of a still-matchable observable)
    kmax = 0
    for t in range(T):
        if kk[t] > kmax:
            kmax = kk[t]
    g = np.empty((T, kmax + 1))
    for t in range(T):
        for r in range(kmax + 1):
            g[t, r] = math.exp(math.log(1.0 - nu / 3.0 ** r) / weight[t])
    out = np.empty((max_meas, n), np.uint8)
    free = np.zeros((max_meas, n), np.bool_)  # qubit touched by no still-matchable unfinished observable (tie -> X)
    M = 0
    remaining = np.empty(T, np.int64)  # unassigned support qubits; -1 = mismatched in this round
    base = np.empty(T)
    while M < max_meas:
        done = True
        for t in range(T):
            if m[t] < target[t]:
                done = False
                break
        if done:
            break
        # per-round base factors (shift-free; only ratios matter within a qubit's choice)
        mn = 1e300
        for t in range(T):
            if m[t] < target[t]:
                v = eta / 2.0 * m[t] / weight[t]
                if v < mn:
                    mn = v
        for t in range(T):
            remaining[t] = kk[t]
            base[t] = math.exp(-(eta / 2.0 * m[t] / weight[t] - mn)) if m[t] < target[t] else 0.0
        for i in range(n):
            best = 0
            bestc = 0.0
            isfree = True
            for e in range(qptr[i], qptr[i + 1]):
                if base[qobs[e]] != 0.0 and remaining[qobs[e]] >= 0:
                    isfree = False
                    break
            free[M, i] = isfree
            for p in range(3):
                c = 0.0
                for e in range(qptr[i], qptr[i + 1]):
                    t = qobs[e]
                    if base[t] == 0.0:
                        continue
                    r = remaining[t]
                    if r < 0:
                        c += base[t]
                    elif qpau[e] == p:
                        c += base[t] * g[t, r - 1]
                    else:
                        c += base[t]
                if p == 0 or c < bestc:
                    best = p
                    bestc = c
            out[M, i] = best
            for e in range(qptr[i], qptr[i + 1]):
                t = qobs[e]
                if remaining[t] < 0:
                    continue
                if qpau[e] == best:
                    remaining[t] -= 1
                else:
                    remaining[t] = -1
        for t in range(T):
            if remaining[t] == 0:
                m[t] += 1.0
        M += 1
    return out[:M], m, free[:M]


def hkp_derandomize(universe, terms, need, have=None, eta=ETA, max_meas=50_000_000):
    """Faithful derandomized measurements for `terms` (universe indices) until every term reaches `need` (array aligned
    with terms), starting from `have` (default 0). Returns (settings (M, n) uint8 in generation order, final counts)."""
    terms = np.asarray(terms, np.int64)
    need = np.asarray(need, np.int64)
    sub = universe.subset(terms)
    Mmax = float(max(1, need.max()))
    weight = (need.astype(np.float64) + 1e-9) / Mmax
    target = np.floor(weight * Mmax)
    assert np.array_equal(target.astype(np.int64), need), "floor(w M) != need"
    h = np.zeros(len(terms)) if have is None else np.asarray(have, np.float64)
    out, m, _ = _derand(universe.n, sub.ptr, sub.q, sub.b, target, h, weight, eta, int(max_meas))
    return out, m


def consolidate(settings):
    """(M, n) one-shot measurements -> list of (setting, shots) over distinct settings (order of first appearance).
    Counts per term do not depend on the order of shots inside one job."""
    if len(settings) == 0:
        return []
    u, first, inv, cnt = np.unique(settings, axis=0, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)
    return [(u[i].astype(np.int64), int(cnt[i])) for i in order]


def hkp_derandomize_until_feasible(universe, terms, need, have, feasible, chunk_frac=0.01, min_chunk=200,
                                   eta=ETA, max_meas=50_000_000, fill_rng=None):
    """The same deterministic HKP sequence as hkp_derandomize, generated in chunks; after each chunk `feasible(m)`
    (a function of the counts only — B2) is evaluated and generation stops as soon as it returns True (query-level
    count-aware stopping, '+qs'). Resuming with the current counts continues the identical sequence (the algorithm's
    state is the count vector). With `fill_rng`, qubits that no still-matchable unfinished observable touches (the
    reference assigns X by tie) get seeded uniform bases instead. Returns (settings, counts of `terms`)."""
    terms = np.asarray(terms, np.int64)
    need = np.asarray(need, np.int64)
    sub = universe.subset(terms)
    Mmax = float(max(1, need.max()))
    weight = (need.astype(np.float64) + 1e-9) / Mmax
    target = np.floor(weight * Mmax)
    m = np.asarray(have, np.float64).copy()
    outs = []
    total = 0
    while total < max_meas:
        if feasible(m) or np.all(m >= target):
            break
        c = int(max(min_chunk, chunk_frac * max(total, 1)))
        out, m, free = _derand(universe.n, sub.ptr, sub.q, sub.b, target, m, weight, eta, c)
        if len(out) == 0:
            break
        if fill_rng is not None:  # random fill of free qubits: changes no deficit term's count (they are all
            # mismatched or finished in that measurement); adds outcome-independent coverage for other terms
            out = out.copy()
            out[free] = fill_rng.integers(0, 3, size=int(free.sum()), dtype=np.uint8)
        outs.append(out)
        total += len(out)
    seq = np.concatenate(outs) if outs else np.zeros((0, universe.n), np.uint8)
    return seq, m
