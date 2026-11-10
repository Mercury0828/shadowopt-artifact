"""Per-answer error bounds (guide §P4.6): matched mean, Hoeffding radius, delta-split, per-term count thresholds.

All functions take counts, coefficients and targets only — never outcomes (B2).
"""
import math

import numpy as np


def radius(m, delta):
    """r(m, delta) = sqrt(2 ln(2/delta) / m); +inf for m = 0."""
    m = np.asarray(m, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(m > 0, np.sqrt(2.0 * np.log(2.0 / np.asarray(delta, dtype=np.float64)) / np.maximum(m, 1)),
                        np.inf)


def _g(x):
    return x * np.sqrt(2.0 * np.log(2.0 / x))


def _ginv(y, lo=1e-300, hi=1.0):
    """Inverse of g(x) = x sqrt(2 ln(2/x)) on (0, 1] (g is increasing there), vectorized bisection in log space."""
    y = np.asarray(y, dtype=np.float64)
    a = np.full_like(y, np.log(lo))
    b = np.full_like(y, np.log(hi))
    for _ in range(80):
        mid = 0.5 * (a + b)
        val = _g(np.exp(mid))
        a = np.where(val < y, mid, a)
        b = np.where(val < y, b, mid)
    return np.exp(0.5 * (a + b))


def delta_split(c, m, delta):
    """Split delta over the terms of a query to minimize sum_j c_j r(m_j, delta_j) s.t. sum_j delta_j = delta.

    c: |coefficients| (> 0), m: counts (>= 1). Stationarity: delta_j sqrt(2 ln(2/delta_j)) = c_j / (lambda sqrt(m_j)).
    Depends on counts and coefficients only. Returns delta_j (sum = delta up to 1e-12 relative)."""
    c = np.asarray(c, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    J = len(c)
    if J == 1:
        return np.array([delta])
    w = c / np.sqrt(m)
    # scale-invariant bracket for the multiplier (C-BOUND R1): at lam = w_max / g(1) the largest term already takes
    # d = 1 >= delta (sum > delta); at lam = w_max / g(delta / J) every d_j <= delta / J (sum <= delta)
    w = w / float(np.max(w))  # the split depends only on ratios w_j / lambda: normalize (no overflow; C-BOUND R1 audits)
    lo = np.log(1.0 / _g(np.array(1.0)))
    hi = np.log(1.0 / _g(np.array(delta / J)))
    for _ in range(100):
        lam = np.exp(0.5 * (lo + hi))
        y = np.minimum(w / lam, _g(np.array(1.0)))
        s = _ginv(y).sum()
        if s > delta:
            lo = np.log(lam)
        else:
            hi = np.log(lam)
    lam = np.exp(hi)
    d = _ginv(np.minimum(w / lam, _g(np.array(1.0))))
    return d * (delta / d.sum())  # tiny renormalization; sum exactly delta


def query_halfwidth(coeffs, m, delta):
    """Stated half-width of a query answer sum_j a_j mu_hat_j: sum_j |a_j| r(m_j, delta_j*) (inf if any m_j = 0)."""
    c = np.abs(np.asarray(coeffs, dtype=np.float64))
    m = np.asarray(m, dtype=np.float64)
    keep = c > 0
    c, m = c[keep], m[keep]
    if len(c) == 0:
        return 0.0
    if np.any(m <= 0):
        return math.inf
    d = delta_split(c, m, delta)
    return float(np.sum(c * radius(m, d)))


def term_thresholds(coeffs, eps, delta):
    """Per-term count thresholds n*_j sufficient for the query target (planning rule, Phase 0):
    with the uniform split delta_j = delta/J, minimizing sum_j n_j s.t. sum_j c_j A / sqrt(n_j) <= eps
    (A = sqrt(2 ln(2J/delta))) gives n_j = A^2 S^2 c_j^(2/3) / eps^2 with S = sum_k c_k^(2/3).
    If every term reaches its threshold the query's half-width with the optimized split is <= eps."""
    c = np.abs(np.asarray(coeffs, dtype=np.float64))
    J = int(np.sum(c > 0))
    A2 = 2.0 * math.log(2.0 * max(J, 1) / delta)
    S = np.sum(c[c > 0] ** (2.0 / 3.0))
    n = np.where(c > 0, np.ceil(A2 * S * S * c ** (2.0 / 3.0) / eps ** 2 - 1e-9), 0).astype(np.int64)
    return n
