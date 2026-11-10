"""Query-level accuracy checking (H-b′3 pipeline): counts for every distinct term of a batch from an access path, the
optimized-delta-split half-width per query (compiled; same definition as bounds.query_halfwidth), and the certified /
uncertified split. The statistical part is shared by every count source, so a timing comparison between count sources
measures the access paths, not interpreter overhead.

Certification semantics: HW <= eps is a SUFFICIENT per-query accuracy condition (B2: counts only).
"""
import math

import numpy as np
from numba import njit, prange

_G1 = math.sqrt(2.0 * math.log(2.0))  # g(1) = 1 * sqrt(2 ln 2)


@njit(cache=True)
def _g(x):
    return x * math.sqrt(2.0 * math.log(2.0 / x))


@njit(cache=True)
def _ginv(y):
    """x in (0, 1] with g(x) = y, g(x) = x sqrt(2 ln(2/x)) (increasing on (0, 1]). Newton in t = ln x on
    h(t) = t + 0.5 ln(2 (ln 2 - t)) - ln y, which is near-linear (h'(t) = 1 - 1 / (2 (ln 2 - t)) in [0.27, 1) for t <= 0);
    bracket [ln 1e-300, 0] safeguard (steps leaving it bisect); relative stopping 2e-15 (A14: the former absolute 1e-15 was
    below the float spacing of t and took a median 43 iterations)."""
    if y >= _G1:
        return 1.0
    if y <= 0.0:
        return 1e-300
    a = math.log(1e-300)
    b = 0.0
    ly = math.log(y)
    t = ly - 0.5 * math.log(2.0 * math.log(2.0 / max(y, 1e-300)))  # x ~ y / sqrt(2 ln(2/y))
    if not (a < t < b):
        t = 0.5 * (a + b)
    for _ in range(100):
        L = math.log(2.0) - t  # ln(2/x) > 0
        h = t + 0.5 * math.log(2.0 * L) - ly
        if h < 0:
            a = t
        else:
            b = t
        dh = 1.0 - 0.5 / L
        tn = t - h / dh
        if not (a < tn < b):
            tn = 0.5 * (a + b)
        if abs(tn - t) <= 2e-15 * max(1.0, abs(t)):
            t = tn
            break
        t = tn
    x = math.exp(t)
    r = math.sqrt(2.0 * math.log(2.0 / x))  # one Newton step in x removes the exp(t) representation error (|t| ulps)
    xn = x - (x * r - y) / (r - 1.0 / r)
    return xn if 0.0 < xn <= 1.0 else x


@njit(cache=True)
def _split_sum(w, lam, J):
    """S(lam) = sum_j ginv(min(w_j / lam, g(1))) and dS/d ln(lam) (clamped terms contribute 0 to the derivative)."""
    s = 0.0
    ds = 0.0
    for j in range(J):
        y = w[j] / lam
        if y >= _G1:
            s += 1.0
            continue
        x = _ginv(y)
        s += x
        r = math.sqrt(2.0 * math.log(2.0 / x))
        ds -= y / (r - 1.0 / r)  # dx/dy = 1 / g'(x), g'(x) = r - 1/r; dy/d ln(lam) = -y
    return s, ds


@njit(cache=True)
def _halfwidth(c, m, delta):
    """Same as bounds.query_halfwidth for c > 0 (inf if any m == 0)."""
    return _halfwidth_d(c, m, delta, np.empty(c.shape[0]))


@njit(cache=True)
def _halfwidth_d(c, m, delta, dout):
    """_halfwidth that also writes the (renormalized) delta split into dout (NaN if some m == 0). A14."""
    J = c.shape[0]
    if J == 0:
        return 0.0
    for j in range(J):
        if m[j] <= 0:
            for k in range(J):
                dout[k] = np.nan
            return np.inf
    if J == 1:
        dout[0] = delta
        return c[0] * math.sqrt(2.0 * math.log(2.0 / delta) / m[0])
    w = np.empty(J)
    for j in range(J):
        w[j] = c[j] / math.sqrt(m[j])
    wmax = 0.0
    for j in range(J):
        if w[j] > wmax:
            wmax = w[j]
    for j in range(J):
        w[j] = w[j] / wmax  # normalize: the split depends only on ratios (no overflow)
    lo = math.log(1.0 / _G1)
    hi = math.log(1.0 / _g(delta / J))
    d = np.empty(J)
    # A14: safeguarded Newton on F(u) = ln S(e^u) - ln delta (S decreasing in u) instead of 64 fixed bisection steps.
    # Any lam gives a VALID split after the renormalization below; the optimum only makes it tight.
    u = 0.5 * (lo + hi)
    ldelta = math.log(delta)
    for _ in range(100):
        s, ds = _split_sum(w, math.exp(u), J)
        F = math.log(s) - ldelta
        if F > 0:
            lo = u
        else:
            hi = u
        un = u - F * s / ds if ds < 0 else 0.5 * (lo + hi)
        if not (lo < un < hi):
            un = 0.5 * (lo + hi)
        if abs(un - u) <= 1e-13 * max(1.0, abs(u)) or hi - lo <= 1e-13 * max(1.0, abs(u)):
            u = un
            break
        u = un
    lam = math.exp(u)
    tot = 0.0
    for j in range(J):
        d[j] = _ginv(min(w[j] / lam, _G1))
        tot += d[j]
    hw = 0.0
    for j in range(J):
        dj = d[j] * (delta / tot)
        dout[j] = dj
        hw += c[j] * math.sqrt(2.0 * math.log(2.0 / dj) / m[j])
    return hw


@njit(parallel=True, cache=True)
def _batch_halfwidths(qptr, qterm, qcoef, qdelta, m_by_slot):
    Q = qptr.shape[0] - 1
    out = np.empty(Q)
    for i in prange(Q):
        a, b = qptr[i], qptr[i + 1]
        cnt = 0
        for j in range(a, b):
            if qcoef[j] != 0.0:
                cnt += 1
        c = np.empty(cnt)
        m = np.empty(cnt)
        k = 0
        for j in range(a, b):
            if qcoef[j] != 0.0:
                c[k] = abs(qcoef[j])
                m[k] = m_by_slot[qterm[j]]
                k += 1
        out[i] = _halfwidth(c, m, qdelta[i])
    return out


@njit(parallel=True, cache=True)
def _batch_hw_split(qptr, qterm, qcoef, qdelta, m_by_slot, dflat):
    """_batch_halfwidths that also stores each query's split in dflat (aligned with qterm; 0 for zero coefficients)."""
    Q = qptr.shape[0] - 1
    out = np.empty(Q)
    for i in prange(Q):
        a, b = qptr[i], qptr[i + 1]
        cnt = 0
        for j in range(a, b):
            if qcoef[j] != 0.0:
                cnt += 1
        c = np.empty(cnt)
        m = np.empty(cnt)
        d = np.empty(cnt)
        k = 0
        for j in range(a, b):
            if qcoef[j] != 0.0:
                c[k] = abs(qcoef[j])
                m[k] = m_by_slot[qterm[j]]
                k += 1
        out[i] = _halfwidth_d(c, m, qdelta[i], d)
        k = 0
        for j in range(a, b):
            if qcoef[j] != 0.0:
                dflat[j] = d[k]
                k += 1
            else:
                dflat[j] = 0.0
    return out


@njit(parallel=True, cache=True)
def _batch_hw_bounds(qptr, qterm, qcoef, qdelta, m_by_slot, dflat):
    """Per query: lb = sum |a_j| sqrt(2 ln(2/delta)/m_j) <= HW_opt (every delta_j <= delta) and ub = the half-width at the
    given split dflat (any split summing to delta is valid, so HW_opt <= ub; inf if the split is undefined). A14."""
    Q = qptr.shape[0] - 1
    lb = np.empty(Q)
    ub = np.empty(Q)
    for i in prange(Q):
        a, b = qptr[i], qptr[i + 1]
        l = 0.0
        u = 0.0
        L = 2.0 * math.log(2.0 / qdelta[i])
        for j in range(a, b):
            if qcoef[j] == 0.0:
                continue
            mj = m_by_slot[qterm[j]]
            if mj <= 0:
                l = np.inf
                u = np.inf
                break
            c = abs(qcoef[j])
            l += c * math.sqrt(L / mj)
            dj = dflat[j]
            if not (dj > 0.0):
                u = np.inf
            else:
                u += c * math.sqrt(2.0 * math.log(2.0 / dj) / mj)
        lb[i] = l
        ub[i] = u
    return lb, ub


class BatchCSR:
    """A batch of queries in CSR form over the batch's distinct terms (slots)."""

    def __init__(self, queries):
        terms = sorted({int(u) for q in queries for u in q.terms})
        slot = {u: i for i, u in enumerate(terms)}
        self.terms = np.array(terms, np.int64)
        ptr, qt, qc = [0], [], []
        for q in queries:
            qt += [slot[int(u)] for u in q.terms]
            qc += [float(x) for x in q.coeffs]
            ptr.append(len(qt))
        self.qptr = np.array(ptr, np.int64)
        self.qterm = np.array(qt, np.int64)
        self.qcoef = np.array(qc, np.float64)
        self.qdelta = np.array([q.delta for q in queries], np.float64)
        self.qeps = np.array([q.eps for q in queries], np.float64)


def certify(csr, slot_counts):
    """slot_counts: counts of csr.terms (aligned). Returns (halfwidths, certified mask)."""
    hw = _batch_halfwidths(csr.qptr, csr.qterm, csr.qcoef, csr.qdelta, np.asarray(slot_counts, np.float64))
    return hw, hw <= csr.qeps  # the frozen condition HW <= eps, no tolerance


# ----------------------------------------------------------------------------------------------------------------------
# Count sources: every source returns the exact counts of csr.terms over the store's current rows.
# ----------------------------------------------------------------------------------------------------------------------
class CountSource:
    """kind in {"bitmap", "sstar"} x cached in {False, True}. Uncached: one select-only pass over all rows per batch.
    Cached: a counts-only TermCache (watermarks; only the delta since the last check is evaluated)."""

    def __init__(self, index, kind="bitmap", cached=False, block=2048):
        from .engine import TermCache
        assert not (cached and kind == "shared"), "cached sources evaluate deltas with the bitmap or S* range kernels"
        self.ix, self.kind, self.cached, self.block = index, kind, cached, block
        self.cache = TermCache(index, kind, block, select_only=True) if cached else None
        self._sub = None  # (universe, csr, subset) memo of the last batch (A14: no per-call term-subset rebuild)
        self.plan_memo = {} if cached else None  # A14: unchanged counts -> unchanged completion plan (cached sources)

    def _subset(self, universe, csr):
        if self._sub is None or self._sub[0] is not universe or self._sub[1] is not csr:
            self._sub = (universe, csr, universe.subset(csr.terms))
        return self._sub[2]

    def counts(self, universe, csr):
        sub = self._subset(universe, csr)
        if self.cached:
            return self.cache.answer(sub)[0]
        from .engine import _bitmap_range
        from .scan import encode_terms_for_scan, _sstar
        self.ix.refresh()
        N = self.ix.N_rep
        if self.kind == "shared":  # A16: shared-prefix trie over the batch's literals (select-only); the plan is rebuilt
            # on every call (no memo), so a cold check always pays for it. Rows >= N_rep have all-zero bitmap words.
            from .sharedplan import Trie, _shared_batch
            tr = Trie(sub)
            return _shared_batch(self.ix.Z, self.ix.O, tr.node_q, tr.node_b, tr.node_depth, tr.term_ptr, tr.term_ids,
                                 tr.T, tr.max_depth, 2048, True)[0]
        if self.kind == "bitmap":
            return _bitmap_range(self.ix.Z, self.ix.O, sub.ptr, sub.q, sub.b, 0, N, True)[0]
        enc = encode_terms_for_scan(sub, self.ix.n, True)
        return _sstar(self.ix.rows[:N], *enc, self.block, True)[0]


@njit(cache=True, nogil=True)
def _first_fit(n, ptr, q, b, order):
    """Compiled first-fit qubit-wise-compatible grouping in the given term order; identical to planner.group_terms
    (a term joins the first group whose assigned bases agree on its support). Returns (assign G x n, -1 = unassigned;
    group id per position of `order`)."""
    T = order.shape[0]
    asg = np.full((max(T, 1), n), -1, np.int64)
    gid = np.empty(T, np.int64)
    G = 0
    for i in range(T):
        u = order[i]
        placed = -1
        for g in range(G):
            fits = True
            for j in range(ptr[u], ptr[u + 1]):
                a = asg[g, q[j]]
                if a != -1 and a != b[j]:
                    fits = False
                    break
            if fits:
                placed = g
                break
        if placed == -1:
            placed = G
            G += 1
        for j in range(ptr[u], ptr[u + 1]):
            asg[placed, q[j]] = b[j]
        gid[i] = placed
    return asg[:G], gid


_SERIAL = None


def _threads():
    import numba
    return numba.get_num_threads()


def _serial_kernels():
    """Serial, GIL-free variants of the completion kernels for the per-order threads (prange runs as range)."""
    global _SERIAL
    if _SERIAL is None:
        mk = lambda f: njit(nogil=True)(f.py_func)
        _SERIAL = (mk(_compat), mk(_batch_hw_split), mk(_batch_hw_bounds), mk(_batch_halfwidths))
    return _SERIAL


def fast_group_terms(universe, terms, order_key=None):
    """Same output as planner.group_terms (list of (setting with Z = 2 on unassigned qubits, members in order))."""
    w = universe.weights()
    terms = sorted(terms, key=order_key or (lambda u: (-w[u], u)))
    if not terms:
        return []
    asg, gid = _first_fit(universe.n, universe.ptr, universe.q, universe.b, np.array(terms, np.int64))
    S = np.where(asg < 0, 2, asg).astype(np.int64)
    mem = [[] for _ in range(S.shape[0])]
    for u, g in zip(terms, gid):
        mem[int(g)].append(int(u))
    return [(S[g].copy(), mem[g]) for g in range(S.shape[0])]


@njit(parallel=True, cache=True)
def _compat(S, ptr, q, b):
    """C[t, g] = 1 iff term t is compatible with setting g (the same predicate as planner._setting_counts)."""
    T = ptr.shape[0] - 1
    G = S.shape[0]
    C = np.zeros((T, G), np.int64)
    for t in prange(T):
        for g in range(G):
            ok = 1
            for j in range(ptr[t], ptr[t + 1]):
                if S[g, q[j]] != b[j]:
                    ok = 0
                    break
            C[t, g] = ok
    return C


def _plan_search(alpha_of, shots_of, feas_of, max_doublings):
    """Minimal feasible alpha on the dyadic grid of the reference bisection (hi = 2^d after doubling, then 30 halvings:
    grid k * hi / 2^30). Feasibility depends on alpha only through shots(alpha) = ceil(alpha * base) and is monotone in
    alpha, so the bisection returns the smallest feasible grid point; this search finds the same point with interpolation
    probes and plateau jumps (all k with identical shots share feasibility). Returns hi or None (fall back)."""
    hi = 1.0
    verified = False
    for _ in range(max_doublings):
        f, r = feas_of(shots_of(hi))
        if f:
            verified = True
            break
        hi *= 2
    if not verified:
        return None
    K = 1 << 30
    a = lambda k: k * hi / K  # exact dyadic (k < 2^31, hi a power of 2)
    klo, khi = 0, K
    rlo, rhi = math.inf, r
    s_hi = shots_of(hi)

    def low_end(k, s):  # smallest k' in [klo + 1, k] with shots(k') == s (shots monotone in k)
        lo_, hi_ = klo + 1, k
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if np.array_equal(shots_of(a(mid)), s):
                hi_ = mid
            else:
                lo_ = mid + 1
        return lo_

    def high_end(k, s):  # largest k' in [k, khi - 1] with shots(k') == s
        lo_, hi_ = k, khi - 1
        while lo_ < hi_:
            mid = (lo_ + hi_ + 1) // 2
            if np.array_equal(shots_of(a(mid)), s):
                lo_ = mid
            else:
                hi_ = mid - 1
        return lo_

    khi = low_end(khi, s_hi)
    stall = 0
    while khi - klo > 1:
        width = khi - klo
        if stall >= 2:
            kt = (klo + khi) // 2
        elif klo == 0 or not math.isfinite(rlo):  # one-sided model: ratio ~ alpha^(-1/2)
            kt = int(math.floor(a(khi) * rhi * rhi * K / hi))
        else:  # interpolate ln ratio linearly in ln alpha between the bracket ends
            la, lb = math.log(a(klo)), math.log(a(khi))
            ra, rb = math.log(rlo), math.log(rhi)
            kt = int(math.floor(math.exp(la + (0.0 - ra) * (lb - la) / (rb - ra)) * K / hi)) if rb < ra else (klo + khi) // 2
        kt = min(max(kt, klo + 1), khi - 1)
        s = shots_of(a(kt))
        f, r = feas_of(s)
        if f:
            khi, rhi = low_end(kt, s), r
        else:
            klo, rlo = high_end(kt, s), r
        stall = stall + 1 if (khi - klo) > width // 2 else 0
    return a(khi)


def complete(universe, csr, m, hw, ok, max_doublings=40, cost=None, n_random_orders=4, orders=None):
    """Cost-aware FEASIBLE completion for the uncertified queries (not claimed optimal): the same plan as
    `complete_reference` (greedy QWC grouping per term order, s_g = ceil(alpha * base_g), alpha minimal on the bisection's
    dyadic grid such that every uncertified query is certified at the projected counts; cheapest order under `cost`),
    computed faster (A14): feasibility on the uncertified queries only, compiled grouping, a compatibility matrix per
    order, and an interpolation search with plateau jumps for alpha. Returns (settings, projected counts)."""
    from .bounds import term_thresholds
    orders_arg = orders  # passed on unchanged to the reference fallback
    bad = np.nonzero(~ok)[0]
    if len(bad) == 0:
        return [], m
    slot_need = np.zeros(len(csr.terms), np.int64)
    for i in bad:
        a, b = csr.qptr[i], csr.qptr[i + 1]
        th = term_thresholds(csr.qcoef[a:b], csr.qeps[i], csr.qdelta[i])
        for j, t in zip(range(a, b), th):
            slot_need[csr.qterm[j]] = max(slot_need[csr.qterm[j]], int(t))
    deficit_slots = [s for s in np.nonzero(slot_need > 0)[0] if m[s] < slot_need[s]]
    deficit_terms = [int(csr.terms[s]) for s in deficit_slots]
    slot_of = {int(u): i for i, u in enumerate(csr.terms)}
    sub = universe.subset(csr.terms)
    w = universe.weights()
    need_of = {int(csr.terms[s]): int(slot_need[s] - m[s]) for s in deficit_slots}
    if orders is not None:  # A15: caller-supplied order keys (functions of (term, need)); default = the A13 orders
        orders = [(lambda u, f=f: f(u, need_of[u])) for f in orders]
    elif cost is None:
        orders = [None]
    else:
        orders = [None, lambda u: (-need_of[u], u), lambda u: (w[u], u)]
        for r in range(n_random_orders):
            perm = np.random.default_rng([20261108, r]).permutation(universe.T)
            orders.append(lambda u, perm=perm: perm[u])
    # the uncertified queries only, in CSR over the same slots
    bptr, bterm, bcoef = [0], [], []
    for i in bad:
        a, b = csr.qptr[i], csr.qptr[i + 1]
        bterm += list(csr.qterm[a:b])
        bcoef += list(csr.qcoef[a:b])
        bptr.append(len(bterm))
    bptr, bterm, bcoef = np.array(bptr, np.int64), np.array(bterm, np.int64), np.array(bcoef, np.float64)
    bdelta, beps = csr.qdelta[bad], csr.qeps[bad]
    m64 = np.asarray(m, np.int64)
    rows = np.unique(bterm)  # slots used by the uncertified queries; feasibility needs only these counts
    bterm_loc = np.searchsorted(rows, bterm).astype(np.int64)
    m_rows = m64[rows]

    K = _serial_kernels() if (cost is not None and len(orders) > 1 and _threads() > 1) else None
    k_compat, k_split, k_bounds, k_hw = K if K else (_compat, _batch_hw_split, _batch_hw_bounds, _batch_halfwidths)

    def plan_for(groups):
        base = np.array([max(1, max(int(slot_need[slot_of[u]] - m[slot_of[u]]) for u in mem)) for _, mem in groups],
                        np.float64)
        S = np.array([s for s, _ in groups], np.int64)
        C = k_compat(S, sub.ptr, sub.q, sub.b)
        Cr = np.ascontiguousarray(C[rows])
        shots_of = lambda alpha: np.ceil(alpha * base).astype(np.int64)

        split = [None]  # the split of the last exact evaluation (warm upper bound)

        def feas_of(s):
            """Exact decision 'every uncertified query has HW <= eps at the projected counts'. Bounds decide only with a
            1e-12 relative margin (lb > eps: infeasible; ub <= eps: feasible; lb <= HW <= ub mathematically); otherwise
            the compiled half-width decides (A14)."""
            mc = (m_rows + Cr @ s).astype(np.float64)
            if split[0] is not None:
                lb, ub = k_bounds(bptr, bterm_loc, bcoef, bdelta, mc, split[0])
                if np.any(lb > beps * (1 + 1e-12)):
                    return False, float(np.max(lb / beps))
                if np.all(ub <= beps * (1 - 1e-12)):
                    return True, float(np.max(ub / beps))
            d = np.empty(len(bterm_loc))
            h = k_split(bptr, bterm_loc, bcoef, bdelta, mc, d)
            split[0] = d
            return bool(np.all(h <= beps)), float(np.max(h / beps))

        alpha = _plan_search(None, shots_of, feas_of, max_doublings)
        if alpha is None:  # never feasible within the doublings: the reference behaviour
            return None
        s = shots_of(alpha)
        mp = m64 + C @ s
        h = k_hw(bptr, bterm, bcoef, bdelta, mp.astype(np.float64))  # exact verification of the plan
        if not np.all(h <= beps):
            return None
        return [(S[g].copy(), int(s[g])) for g in range(len(groups))], mp

    run = lambda key: plan_for(fast_group_terms(universe, deficit_terms, key))
    if K:  # A14: the orders are independent -> one thread per order with serial nogil kernels; results kept in order
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(orders), _threads())) as ex:
            results = list(ex.map(run, orders))
    else:
        results = [run(key) for key in orders]
    best = None
    for res in results:  # the same selection as the reference: first order with strictly smallest cost
        if res is None:
            return complete_reference(universe, csr, m, hw, ok, max_doublings, cost, n_random_orders, orders_arg)
        ss, mp = res
        u = cost.target([k for _, k in ss]) if cost is not None else 0.0
        if best is None or u < best[0]:
            best = (u, ss, mp)
    return best[1], best[2]


def complete_reference(universe, csr, m, hw, ok, max_doublings=40, cost=None, n_random_orders=4, orders=None):
    """Cost-aware FEASIBLE completion for the uncertified queries (not claimed optimal): greedy qubit-wise-compatible
    grouping of their deficit terms, one setting per group with s_g = ceil(alpha * base_g) shots, base_g = the group's
    largest per-term deficit to its uniform-split threshold; alpha minimal by bisection such that every uncertified query
    is certified at the PROJECTED counts (verified with the same compiled checker). Returns (settings, projected counts)."""
    from .bounds import term_thresholds
    from .planner import group_terms, setting_counts
    bad = np.nonzero(~ok)[0]
    if len(bad) == 0:
        return [], m
    slot_need = np.zeros(len(csr.terms), np.int64)
    for i in bad:
        a, b = csr.qptr[i], csr.qptr[i + 1]
        th = term_thresholds(csr.qcoef[a:b], csr.qeps[i], csr.qdelta[i])
        for j, t in zip(range(a, b), th):
            slot_need[csr.qterm[j]] = max(slot_need[csr.qterm[j]], int(t))
    deficit_slots = [s for s in np.nonzero(slot_need > 0)[0] if m[s] < slot_need[s]]
    deficit_terms = [int(csr.terms[s]) for s in deficit_slots]
    slot_of = {int(u): i for i, u in enumerate(csr.terms)}
    sub = universe.subset(csr.terms)
    w = universe.weights()
    need_of = {int(csr.terms[s]): int(slot_need[s] - m[s]) for s in deficit_slots}
    if orders is not None:  # A15: caller-supplied order keys (functions of (term, need))
        orders = [(lambda u, f=f: f(u, need_of[u])) for f in orders]
    elif cost is None:
        orders = [None]  # Phase-0/first-version behaviour: first-fit by weight descending only
    else:  # A13: a fixed set of term orders; the cheapest feasible plan under the cost model is kept (counts only)
        orders = [None, lambda u: (-need_of[u], u), lambda u: (w[u], u)]
        for r in range(n_random_orders):
            perm = np.random.default_rng([20261108, r]).permutation(universe.T)
            orders.append(lambda u, perm=perm: perm[u])

    def plan_for(groups):
        base = [max(1, max(int(slot_need[slot_of[u]] - m[slot_of[u]]) for u in mem)) for _, mem in groups]

        def project(alpha):
            ss = [(s, int(np.ceil(alpha * b0))) for (s, _), b0 in zip(groups, base)]
            return ss, m + setting_counts(sub, ss)

        def feasible(alpha):
            _, mp = project(alpha)
            _, okp = certify(csr, mp)
            return bool(np.all(okp[bad]))

        hi = 1.0
        for _ in range(max_doublings):  # thresholds imply feasibility at alpha = 1; doubling is a guard only
            if feasible(hi):
                break
            hi *= 2
        lo = 0.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if feasible(mid):
                hi = mid
            else:
                lo = mid
        return project(hi)

    best = None
    for key in orders:
        ss, mp = plan_for(group_terms(universe, deficit_terms, key))
        u = cost.target([k for _, k in ss]) if cost is not None else 0.0
        if best is None or u < best[0]:
            best = (u, ss, mp)
    return best[1], best[2]


def check_and_plan(universe, csr, source, cost=None):
    """The full checking pipeline: counts -> half-widths -> certified split -> completion plan with verified projection.
    With `cost`, the completion keeps the cheapest of a fixed set of grouping orders (A13)."""
    m = source.counts(universe, csr)
    hw, ok = certify(csr, m)
    memo = getattr(source, "plan_memo", None)  # cached sources only: the plan is a deterministic function of the counts
    key = (id(universe), id(csr), id(cost), np.asarray(m, np.int64).tobytes())
    if memo is not None and memo.get("key") == key and memo.get("refs") == (universe, csr, cost):
        plan, mproj = memo["plan"]
    else:
        plan, mproj = complete(universe, csr, m, hw, ok, cost=cost)
        if memo is not None:
            memo.clear()
            memo.update(key=key, refs=(universe, csr, cost), plan=(plan, mproj))
    return {"m": m, "hw": hw, "ok": ok, "plan": plan, "m_projected": mproj}
