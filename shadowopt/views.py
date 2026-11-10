"""Materialized marginal views over supports (data-cube style, G17).

V_T for a support T = (T_0 < ... < T_{s-1}) stores, for every basis pattern beta in {X,Y,Z}^s and outcome pattern
omega in {0,1}^s on T, the number of snapshots with that pattern: counts[beta_idx * 2^s + omega_idx] with
beta_idx = sum_j beta_j 3^j and omega_idx = sum_j omega_j 2^j. Maintenance under append is one increment per view per
snapshot. At query time a view is "prepared" once (Walsh-Hadamard transform over omega for each beta:
W[beta, A] = sum_omega counts[beta, omega] (-1)^{|omega & A|}); a term P with S(P) inside T is then answered by
  m_P = sum_{beta consistent with P on S(P)} W[beta, 0],   s_P = sum_{beta consistent} W[beta, A(P)],
where A(P) is the position mask of S(P) inside T. Cost per term: 3^(s-k) additions, independent of N.

View selection: a greedy benefit-per-space rule over the lattice of supports (HRU-style, G17), described in
`select_views`.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _build_view(bases, outcomes, T, nthreads_chunks):
    N = bases.shape[0]
    s = T.shape[0]
    size = (3 ** s) * (2 ** s)
    part = np.zeros((nthreads_chunks, size), np.int64)
    step = (N + nthreads_chunks - 1) // nthreads_chunks
    for c in prange(nthreads_chunks):
        r0 = c * step
        r1 = min(N, r0 + step)
        for r in range(r0, r1):
            bi = 0
            oi = 0
            p3 = 1
            for j in range(s):
                bi += np.int64(bases[r, T[j]]) * p3
                oi |= np.int64(outcomes[r, T[j]]) << j
                p3 *= 3
            part[c, bi * (2 ** s) + oi] += 1
    return part.sum(axis=0)


@njit(cache=True)
def _wht(counts, s):
    nb = 3 ** s
    no = 2 ** s
    W = counts.reshape(nb, no).copy()
    for bi in range(nb):
        h = 1
        while h < no:
            for i in range(0, no, 2 * h):
                for j in range(i, i + h):
                    a = W[bi, j]
                    c = W[bi, j + h]
                    W[bi, j] = a + c
                    W[bi, j + h] = a - c
            h *= 2
    return W


class View:
    def __init__(self, support):
        self.T = np.array(sorted(support), np.int64)
        self.s = len(self.T)
        self.counts = np.zeros((3 ** self.s) * (2 ** self.s), np.int64)
        self.W = None

    def build(self, store, chunks=64):
        self.counts = _build_view(store.bases, store.outcomes, self.T, chunks)
        self.W = None
        return self

    def append(self, bases, outcomes):
        self.counts += _build_view(bases, outcomes, self.T, 1)
        self.W = None

    def nbytes(self):
        return 2 * self.counts.nbytes  # counts + prepared transform

    def prepare(self):
        if self.W is None:
            self.W = _wht(self.counts, self.s)
        return self.W


@njit(parallel=True, cache=True)
def _views_answer_all(Wflat, Woff, svec, vid, pos, req, pos_ptr):
    T = pos_ptr.shape[0] - 1
    M = np.zeros(T, np.int64)
    S = np.zeros(T, np.int64)
    for t in prange(T):
        v = vid[t]
        s = svec[v]
        no = 2 ** s
        nb = 3 ** s
        base = Woff[v]
        p0 = pos_ptr[t]
        p1 = pos_ptr[t + 1]
        amask = 0
        for j in range(p0, p1):
            amask |= 1 << pos[j]
        # enumerate consistent basis indices: fixed digits at pos, free digits elsewhere
        fixed = 0
        fixmask = 0
        for j in range(p0, p1):
            fixed += req[j] * (3 ** pos[j])
            fixmask |= 1 << pos[j]
        free = np.empty(s, np.int64)
        nf = 0
        for j in range(s):
            if not (fixmask >> j) & 1:
                free[nf] = j
                nf += 1
        m = 0
        sm = 0
        for c in range(3 ** nf):
            bi = fixed
            cc = c
            for f in range(nf):
                bi += (cc % 3) * (3 ** free[f])
                cc //= 3
            m += Wflat[base + bi * no]
            sm += Wflat[base + bi * no + amask]
        M[t] = m
        S[t] = sm
    return M, S


class ViewSet:
    def __init__(self, supports):
        self.views = [View(T) for T in supports]

    def build(self, store):
        for v in self.views:
            v.build(store)
        return self

    def append(self, bases, outcomes):
        for v in self.views:
            v.append(bases, outcomes)

    def nbytes(self):
        return sum(v.nbytes() for v in self.views)

    def assign(self, terms):
        """For each term, the smallest view whose support contains S(P), or -1 (qubit sets as integer bitmasks)."""
        vmask = [sum(1 << int(q) for q in v.T) for v in self.views]
        order = sorted(range(len(self.views)), key=lambda i: (self.views[i].s, i))
        out = np.full(terms.T, -1, np.int64)
        ptr, q = terms.ptr, terms.q.tolist()
        for t in range(terms.T):
            tm = 0
            for qi in q[ptr[t]:ptr[t + 1]]:
                tm |= 1 << qi
            for vi in order:
                if tm & ~vmask[vi] == 0:
                    out[t] = vi
                    break
        return out

    def encode(self, terms, assignment):
        idx = np.nonzero(assignment >= 0)[0]
        where = [{q: j for j, q in enumerate(v.T.tolist())} for v in self.views]
        pos, req, ptr = [], [], [0]
        tp, tq, tb = terms.ptr, terms.q.tolist(), terms.b.tolist()
        for t in idx.tolist():
            wv = where[assignment[t]]
            for j in range(tp[t], tp[t + 1]):
                pos.append(wv[tq[j]])
                req.append(tb[j])
            ptr.append(len(pos))
        return (idx, assignment[idx].astype(np.int64), np.array(pos, np.int64), np.array(req, np.int64),
                np.array(ptr, np.int64))

    def answer(self, enc):
        """Answer the encoded (view-covered) terms; prepares (transforms) dirty views first."""
        idx, vid, pos, req, ptr = enc
        Ws = [v.prepare() for v in self.views]
        Woff = np.zeros(len(Ws), np.int64)
        off = 0
        for i, W in enumerate(Ws):
            Woff[i] = off
            off += W.size
        Wflat = np.concatenate([W.ravel() for W in Ws]) if Ws else np.zeros(0, np.int64)
        svec = np.array([v.s for v in self.views], np.int64)
        M, S = _views_answer_all(Wflat, Woff, svec, vid, pos, req, ptr)
        return idx, M, S


def view_bytes(s):
    return 2 * (3 ** s) * (2 ** s) * 8


def select_views(term_supports, term_freq, NW, cap_bytes, s_max=6, n=None):
    """Greedy benefit-per-space view selection (HRU-style) over supports of size <= s_max.

    term_supports: list of frozensets (one per distinct workload term, k = |support|); term_freq: how often each is asked
    in the selection window. Benefit of a view T for an uncovered term P with S(P) inside T, per batch:
    freq_P * (2 k NW - 3^(s-k)) word operations saved against the bitmap path. Candidates are grown from the uncovered
    term with the largest benefit: starting from its support, repeatedly add the qubit that maximizes
    (benefit / space) among supports of the next size, keeping the best ratio seen; the best grown candidate is chosen
    if it fits the remaining space. Stops when nothing fits or nothing uncovered remains.
    """
    terms = [(frozenset(S), f) for S, f in zip(term_supports, term_freq) if 0 < len(S) <= s_max]
    uncovered = {i for i in range(len(terms))}
    chosen = []
    used = 0

    def benefit(Tset):
        s = len(Tset)
        b = 0.0
        cov = []
        for i in uncovered:
            S, f = terms[i]
            if S <= Tset:
                k = len(S)
                b += f * (2 * k * NW - 3 ** (s - k))
                cov.append(i)
        return b, cov

    qubits_all = sorted(set().union(*[S for S, _ in terms])) if terms else []
    tried_seeds = set()
    while uncovered:
        # seed: uncovered term with the largest single benefit
        seed = max(uncovered - tried_seeds, key=lambda i: (terms[i][1] * 2 * len(terms[i][0]) * NW, -i),
                   default=None)
        if seed is None:
            break
        cur = set(terms[seed][0])
        best = None
        while True:
            b, cov = benefit(frozenset(cur))
            sp = view_bytes(len(cur))
            if b > 0 and used + sp <= cap_bytes:
                r = b / sp
                if best is None or r > best[0]:
                    best = (r, frozenset(cur), cov, sp)
            if len(cur) >= s_max:
                break
            # candidate qubits: those in uncovered supports touching cur
            cand = set()
            for i in uncovered:
                S, _ = terms[i]
                if S & cur and not S <= cur and len(S | cur) <= s_max:
                    cand |= (S - cur)
            if not cand:
                break
            scored = []
            for qn in cand:
                b2, _ = benefit(frozenset(cur | {qn}))
                scored.append((b2, -qn, qn))
            scored.sort(reverse=True)
            cur = cur | {scored[0][2]}
        if best is None:
            tried_seeds.add(seed)
            continue
        _, Tset, cov, sp = best
        chosen.append(tuple(sorted(Tset)))
        used += sp
        uncovered -= set(cov)
        tried_seeds.clear()
    return chosen, used


class ViewPath:
    """Views for covered terms; the shared-plan bitmap path for the rest (needed to answer the whole batch)."""

    def __init__(self, viewset, shared_path):
        self.vs = viewset
        self.shared = shared_path

    def plan(self, terms):
        a = self.vs.assign(terms)
        enc = self.vs.encode(terms, a)
        rest = np.nonzero(a < 0)[0]
        rest_plan = self.shared.plan(terms.subset(rest)) if len(rest) else None
        return (terms.T, enc, rest, rest_plan)

    def execute(self, plan):
        T, enc, rest, rest_plan = plan
        M = np.zeros(T, np.int64)
        S = np.zeros(T, np.int64)
        idx, Mv, Sv = self.vs.answer(enc)
        M[idx] = Mv
        S[idx] = Sv
        if rest_plan is not None:
            Mr, Sr = self.shared.execute(rest_plan)
            M[rest] = Mr
            S[rest] = Sr
        return M, S

    def run(self, terms):
        return self.execute(self.plan(terms))
