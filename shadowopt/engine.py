"""M1 engine layer (C-INDEX R1 resolution): append-only store with amortized appends, incremental bit-sliced bitmaps and
packed rows with an explicit represented prefix (freshness contract), an exact row-range kernel (boundary-word masks),
and a term-answer cache with append watermarks — usable over the bitmap path or over S* (the added "S* + cache" scan
baseline, so recurring-workload gains are never credited to indexing alone).

Freshness contract: every structure records `N_rep`, the number of store rows it represents. `refresh()` brings it to the
store's current N (incremental, O(delta n)); every `run` refreshes first. A cache entry (r, m, s) is advanced to N only
after the complete delta [r, N) has been evaluated; it never sees outcomes before deciding anything (B2: admission,
refresh and eviction depend on term keys and counts only).
"""
import numpy as np
from numba import njit, prange

from ._bits import popcnt64
from .store import SRC_CODES, row_words, n_words
from .scan import encode_terms_for_scan, _sstar


class AppendStore:
    """Append-only snapshot log with capacity doubling (amortized O(delta n) per append, no full copy per append)."""

    def __init__(self, n, cap=1 << 16):
        self.n = n
        self.N = 0
        self._b = np.zeros((cap, n), np.uint8)
        self._o = np.zeros((cap, n), np.uint8)
        self._e = np.zeros(cap, np.int32)
        self._s = np.zeros(cap, np.uint8)

    def _grow(self, need):
        cap = self._b.shape[0]
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for name in ("_b", "_o"):
            a = np.zeros((new, self.n), np.uint8)
            a[:self.N] = getattr(self, name)[:self.N]
            setattr(self, name, a)
        for name, dt in (("_e", np.int32), ("_s", np.uint8)):
            a = np.zeros(new, dt)
            a[:self.N] = getattr(self, name)[:self.N]
            setattr(self, name, a)

    def append(self, bases, outcomes, epoch=0, src="rand"):
        bases = np.asarray(bases, np.uint8)
        outcomes = np.asarray(outcomes, np.uint8)
        k = bases.shape[0]
        assert bases.shape == outcomes.shape and bases.shape[1] == self.n
        self._grow(self.N + k)
        self._b[self.N:self.N + k] = bases
        self._o[self.N:self.N + k] = outcomes
        self._e[self.N:self.N + k] = epoch
        self._s[self.N:self.N + k] = SRC_CODES[src]
        self.N += k

    @property
    def bases(self):
        return self._b[:self.N]

    @property
    def outcomes(self):
        return self._o[:self.N]


# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _set_bits(Z, O, bases, outcomes, r0):
    """Write rows r0 .. r0+len into bitmaps (bits of rows >= N stay 0). Parallel over qubits (disjoint writes)."""
    k, n = bases.shape
    for i in prange(n):
        for t in range(k):
            r = r0 + t
            w = r >> 6
            bit = np.uint64(1) << np.uint64(r & 63)
            Z[i, bases[t, i], w] |= bit
            if outcomes[t, i]:
                O[i, w] |= bit


@njit(cache=True)
def _pack_into(rows, bases, outcomes, r0, W):
    k, n = bases.shape
    for t in range(k):
        r = r0 + t
        for w in range(W):
            rows[r, w] = 0
        for i in range(n):
            code = np.uint64(bases[t, i] + 1)
            bit = 2 * i
            rows[r, bit >> 6] |= code << np.uint64(bit & 63)
            if outcomes[t, i]:
                ob = 2 * n + i
                rows[r, ob >> 6] |= np.uint64(1) << np.uint64(ob & 63)


class IncrementalIndex:
    """Bit-sliced bitmaps + packed rows of an AppendStore, refreshed incrementally. N_rep = rows represented."""

    def __init__(self, store, cap_rows=1 << 16):
        self.store = store
        self.n = store.n
        self.N_rep = 0
        self.Z = np.zeros((self.n, 3, n_words(cap_rows)), np.uint64)
        self.O = np.zeros((self.n, n_words(cap_rows)), np.uint64)
        self.W = row_words(self.n)
        self.rows = np.zeros((cap_rows, self.W), np.uint64)
        self.refresh()

    def _grow(self, need_rows):
        if need_rows <= self.rows.shape[0]:
            return
        new = max(need_rows, 2 * self.rows.shape[0])
        nw_old, nw = self.Z.shape[2], n_words(new)
        Z = np.zeros((self.n, 3, nw), np.uint64)
        Z[:, :, :nw_old] = self.Z
        O = np.zeros((self.n, nw), np.uint64)
        O[:, :nw_old] = self.O
        R = np.zeros((new, self.W), np.uint64)
        R[:self.N_rep] = self.rows[:self.N_rep]
        self.Z, self.O, self.rows = Z, O, R

    def refresh(self):
        N = self.store.N
        if N == self.N_rep:
            return 0
        assert N > self.N_rep, "append-only store shrank"
        self._grow(N)
        b = self.store.bases[self.N_rep:N]
        o = self.store.outcomes[self.N_rep:N]
        _set_bits(self.Z, self.O, b, o, self.N_rep)
        _pack_into(self.rows, b, o, self.N_rep, self.W)
        k = N - self.N_rep
        self.N_rep = N
        return k


# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _bitmap_range(Z, O, ptr, q, b, r0, r1, sel=False):
    """Exact (m, s) of every term over rows [r0, r1): first and last word masked (GAP-1 of the C-INDEX R1 audit).
    sel=True: counts only (outcome planes never read; the returned s equals m and must not be used)."""
    T = ptr.shape[0] - 1
    M = np.zeros(T, np.int64)
    S = np.zeros(T, np.int64)
    if r1 <= r0:
        return M, S
    w0 = r0 >> 6
    wl = (r1 - 1) >> 6
    full = ~np.uint64(0)
    m_first = full << np.uint64(r0 & 63)
    tail = r1 - (wl << 6)  # 1..64 bits valid in the last word
    m_last = full if tail == 64 else (np.uint64(1) << np.uint64(tail)) - np.uint64(1)
    for t in prange(T):
        j0 = ptr[t]
        j1 = ptr[t + 1]
        m = 0
        neg = 0
        for w in range(w0, wl + 1):
            a = full
            if w == w0:
                a &= m_first
            if w == wl:
                a &= m_last
            x = np.uint64(0)
            for j in range(j0, j1):
                a &= Z[q[j], b[j], w]
                if a == 0:
                    break
                if not sel:
                    x ^= O[q[j], w]
            if a != 0:
                m += popcnt64(a)
                if not sel:
                    neg += popcnt64(a & x)
        M[t] = m
        S[t] = m - 2 * neg
    return M, S


class TermCache:
    """Exact answers per term key with an append watermark r: (m, s) over rows [0, r). `answer` refreshes the index,
    evaluates only [r, N) for cached terms (and [0, N) for new ones) with the chosen delta evaluator ("bitmap" or
    "sstar"), then advances watermarks. Keys are (support, bases) tuples."""

    def __init__(self, index, evaluator="bitmap", block=2048, select_only=False):
        self.ix = index
        self.ev = evaluator
        self.block = block
        self.sel = select_only  # counts-only cache (checking pipeline); s is then meaningless
        self.slot = {}  # key -> slot (A14: per-slot arrays instead of per-key lists)
        self.r = np.zeros(0, np.int64)  # watermark per slot
        self.m = np.zeros(0, np.int64)
        self.s = np.zeros(0, np.int64)
        self._memo = []  # [(terms object, slots per term)] for the last few term batches (held refs: ids stay valid)

    def _eval(self, terms, r0, r1):
        if self.ev == "bitmap":
            return _bitmap_range(self.ix.Z, self.ix.O, terms.ptr, terms.q, terms.b, r0, r1, self.sel)
        enc = encode_terms_for_scan(terms, self.ix.n, self.sel)
        return _sstar(self.ix.rows[r0:r1], *enc, self.block, self.sel)

    def _slots(self, terms):
        for obj, sl in self._memo:
            if obj is terms:
                return sl
        sl = np.empty(terms.T, np.int64)
        for t in range(terms.T):
            qq, bb = terms.term(t)
            k = (tuple(int(x) for x in qq), tuple(int(x) for x in bb))
            if k not in self.slot:
                self.slot[k] = len(self.slot)
            sl[t] = self.slot[k]
        grow = len(self.slot) - len(self.r)
        if grow > 0:
            z = np.zeros(grow, np.int64)
            self.r, self.m, self.s = np.concatenate([self.r, z]), np.concatenate([self.m, z]), np.concatenate([self.s, z])
        self._memo = ([(terms, sl)] + self._memo)[:8]
        return sl

    def answer(self, terms):
        self.ix.refresh()
        N = self.ix.N_rep
        sl = self._slots(terms)
        # one evaluation per DISTINCT key (a batch may repeat a term), grouped by watermark so each delta range is
        # evaluated once per group
        uniq, first = np.unique(sl, return_index=True)
        rr = self.r[uniq]
        for r in np.unique(rr[rr < N]):
            sel = rr == r
            ts = np.sort(first[sel])
            sub = terms.subset(ts)
            dm, ds = self._eval(sub, int(r), N)
            u = sl[ts]
            self.m[u] += np.asarray(dm, np.int64)
            self.s[u] += np.asarray(ds, np.int64)
            self.r[u] = N  # advanced after the complete delta
        return self.m[sl].copy(), self.s[sl].copy()


class IncrementalViews:
    """Marginal views (views.ViewSet) bound to an AppendStore with the same freshness contract as IncrementalIndex:
    N_rep = rows represented; refresh() adds rows [N_rep, N) to every view (one increment per view per row) and
    invalidates the prepared transforms (re-prepared lazily on the next answer). Every answer refreshes first."""

    def __init__(self, store, supports):
        from .views import ViewSet
        self.store = store
        self.vs = ViewSet(supports)
        self.N_rep = 0
        self.refresh()

    def refresh(self):
        N = self.store.N
        if N == self.N_rep:
            return 0
        assert N > self.N_rep, "append-only store shrank"
        self.vs.append(self.store.bases[self.N_rep:N], self.store.outcomes[self.N_rep:N])  # also resets prepared W
        k = N - self.N_rep
        self.N_rep = N
        return k

    def answer(self, terms):
        """(covered term indices, m, s) for the terms some view covers; uncovered terms go to another path."""
        self.refresh()
        a = self.vs.assign(terms)
        enc = self.vs.encode(terms, a)
        return self.vs.answer(enc)
