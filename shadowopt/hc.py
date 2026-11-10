"""H-c replicate machinery (prereg/M1.md §6; C-BOUND Lemmas 1–2). One replicate = one independent run of the engine's
decisions on a fresh random store with fresh outcomes:

- random store: N0 uniform rows (fresh basis seed), outcomes sampled per row by the state's sampler;
- per batch: counts (index counts of the random rows + exact counts of the targeted settings acquired so far) →
  certify.check (compiled optimized split) → certify.complete plan when needed (counts only, B2);
- targeted acquisition: a setting repeated s times is sampled EXACTLY as one multinomial draw of its outcome histogram
  (i.i.d. shots of the same measurement) — statevector states only (n <= 20); per-row sampling otherwise;
- answers: (m, s) = random-row part from the bitmap path + targeted part from the histograms; checkpoint misses
  |answer - truth| > HW recorded; m = 0 recorded as "unavailable" (never as a miss).
`validate_against_engine` checks, on a small case, that the counts and decisions equal the real engine's (seq_runner)
and that the multinomial path reproduces per-row sampling in distribution (two-sample test on a term's parity sum).
"""
import numpy as np
from numba import njit, prange

from .engine import AppendStore, IncrementalIndex, _bitmap_range
from .certify import BatchCSR, certify, complete
from .planner import setting_counts
from .sampling import uniform_bases


@njit(cache=True)
def _parity_from_hist(hist, masks):
    """s for each mask: sum_o hist[o] (-1)^{popcount(o & mask)}  (outcome index o: bit q = outcome of qubit q)."""
    T = masks.shape[0]
    out = np.zeros(T, np.int64)
    for t in range(T):
        mk = masks[t]
        s = 0
        for o in range(hist.shape[0]):
            h = hist[o]
            if h == 0:
                continue
            x = o & mk
            c = 0
            while x:
                x &= x - 1
                c += 1
            s += h if (c & 1) == 0 else -h
        out[t] = s
    return out


def setting_probs(psi, n, setting):
    """Outcome distribution of measuring the n-qubit statevector psi (little-endian: bit q of the index = qubit q) in the
    basis `setting` (0 = X, 1 = Y, 2 = Z)."""
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    Sdg = np.array([[1, 0], [0, -1j]], complex)
    rot = {0: H, 1: H @ Sdg, 2: np.eye(2, dtype=complex)}
    a = np.asarray(psi, complex).reshape([2] * n)  # axis k <-> qubit n-1-k (little-endian index)
    for q in range(n):
        if setting[q] == 2:
            continue
        ax = n - 1 - q
        a = np.moveaxis(np.tensordot(rot[int(setting[q])], a, axes=([1], [ax])), 0, ax)
    p = np.abs(a.reshape(-1)) ** 2
    return p / p.sum()


class TargetedAggregates:
    """Exact per-term (m, s) of targeted acquisitions, sampled by multinomial histograms (statevector states)."""

    def __init__(self, universe, psi, n, sampler=None):
        """psi given: exact multinomial histograms; psi None: per-row sampling with `sampler` (e.g. GHZ, n = 50)."""
        self.U, self.psi, self.n, self.sampler = universe, psi, n, sampler
        self.m = np.zeros(universe.T, np.int64)
        self.s = np.zeros(universe.T, np.int64)
        self._p = {}

    def acquire(self, plan, rng):
        from .planner import compat_setting
        for setting, shots in plan:
            if shots <= 0:
                continue
            key = tuple(int(x) for x in setting)
            if self.psi is None:
                comp = np.nonzero(compat_setting(self.U, np.array(key)))[0]
                B = np.tile(np.array(key, np.uint8)[None, :], (int(shots), 1))
                O = self.sampler(B, rng).astype(np.int64)
                for t in comp:
                    qq = self.U.term(int(t))[0]
                    par = O[:, qq].sum(axis=1) & 1
                    self.m[t] += int(shots)
                    self.s[t] += int(shots) - 2 * int(par.sum())
                continue
            if key not in self._p:
                self._p[key] = setting_probs(self.psi, self.n, np.array(key))
            hist = rng.multinomial(int(shots), self._p[key]).astype(np.int64)
            comp = np.nonzero(compat_setting(self.U, np.array(key)))[0]
            masks = np.array([int(sum(1 << int(q) for q in self.U.term(int(t))[0])) for t in comp], np.int64)
            self.m[comp] += int(shots)
            self.s[comp] += _parity_from_hist(hist, masks)


def run_replicate(fam, truth, n, sampler, psi, N0, seed, checkpoints):
    """One replicate. checkpoints: set of (batch index, query index). Returns list of (tau, qi, err, hw, m_min)."""
    U = fam.universe
    rb = np.random.default_rng([seed, 1])
    ro = np.random.default_rng([seed, 2])
    rt = np.random.default_rng([seed, 3])
    B = uniform_bases(rb, N0, n)
    O = sampler(B, ro)
    app = AppendStore(n, cap=max(N0, 1))
    app.append(B, O)
    ix = IncrementalIndex(app, cap_rows=max(N0, 1))
    tg = TargetedAggregates(U, psi, n, sampler)
    out = []
    for tau, batch in enumerate(fam.batches):
        csr = BatchCSR(batch)
        sub = U.subset(csr.terms)
        mR, sR = _bitmap_range(ix.Z, ix.O, sub.ptr, sub.q, sub.b, 0, ix.N_rep, False)
        m = mR + tg.m[csr.terms]
        hw, ok = certify(csr, m)
        if not np.all(ok):
            plan, _ = complete(U, csr, m, hw, ok)
            tg.acquire(plan, rt)
            m = mR + tg.m[csr.terms]
            hw, ok = certify(csr, m)
            assert np.all(ok)
        s = sR + tg.s[csr.terms]
        mu = np.where(m > 0, s / np.maximum(m, 1), 0.0)
        for qi, q in enumerate(batch):
            if (tau, qi) not in checkpoints:
                continue
            a, b = csr.qptr[qi], csr.qptr[qi + 1]
            slots = csr.qterm[a:b]
            mm = int(np.min(m[slots]))
            est = float(np.sum(csr.qcoef[a:b] * mu[slots])) + q.const
            tru = float(np.sum(csr.qcoef[a:b] * truth[csr.terms[slots]])) + q.const
            out.append((tau, qi, abs(est - tru), float(hw[qi]), mm))
    return out
