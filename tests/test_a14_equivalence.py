"""A14 (H-b′3 repair #1): the accelerated pipeline returns the same decisions as the reference implementations."""
import math
import os

import numpy as np

from shadowopt.certify import (BatchCSR, CountSource, _ginv, _g, certify, complete, complete_reference,
                               fast_group_terms)
from shadowopt.cost import hw1_fitted
from shadowopt.planner import group_terms
from shadowopt.store import TermBatch
from shadowopt.workloads import w1_energy, w3_local_fidelities, w4_zipf, w2_correlators

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _subset_reference(U, idx):
    qs, bs, ptr = [], [], [0]
    for t in idx:
        qq, bb = U.term(t)
        qs.append(qq)
        bs.append(bb)
        ptr.append(ptr[-1] + len(qq))
    return (np.array(ptr, np.int64), np.concatenate(qs).astype(np.int64) if qs else np.zeros(0, np.int64),
            np.concatenate(bs).astype(np.int64) if bs else np.zeros(0, np.int64))


def _lih():
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    U = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
    return w1_energy(U, 0.0, eps_seq=(0.1,))


def test_ginv_inverts_g():
    for y in np.exp(np.linspace(np.log(1e-250), np.log(1.17), 4000)):
        x = _ginv(float(y))
        assert abs(_g(x) / y - 1) < 1e-12, (y, x)


def test_subset_identical():
    rng = np.random.default_rng(3)
    U = _lih().universe
    for _ in range(20):
        idx = rng.integers(0, U.T, size=int(rng.integers(0, 50)))
        s = U.subset(idx)
        p, q, b = _subset_reference(U, idx)
        assert np.array_equal(s.ptr, p) and np.array_equal(s.q, q) and np.array_equal(s.b, b)
        assert np.array_equal(s.coeff, U.coeff[idx])


def test_grouping_identical():
    rng = np.random.default_rng(4)
    for fam in (_lih(), w3_local_fidelities(50), w4_zipf(20, seed=1)):
        U = fam.universe
        w = U.weights()
        for r in range(6):
            terms = list(rng.choice(U.T, size=min(U.T, int(rng.integers(1, 400))), replace=False))
            key = None if r == 0 else (lambda u, perm=rng.permutation(U.T): perm[u])
            a, b = group_terms(U, terms, key), fast_group_terms(U, terms, key)
            assert len(a) == len(b)
            for (s1, m1), (s2, m2) in zip(a, b):
                assert np.array_equal(s1, s2) and list(m1) == list(m2)


def test_completion_identical_to_reference():
    rng = np.random.default_rng(6)
    cost = hw1_fitted()
    fams = [(_lih(), 0), (w3_local_fidelities(50), 0), (w4_zipf(20, seed=1), 2), (w2_correlators(20), 0)]
    for fam, tau in fams:
        csr = BatchCSR(fam.batches[tau])
        for trial in range(4):
            hi = int(rng.choice([50, 2000, 60000]))
            m = rng.integers(0, hi, size=len(csr.terms))
            hw, ok = certify(csr, m)
            for c in (None, cost):
                p1, m1 = complete_reference(fam.universe, csr, m, hw, ok, cost=c)
                p2, m2 = complete(fam.universe, csr, m, hw, ok, cost=c)
                assert np.array_equal(m1, m2), (trial, c)
                assert len(p1) == len(p2)
                for (s1, k1), (s2, k2) in zip(p1, p2):
                    assert np.array_equal(s1, s2) and k1 == k2
                _, ok2 = certify(csr, m2)
                assert np.all(ok2)


def test_cache_answers_unchanged_after_repeated_batches():
    from shadowopt.engine import AppendStore, IncrementalIndex
    rng = np.random.default_rng(8)
    fam = w4_zipf(20, seed=1)
    U = fam.universe
    app = AppendStore(U.n)
    ix = IncrementalIndex(app)
    cached = CountSource(ix, "bitmap", True)
    plain = CountSource(ix, "sstar", False, block=61)
    for tau, k in enumerate((300, 0, 4000, 17, 0, 2500)):
        if k:
            app.append(rng.integers(0, 3, (k, U.n)), rng.integers(0, 2, (k, U.n)))
        csr = BatchCSR(fam.batches[tau % len(fam.batches)])
        for _ in range(2):  # the same csr twice: memoized subset + slots
            assert np.array_equal(cached.counts(U, csr), plain.counts(U, csr))
