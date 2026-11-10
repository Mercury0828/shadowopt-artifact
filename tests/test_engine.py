"""Engine layer: incremental index and term cache are bit-identical to the naive reference across appends that end
mid-word, stale-cache refreshes and repeated batches (C-INDEX R1 audit GAP-1/GAP-2)."""
import numpy as np

from shadowopt.engine import AppendStore, IncrementalIndex, TermCache, _bitmap_range
from shadowopt.store import Store, TermBatch, terms_from_pairs
from shadowopt.scan import naive_reference


def _rand_terms(rng, n, T, kmax=4):
    pairs = []
    for _ in range(T):
        k = int(rng.integers(1, kmax + 1))
        qs = sorted(rng.choice(n, size=k, replace=False).tolist())
        pairs.append([(q, int(rng.integers(0, 3))) for q in qs])
    return terms_from_pairs(n, pairs)


def _ref(app, terms):
    st = Store(app.n, app.bases.copy(), app.outcomes.copy(), np.zeros(app.N, np.int32), np.zeros(app.N, np.uint8))
    return naive_reference(st, terms)


def test_cache_and_index_match_reference_across_appends():
    rng = np.random.default_rng(0)
    n = 12
    app = AppendStore(n, cap=100)
    ix = IncrementalIndex(app, cap_rows=64)
    terms = _rand_terms(rng, n, 200)
    cb = TermCache(ix, "bitmap")
    cs = TermCache(ix, "sstar", block=37)
    for k in (1, 63, 64, 65, 1000, 7, 4096, 129, 3):  # word-boundary and mid-word appends; capacity growth
        app.append(rng.integers(0, 3, (k, n)), rng.integers(0, 2, (k, n)))
        sub = terms.subset(np.sort(rng.choice(terms.T, size=120, replace=False)))  # partial overlap -> mixed watermarks
        Mr, Sr = _ref(app, sub)
        for c in (cb, cs):
            M, S = c.answer(sub)
            assert np.array_equal(M, Mr) and np.array_equal(S, Sr)
        M2, S2 = cb.answer(sub)  # repeat without append: pure cache hit
        assert np.array_equal(M2, Mr) and np.array_equal(S2, Sr)


def test_range_kernel_boundaries():
    rng = np.random.default_rng(1)
    n = 6
    app = AppendStore(n)
    app.append(rng.integers(0, 3, (300, n)), rng.integers(0, 2, (300, n)))
    ix = IncrementalIndex(app)
    terms = _rand_terms(rng, n, 50, kmax=2)
    full_M, full_S = _ref(app, terms)
    for cut in (0, 1, 63, 64, 65, 127, 128, 200, 299, 300):
        a = _bitmap_range(ix.Z, ix.O, terms.ptr, terms.q, terms.b, 0, cut)
        bb = _bitmap_range(ix.Z, ix.O, terms.ptr, terms.q, terms.b, cut, 300)
        assert np.array_equal(a[0] + bb[0], full_M) and np.array_equal(a[1] + bb[1], full_S)


def test_negative_control_unmasked_range_fails():
    """Evaluating whole words without the boundary mask must double count (the audit's failing case)."""
    rng = np.random.default_rng(2)
    n = 4
    app = AppendStore(n)
    app.append(rng.integers(0, 3, (130, n)), rng.integers(0, 2, (130, n)))
    ix = IncrementalIndex(app)
    terms = _rand_terms(rng, n, 30, kmax=1)
    a = _bitmap_range(ix.Z, ix.O, terms.ptr, terms.q, terms.b, 0, 100)
    wrong = _bitmap_range(ix.Z, ix.O, terms.ptr, terms.q, terms.b, 64, 130)  # word-aligned start, overlaps 64..99
    full = _ref(app, terms)[0]
    assert not np.array_equal(a[0] + wrong[0], full)


def test_multiword_rows_and_interior_ranges():
    """n = 50 and 100 (rows span several packed words); ranges strictly inside one 64-bit word."""
    rng = np.random.default_rng(3)
    for n in (50, 100):
        app = AppendStore(n, cap=10)
        ix = IncrementalIndex(app, cap_rows=10)
        terms = _rand_terms(rng, n, 80, kmax=5)
        cs = TermCache(ix, "sstar", block=50)
        cb = TermCache(ix, "bitmap")
        for k in (5, 70, 300, 1):
            app.append(rng.integers(0, 3, (k, n)), rng.integers(0, 2, (k, n)))
            Mr, Sr = _ref(app, terms)
            for c in (cs, cb):
                M, S = c.answer(terms)
                assert np.array_equal(M, Mr) and np.array_equal(S, Sr)
        full = _ref(app, terms)
        parts = [(0, 3), (3, 17), (17, 63), (63, 64), (64, 70), (70, app.N)]
        acc = [np.zeros(terms.T, np.int64), np.zeros(terms.T, np.int64)]
        for r0, r1 in parts:
            a = _bitmap_range(ix.Z, ix.O, terms.ptr, terms.q, terms.b, r0, r1)
            acc[0] += a[0]
            acc[1] += a[1]
        assert np.array_equal(acc[0], full[0]) and np.array_equal(acc[1], full[1])


def test_incremental_views_fresh_across_appends():
    from shadowopt.engine import IncrementalViews
    rng = np.random.default_rng(7)
    n = 10
    app = AppendStore(n)
    terms = _rand_terms(rng, n, 120, kmax=3)
    iv = IncrementalViews(app, [tuple(range(0, 5)), tuple(range(4, 9)), (1, 3, 7, 9)])
    for k in (100, 1, 63, 2000):
        app.append(rng.integers(0, 3, (k, n)), rng.integers(0, 2, (k, n)))
        idx, M, S = iv.answer(terms)
        Mr, Sr = _ref(app, terms)
        assert len(idx) > 0
        assert np.array_equal(M, Mr[idx]) and np.array_equal(S, Sr[idx])
