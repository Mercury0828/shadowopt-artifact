"""Correctness tests for the store, samplers, truth and every access path (run: python -m pytest -q tests)."""
import numpy as np
import pytest

from shadowopt.store import Store, pack_rows, build_bitmaps, terms_from_pairs, terms_from_strings
from shadowopt.scan import naive_reference, SStar
from shadowopt.bitmaps import BitmapPath
from shadowopt.sharedplan import SharedPlanPath
from shadowopt.views import ViewSet, ViewPath, select_views
from shadowopt.sampling import (uniform_bases, sample_statevector, MPSSampler, sample_ghz, ROT)
from shadowopt.states import (pauli_expectation, statevector_to_mps, mps_to_statevector, ghz_expectation,
                              ghz_statevector)


def random_terms(rng, n, T, kmax):
    pairs = []
    for _ in range(T):
        k = int(rng.integers(1, kmax + 1))
        qs = rng.choice(n, size=k, replace=False)
        pairs.append([(int(q), int(rng.integers(0, 3))) for q in qs])
    return terms_from_pairs(n, pairs)


def random_store(rng, N, n):
    st = Store(n)
    st.append(uniform_bases(rng, N, n), rng.integers(0, 2, size=(N, n), dtype=np.uint8))
    return st


@pytest.mark.parametrize("n,N", [(12, 5000), (20, 4099), (50, 3001), (70, 1500)])
def test_paths_bit_identical(n, N):
    rng = np.random.default_rng(n * 7 + N)
    st = random_store(rng, N, n)
    terms = random_terms(rng, n, 400, min(n, 10))
    M0, S0 = naive_reference(st, terms)
    rows = pack_rows(st)
    Z, O = build_bitmaps(st)
    for path in (SStar(rows, n, block=257), BitmapPath(Z, O, N, chunk=7), SharedPlanPath(Z, O, N, chunk=5)):
        M, S = path.run(terms)
        assert np.array_equal(M, M0) and np.array_equal(S, S0), type(path)
    sup = [frozenset(terms.term(t)[0].tolist()) for t in range(terms.T)]
    chosen, _ = select_views(sup, [1] * len(sup), (N + 63) // 64, cap_bytes=10 ** 7, s_max=5)
    vs = ViewSet(chosen).build(st)
    vp = ViewPath(vs, SharedPlanPath(Z, O, N))
    M, S = vp.run(terms)
    assert np.array_equal(M, M0) and np.array_equal(S, S0)
    assert (vs.assign(terms) >= 0).sum() > 0


def test_negative_control_dropped_bit():
    """A path that drops one bit of the mask must fail the bit-identical check."""
    rng = np.random.default_rng(1)
    n, N = 12, 3000
    st = random_store(rng, N, n)
    terms = random_terms(rng, n, 200, 6)
    M0, S0 = naive_reference(st, terms)
    Z, O = build_bitmaps(st)
    # drop one bit that matters: a row compatible with the first term with m > 0, on that term's first literal
    t0 = int(np.nonzero(M0 > 0)[0][0])
    qq, bb = terms.term(t0)
    r = int(np.nonzero(np.all(st.bases[:, qq] == bb.astype(np.uint8), axis=1))[0][0])
    Zbad = Z.copy()
    Zbad[qq[0], bb[0], r >> 6] &= ~(np.uint64(1) << np.uint64(r & 63))
    M, S = BitmapPath(Zbad, O, N).run(terms)
    assert not (np.array_equal(M, M0) and np.array_equal(S, S0))


def test_view_maintenance_under_append():
    rng = np.random.default_rng(2)
    n = 10
    st = random_store(rng, 2000, n)
    terms = random_terms(rng, n, 100, 4)
    vs = ViewSet([(0, 1, 2, 3), (2, 4, 6, 8, 9)]).build(st)
    nb = uniform_bases(rng, 777, n)
    no = rng.integers(0, 2, size=(777, n), dtype=np.uint8)
    st.append(nb, no)
    vs.append(nb, no)
    Z, O = build_bitmaps(st)
    M0, S0 = naive_reference(st, terms)
    M, S = ViewPath(vs, SharedPlanPath(Z, O, st.N)).run(terms)
    assert np.array_equal(M, M0) and np.array_equal(S, S0)


def dense_pauli(n, qq, bb):
    P = {0: np.array([[0, 1], [1, 0]]), 1: np.array([[0, -1j], [1j, 0]]), 2: np.diag([1, -1]), -1: np.eye(2)}
    ops = [-1] * n
    for q, b in zip(qq, bb):
        ops[q] = b
    M = np.array([[1.0 + 0j]])
    for q in range(n):  # little-endian: qubit q is bit q -> kron order reversed
        M = np.kron(P[ops[q]], M)
    return M


def test_pauli_expectation_dense():
    rng = np.random.default_rng(3)
    n = 4
    psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi /= np.linalg.norm(psi)
    for _ in range(60):
        k = int(rng.integers(1, n + 1))
        qq = sorted(rng.choice(n, size=k, replace=False).tolist())
        bb = rng.integers(0, 3, size=k).tolist()
        ref = np.real(np.vdot(psi, dense_pauli(n, qq, bb) @ psi))
        assert abs(pauli_expectation(psi, qq, bb) - ref) < 1e-12


def exact_probs(psi, n, b):
    v = psi.reshape([2] * n)  # axes q_{n-1}..q0
    for q in range(n):
        ax = n - 1 - q
        v = np.moveaxis(np.tensordot(ROT[b[q]], v, axes=([1], [ax])), 0, ax)
    return np.abs(v.reshape(-1)) ** 2


def _check_sampler(sample_fn, psi, n, rng, nb=12, shots=20000):
    for _ in range(nb):
        b = rng.integers(0, 3, size=n).astype(np.uint8)
        bases = np.repeat(b[None, :], shots, axis=0)
        out = sample_fn(bases)
        idx = (out.astype(np.int64) << np.arange(n)).sum(axis=1)
        emp = np.bincount(idx, minlength=2 ** n) / shots
        p = exact_probs(psi, n, b)
        tv = 0.5 * np.abs(emp - p).sum()
        assert tv < 0.03, (b, tv)


def test_statevector_sampler():
    rng = np.random.default_rng(4)
    n = 4
    psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi /= np.linalg.norm(psi)
    _check_sampler(lambda B: sample_statevector(psi, n, B, rng), psi, n, rng)


def test_mps_roundtrip_and_sampler():
    rng = np.random.default_rng(5)
    n = 6
    psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi /= np.linalg.norm(psi)
    tens, disc = statevector_to_mps(psi, n, chi_max=64, cutoff=1e-14)
    back = mps_to_statevector(tens)
    assert abs(abs(np.vdot(back, psi)) - 1) < 1e-10
    sampler = MPSSampler(tens)
    _check_sampler(lambda B: sampler.sample(B, rng), back, n, rng)


def test_ghz_sampler_and_truth():
    rng = np.random.default_rng(6)
    n = 5
    psi = ghz_statevector(n)
    _check_sampler(lambda B: sample_ghz(B, rng), psi, n, rng, nb=20)
    for _ in range(300):
        k = int(rng.integers(1, n + 1))
        qq = sorted(rng.choice(n, size=k, replace=False).tolist())
        bb = rng.integers(0, 3, size=k).tolist()
        assert abs(ghz_expectation(n, qq, bb) - pauli_expectation(psi, qq, bb)) < 1e-12
    # all-X/Y full-support strings
    for _ in range(50):
        bb = rng.integers(0, 2, size=n).tolist()
        assert abs(ghz_expectation(n, list(range(n)), bb) - pauli_expectation(psi, list(range(n)), bb)) < 1e-12


@pytest.mark.parametrize("n", [12, 20, 50, 70])
def test_sstar_select_only_and_multiword(n):
    rng = np.random.default_rng(n)
    st = random_store(rng, 3001, n)
    terms = random_terms(rng, n, 500, min(n, 10))
    M0, S0 = naive_reference(st, terms)
    ss = SStar(pack_rows(st), n, block=300)
    M, S = ss.run(terms)
    Ms, _ = ss.run(terms, select_only=True)
    assert np.array_equal(M, M0) and np.array_equal(S, S0)
    assert np.array_equal(Ms, M0)


def test_sstar_fast_encoding_equal():
    from shadowopt.scan import SStarFast, encode_terms_for_scan, _encode_nb
    from shadowopt.store import row_words
    for n in (12, 20, 50, 70):
        rng = np.random.default_rng(100 + n)
        st = random_store(rng, 2000, n)
        terms = random_terms(rng, n, 300, min(n, 10))
        for sel in (False, True):
            a = encode_terms_for_scan(terms, n, sel)
            b = _encode_nb(terms.ptr, terms.q, terms.b, n, row_words(n), sel)
            assert all(np.array_equal(x, y) for x, y in zip(a, b))
        M0, S0 = naive_reference(st, terms)
        M, S = SStarFast(pack_rows(st), n).run(terms)
        assert np.array_equal(M, M0) and np.array_equal(S, S0)
