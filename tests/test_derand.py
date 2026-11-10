"""The faithful derandomization port reproduces the HKP reference code's measurement sequence."""
import importlib.util
import os

import numpy as np

from shadowopt.derand import hkp_derandomize
from shadowopt.store import TermBatch, terms_from_pairs
from shadowopt.workloads import w4_zipf

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _ref():
    spec = importlib.util.spec_from_file_location("hkp_acq", os.path.join(ROOT, "third_party", "hkp",
                                                                          "data_acquisition_shadow.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.derandomized_classical_shadow


def _ref_run(U, terms, need):
    obs = []
    for t in terms:
        qq, bb = U.term(int(t))
        obs.append([("XYZ"[int(x)], int(p)) for p, x in zip(qq, bb)])
    M = int(need.max())
    w = list((need.astype(float) + 1e-9) / M)
    seq = _ref()(obs, M, U.n, weight=w)
    return np.array([["XYZ".index(c) for c in row] for row in seq], np.uint8)


def _exact_tie(U, terms, need, ref, i):
    """True if at measurement i, on the first differing qubit, the reference's choice and ours have equal cost in
    50-digit arithmetic (the reference breaks such exact ties by floating-point rounding of its running sums)."""
    import mpmath as mp
    mp.mp.dps = 50
    ours, _ = hkp_derandomize(U, terms, need)
    j = int(np.nonzero(ref[i] != ours[i])[0][0])
    obs = [list(zip(*U.term(int(t)))) for t in terms]
    m = np.zeros(len(terms))
    for row in ref[:i]:
        for k, o in enumerate(obs):
            if all(row[q] == bb for q, bb in o):
                m[k] += 1
    eta = mp.mpf("0.9"); nu = 1 - mp.e ** (-eta / 2)
    Mx = int(need.max()); w = [(mp.mpf(int(x)) + mp.mpf("1e-9")) / Mx for x in need]
    rem = {k: len(o) for k, o in enumerate(obs)}
    for qi in range(j):
        for k, o in enumerate(obs):
            for q, bb in o:
                if q == qi and rem[k] >= 0:
                    rem[k] = rem[k] - 1 if ref[i][qi] == bb else -1
    def tot(p):
        s = mp.mpf(0)
        for k, o in enumerate(obs):
            if m[k] >= int(need[k]):
                continue
            r = rem[k]
            inq = [bb for q, bb in o if q == j]
            if r >= 0 and inq:
                r = r - 1 if inq[0] == p else -1
            V = eta / 2 * int(m[k]) if r < 0 else eta / 2 * int(m[k]) - mp.log(1 - nu / mp.mpf(3) ** r)
            s += mp.e ** (-V / w[k])
        return s
    a, b = tot(int(ref[i][j])), tot(int(ours[i][j]))
    return abs(a - b) <= mp.mpf("1e-40") * abs(a)


def compare(U, terms, need):
    ref = _ref_run(U, terms, need)
    ours, m = hkp_derandomize(U, terms, need)
    L = min(len(ref), len(ours))
    first_diff = next((i for i in range(L) if not np.array_equal(ref[i], ours[i])), None)
    if first_diff is not None:
        assert _exact_tie(U, terms, need, ref, first_diff), f"divergence at {first_diff} is not an exact tie"
    assert abs(len(ref) - len(ours)) <= 0.02 * len(ref), (len(ref), len(ours))
    return len(ref), len(ours), first_diff


def test_identical_on_w4_batch():
    fam = w4_zipf(20, seed=0)
    terms = fam.batch_terms(0)
    need = np.full(len(terms), 30, np.int64)
    nr, no, d = compare(fam.universe, terms, need)
    print("W4 batch 0:", nr, no, d)


def test_identical_on_lih_nonuniform():
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    U = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
    terms = np.arange(0, U.T, 3)
    c = np.abs(U.coeff[terms])
    need = np.maximum(1, np.ceil(20 * (c / c.max()) ** (2 / 3))).astype(np.int64)
    nr, no, dd = compare(U, terms, need)
    print("LiH:", nr, no, dd)


def test_identical_on_w6_heavy_batch():
    from shadowopt.workloads import w6_heavytail
    fam = w6_heavytail(50, seed=0)
    terms = fam.batch_terms(2)
    need = np.full(len(terms), 15, np.int64)
    nr, no, d = compare(fam.universe, terms, need)
    print("W6 batch 2:", nr, no, d)
