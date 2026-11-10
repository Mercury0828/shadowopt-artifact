"""Scan baselines: the naive reference (correctness oracle), the strongest scan S*, and wrappers for the HKP reference
code (G2) and PennyLane's pauli_expval (G4).

Every path returns, for each term P of a batch, (m_P, s_P) with m_P = |C(P)| and s_P = sum_{t in C(P)} (-1)^{parity_t}.
"""
import importlib.util
import os

import numpy as np
from numba import njit, prange

from ._bits import popcnt64
from .store import row_words


# ----------------------------------------------------------------------------------------------------------------------
# Naive reference (numpy, per term, on the raw uint8 arrays; deliberately independent of every packed layout)
# ----------------------------------------------------------------------------------------------------------------------
def naive_reference(store, terms):
    M = np.zeros(terms.T, np.int64)
    S = np.zeros(terms.T, np.int64)
    for t in range(terms.T):
        qq, bb = terms.term(t)
        if len(qq) == 0:
            M[t] = store.N
            S[t] = store.N
            continue
        mask = np.all(store.bases[:, qq] == bb[None, :].astype(np.uint8), axis=1)
        m = int(mask.sum())
        par = (store.outcomes[mask][:, qq].sum(axis=1) & 1)
        M[t] = m
        S[t] = m - 2 * int(par.sum())
    return M, S


# ----------------------------------------------------------------------------------------------------------------------
# S*: bit-packed row-major scan, all terms of the batch in one pass over the store, row blocks shared by all terms.
# Term encoding: for each term, the list of row words it touches, with (compare mask, compare value, parity mask).
# ----------------------------------------------------------------------------------------------------------------------
def encode_terms_for_scan(terms, n, select_only=False):
    """select_only: keep only the words carrying basis bits (compare mask != 0); parity masks are dropped."""
    W = row_words(n)
    tw_ptr = [0]
    tw_word, tw_cm, tw_cv, tw_pm = [], [], [], []
    for t in range(terms.T):
        cm = np.zeros(W, np.uint64)
        cv = np.zeros(W, np.uint64)
        pm = np.zeros(W, np.uint64)
        qq, bb = terms.term(t)
        for qi, bi in zip(qq, bb):
            bit = 2 * int(qi)
            cm[bit >> 6] |= np.uint64(3) << np.uint64(bit & 63)
            cv[bit >> 6] |= np.uint64(int(bi) + 1) << np.uint64(bit & 63)
            ob = 2 * n + int(qi)
            pm[ob >> 6] |= np.uint64(1) << np.uint64(ob & 63)
        if select_only:
            pm[:] = 0
        for w in range(W):
            if cm[w] or pm[w]:
                tw_word.append(w)
                tw_cm.append(cm[w])
                tw_cv.append(cv[w])
                tw_pm.append(pm[w])
        tw_ptr.append(len(tw_word))
    return (np.array(tw_ptr, np.int64), np.array(tw_word, np.int64), np.array(tw_cm, np.uint64),
            np.array(tw_cv, np.uint64), np.array(tw_pm, np.uint64))


@njit(parallel=True, cache=True, fastmath=True)
def _sstar(rows, tw_ptr, tw_word, tw_cm, tw_cv, tw_pm, block, select_only):
    N = rows.shape[0]
    T = tw_ptr.shape[0] - 1
    nb = (N + block - 1) // block
    Mp = np.zeros((nb, T), np.int64)
    Np = np.zeros((nb, T), np.int64)
    one = np.uint64(1)
    for bi in prange(nb):
        r0 = bi * block
        r1 = min(N, r0 + block)
        for t in range(T):
            j0 = tw_ptr[t]
            j1 = tw_ptr[t + 1]
            m = 0
            neg = 0
            if j1 - j0 == 1:
                w = tw_word[j0]
                cm = tw_cm[j0]
                cv = tw_cv[j0]
                pm = tw_pm[j0]
                if select_only:
                    for r in range(r0, r1):
                        m += (rows[r, w] & cm) == cv
                else:
                    for r in range(r0, r1):
                        x = rows[r, w]
                        ok = np.int64((x & cm) == cv)
                        m += ok
                        neg += ok & np.int64(popcnt64(x & pm) & one)
            elif j1 - j0 == 2:
                wa = tw_word[j0]
                wb = tw_word[j0 + 1]
                cma = tw_cm[j0]
                cva = tw_cv[j0]
                pma = tw_pm[j0]
                cmb = tw_cm[j0 + 1]
                cvb = tw_cv[j0 + 1]
                pmb = tw_pm[j0 + 1]
                if select_only:
                    for r in range(r0, r1):
                        m += ((rows[r, wa] & cma) == cva) & ((rows[r, wb] & cmb) == cvb)
                else:
                    for r in range(r0, r1):
                        xa = rows[r, wa]
                        xb = rows[r, wb]
                        ok = np.int64(((xa & cma) == cva) & ((xb & cmb) == cvb))
                        m += ok
                        neg += ok & np.int64(popcnt64((xa & pma) ^ (xb & pmb)) & one)
            elif j1 - j0 == 3:
                wa = tw_word[j0]
                wb = tw_word[j0 + 1]
                wc = tw_word[j0 + 2]
                cma = tw_cm[j0]
                cva = tw_cv[j0]
                pma = tw_pm[j0]
                cmb = tw_cm[j0 + 1]
                cvb = tw_cv[j0 + 1]
                pmb = tw_pm[j0 + 1]
                cmc = tw_cm[j0 + 2]
                cvc = tw_cv[j0 + 2]
                pmc = tw_pm[j0 + 2]
                if select_only:
                    for r in range(r0, r1):
                        m += (((rows[r, wa] & cma) == cva) & ((rows[r, wb] & cmb) == cvb)
                              & ((rows[r, wc] & cmc) == cvc))
                else:
                    for r in range(r0, r1):
                        xa = rows[r, wa]
                        xb = rows[r, wb]
                        xc = rows[r, wc]
                        ok = np.int64(((xa & cma) == cva) & ((xb & cmb) == cvb) & ((xc & cmc) == cvc))
                        m += ok
                        neg += ok & np.int64(popcnt64((xa & pma) ^ (xb & pmb) ^ (xc & pmc)) & one)
            else:
                for r in range(r0, r1):
                    ok = True
                    par = np.uint64(0)
                    for j in range(j0, j1):
                        x = rows[r, tw_word[j]]
                        ok = ok & ((x & tw_cm[j]) == tw_cv[j])
                        par ^= x & tw_pm[j]
                    oki = np.int64(ok)
                    m += oki
                    if not select_only:
                        neg += oki & np.int64(popcnt64(par) & one)
            Mp[bi, t] = m
            Np[bi, t] = neg
    M = Mp.sum(axis=0)
    S = M - 2 * Np.sum(axis=0)
    return M, S


class SStar:
    """S* bound to one packed store. `prepare(terms)` encodes a batch (counted inside the timed batch call)."""

    def __init__(self, rows, n, block=2048):
        self.rows = rows
        self.n = n
        self.block = block

    def plan(self, terms, select_only=False):
        return encode_terms_for_scan(terms, self.n, select_only)

    def execute(self, enc, select_only=False):
        return _sstar(self.rows, *enc, self.block, select_only)

    def run(self, terms, select_only=False):
        return self.execute(self.plan(terms, select_only), select_only)


@njit(cache=True)
def _encode_nb(ptr, q, b, n, W, select_only):
    """Numba version of encode_terms_for_scan (same output), so S*'s per-batch planning is compiled like its kernel."""
    T = ptr.shape[0] - 1
    tw_ptr = np.zeros(T + 1, np.int64)
    cap = T * W
    tw_word = np.zeros(cap, np.int64)
    tw_cm = np.zeros(cap, np.uint64)
    tw_cv = np.zeros(cap, np.uint64)
    tw_pm = np.zeros(cap, np.uint64)
    cm = np.zeros(W, np.uint64)
    cv = np.zeros(W, np.uint64)
    pm = np.zeros(W, np.uint64)
    k = 0
    for t in range(T):
        cm[:] = 0
        cv[:] = 0
        pm[:] = 0
        for j in range(ptr[t], ptr[t + 1]):
            bit = 2 * q[j]
            cm[bit >> 6] |= np.uint64(3) << np.uint64(bit & 63)
            cv[bit >> 6] |= np.uint64(b[j] + 1) << np.uint64(bit & 63)
            if not select_only:
                ob = 2 * n + q[j]
                pm[ob >> 6] |= np.uint64(1) << np.uint64(ob & 63)
        for w in range(W):
            if cm[w] != 0 or pm[w] != 0:
                tw_word[k] = w
                tw_cm[k] = cm[w]
                tw_cv[k] = cv[w]
                tw_pm[k] = pm[w]
                k += 1
        tw_ptr[t + 1] = k
    return tw_ptr, tw_word[:k].copy(), tw_cm[:k].copy(), tw_cv[:k].copy(), tw_pm[:k].copy()


class SStarFast(SStar):
    """S* with its per-batch term encoding compiled (numba) — same kernel; added 2026-11-07 (PREREG addendum A1)."""

    def plan(self, terms, select_only=False):
        return _encode_nb(terms.ptr, terms.q, terms.b, self.n, row_words(self.n), select_only)


# ----------------------------------------------------------------------------------------------------------------------
# HKP reference code (G2): third_party/hkp/prediction_shadow.py::estimate_exp, called as published.
# ----------------------------------------------------------------------------------------------------------------------
_HKP_PATH = os.path.join(os.path.dirname(__file__), "..", "third_party", "hkp", "prediction_shadow.py")


def load_hkp():
    spec = importlib.util.spec_from_file_location("hkp_prediction_shadow", os.path.abspath(_HKP_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_hkp_measurements(store):
    """HKP format: list over snapshots of lists over qubits of (pauli char, +1/-1)."""
    chars = np.array(list("XYZ"))
    B = chars[store.bases]
    V = (1 - 2 * store.outcomes.astype(np.int64))
    return [list(zip(B[r].tolist(), V[r].tolist())) for r in range(store.N)]


def to_hkp_observable(terms, t):
    qq, bb = terms.term(t)
    return [("XYZ"[int(b)], int(q)) for q, b in zip(qq, bb)]


def hkp_select_only(full_measurement, one_observable):
    """Copy of HKP's estimate_exp with the outcome product removed (selection only), used to measure the selection
    share of HKP's batch time. Loop structure and the mismatch test are identical to the published code."""
    cnt_match = 0
    for single_measurement in full_measurement:
        not_match = 0
        for pauli_XYZ, position in one_observable:
            if pauli_XYZ != single_measurement[position][0]:
                not_match = 1
                break
        if not_match == 1:
            continue
        cnt_match += 1
    return cnt_match


def hkp_batch(hkp_mod, meas, terms, select_only=False):
    M = np.zeros(terms.T, np.int64)
    S = np.zeros(terms.T, np.int64)
    for t in range(terms.T):
        obs = to_hkp_observable(terms, t)
        if select_only:
            M[t] = hkp_select_only(meas, obs)
        else:
            s, m = hkp_mod.estimate_exp(meas, obs)
            M[t] = m
            S[t] = s
    return M, S


# ----------------------------------------------------------------------------------------------------------------------
# PennyLane pauli_expval (G4), as published, called on chunks of terms so that its (N, b, n) broadcast fits in memory.
# ----------------------------------------------------------------------------------------------------------------------
def pennylane_words(terms):
    W = -np.ones((terms.T, terms.n), np.int64)
    for t in range(terms.T):
        qq, bb = terms.term(t)
        W[t, qq] = bb
    return W


def pennylane_inputs(store):
    """PennyLane's native inputs (bits, recipes as int64), prepared once per store outside timed batches."""
    return store.outcomes.astype(np.int64), store.bases.astype(np.int64)


def pennylane_batch(store, terms, mem_bytes=2 * 1024 ** 3, select_only=False, inputs=None):
    """Returns per-term (sum of pauli_expval values over snapshots, number of nonzero values) and the estimator
    3^k * S / N as PennyLane computes it. Chunk size so that N * chunk * n * 8 bytes <= mem_bytes."""
    from pennylane.shadows.classical_shadow import pauli_expval
    from pennylane import math

    bits, recipes = inputs if inputs is not None else pennylane_inputs(store)
    words = pennylane_words(terms)
    N, n = bits.shape
    chunk = max(1, int(mem_bytes // (N * n * 8)))
    tot = np.zeros(terms.T, np.float64)
    nnz = np.zeros(terms.T, np.int64)
    for c0 in range(0, terms.T, chunk):
        w = words[c0:c0 + chunk]
        if select_only:
            # verbatim copy of the selection lines of pauli_expval (the `indices` computation)
            T_, n_ = recipes.shape
            b_ = w.shape[0]
            id_mask = w == -1
            indices = math.equal(math.reshape(recipes, (T_, 1, n_)), math.reshape(w, (1, b_, n_)))
            indices = np.logical_or(indices, math.tile(math.reshape(id_mask, (1, b_, n_)), (T_, 1, 1)))
            indices = math.all(indices, axis=2)
            nnz[c0:c0 + chunk] = indices.sum(axis=0)
        else:
            ev = pauli_expval(bits, recipes, w)
            tot[c0:c0 + chunk] = ev.sum(axis=0)
            nnz[c0:c0 + chunk] = np.count_nonzero(ev, axis=0)
    return tot, nnz
