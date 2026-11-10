"""Bit-sliced compatibility and parity kernels.

mask_P = AND_{i in S(P)} Z[i, P_i];  m_P = popcount(mask_P);  s_P = m_P - 2 * popcount(mask_P AND XOR_{i in S(P)} O[i]).
Word chunks of the bitmaps are processed in parallel; inside a chunk all terms of the batch reuse the chunk (cache).
"""
import numpy as np
from numba import njit, prange

from ._bits import popcnt64


@njit(parallel=True, cache=True, fastmath=True)
def _bitmap_batch(Z, O, ptr, q, b, chunk, select_only):
    NW = Z.shape[2]
    T = ptr.shape[0] - 1
    nc = (NW + chunk - 1) // chunk
    Mp = np.zeros((nc, T), np.int64)
    Np = np.zeros((nc, T), np.int64)
    for c in prange(nc):
        w0 = c * chunk
        w1 = min(NW, w0 + chunk)
        L = w1 - w0
        A = np.empty(L, np.uint64)
        X = np.empty(L, np.uint64)
        for t in range(T):
            j0 = ptr[t]
            j1 = ptr[t + 1]
            m = 0
            neg = 0
            if j1 - j0 == 0:
                continue
            q0 = q[j0]
            b0 = b[j0]
            for w in range(L):
                A[w] = Z[q0, b0, w0 + w]
            if not select_only:
                for w in range(L):
                    X[w] = O[q0, w0 + w]
            for j in range(j0 + 1, j1):
                qj = q[j]
                bj = b[j]
                for w in range(L):
                    A[w] &= Z[qj, bj, w0 + w]
                if not select_only:
                    for w in range(L):
                        X[w] ^= O[qj, w0 + w]
            for w in range(L):
                m += popcnt64(A[w])
            if not select_only:
                for w in range(L):
                    neg += popcnt64(A[w] & X[w])
            Mp[c, t] = m
            Np[c, t] = neg
    M = Mp.sum(axis=0)
    S = M - 2 * Np.sum(axis=0)
    # empty terms (identity) are not used by any workload; they are answered by the caller
    return M, S


class BitmapPath:
    def __init__(self, Z, O, N, chunk=2048):
        self.Z, self.O, self.N, self.chunk = Z, O, N, chunk

    def plan(self, terms):
        return terms  # the CSR term arrays are the plan

    def execute(self, terms, select_only=False):
        return _bitmap_batch(self.Z, self.O, terms.ptr, terms.q, terms.b, self.chunk, select_only)

    def run(self, terms, select_only=False):
        return self.execute(self.plan(terms), select_only)
