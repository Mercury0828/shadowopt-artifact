"""Store layer: snapshot schema, append, epochs, packed row layout (for S*), bit-sliced bitmaps, manifests.

A snapshot t is (b_t, o_t, e_t, src_t): basis codes b_t in {0,1,2}^n (X,Y,Z), outcome bits o_t in {0,1}^n,
calibration epoch e_t, source src_t in {rand, biased, derand, target}. The store is an append-only log.

A term batch is stored in CSR form: term t has support qubits q[ptr[t]:ptr[t+1]] (sorted ascending, distinct) and the
required basis codes b[ptr[t]:ptr[t+1]].
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

from ._bits import n_words

SRC_CODES = {"rand": 0, "biased": 1, "derand": 2, "target": 3}


# ----------------------------------------------------------------------------------------------------------------------
# Terms
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class TermBatch:
    n: int
    ptr: np.ndarray  # int64, len T+1
    q: np.ndarray  # int64 qubit indices
    b: np.ndarray  # int64 basis codes 0/1/2
    coeff: np.ndarray = None  # optional float64 coefficients
    labels: list = None

    @property
    def T(self):
        return len(self.ptr) - 1

    def weights(self):
        return np.diff(self.ptr)

    def term(self, t):
        return self.q[self.ptr[t]:self.ptr[t + 1]], self.b[self.ptr[t]:self.ptr[t + 1]]

    def to_strings(self):
        out = []
        for t in range(self.T):
            s = ["I"] * self.n
            qq, bb = self.term(t)
            for qi, bi in zip(qq, bb):
                s[qi] = "XYZ"[bi]
            out.append("".join(s))
        return out

    def subset(self, idx):
        """Terms idx (in order, repeats allowed). Vectorized (A14); same output as the former per-term loop."""
        idx = np.asarray(idx)
        ii = idx.astype(np.int64).reshape(-1)
        p = np.asarray(self.ptr, np.int64)
        lens = p[ii + 1] - p[ii]
        ptr = np.zeros(len(ii) + 1, np.int64)
        np.cumsum(lens, out=ptr[1:])
        pos = np.repeat(p[ii] - ptr[:-1], lens) + np.arange(ptr[-1], dtype=np.int64)
        return TermBatch(self.n, ptr, np.asarray(self.q)[pos].astype(np.int64), np.asarray(self.b)[pos].astype(np.int64),
                         None if self.coeff is None else self.coeff[idx],
                         None if self.labels is None else [self.labels[i] for i in idx])


def terms_from_strings(strings, coeffs=None):
    n = len(strings[0])
    ptr, qs, bs = [0], [], []
    for s in strings:
        assert len(s) == n
        for i, c in enumerate(s):
            if c != "I":
                qs.append(i)
                bs.append("XYZ".index(c))
        ptr.append(len(qs))
    return TermBatch(n, np.array(ptr, np.int64), np.array(qs, np.int64), np.array(bs, np.int64),
                     None if coeffs is None else np.asarray(coeffs, np.float64), list(strings))


def terms_from_pairs(n, list_of_pairs, coeffs=None):
    """list_of_pairs: for each term, a list of (qubit, basis_code) pairs (any order, distinct qubits)."""
    ptr, qs, bs = [0], [], []
    for pairs in list_of_pairs:
        pairs = sorted(pairs)
        assert len(set(p[0] for p in pairs)) == len(pairs), "duplicate qubit in term"
        for qi, bi in pairs:
            qs.append(qi)
            bs.append(bi)
        ptr.append(len(qs))
    return TermBatch(n, np.array(ptr, np.int64), np.array(qs, np.int64), np.array(bs, np.int64),
                     None if coeffs is None else np.asarray(coeffs, np.float64))


# ----------------------------------------------------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Store:
    n: int
    bases: np.ndarray = None  # uint8 (N, n)
    outcomes: np.ndarray = None  # uint8 (N, n)
    epoch: np.ndarray = None  # int32 (N,)
    src: np.ndarray = None  # uint8 (N,)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.bases is None:
            self.bases = np.zeros((0, self.n), np.uint8)
            self.outcomes = np.zeros((0, self.n), np.uint8)
            self.epoch = np.zeros(0, np.int32)
            self.src = np.zeros(0, np.uint8)

    @property
    def N(self):
        return self.bases.shape[0]

    def append(self, bases, outcomes, epoch=0, src="rand"):
        bases = np.ascontiguousarray(bases, np.uint8)
        outcomes = np.ascontiguousarray(outcomes, np.uint8)
        assert bases.shape == outcomes.shape and bases.shape[1] == self.n
        assert bases.max(initial=0) <= 2 and outcomes.max(initial=0) <= 1
        k = bases.shape[0]
        self.bases = np.concatenate([self.bases, bases])
        self.outcomes = np.concatenate([self.outcomes, outcomes])
        self.epoch = np.concatenate([self.epoch, np.full(k, epoch, np.int32)])
        self.src = np.concatenate([self.src, np.full(k, SRC_CODES[src], np.uint8)])

    def head(self, N):
        """The first N snapshots (a prefix of the append log)."""
        return Store(self.n, self.bases[:N], self.outcomes[:N], self.epoch[:N], self.src[:N], dict(self.meta))

    # -- persistence ---------------------------------------------------------------------------------------------------
    def save(self, directory, manifest_extra=None):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "store.npz")
        np.savez(path, bases=self.bases, outcomes=self.outcomes, epoch=self.epoch, src=self.src)
        sha = sha256_file(path)
        man = dict(self.meta)
        man.update(manifest_extra or {})
        man.update({"n": self.n, "N": int(self.N), "file": "store.npz", "sha256": sha,
                    "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        with open(os.path.join(directory, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        return man

    @staticmethod
    def load(directory, verify=True):
        with open(os.path.join(directory, "manifest.json")) as f:
            man = json.load(f)
        path = os.path.join(directory, man["file"])
        if verify:
            sha = sha256_file(path)
            if sha != man["sha256"]:
                raise ValueError(f"sha256 mismatch for {path}: {sha} != {man['sha256']}")
        d = np.load(path)
        return Store(man["n"], d["bases"], d["outcomes"], d["epoch"], d["src"], man)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# Packed row-major layout (S*): per row, bits [0, 2n) hold the basis codes (2 bits per qubit; code+1 so that X=01,
# Y=10, Z=11), bits [2n, 3n) hold the outcome bits; ceil(3n/64) uint64 words per row.
# ----------------------------------------------------------------------------------------------------------------------
def row_words(n):
    return n_words(3 * n)


@njit(parallel=True, cache=True)
def _pack_rows(bases, outcomes, W):
    N, n = bases.shape
    out = np.zeros((N, W), np.uint64)
    for r in prange(N):
        for i in range(n):
            code = np.uint64(bases[r, i] + 1)
            bit = 2 * i
            out[r, bit >> 6] |= code << np.uint64(bit & 63)
            if outcomes[r, i]:
                ob = 2 * n + i
                out[r, ob >> 6] |= np.uint64(1) << np.uint64(ob & 63)
    return out


def pack_rows(store):
    return _pack_rows(store.bases, store.outcomes, row_words(store.n))


def packed_bytes(n, N):
    return N * row_words(n) * 8


# ----------------------------------------------------------------------------------------------------------------------
# Bit-sliced bitmaps: Z[i, beta] (N bits) = rows whose basis on qubit i is beta; O[i] = rows whose outcome on i is 1.
# Bit r of the store is bit (r & 63) of word r >> 6.
# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _build_bitmaps(bases, outcomes, NW):
    N, n = bases.shape
    Z = np.zeros((n, 3, NW), np.uint64)
    O = np.zeros((n, NW), np.uint64)
    for i in prange(n):
        for w in range(NW):
            r0 = w * 64
            r1 = min(r0 + 64, N)
            zx = np.uint64(0)
            zy = np.uint64(0)
            zz = np.uint64(0)
            oo = np.uint64(0)
            for r in range(r0, r1):
                bit = np.uint64(1) << np.uint64(r - r0)
                c = bases[r, i]
                if c == 0:
                    zx |= bit
                elif c == 1:
                    zy |= bit
                else:
                    zz |= bit
                if outcomes[r, i]:
                    oo |= bit
            Z[i, 0, w] = zx
            Z[i, 1, w] = zy
            Z[i, 2, w] = zz
            O[i, w] = oo
    return Z, O


def build_bitmaps(store):
    NW = n_words(store.N)
    return _build_bitmaps(store.bases, store.outcomes, NW)


def bitmap_bytes(n, N):
    return 4 * n * n_words(N) * 8
