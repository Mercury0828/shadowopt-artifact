"""Basis distributions and one-shot-per-basis sampling.

Measuring qubit q in basis b: apply U_b (X: H; Y: H S^dagger; Z: identity), then measure Z. Outcome bit 0 = +1.
All randomness comes from numpy Generators (PCG64) seeded by the caller; numba kernels consume pre-drawn uniforms.
"""
import numpy as np
from numba import njit, prange

_SQ = 1 / np.sqrt(2)
# rotation matrices U_b, rows = outcome o, cols = computational state s
ROT = np.array([
    [[_SQ, _SQ], [_SQ, -_SQ]],  # X: H
    [[_SQ, -1j * _SQ], [_SQ, 1j * _SQ]],  # Y: H S^dagger
    [[1, 0], [0, 1]],  # Z
], dtype=np.complex128)


def uniform_bases(rng, N, n):
    return rng.integers(0, 3, size=(N, n), dtype=np.uint8)


def biased_bases(rng, N, probs):
    """probs: (n, 3) per-qubit basis probabilities (locally biased shadows, G10)."""
    n = probs.shape[0]
    u = rng.random((N, n))
    c = np.cumsum(probs, axis=1)
    return (u[:, :, None] > c[None, :, :2]).sum(axis=2).astype(np.uint8)


# ----------------------------------------------------------------------------------------------------------------------
# statevector sampler
# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _sample_sv(psi, n, bases, u, rot):
    N = bases.shape[0]
    dim = psi.shape[0]
    out = np.zeros((N, n), np.uint8)
    for t in prange(N):
        v = psi.copy()
        for q in range(n):
            b = bases[t, q]
            if b == 2:
                continue
            u00 = rot[b, 0, 0]
            u01 = rot[b, 0, 1]
            u10 = rot[b, 1, 0]
            u11 = rot[b, 1, 1]
            step = 1 << q
            for i in range(dim):
                if (i >> q) & 1 == 0:
                    a0 = v[i]
                    a1 = v[i + step]
                    v[i] = u00 * a0 + u01 * a1
                    v[i + step] = u10 * a0 + u11 * a1
        tot = 0.0
        for i in range(dim):
            tot += v[i].real * v[i].real + v[i].imag * v[i].imag
        thr = u[t] * tot
        acc = 0.0
        pick = dim - 1
        for i in range(dim):
            acc += v[i].real * v[i].real + v[i].imag * v[i].imag
            if acc > thr:
                pick = i
                break
        for q in range(n):
            out[t, q] = (pick >> q) & 1
    return out


def sample_statevector(psi, n, bases, rng):
    u = rng.random(bases.shape[0])
    return _sample_sv(psi, n, np.ascontiguousarray(bases), u, ROT)


# ----------------------------------------------------------------------------------------------------------------------
# MPS sampler (right-canonical tensors; sequential conditional sampling)
# ----------------------------------------------------------------------------------------------------------------------
def pack_mps(tensors):
    """Pre-rotate: R[q][b][o] = sum_s ROT[b,o,s] B_q[:, s, :]; packed into one array with offsets."""
    n = len(tensors)
    chil = np.array([A.shape[0] for A in tensors], np.int64)
    chir = np.array([A.shape[2] for A in tensors], np.int64)
    offs = np.zeros(n + 1, np.int64)
    for q in range(n):
        offs[q + 1] = offs[q] + 6 * chil[q] * chir[q]
    flat = np.zeros(offs[-1], np.complex128)
    for q, A in enumerate(tensors):
        R = np.einsum("bos,lsr->bolr", ROT, A)  # (3,2,chil,chir)
        flat[offs[q]:offs[q + 1]] = R.ravel()
    return flat, offs, chil, chir


@njit(parallel=True, cache=True)
def _sample_mps(flat, offs, chil, chir, bases, u, chimax):
    N, n = bases.shape
    out = np.zeros((N, n), np.uint8)
    for t in prange(N):
        L = np.zeros(chimax, np.complex128)
        L[0] = 1.0
        v0 = np.zeros(chimax, np.complex128)
        v1 = np.zeros(chimax, np.complex128)
        for q in range(n):
            cl = chil[q]
            cr = chir[q]
            b = bases[t, q]
            base0 = offs[q] + (b * 2 + 0) * cl * cr
            base1 = offs[q] + (b * 2 + 1) * cl * cr
            p0 = 0.0
            p1 = 0.0
            for r in range(cr):
                a0 = 0j
                a1 = 0j
                for l in range(cl):
                    a0 += L[l] * flat[base0 + l * cr + r]
                    a1 += L[l] * flat[base1 + l * cr + r]
                v0[r] = a0
                v1[r] = a1
                p0 += a0.real * a0.real + a0.imag * a0.imag
                p1 += a1.real * a1.real + a1.imag * a1.imag
            if u[t, q] * (p0 + p1) < p0:
                o = 0
                nrm = np.sqrt(p0)
                for r in range(cr):
                    L[r] = v0[r] / nrm
            else:
                o = 1
                nrm = np.sqrt(p1)
                for r in range(cr):
                    L[r] = v1[r] / nrm
            out[t, q] = o
    return out


class MPSSampler:
    def __init__(self, tensors):
        self.flat, self.offs, self.chil, self.chir = pack_mps(tensors)
        self.chimax = int(max(self.chil.max(), self.chir.max()))
        self.n = len(tensors)

    def sample(self, bases, rng):
        u = rng.random(bases.shape)
        return _sample_mps(self.flat, self.offs, self.chil, self.chir, np.ascontiguousarray(bases), u, self.chimax)


# ----------------------------------------------------------------------------------------------------------------------
# GHZ_n analytic sampler
# p(o | b) = 2^-n (1 + cos(pi #Y / 2) (-1)^{sum o}) if b has no Z; else Z outcomes all equal (prob 1/2 each) and the
# X/Y outcomes are independent uniform bits.
# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _sample_ghz(bases, u):
    N, n = bases.shape
    out = np.zeros((N, n), np.uint8)
    for t in prange(N):
        nz = 0
        ny = 0
        for q in range(n):
            if bases[t, q] == 2:
                nz += 1
            elif bases[t, q] == 1:
                ny += 1
        c = 1 if u[t, n] < 0.5 else 0
        par = 0
        for q in range(n):
            if bases[t, q] == 2:
                out[t, q] = c
            else:
                bit = 1 if u[t, q] < 0.5 else 0
                out[t, q] = bit
                par ^= bit
        if nz == 0 and ny % 2 == 0:
            want = 0 if ny % 4 == 0 else 1  # <P> = +1 -> even parity; -1 -> odd parity
            if par != want:
                out[t, n - 1] ^= 1
    return out


def sample_ghz(bases, rng):
    N, n = bases.shape
    u = rng.random((N, n + 1))
    return _sample_ghz(np.ascontiguousarray(bases), u)


# ----------------------------------------------------------------------------------------------------------------------
# Stabilizer states (TEST S3: cluster n = 100, random Clifford n = 50): exact per-row sampling with stim
# ----------------------------------------------------------------------------------------------------------------------
class StabilizerSampler:
    """Exact samples of a stabilizer state measured in per-row product bases. One stim TableauSimulator, seeded per call
    from `rng` (reproducible for a fixed stim version); per row: reset to the state, rotate X -> Z (H) or Y -> Z (H_YZ),
    measure every qubit in Z."""

    def __init__(self, stabs):
        import stim
        sim = stim.TableauSimulator()
        sim.set_state_from_stabilizers(stabs)
        self.inv = sim.current_inverse_tableau()
        self.n = len(stabs)

    def sample(self, bases, rng):
        import stim
        sim = stim.TableauSimulator(seed=int(rng.integers(0, 2 ** 62)))
        N, n = bases.shape
        out = np.zeros((N, n), np.uint8)
        qs = list(range(n))
        for t in range(N):
            sim.set_inverse_tableau(self.inv)
            b = bases[t]
            xs = np.nonzero(b == 0)[0].tolist()
            ys = np.nonzero(b == 1)[0].tolist()
            if xs:
                sim.h(*xs)
            if ys:
                sim.h_yz(*ys)
            out[t] = sim.measure_many(*qs)
        return out
