"""State generators (S1-S3 for Phase 0) and exact expectation values.

S1 LiH (STO-3G, Jordan-Wigner, 12 qubits): pyscf + openfermion; ground state in the 4-electron sector; statevector.
S2 TFIM chain H = -J sum Z_i Z_{i+1} - h sum X_i (open boundary): exact ground state by Lanczos, then converted to a
   right-canonical MPS with a stated truncation; the stored state IS the MPS (its statevector is used for exact truth).
S3 GHZ_n: analytic sampler and analytic expectation values (cross-checked against a statevector and stim).

All statevectors are little-endian: qubit q is bit q of the index.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------------------------------------------------
# Pauli expectation on a little-endian statevector
# ----------------------------------------------------------------------------------------------------------------------
def pauli_masks(qq, bb):
    x = 0
    z = 0
    ny = 0
    for q, b in zip(np.asarray(qq).tolist(), np.asarray(bb).tolist()):
        if b == 0:
            x |= 1 << q
        elif b == 1:
            x |= 1 << q
            z |= 1 << q
            ny += 1
        else:
            z |= 1 << q
    return x, z, ny


def pauli_expectation(psi, qq, bb):
    """<psi| P |psi> with P = prod_q sigma_{b_q}(q); uses Y = i X Z, so P|j> = i^{#Y} (-1)^{|j & z|} |j ^ x>."""
    x, z, ny = pauli_masks(qq, bb)
    idx = np.arange(psi.shape[0], dtype=np.int64)
    pc = np.bitwise_count(idx & z).astype(np.int64)  # numpy >= 2.0 (uint8 result: cast before arithmetic)
    sign = 1 - 2 * (pc & 1)
    val = np.vdot(psi[idx ^ x], sign * psi) * (1j ** ny)
    return float(np.real(val))


def expectations_statevector(psi, terms):
    return np.array([pauli_expectation(psi, *terms.term(t)) for t in range(terms.T)])


# ----------------------------------------------------------------------------------------------------------------------
# S1: LiH
# ----------------------------------------------------------------------------------------------------------------------
def big_to_little(psi_be, n):
    return np.ascontiguousarray(psi_be.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1))


def lih_state(bond_length=1.5949, basis="sto-3g"):
    """Returns (n, psi_le, terms TermBatch with coefficients (no identity), constant, E0, info)."""
    import openfermion as of
    from openfermionpyscf import run_pyscf
    from .store import terms_from_pairs

    geom = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))]
    mol = of.chem.MolecularData(geom, basis, multiplicity=1, charge=0)
    mol = run_pyscf(mol, run_scf=1, run_fci=1)
    ham = mol.get_molecular_hamiltonian()
    qop = of.jordan_wigner(of.get_fermion_operator(ham))
    n = of.count_qubits(qop)
    H = of.get_sparse_operator(qop, n_qubits=n).tocsc()  # big-endian
    # restrict to the 4-electron sector (n_electrons of the neutral molecule)
    ne = mol.n_electrons
    dim = 2 ** n
    pc = np.array([bin(i).count("1") for i in range(dim)])
    sec = np.nonzero(pc == ne)[0]
    Hs = H[sec][:, sec].toarray()
    w, v = np.linalg.eigh(Hs)
    psi_be = np.zeros(dim, np.complex128)
    psi_be[sec] = v[:, 0]
    psi = big_to_little(psi_be, n)
    const = 0.0
    pairs, coeffs = [], []
    for term, c in qop.terms.items():
        if len(term) == 0:
            const = float(np.real(c))
            continue
        assert abs(np.imag(c)) < 1e-12
        pairs.append([(q, "XYZ".index(p)) for q, p in term])
        coeffs.append(float(np.real(c)))
    terms = terms_from_pairs(n, pairs, coeffs)
    info = {"bond_length_A": bond_length, "basis": basis, "n_electrons": ne, "E0_sector": float(w[0]),
            "E_fci_pyscf": float(mol.fci_energy), "E_hf": float(mol.hf_energy), "n_terms": terms.T}
    return n, psi, terms, const, float(w[0]), info


# ----------------------------------------------------------------------------------------------------------------------
# S2: TFIM chain and MPS
# ----------------------------------------------------------------------------------------------------------------------
def tfim_hamiltonian(n, J=1.0, h=1.0):
    dim = 2 ** n
    idx = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim)
    for i in range(n - 1):
        zi = 1 - 2 * ((idx >> i) & 1)
        zj = 1 - 2 * ((idx >> (i + 1)) & 1)
        diag += -J * zi * zj
    rows = [idx]
    cols = [idx]
    vals = [diag]
    for i in range(n):
        rows.append(idx)
        cols.append(idx ^ (1 << i))
        vals.append(np.full(dim, -h))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))


def tfim_ground_state(n, J=1.0, h=1.0, seed=0):
    H = tfim_hamiltonian(n, J, h)
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(2 ** n)
    w, v = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-12)
    return float(w[0]), v[:, 0].astype(np.complex128), H


def statevector_to_mps(psi_le, n, chi_max=64, cutoff=1e-10):
    """Right-canonical MPS B[0..n-1], B[q] shape (chi_l, 2, chi_r), physical index = bit of qubit q.
    Truncation: drop singular values below cutoff * largest and keep at most chi_max."""
    T = psi_le.reshape([2] * n).transpose(list(range(n))[::-1])  # axes q0..q_{n-1}
    tensors = [None] * n
    rem = T.reshape(-1, 2)  # (2^{n-1}, 2): last axis q_{n-1}
    chi_r = 1
    discarded = 0.0
    for q in range(n - 1, 0, -1):
        # rem shape: (2^q, 2*chi_r) with columns (s_q, right)
        U, S, Vh = np.linalg.svd(rem, full_matrices=False)
        keep = int(min(chi_max, np.sum(S > cutoff * S[0])))
        discarded += float(np.sum(S[keep:] ** 2))
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        tensors[q] = Vh.reshape(keep, 2, chi_r)
        chi_r = keep
        rem = (U * S).reshape(-1, 2 * chi_r)  # rows: q0..q_{q-1}; cols: (s_{q-1}, chi)
    tensors[0] = rem.reshape(1, 2, chi_r)
    nrm = np.linalg.norm(tensors[0])
    tensors[0] = tensors[0] / nrm
    return tensors, discarded


def mps_to_statevector(tensors):
    n = len(tensors)
    v = tensors[0].reshape(2, -1)  # (s0, chi)
    for q in range(1, n):
        A = tensors[q]
        v = np.einsum("pa,asb->psb", v, A).reshape(-1, A.shape[2])
    v = v.reshape([2] * n)  # axes q0..q_{n-1}
    return np.ascontiguousarray(v.transpose(list(range(n))[::-1]).reshape(-1))


# ----------------------------------------------------------------------------------------------------------------------
# S3: GHZ_n analytic expectation
# ----------------------------------------------------------------------------------------------------------------------
def ghz_expectation(n, qq, bb):
    qq = list(np.asarray(qq).tolist())
    bb = list(np.asarray(bb).tolist())
    if len(qq) == 0:
        return 1.0
    nz = sum(1 for b in bb if b == 2)
    nxy = len(bb) - nz
    if nxy == 0:
        return 1.0 if nz % 2 == 0 else 0.0
    if len(qq) == n and nz == 0:
        ny = sum(1 for b in bb if b == 1)
        return float(np.round(np.cos(np.pi * ny / 2)))
    return 0.0


def ghz_statevector(n):
    psi = np.zeros(2 ** n, np.complex128)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    return psi
