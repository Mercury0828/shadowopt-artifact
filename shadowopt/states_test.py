"""TEST states (guide §P6.2; prereg/M1.md §2 and M1-A3). Little-endian statevectors (qubit q = bit q of the index).

S1 molecules (BeH2, H2O, N2; STO-3G, Jordan-Wigner, no frozen core): the FCI ground state of pyscf, mapped from the
   determinant (alpha string, beta string) basis to the qubit basis of openfermion's JW ordering (mode 2i = alpha i,
   2i+1 = beta i). Validated by recomputing the energy from the qubit terms.
S2 Heisenberg chains (open boundary, J = 1, H = sum X X + Y Y + Z Z): DMRG ground state (quimb), stored as a
   right-canonical MPS B[q] (chi_l, 2, chi_r) in the same format as the DEV TFIM state; truth from the MPS itself.
S3 cluster state n = 100 (H on all, CZ on neighbours) and a seeded random Clifford state n = 50 (qiskit random_clifford
   applied to |0...0>): stabilizer generators as stim PauliStrings; truth by stim's exact expectation.
"""
import numpy as np

GEOMETRIES = {  # M1-A3
    "BeH2": [("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.326)), ("H", (0.0, 0.0, -1.326))],
    "H2O": [("O", (0.0, 0.0, 0.0)),
            ("H", (0.9584 * np.sin(np.radians(104.45 / 2)), 0.0, 0.9584 * np.cos(np.radians(104.45 / 2)))),
            ("H", (-0.9584 * np.sin(np.radians(104.45 / 2)), 0.0, 0.9584 * np.cos(np.radians(104.45 / 2))))],
    "N2": [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
}


def molecule_state(name, basis="sto-3g"):
    """Returns (n, psi_le, terms, const, E_fci, info)."""
    import openfermion as of
    from openfermionpyscf import run_pyscf
    from pyscf import fci
    from pyscf.fci import cistring
    from .store import terms_from_pairs

    mol = of.chem.MolecularData(GEOMETRIES[name], basis, multiplicity=1, charge=0)
    mol = run_pyscf(mol, run_scf=1, run_fci=0)
    mf = mol._pyscf_data["scf"]
    norb = mf.mo_coeff.shape[1]
    ne = mol.n_electrons
    na = nb = ne // 2
    cis = fci.FCI(mf)
    cis.conv_tol = 1e-12
    E_fci, civec = cis.kernel()
    sa = cistring.make_strings(range(norb), na)
    sb = cistring.make_strings(range(norb), nb)
    n = 2 * norb
    psi = np.zeros(2 ** n, np.complex128)
    for ia, a in enumerate(sa):
        a = int(a)
        aocc = [i for i in range(norb) if (a >> i) & 1]
        for ib, b in enumerate(sb):
            c = civec[ia, ib]
            if c == 0:
                continue
            b = int(b)
            idx = 0
            for i in aocc:
                idx |= 1 << (2 * i)
            sign_par = 0
            for i in range(norb):
                if (b >> i) & 1:
                    idx |= 1 << (2 * i + 1)
                    sign_par += sum(1 for j in aocc if j > i)  # beta_i moves left past alpha_j with j > i
            psi[idx] += c * (-1) ** (sign_par & 1)
    ham = mol.get_molecular_hamiltonian()
    qop = of.jordan_wigner(of.get_fermion_operator(ham))
    const, pairs, coeffs = 0.0, [], []
    for term, cc in qop.terms.items():
        if len(term) == 0:
            const = float(np.real(cc))
            continue
        assert abs(np.imag(cc)) < 1e-12
        pairs.append([(q, "XYZ".index(p)) for q, p in term])
        coeffs.append(float(np.real(cc)))
    terms = terms_from_pairs(n, pairs, coeffs)
    info = {"state": name, "basis": basis, "geometry": GEOMETRIES[name], "n_electrons": ne, "norb": norb,
            "E_fci_pyscf": float(E_fci), "E_hf": float(mol.hf_energy), "n_terms": terms.T,
            "fci_vector_norm": float(np.linalg.norm(psi))}
    return n, psi, terms, const, float(E_fci), info


# ----------------------------------------------------------------------------------------------------------------------
def _patch_cotengra_constants():
    """Work around a cotengra 0.8.2 / autoray 0.8.4 incompatibility in the pinned stack: constant folding through lazy
    tracing returns a plain ndarray ("'numpy.ndarray' object has no attribute 'get_function'"). The fallback builds the
    same contraction without constants and inserts the constant arrays at call time (same arithmetic, no folding)."""
    import cotengra.interface as ci
    if getattr(ci, "_shadowopt_patched", False):
        return

    def _with_constants(inputs, output, size_dict, constants, optimize="auto", implementation=None,
                        prefer_einsum=False, autojit=False, via=None, sort_contraction_indices=False, cache=True):
        full = ci.array_contract_expression(inputs, output, size_dict, optimize=optimize, constants=None,
                                            implementation=implementation, autojit=False, prefer_einsum=prefer_einsum,
                                            sort_contraction_indices=sort_contraction_indices, cache=cache)
        consts = dict(constants)
        if via is not None:
            consts = {k: via[0](v) for k, v in consts.items()}

        def fn(*arrays, backend=None):
            it = iter(arrays)
            args = [consts[i] if i in consts else next(it) for i in range(len(inputs))]
            out = full(*args) if backend is None else full(*args, backend=backend)
            return via[1](out) if via is not None else out
        return fn

    ci._array_contract_expression_with_constants = _with_constants
    ci._shadowopt_patched = True


def heisenberg_mps(n, chi_max=128, cutoff=1e-10, seed=0):
    """DMRG ground state of H = sum_i (X X + Y Y + Z Z)_{i,i+1} (open chain). Returns (B tensors, info)."""
    import quimb.tensor as qtn
    _patch_cotengra_constants()
    H = qtn.MPO_ham_heis(n, j=4.0, bz=0.0, cyclic=False)  # quimb uses spin-1/2 operators: j = 4 gives Pauli units
    dmrg = qtn.DMRG2(H, bond_dims=[16, 32, 64, chi_max], cutoffs=cutoff)
    np.random.seed(seed)
    ok = dmrg.solve(tol=1e-9, max_sweeps=30, verbosity=0)
    psi = dmrg.state
    psi.right_canonize()
    B = []
    for q in range(n):
        t = psi[q]
        site = psi.site_ind(q)
        left = [i for i in t.inds if i != site and q > 0 and i in psi[q - 1].inds]
        right = [i for i in t.inds if i != site and q < n - 1 and i in psi[q + 1].inds]
        arr = t.transpose(*(left + [site] + right)).data
        if not left:
            arr = arr[None]
        if not right:
            arr = arr[..., None]
        B.append(np.ascontiguousarray(arr, dtype=np.complex128))
    nrm = mps_norm(B)
    B[0] = B[0] / np.sqrt(nrm)
    info = {"n": n, "E_dmrg": float(np.real(dmrg.energy)), "converged": bool(ok), "chi_max": chi_max,
            "cutoff": cutoff, "chi_used": int(max(b.shape[2] for b in B)), "norm_before": float(nrm)}
    return B, info


def mps_norm(B):
    L = np.ones((1, 1), np.complex128)
    for A in B:
        L = np.einsum("ab,asc,bsd->cd", L, A.conj(), A)
    return float(np.real(L[0, 0]))


_PAULI = [np.array([[0, 1], [1, 0]], np.complex128), np.array([[0, -1j], [1j, 0]], np.complex128),
          np.array([[1, 0], [0, -1]], np.complex128)]


def mps_two_point(B, basis):
    """C[i, j] = <sigma_i^a sigma_j^a> for all i < j (a = basis code), O(n^2 chi^3)."""
    n = len(B)
    P = _PAULI[basis]
    C = np.zeros((n, n))
    L = np.ones((1, 1), np.complex128)
    for i in range(n):
        Li = np.einsum("ab,asc,st,btd->cd", L, B[i].conj(), P, B[i])
        M = Li
        for j in range(i + 1, n):
            v = np.einsum("ab,asc,st,btd->cd", M, B[j].conj(), P, B[j])
            # right-canonical: the right environment is the identity
            C[i, j] = float(np.real(np.trace(v)))
            M = np.einsum("ab,asc,bsd->cd", M, B[j].conj(), B[j])
        L = np.einsum("ab,asc,bsd->cd", L, B[i].conj(), B[i])
    return C


# ----------------------------------------------------------------------------------------------------------------------
def cluster_stabilizers(n):
    import stim
    out = []
    for i in range(n):
        s = ["_"] * n
        s[i] = "X"
        if i > 0:
            s[i - 1] = "Z"
        if i < n - 1:
            s[i + 1] = "Z"
        out.append(stim.PauliString("".join(s)))
    return out


def random_clifford_stabilizers(n, seed):
    import stim
    from qiskit.quantum_info import random_clifford
    cl = random_clifford(n, seed=seed)
    labels = cl.to_labels(mode="S")  # qiskit labels: sign + qubits in little-endian order (rightmost = qubit 0)
    out = []
    for lab in labels:
        sign = "-" if lab.startswith("-") else "+"
        body = lab.lstrip("+-")[::-1].replace("I", "_")
        out.append(stim.PauliString(sign + body))
    return out


def stabilizer_simulator(stabs):
    import stim
    sim = stim.TableauSimulator()
    sim.set_state_from_stabilizers(stabs)
    return sim


def stabilizer_expectation(sim, n, qq, bb):
    import stim
    s = ["_"] * n
    for q, b in zip(np.asarray(qq).tolist(), np.asarray(bb).tolist()):
        s[q] = "XYZ"[b]
    return float(sim.peek_observable_expectation(stim.PauliString("".join(s))))
