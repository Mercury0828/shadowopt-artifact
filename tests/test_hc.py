"""H-c machinery: the multinomial histogram path matches the project's per-row sampler and exact expectations."""
import os
import numpy as np

from shadowopt.hc import setting_probs, _parity_from_hist
from shadowopt.sampling import sample_statevector
from shadowopt.states import pauli_expectation

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_setting_probs_exact_and_match_row_sampler():
    psi = np.load(os.path.join(ROOT, "stores", "states", "LiH", "state.npz"))["psi"]
    n = 12
    rng = np.random.default_rng(0)
    for _ in range(4):
        setting = rng.integers(0, 3, n)
        p = setting_probs(psi, n, setting)
        B = np.tile(setting.astype(np.uint8)[None, :], (40000, 1))
        O = sample_statevector(psi, n, B, np.random.default_rng(1))
        idx = (O.astype(np.int64) << np.arange(n)[None, :]).sum(axis=1)
        hist = np.bincount(idx, minlength=2 ** n).astype(np.int64)
        for _ in range(12):
            qs = sorted(rng.choice(n, size=int(rng.integers(1, 5)), replace=False).tolist())
            mk = int(sum(1 << q for q in qs))
            # exact: parity expectation from the outcome distribution == <P> of the state (P = setting on qs)
            sign = np.array([1 - 2 * (bin(o & mk).count("1") & 1) for o in range(2 ** n)])
            e_p = float(np.dot(p, sign))
            e_true = float(pauli_expectation(psi, np.array(qs), setting[qs]))
            assert abs(e_p - e_true) < 1e-9, (e_p, e_true)
            # histogram path == rows (exact identity on the same samples)
            sh = int(_parity_from_hist(hist, np.array([mk], np.int64))[0])
            par = (O[:, qs].astype(np.int64).sum(axis=1) & 1)
            assert sh == int(len(par)) - 2 * int(par.sum())
            # row sampler unbiased for <P> (5 sigma)
            assert abs(sh / len(par) - e_true) < 5 / np.sqrt(len(par))
