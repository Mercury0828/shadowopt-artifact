"""A15: the allocated completion is verified-feasible and never costlier than certify.complete on the same input."""
import os

import numpy as np

from shadowopt.alloc import complete_alloc
from shadowopt.bounds import query_halfwidth
from shadowopt.certify import BatchCSR, certify, complete
from shadowopt.cost import hw1_fitted
from shadowopt.planner import setting_counts
from shadowopt.store import TermBatch
from shadowopt.workloads import w1_energy, w3_local_fidelities, w4_zipf

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _lih():
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    U = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
    return w1_energy(U, 0.0, eps_seq=(0.3,))


def test_alloc_feasible_and_not_costlier():
    rng = np.random.default_rng(15)
    cost = hw1_fitted()
    for fam, tau in [(_lih(), 0), (w3_local_fidelities(20), 0), (w4_zipf(20, seed=1), 3)]:
        csr = BatchCSR(fam.batches[tau])
        for trial in range(3):
            hi = int(rng.choice([1, 300, 3000]))
            m = rng.integers(0, hi, size=len(csr.terms))
            hw, ok = certify(csr, m)
            p0, _ = complete(fam.universe, csr, m, hw, ok, cost=cost)
            p1, mp, info = complete_alloc(fam.universe, csr, m, hw, ok, cost)
            assert cost.target([k for _, k in p1]) <= cost.target([k for _, k in p0]) + 1e-12
            # independent re-check with the Python reference half-width on the plan's own counts
            sub = fam.universe.subset(csr.terms)
            m2 = m + setting_counts(sub, p1)
            assert np.array_equal(m2, mp)
            for i in range(len(csr.qptr) - 1):
                a, b = csr.qptr[i], csr.qptr[i + 1]
                h = query_halfwidth(csr.qcoef[a:b], m2[csr.qterm[a:b]], csr.qdelta[i])
                assert h <= csr.qeps[i] * (1 + 1e-9), (trial, i, h, csr.qeps[i])


def test_alloc_strictly_better_on_w3():
    """The case that motivated A15 (DEV exploration): shared windows, one setting serving many queries."""
    cost = hw1_fitted()
    fam = w3_local_fidelities(20)
    csr = BatchCSR(fam.batches[0])
    m = np.zeros(len(csr.terms), np.int64)
    hw, ok = certify(csr, m)
    p0, _ = complete(fam.universe, csr, m, hw, ok, cost=cost)
    p1, _, info = complete_alloc(fam.universe, csr, m, hw, ok, cost)
    assert info["chosen"].startswith("alloc")
    assert cost.target([k for _, k in p1]) < 0.9 * cost.target([k for _, k in p0])
