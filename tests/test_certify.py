"""Compiled query half-widths equal bounds.query_halfwidth; certification depends on counts only."""
import os
import numpy as np

from shadowopt.certify import BatchCSR, certify
from shadowopt.bounds import query_halfwidth
from shadowopt.store import TermBatch
from shadowopt.workloads import w1_energy, w3_local_fidelities, w4_zipf

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _check(queries, rng):
    csr = BatchCSR(queries)
    m = rng.integers(1, 50000, size=len(csr.terms))
    hw, ok = certify(csr, m)
    slot = {int(u): i for i, u in enumerate(csr.terms)}
    for i, q in enumerate(queries):
        ref = query_halfwidth(q.coeffs, np.array([m[slot[int(u)]] for u in q.terms]), q.delta)
        assert abs(hw[i] - ref) <= 1e-9 * ref, (i, hw[i], ref)
        assert ok[i] == (hw[i] <= q.eps)


def test_matches_reference_halfwidths():
    rng = np.random.default_rng(0)
    d = np.load(os.path.join(ROOT, "stores", "states", "LiH", "truth_W1.npz"))
    U = TermBatch(12, d["ptr"], d["q"], d["b"], d["coeff"])
    _check(w1_energy(U, 0.0, eps_seq=(0.1,)).batches[0], rng)
    _check(w3_local_fidelities(50).batches[0], rng)
    _check(w4_zipf(20, seed=1).batches[0], rng)


def test_count_sources_agree_and_completion_verifies():
    from shadowopt.engine import AppendStore, IncrementalIndex
    from shadowopt.certify import CountSource, check_and_plan
    rng = np.random.default_rng(5)
    fam = w4_zipf(20, seed=1)
    U = fam.universe
    app = AppendStore(U.n)
    ix = IncrementalIndex(app)
    srcs = [CountSource(ix, k, c, block=97) for k in ("bitmap", "sstar") for c in (False, True)]
    for tau, k in enumerate((500, 3000, 64, 20000)):
        app.append(rng.integers(0, 3, (k, U.n)), rng.integers(0, 2, (k, U.n)))
        csr = BatchCSR(fam.batches[tau])
        outs = [check_and_plan(U, csr, s) for s in srcs]
        for o in outs[1:]:
            assert np.array_equal(o["m"], outs[0]["m"]) and np.array_equal(o["ok"], outs[0]["ok"])
            assert len(o["plan"]) == len(outs[0]["plan"])  # decisions depend on counts only
        hw2, ok2 = certify(csr, outs[0]["m_projected"])
        assert np.all(ok2)  # every query certified at the projected counts


def test_boundary_regression_no_tolerance():
    """: a half-width just above eps must not be certified."""
    from shadowopt.workloads import Query
    q = Query(np.array([0]), np.array([1.0]), 0.2716203031479881, 0.05)
    csr = BatchCSR([q])
    hw, ok = certify(csr, np.array([100]))
    assert hw[0] > q.eps and not ok[0]


def test_delta_split_scale_invariant_minimizer():
    """with c = (1e14, 2e14), m = (100, 100), delta = .05 the minimizing split is ~(.0157052, .0342948)."""
    from shadowopt.bounds import delta_split
    from shadowopt.workloads import Query
    d = delta_split(np.array([1e14, 2e14]), np.array([100.0, 100.0]), 0.05)
    assert abs(d[0] - 0.0157052311) < 1e-6 and abs(d[1] - 0.0342947689) < 1e-6
    q = Query(np.array([0, 1]), np.array([1e14, 2e14]), 8.85e13, 0.05)
    hw, ok = certify(BatchCSR([q]), np.array([100, 100]))
    assert abs(hw[0] - 8.8167585337e13) < 1e4 and ok[0]


def test_delta_split_no_overflow_near_float_max():
    """Extreme coefficients: c = (1e308, 1.5e308), m = (100, 100): optimum ~(.0193802, .0306198), H ~ 7.3818730151e307."""
    from shadowopt.bounds import delta_split, query_halfwidth
    from shadowopt.workloads import Query
    c = np.array([1e308, 1.5e308])
    d = delta_split(c, np.array([100.0, 100.0]), 0.05)
    assert abs(d[0] - 0.019380203712317) < 1e-7 and abs(d[1] - 0.030619796287683) < 1e-7
    assert abs(query_halfwidth(c, np.array([100, 100]), 0.05) / 7.381873015094e307 - 1) < 1e-9
    q = Query(np.array([0, 1]), c, 7.39e307, 0.05)
    hw, ok = certify(BatchCSR([q]), np.array([100, 100]))
    assert abs(hw[0] / 7.381873015094e307 - 1) < 1e-9 and ok[0]
