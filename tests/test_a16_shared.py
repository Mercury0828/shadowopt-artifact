"""A16: the shared-trie count source returns exactly the bitmap/S* counts, including after appends (capacity words beyond
N_rep are zero) and with repeated terms."""
import numpy as np

from shadowopt.certify import BatchCSR, CountSource
from shadowopt.engine import AppendStore, IncrementalIndex
from shadowopt.workloads import w3_local_fidelities, w4_zipf


def test_shared_equals_bitmap_and_sstar():
    rng = np.random.default_rng(16)
    for fam in (w4_zipf(20, seed=1), w3_local_fidelities(20)):
        U = fam.universe
        app = AppendStore(U.n, cap=64)
        ix = IncrementalIndex(app, cap_rows=64)
        srcs = [CountSource(ix, "shared"), CountSource(ix, "bitmap"), CountSource(ix, "sstar", block=61)]
        for tau, k in enumerate((0, 1, 63, 64, 5000, 3, 0, 777)):
            if k:
                app.append(rng.integers(0, 3, (k, U.n)), rng.integers(0, 2, (k, U.n)))
            csr = BatchCSR(fam.batches[tau % len(fam.batches)])
            a, b, c = (s.counts(U, csr) for s in srcs)
            assert np.array_equal(a, b) and np.array_equal(a, c), (tau, k)
