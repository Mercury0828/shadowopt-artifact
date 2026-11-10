"""D1 supplementary run (PREREG addendum A1): SStarFast (compiled encoder) under the same protocol. Lock held."""
import os
import sys

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.store import Store, pack_rows
from shadowopt.scan import SStarFast, naive_reference
from scripts.d0_anchors import load_truth
from scripts.d1_timing import ROOT, STATES, done_keys, emit, timeit, tuned_params


def main():
    params = tuned_params()
    for state, wl in STATES:
        big = Store.load(os.path.join(ROOT, "stores", state, "1e6"))
        terms, _ = load_truth(state)
        for N in (100_000, 1_000_000):
            st = big if N == big.N else big.head(N)
            M0, S0 = naive_reference(st, terms)
            rows = pack_rows(st)
            base = {"state": state, "workload": wl, "N": N, "n": st.n, "n_terms": terms.T,
                    "store_sha256_1e6": big.meta["sha256"], "addendum": "A1"}
            for T in (16, 1):
                numba.set_num_threads(T)
                p = SStarFast(rows, st.n, block=params["S*"])
                M, S = p.run(terms)
                assert np.array_equal(M, M0) and np.array_equal(S, S0)
                done = done_keys()
                k = f"batch|{state}|{N}|S*fast|T{T}"
                if k not in done:
                    r = timeit(lambda: p.run(terms), 5)
                    r.update({**base, "key": k, "kind": "batch", "method": "S*fast", "threads": T, "plan": "included"})
                    emit(r)
                k = f"batchcached|{state}|{N}|S*fast|T{T}"
                if k not in done:
                    plan = p.plan(terms)
                    r = timeit(lambda: p.execute(plan), 5)
                    r.update({**base, "key": k, "kind": "batch", "method": "S*fast", "threads": T, "plan": "cached"})
                    emit(r)
                k = f"select|{state}|{N}|S*fast|T{T}"
                if k not in done:
                    r = timeit(lambda: p.run(terms, select_only=True), 5)
                    r.update({**base, "key": k, "kind": "select", "method": "S*fast", "threads": T})
                    emit(r)


if __name__ == "__main__":
    main()
