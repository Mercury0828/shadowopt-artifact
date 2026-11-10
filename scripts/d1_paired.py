"""D1 addendum A4: interleaved paired re-time (S* / S*fast vs each path), N = 1e6, T = 16 and 1, 7 pairs. Lock held."""
import os
import sys
import time

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shadowopt.store import Store, pack_rows, build_bitmaps, packed_bytes
from shadowopt.scan import SStar, SStarFast, naive_reference
from shadowopt.bitmaps import BitmapPath
from shadowopt.sharedplan import SharedPlanPath
from shadowopt.views import ViewSet, ViewPath, select_views
from scripts.d0_anchors import load_truth
from scripts.d1_timing import ROOT, STATES, done_keys, emit, tuned_params, cpu_load, query_frequencies


def main(pairs=7):
    params = tuned_params()
    for state, wl in STATES:
        st = Store.load(os.path.join(ROOT, "stores", state, "1e6"))
        terms, _ = load_truth(state)
        M0, S0 = naive_reference(st, terms)
        rows = pack_rows(st)
        Z, O = build_bitmaps(st)
        sup = [frozenset(terms.term(t)[0].tolist()) for t in range(terms.T)]
        chosen, _ = select_views(sup, query_frequencies(state, terms.T), (st.N + 63) // 64,
                                5 / 3 * packed_bytes(st.n, st.N), s_max=6)
        vs = ViewSet(chosen).build(st)
        for T in (16, 1):
            numba.set_num_threads(T)
            scans = {"S*": SStar(rows, st.n, block=params["S*"]), "S*fast": SStarFast(rows, st.n, block=params["S*"])}
            paths = {"bitmaps": BitmapPath(Z, O, st.N, chunk=params["bitmaps"]),
                     "shared": SharedPlanPath(Z, O, st.N, chunk=params["shared"]),
                     "views": ViewPath(vs, SharedPlanPath(Z, O, st.N, chunk=params["shared"]))}
            for p in list(scans.values()) + list(paths.values()):
                M, S = p.run(terms)
                assert np.array_equal(M, M0) and np.array_equal(S, S0)
            for sn, sp in scans.items():
                for pn, pp in paths.items():
                    key = f"paired|{state}|1000000|{sn}|{pn}|T{T}"
                    if key in done_keys():
                        continue
                    ratios, ts, tp, loads = [], [], [], []
                    sp.run(terms)
                    pp.run(terms)
                    for i in range(pairs):
                        loads.append(cpu_load())
                        order = (sp, pp) if i % 2 == 0 else (pp, sp)
                        tt = {}
                        for obj in order:
                            t0 = time.perf_counter()
                            obj.run(terms)
                            tt[id(obj)] = time.perf_counter() - t0
                        ts.append(tt[id(sp)])
                        tp.append(tt[id(pp)])
                        ratios.append(tt[id(sp)] / tt[id(pp)])
                    emit({"key": key, "kind": "paired", "state": state, "workload": wl, "N": 1_000_000,
                          "scan": sn, "path": pn, "threads": T, "pairs": pairs, "ratios": ratios,
                          "median_ratio": float(np.median(ratios)), "q25_ratio": float(np.quantile(ratios, 0.25)),
                          "q75_ratio": float(np.quantile(ratios, 0.75)), "scan_times_s": ts, "path_times_s": tp,
                          "cpu_load_before_pct": loads, "addendum": "A4", "median_s": float(np.median(ratios)),
                          "q25_s": float(np.quantile(ratios, 0.25)), "q75_s": float(np.quantile(ratios, 0.75))})


if __name__ == "__main__":
    main()
