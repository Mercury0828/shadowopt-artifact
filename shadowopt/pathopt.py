"""Access-path cost model and choice (C-INDEX; guide §3c module "pathopt"). Plan-level, from EXACT work counts:

  T_hat(plan) = tau_fixed + sum over stages of max(work_stage / rate_stage)

Work counts per batch (computed from the terms, the store size and the structures' state — never from outcomes):
- S*:        row-words touched  = N * sum_q d_q           (d_q = packed words term q reads; select-only: basis words)
- bitmaps:   bitmap words       = W * (K + Q) / survive   (W = ceil(N/64), K = sum of weights, Q terms; full eval)
- shared:    bitmap words       = W * (E + Q)             (E = trie nodes)
- views:     cells              = sum over covered terms 3^(s_v - k_q)  + prepare s_v 6^s_v for dirty views
             + the shared plan for uncovered terms
- cache:     delta rows only    = the chosen evaluator on [r, N) for cached keys, full for new keys
Rates (words/s, cells/s) and tau_fixed per path are CALIBRATED on independent synthetic stores (scripts/pathopt_calib.py,
under the CPU lock; never on DEV/TEST timings), then frozen. Choice = argmin T_hat over the available paths, including
construction or maintenance owed before execution (lifecycle accounting); S* is always a candidate.
"""
from dataclasses import dataclass, field

import numpy as np

from .store import row_words, n_words


@dataclass
class Rates:
    """Calibrated constants (planning placeholders until scripts/pathopt_calib.py fits them; label says which)."""
    sstar_words_per_s: float = 2.0e10
    bitmap_words_per_s: float = 2.0e10
    shared_words_per_s: float = 2.0e10
    view_cells_per_s: float = 2.0e9
    tau_fixed: dict = field(default_factory=lambda: {"sstar": 2e-4, "bitmap": 1e-4, "shared": 3e-4, "views": 3e-4})
    label: str = "planning placeholders (not calibrated)"


def sstar_words(terms, n, N, select_only=False):
    """Packed row words each term touches (basis bits 2i, 2i+1; outcome bit 2n+i), summed over terms, times N."""
    tot = 0
    for t in range(terms.T):
        qq, _ = terms.term(t)
        ws = set()
        for q in qq.tolist():
            ws.add((2 * q) >> 6)
            ws.add((2 * q + 1) >> 6)
            if not select_only:
                ws.add((2 * n + q) >> 6)
        tot += len(ws)
    return N * tot


def bitmap_words(terms, N, select_only=False):
    W = n_words(N)
    K = int(terms.weights().sum())
    return W * (K * (1 if select_only else 2) + terms.T)


def shared_words(trie, N, select_only=False):
    W = n_words(N)
    return W * (trie.n_nodes * (1 if select_only else 2) + trie.T)


def view_cells(viewset, terms):
    a = viewset.assign(terms)
    cells = 0
    prep = 0
    dirty = set()
    for t in range(terms.T):
        if a[t] < 0:
            continue
        v = viewset.views[a[t]]
        cells += 3 ** (v.s - len(terms.term(t)[0]))
        if v.W is None:
            dirty.add(a[t])
    for i in dirty:
        v = viewset.views[i]
        prep += v.s * 6 ** v.s
    return cells, prep, np.nonzero(a < 0)[0]


def predict(terms, n, N, rates, trie=None, viewset=None, select_only=False):
    """Predicted seconds per available path for one batch."""
    out = {"sstar": rates.tau_fixed["sstar"] + sstar_words(terms, n, N, select_only) / rates.sstar_words_per_s,
           "bitmap": rates.tau_fixed["bitmap"] + bitmap_words(terms, N, select_only) / rates.bitmap_words_per_s}
    if trie is not None:
        out["shared"] = rates.tau_fixed["shared"] + shared_words(trie, N, select_only) / rates.shared_words_per_s
    if viewset is not None and trie is not None:
        cells, prep, rest = view_cells(viewset, terms)
        rest_frac = len(rest) / max(1, terms.T)
        out["views"] = (rates.tau_fixed["views"] + (cells + prep) / rates.view_cells_per_s
                        + rest_frac * (out["shared"] - rates.tau_fixed["shared"]))
    return out


def choose(terms, n, N, rates, trie=None, viewset=None, select_only=False):
    p = predict(terms, n, N, rates, trie, viewset, select_only)
    best = min(p, key=p.get)
    return best, p
