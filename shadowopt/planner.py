"""Acquisition planning (guide §P4.7) and the fixed baselines F1-F5, F7 (guide §P6.3), Phase-0 version.

Everything here is count-driven: decisions use drawn bases, counts, the query sequence and past batches only — never
outcomes (B2). Usage comes from one shared CostModel. Feasibility of every answered query is re-checked by an independent
function (`check_feasible`) from the counts.

Model (Phase 0):
- Random store: a seeded uniform basis stream (common to every strategy of a (family, seed)); the store after appends is a
  prefix of length L. Counts m_R(u, L) of universe terms are exact (StreamCounter).
- Targeted samples: a setting = a full basis assignment (the grouped terms' bases on their supports, Z elsewhere), s
  shots; m_T(u) = sum of shots of settings compatible with u. Targeted samples are kept separately (B6 default).
- A term is answered from the source with the larger count (count-driven choice); a query is feasible if every term of it
  reaches its threshold n*_u (bounds.term_thresholds), which implies the stated half-width <= eps.
- Derandomized store (F4 only): its own snapshots, counted exactly.
"""
import math
from dataclasses import dataclass, field

import numba
import numpy as np
from numba import njit, prange

from ._bits import popcnt64
from .bounds import term_thresholds, query_halfwidth
from .cost import CostModel

INF = np.iinfo(np.int64).max // 4


# ----------------------------------------------------------------------------------------------------------------------
# Exact counts over a seeded uniform basis stream
# ----------------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _chunk_counts(bases, ptr, q, b):
    """counts per term over the rows of one chunk (bit-sliced, same primitive as the access paths)."""
    R, n = bases.shape
    NW = (R + 63) // 64
    Z = np.zeros((n, 3, NW), np.uint64)
    for i in prange(n):
        for r in range(R):
            Z[i, bases[r, i], r >> 6] |= np.uint64(1) << np.uint64(r & 63)
    T = ptr.shape[0] - 1
    out = np.zeros(T, np.int64)
    for t in prange(T):
        c = 0
        for w in range(NW):
            a = ~np.uint64(0)
            for j in range(ptr[t], ptr[t + 1]):
                a &= Z[q[j], b[j], w]
            c += popcnt64(a)
        out[t] = c
    return out


@njit(cache=True)
def _row_masks_pos(bases, qq, bb, need):
    """0-based row index (within the chunk) of the need-th compatible row, or -1."""
    c = 0
    for r in range(bases.shape[0]):
        ok = True
        for j in range(qq.shape[0]):
            if bases[r, qq[j]] != bb[j]:
                ok = False
                break
        if ok:
            c += 1
            if c == need:
                return r
    return -1


class StreamCounter:
    def __init__(self, universe, seed, chunk=1 << 16, max_rows=1 << 28):
        self.U = universe
        self.n = universe.n
        self.seed = seed
        self.chunk = chunk
        self.max_chunks = max_rows // chunk
        self.counts = np.zeros((0, universe.T), np.int64)  # per chunk
        self.cum = np.zeros((1, universe.T), np.int64)  # cum[c] = counts in chunks < c

    def bases(self, c):
        return np.random.default_rng([self.seed, c]).integers(0, 3, size=(self.chunk, self.n), dtype=np.uint8)

    def _ensure(self, nchunks):
        nchunks = min(nchunks, self.max_chunks)
        have = self.counts.shape[0]
        if nchunks <= have:
            return
        new = np.zeros((nchunks - have, self.U.T), np.int64)
        for c in range(have, nchunks):
            new[c - have] = _chunk_counts(self.bases(c), self.U.ptr, self.U.q, self.U.b)
        self.counts = np.vstack([self.counts, new])
        self.cum = np.vstack([np.zeros((1, self.U.T), np.int64), np.cumsum(self.counts, axis=0)])

    def count(self, u_idx, L):
        """exact m_R(u, L) for an array of universe indices."""
        u_idx = np.asarray(u_idx, np.int64)
        c, r = divmod(int(L), self.chunk)
        self._ensure(c + 1)
        out = self.cum[min(c, self.cum.shape[0] - 1), u_idx].copy()
        if r and c < self.max_chunks:
            B = self.bases(c)[:r]
            sub = self.U.subset(u_idx)
            out += _chunk_counts(B, sub.ptr, sub.q, sub.b)
        return out

    def position(self, u, need):
        """Smallest prefix length L with m_R(u, L) >= need (INF if beyond max_rows)."""
        if need <= 0:
            return 0
        # grow until the cumulative count reaches need
        while True:
            have = self.counts.shape[0]
            if self.cum[have, u] >= need:
                break
            if have >= self.max_chunks:
                return INF
            self._ensure(min(self.max_chunks, max(have * 2, have + 8)))
        c = int(np.searchsorted(self.cum[:, u], need, side="left")) - 1  # chunk containing the need-th row
        before = self.cum[c, u]
        qq, bb = self.U.term(u)
        r = _row_masks_pos(self.bases(c), qq, bb, need - before)
        return c * self.chunk + r + 1


# ----------------------------------------------------------------------------------------------------------------------
# Targeted settings
# ----------------------------------------------------------------------------------------------------------------------
def group_terms(universe, terms, order_key=None):
    """Greedy qubit-wise-compatible grouping (first fit; terms sorted by weight desc). Returns list of (setting, members)
    where setting is a full basis array (Z = 2 on unassigned qubits... assigned lazily) and members universe indices."""
    n = universe.n
    w = universe.weights()
    terms = sorted(terms, key=order_key or (lambda u: (-w[u], u)))
    groups = []  # (assign dict, members)
    for u in terms:
        qq, bb = universe.term(u)
        placed = False
        for g in groups:
            asg = g[0]
            if all(asg.get(int(qi), int(bi)) == int(bi) for qi, bi in zip(qq, bb)):
                for qi, bi in zip(qq, bb):
                    asg[int(qi)] = int(bi)
                g[1].append(u)
                placed = True
                break
        if not placed:
            groups.append(({int(qi): int(bi) for qi, bi in zip(qq, bb)}, [u]))
    out = []
    for asg, mem in groups:
        s = np.full(n, 2, np.int64)
        for qi, bi in asg.items():
            s[qi] = bi
        out.append((s, mem))
    return out


@njit(parallel=True, cache=True)
def _setting_counts(S, shots, ptr, q, b):
    T = ptr.shape[0] - 1
    out = np.zeros(T, np.int64)
    for t in prange(T):
        c = 0
        for r in range(S.shape[0]):
            ok = True
            for j in range(ptr[t], ptr[t + 1]):
                if S[r, q[j]] != b[j]:
                    ok = False
                    break
            if ok:
                c += shots[r]
        out[t] = c
    return out


def setting_counts(universe, settings_shots):
    """Per universe term: total shots of the compatible settings (same result as summing compat_setting * shots)."""
    if not settings_shots:
        return np.zeros(universe.T, np.int64)
    S = np.array([s for s, _ in settings_shots], np.int64)
    k = np.array([int(x) for _, x in settings_shots], np.int64)
    return _setting_counts(S, k, universe.ptr, universe.q, universe.b)


def compat_setting(universe, setting):
    """boolean array over the universe: term compatible with a full basis assignment."""
    ok = np.ones(universe.T, bool)
    for t in range(universe.T):
        qq, bb = universe.term(t)
        ok[t] = bool(np.all(setting[qq] == bb))
    return ok


# ----------------------------------------------------------------------------------------------------------------------
# Derandomized measurements (HKP 2021 greedy conditional-expectation rule), F4
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True)
def _derand_one(n, ptr, q, b, weight, eta):
    """Choose one measurement basis assignment greedily qubit by qubit minimizing
    sum_j weight_j * (1 - (1 - e^-eta) * prod_{assigned q in S_j} [match] * (1/3)^{#unassigned q in S_j})."""
    T = ptr.shape[0] - 1
    assign = -np.ones(n, np.int64)
    fac = 1.0 - math.exp(-eta)
    for i in range(n):
        best = 0
        bestv = 1e300
        for cand in range(3):
            assign[i] = cand
            v = 0.0
            for t in range(T):
                if weight[t] == 0.0:
                    continue
                p = 1.0
                for j in range(ptr[t], ptr[t + 1]):
                    a = assign[q[j]]
                    if a < 0:
                        p *= 1.0 / 3.0
                    elif a != b[j]:
                        p = 0.0
                        break
                v += weight[t] * (1.0 - fac * p)
            if v < bestv:
                bestv = v
                best = cand
        assign[i] = best
    return assign


def derandomize_until(universe, deficit, need, have, eta=0.9, max_meas=2_000_000):
    """Generate derandomized single-shot measurements until have[u] >= need[u] for u in deficit.
    Returns list of (setting, shots) with consecutive identical settings merged into one setting with several shots."""
    sub = universe.subset(deficit)
    h = have[deficit].astype(np.float64).copy()
    nd = need[deficit].astype(np.float64)
    out = []
    M = 0
    while np.any(h < nd) and M < max_meas:
        weight = np.where(h < nd, np.exp(-eta * h / np.maximum(nd, 1) * 10.0), 0.0)
        s = _derand_one(universe.n, sub.ptr, sub.q, sub.b, weight, eta)
        # ADAPTATION (not the published shot-by-shot algorithm): repeat the chosen setting for shots = smallest remaining
        # need among the unmet terms it hits; the greedy choice may change within these repetitions (orientation check)
        hit = np.array([np.all(s[sub.term(t)[0]] == sub.term(t)[1]) for t in range(sub.T)])
        rem = (nd - h)[hit & (h < nd)]
        if len(rem) == 0:
            # the greedy choice hits no deficit term (cannot happen unless weights vanish); force a targeted setting
            t0 = int(np.nonzero(h < nd)[0][0])
            s = np.full(universe.n, 2, np.int64)
            qq, bb = sub.term(t0)
            s[qq] = bb
            hit = np.array([np.all(s[sub.term(t)[0]] == sub.term(t)[1]) for t in range(sub.T)])
            rem = (nd - h)[hit & (h < nd)]
        shots = int(max(1, math.ceil(rem.min())))
        h += hit * shots
        out.append((s, shots))
        M += shots
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Engine state shared by every strategy
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class EngineState:
    universe: object
    stream: StreamCounter
    cost: CostModel
    L: int = 0  # random-store prefix length
    mT: np.ndarray = None  # targeted counts per universe term
    mD: np.ndarray = None  # derandomized-store counts (F4)
    usage: float = 0.0
    log: list = field(default_factory=list)
    settings_used: list = field(default_factory=list)
    pooled: bool = False  # True: answers and needs use POOLED counts mR + mT + mD (C-BOUND Lemma 1; harness A9 fix)

    def __post_init__(self):
        if self.mT is None:
            self.mT = np.zeros(self.universe.T, np.int64)
        if self.mD is None:
            self.mD = np.zeros(self.universe.T, np.int64)

    def mR(self, u_idx):
        return self.stream.count(u_idx, self.L)

    def best_counts(self, u_idx):
        u_idx = np.asarray(u_idx, np.int64)
        if self.pooled:
            return self.mR(u_idx) + self.mT[u_idx] + self.mD[u_idx]
        return np.maximum(np.maximum(self.mR(u_idx), self.mT[u_idx]), self.mD[u_idx])

    def do_append(self, n_add, tau, tag):
        if n_add <= 0:
            return 0.0
        self.L += int(n_add)
        u = self.cost.append(n_add)
        self.usage += u
        self.log.append({"tau": tau, "action": "APPEND", "n": int(n_add), "U": u, "tag": tag})
        return u

    def do_target(self, settings_shots, tau, tag):
        settings_shots = [(s, int(k)) for s, k in settings_shots if k > 0]
        if not settings_shots:
            return 0.0
        self.mT += setting_counts(self.universe, settings_shots)
        for s, k in settings_shots:
            self.settings_used.append(tuple(int(x) for x in s))
        u = self.cost.target([k for _, k in settings_shots])
        self.usage += u
        self.log.append({"tau": tau, "action": "TARGET", "settings": len(settings_shots),
                         "shots": int(sum(k for _, k in settings_shots)), "U": u, "tag": tag})
        return u

    def do_derand(self, settings_shots, tau, tag):
        settings_shots = [(s, int(k)) for s, k in settings_shots if k > 0]
        if not settings_shots:
            return 0.0
        self.mD += setting_counts(self.universe, settings_shots)
        # identical consecutive derandomized measurements are consolidated into one setting with several shots and
        # charged through the shared job cost (same as TARGET)
        n_exec = sum(k for _, k in settings_shots)
        u = self.cost.job(len(settings_shots), n_exec)
        self.usage += u
        self.log.append({"tau": tau, "action": "DERAND", "n": int(n_exec), "U": u, "tag": tag})
        return u


def batch_requirements(fam, tau):
    """dict universe index -> threshold n* (max over the batch's queries containing it)."""
    req = {}
    for qy in fam.batches[tau]:
        th = term_thresholds(qy.coeffs, qy.eps, qy.delta)
        for u, t in zip(qy.terms.tolist(), th.tolist()):
            if t > 0:
                req[u] = max(req.get(u, 0), int(t))
    return req


def check_feasible(st, fam, tau):
    """Independent re-check: every query of the batch has stated half-width <= eps with the counts each term is answered
    from (the larger source). Returns (ok, worst ratio)."""
    worst = 0.0
    for qy in fam.batches[tau]:
        m = st.best_counts(qy.terms)
        hw = query_halfwidth(qy.coeffs, m, qy.delta)
        worst = max(worst, hw / qy.eps)
    return worst <= 1.0, worst  # exact predicate HW <= eps


# ----------------------------------------------------------------------------------------------------------------------
# Plan pieces
# ----------------------------------------------------------------------------------------------------------------------
def append_size(st, req_items):
    """Exact minimal N_add so that m_R(u) >= need for all (u, need) — using the drawn stream (counts known before any
    QPU second, DERIVED D3)."""
    if not req_items:
        return 0
    if st.pooled:  # random rows needed beyond what the targeted/derandomized sources already contribute
        req_items = [(u, need - int(st.mT[u]) - int(st.mD[u])) for u, need in req_items]
    pos = max(st.stream.position(u, need) for u, need in req_items)
    return INF if pos >= INF else max(0, pos - st.L)


def target_plan(st, items, reuse=True):
    """Settings and shots for TARGET of (u, need) items: greedy grouping, shots = max remaining targeted need."""
    if not items:
        return []
    need = dict(items)
    groups = group_terms(st.universe, list(need))
    out = []
    for s, mem in groups:
        have = (st.best_counts(mem) if st.pooled else st.mT[mem]) if reuse else np.zeros(len(mem), np.int64)
        shots = int(max(0, max(need[u] - h for u, h in zip(mem, have))))
        out.append((s, shots))
    return out


def plan_candidates(st, deficit, req):
    """All APPEND/TARGET mixtures by weight cut: terms with k <= cut appended, the rest targeted."""
    w = st.universe.weights()
    cuts = [-1] + sorted(set(int(w[u]) for u in deficit))
    cands = []
    for cut in cuts:
        A = [u for u in deficit if w[u] <= cut]
        n_add = append_size(st, [(u, req[u]) for u in A])
        if n_add >= INF:
            continue
        # terms the append already satisfies are dropped from TARGET
        rest = [u for u in deficit if w[u] > cut]
        if rest and n_add > 0:
            m_after = st.stream.count(rest, st.L + n_add)
            rest = [u for u, m in zip(rest, m_after) if m < req[u]]
        tp = target_plan(st, [(u, req[u]) for u in rest])
        U = st.cost.append(n_add) + st.cost.target([k for _, k in tp])
        cands.append({"cut": cut, "n_add": int(n_add), "target": tp, "U": U})
    return cands


def credit(st, cand, history, H, deficit_set):
    """Expected future usage saved by the counts this plan adds to terms seen in past batches (past only):
    t_shot * sum_u min(1, H f_u) * min(Delta_u, need_u) / g, with f_u the fraction of past batches asking u, need_u its
    last threshold minus its current best count, g the mean group size of a TARGET grouping of those terms."""
    if not history:
        return 0.0
    fu = {}
    last = {}
    for req in history:
        for u, t in req.items():
            fu[u] = fu.get(u, 0) + 1
            last[u] = t
    us = np.array(sorted(fu), np.int64)
    f = np.array([fu[u] / len(history) for u in us])
    need = np.maximum(0, np.array([last[u] for u in us]) - st.best_counts(us))
    if not np.any(need > 0):
        return 0.0
    dR = st.stream.count(us, st.L + cand["n_add"]) - st.mR(us) if cand["n_add"] > 0 else np.zeros(len(us), np.int64)
    dT = np.zeros(len(us), np.int64)
    for s, k in cand["target"]:
        dT += compat_setting(st.universe, s)[us] * k
    delta = np.maximum(dR, dT)
    gsize = max(1.0, len(us) / max(1, len(group_terms(st.universe, list(us[need > 0])))))
    return st.cost.t_shot * float(np.sum(np.minimum(1.0, H * f) * np.minimum(delta, need))) / gsize


# ----------------------------------------------------------------------------------------------------------------------
# Strategies
# ----------------------------------------------------------------------------------------------------------------------
def run_strategy(name, fam, seed, cost=None, lam=0.0, H=3, stream=None, max_rounds=50, L0=0, pooled=False):
    """Run one strategy over a family's batch sequence. Returns dict with total usage, per-batch log, feasibility."""
    cost = cost or CostModel()
    stream = stream or StreamCounter(fam.universe, seed)
    st = EngineState(fam.universe, stream, cost)
    st.pooled = bool(pooled)
    st.L = int(L0)  # initial persistent store = the first L0 rows of the seeded stream (A9); 0 = Phase-0 behaviour
    w = fam.universe.weights()
    history = []
    per_batch = []
    for tau in range(len(fam.batches)):
        req = batch_requirements(fam, tau)
        u0 = st.usage
        if name == "F1":  # always direct, no reuse: fresh targeted samples for every term of every batch
            st.mT[:] = 0
            st.do_target(target_plan(st, list(req.items()), reuse=False), tau, "F1")
            ok, worst = _check_sources(st, fam, tau, use_R=False, use_D=False)
            per_batch.append({"tau": tau, "U": st.usage - u0, "ok": ok, "worst": worst})
            st.mT[:] = 0
            history.append(req)
            continue
        for _round in range(max_rounds):
            us = np.array(sorted(req), np.int64)
            have = st.best_counts(us)
            deficit = [int(u) for u, h in zip(us, have) if h < req[int(u)]]
            if not deficit:
                break
            if name.startswith("SHADOWOPT"):
                cands = plan_candidates(st, deficit, req)
                if lam > 0:
                    for c in cands:
                        c["obj"] = c["U"] - lam * credit(st, c, history, H, set(deficit))
                else:
                    for c in cands:
                        c["obj"] = c["U"]
                if name == "SHADOWOPT_APPENDONLY":  # negative control: must reproduce F2
                    cands = [c for c in cands if c["cut"] == max(c2["cut"] for c2 in cands)] or cands
                    if cands[0]["target"] and any(k > 0 for _, k in cands[0]["target"]):
                        return {"name": name, "infeasible": True, "per_batch": per_batch, "log": st.log}
                best = min(cands, key=lambda c: (c["obj"], c["U"]))
                ext = {"tau": tau, "U_append_only": next((c["U"] for c in cands if c["cut"] == max(x["cut"] for x in cands)
                                                         and not any(k > 0 for _, k in c["target"])), None),
                       "U_target_only": next((c["U"] for c in cands if c["cut"] == -1), None), "chosen_cut": best["cut"],
                       "chosen_U": best["U"], "n_deficit": len(deficit)}
                st.log.append({"tau": tau, "action": "CANDIDATES", **ext})
                st.do_append(best["n_add"], tau, f"{name}:cut{best['cut']}")
                st.do_target(best["target"], tau, f"{name}:cut{best['cut']}")
            elif name == "F2":
                n_add = append_size(st, [(u, req[u]) for u in deficit])
                if n_add >= INF:
                    return {"name": name, "infeasible": True, "usage_lower_bound": st.usage + cost.append(
                        st.stream.max_chunks * st.stream.chunk - st.L), "per_batch": per_batch, "log": st.log}
                st.do_append(n_add, tau, "F2")
            elif name == "F3":
                st.do_target(target_plan(st, [(u, req[u]) for u in deficit]), tau, "F3")
            elif name == "F4p":  # F4' = the Phase-0 ADAPTED derandomization (kept for the record; not the published one)
                need = np.zeros(fam.universe.T, np.int64)
                for u in deficit:
                    need[u] = req[u]
                ss = derandomize_until(fam.universe, np.array(deficit), need,
                                       st.best_counts(np.arange(fam.universe.T)) if st.pooled else st.mD)
                st.do_derand(ss, tau, "F4p")
            elif name == "F4qs":  # F4 + query-level count-aware stopping (strengthened baseline; B2-safe: counts only)
                from .derand import hkp_derandomize_until_feasible, consolidate
                dd = np.array(deficit, np.int64)
                pos = {u: i for i, u in enumerate(deficit)}
                qs_ = [qy for qy in fam.batches[tau] if any(u in pos for u in qy.terms.tolist())]

                def _feas(mvec, qs_=qs_, pos=pos):
                    for qy in qs_:
                        mm = st.best_counts(qy.terms).astype(np.float64)
                        for k_, u in enumerate(qy.terms.tolist()):
                            if u in pos:
                                mm[k_] = max(mm[k_], mvec[pos[u]])
                        if query_halfwidth(qy.coeffs, mm, qy.delta) > qy.eps:
                            return False
                    return True
                seq, _ = hkp_derandomize_until_feasible(fam.universe, dd, np.array([req[u] for u in deficit]),
                                                        st.best_counts(dd) if st.pooled else st.mD[dd], _feas)
                st.do_derand(consolidate(seq), tau, "F4qs")
                if check_feasible(st, fam, tau)[0]:
                    break
            elif name == "F4":  # published HKP derandomization (shadowopt/derand.py, validated against the reference)
                from .derand import hkp_derandomize, consolidate
                dd = np.array(deficit, np.int64)
                seq, _ = hkp_derandomize(fam.universe, dd, np.array([req[u] for u in deficit]),
                                         have=st.best_counts(dd) if st.pooled else st.mD[dd])
                st.do_derand(consolidate(seq), tau, "F4")
            elif name == "F6":
                # ShadowGrouping (Gresch & Kliesch, code gitlab.com/GreschAI/shadowgrouping @ ac472cd6), Bernstein-bound
                # weights (alpha = 1) as in its tutorial; hits initialized from our current counts (reuse); one shot per
                # setting; settings stored as targeted samples. Time-boxed (raises TimeoutError).
                import sys as _sys, os as _os, time as _time
                _p = _os.path.join(_os.path.dirname(__file__), "..", "third_party", "shadowgrouping")
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
                from shadowgrouping.measurement_schemes import Shadow_Grouping
                from shadowgrouping.weight_functions import Bernstein_bound
                U = fam.universe
                obs = np.zeros((len(deficit), U.n), int)
                wts = np.ones(len(deficit))
                coef = {}
                for qy in fam.batches[tau]:
                    for u, c in zip(qy.terms.tolist(), qy.coeffs.tolist()):
                        coef[u] = max(coef.get(u, 0.0), abs(c))
                for i, u in enumerate(deficit):
                    qq, bb = U.term(u)
                    obs[i, qq] = bb + 1
                    wts[i] = coef.get(u, 1.0) or 1.0
                eps = min(qy.eps for qy in fam.batches[tau])
                sg = Shadow_Grouping(obs, wts, eps, Bernstein_bound(alpha=1)())
                sg.N_hits = st.mT[np.array(deficit)].astype(int)  # hits counted in the targeted source F6 adds to
                need = np.array([req[u] for u in deficit])
                t_start = _time.time()
                ss = []
                while np.any(sg.N_hits < need):
                    setting, _ = sg.find_setting()
                    s_full = np.where(setting > 0, setting - 1, 2).astype(np.int64)
                    if ss and np.array_equal(ss[-1][0], s_full):
                        ss[-1] = (s_full, ss[-1][1] + 1)
                    else:
                        ss.append((s_full, 1))
                    if _time.time() - t_start > getattr(fam, "f6_timebox_s", 900):
                        raise TimeoutError(f"F6 time box: batch {tau}, {sum(k for _, k in ss)} settings in "
                                           f"{_time.time() - t_start:.0f} s, remaining need {int(np.sum(np.maximum(0, need - sg.N_hits)))}")
                # one execution per setting-shot; stored as targeted samples (reused later)
                for s_full, k in ss:
                    st.mT += compat_setting(st.universe, s_full) * k
                n_exec = sum(k for _, k in ss)
                u_ = cost.job(len(ss), n_exec)
                st.usage += u_
                st.log.append({"tau": tau, "action": "F6", "n": int(n_exec), "settings": len(ss), "U": u_})
            elif name == "F5":
                # Ma-Liu static rule on the batch (Statements 1 and 2; hardware times of their Section 2)
                qs = [qy for qy in fam.batches[tau] if any(u in deficit for u in qy.terms.tolist())]
                M = max(1, len(qs))
                Lq = max(1.0, np.mean([len(qy.terms) for qy in qs])) if qs else 1.0
                wmax = max(int(w[u]) for u in deficit)
                eps = min(qy.eps for qy in qs) if qs else 0.05
                dl = min(qy.delta for qy in qs) if qs else 0.05
                T_sh = 17 * Lq * 3 ** wmax / eps ** 2 * math.log(2 * M / dl)
                G = fam.universe.n * T_sh
                C = M * Lq * (T_sh * (1 / 3) ** wmax * (wmax + 1) + 2 * math.log(2 * M / dl) + 2)
                t_sh = G * 10e-9 + T_sh * 10e-6 + C / 1e15
                T_dir = 0.5 * M * Lq ** 3 / eps ** 2 * math.log(2 * M * Lq / dl)
                t_dir = T_dir * 10e-6 + fam.universe.n * T_dir * 10e-9
                if t_sh <= t_dir:
                    n_add = append_size(st, [(u, req[u]) for u in deficit])
                    if n_add >= INF:  # cannot append enough within the stream cap: fall back to TARGET
                        st.do_target(target_plan(st, [(u, req[u]) for u in deficit]), tau, "F5:fallback")
                    else:
                        st.do_append(n_add, tau, "F5:shadow")
                else:
                    st.do_target(target_plan(st, [(u, req[u]) for u in deficit]), tau, "F5:direct")
            elif name == "F7":
                mR = st.best_counts(np.array(deficit)) if st.pooled else st.mR(np.array(deficit))
                n_exp = max(int(3 ** int(w[u]) * max(0, req[u] - m)) for u, m in zip(deficit, mR))
                tp = target_plan(st, [(u, req[u]) for u in deficit])
                if cost.append(n_exp) <= cost.target([k for _, k in tp]) and n_exp < st.stream.max_chunks * st.stream.chunk:
                    st.do_append(max(1, n_exp), tau, "F7:append")
                else:
                    st.do_target(tp, tau, "F7:target")
            else:
                raise KeyError(name)
        ok, worst = check_feasible(st, fam, tau)
        per_batch.append({"tau": tau, "U": st.usage - u0, "ok": ok, "worst": worst})
        history.append(req)
    return {"name": name, "lam": lam, "H": H, "usage": st.usage, "per_batch": per_batch, "log": st.log,
            "all_ok": all(p["ok"] for p in per_batch), "L_final": st.L, "settings_used": st.settings_used}


def _check_sources(st, fam, tau, use_R=True, use_D=True):
    worst = 0.0
    for qy in fam.batches[tau]:
        m = st.mT[qy.terms].copy()
        if use_R:
            m = np.maximum(m, st.mR(qy.terms))
        if use_D:
            m = np.maximum(m, st.mD[qy.terms])
        worst = max(worst, query_halfwidth(qy.coeffs, m, qy.delta) / qy.eps)
    return worst <= 1.0, worst  # exact predicate HW <= eps
