"""OPT_hind: offline hindsight reference for a whole batch sequence (guide §P4.7), as a MIP (scipy / HiGHS) with its LP
relaxation as a lower bound. A reference for H-b, never a claim of the planner.

Action space: uniform appends from the same seeded stream (prefix length L_tau, nondecreasing) and targeted settings from
a candidate set G (the union of settings used by the strategies plus one singleton setting per universe term);
shots are continuous (a relaxation: OPT_hind <= the integer optimum). A term u asked at batch tau is satisfied if
L_tau >= Lstar_u (the prefix where m_R reaches n*_u) OR the targeted shots of compatible settings up to tau reach n*_u.
Cost: t_shot * (L_T + sum x) + t_sub * (sub-jobs), sub-jobs >= ceil-style integer variables per action and batch
(t_set = 0 in the planning model; asserted).
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

from .planner import batch_requirements, compat_setting, INF


def solve_hindsight(fam, stream, cost, settings, time_limit=600, Lcap=None):
    assert cost.t_set == 0.0, "MIP assumes t_set = 0 (planning model)"
    U = fam.universe
    Tb = len(fam.batches)
    reqs = [batch_requirements(fam, t) for t in range(Tb)]
    # candidate settings
    G = {tuple(s) for s in settings}
    for u in sorted({u for r in reqs for u in r}):
        s = np.full(U.n, 2, np.int64)
        qq, bb = U.term(u)
        s[qq] = bb
        G.add(tuple(int(x) for x in s))
    G = [np.array(g) for g in sorted(G)]
    C = np.array([compat_setting(U, g) for g in G])  # (|G|, U.T)
    Lstar = {u: stream.position(u, n) for r in reqs for u, n in r.items()}
    Lmax = Lcap or int(max([v for v in Lstar.values() if v < INF] + [1]))
    BigS = max(n for r in reqs for n in r.values()) * 2.0
    E = cost.exec_per_subjob
    # variable layout
    idx = {}
    nv = 0

    def var(key):
        nonlocal nv
        idx[key] = nv
        nv += 1
        return nv - 1

    for t in range(Tb):
        var(("L", t)); var(("a", t)); var(("y", t)); var(("nsA", t)); var(("nsT", t))
        for g in range(len(G)):
            var(("x", t, g))
        for u in reqs[t]:
            var(("z", t, u))
    c = np.zeros(nv)
    integrality = np.zeros(nv)
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    for t in range(Tb):
        for g in range(len(G)):
            c[idx[("x", t, g)]] = cost.t_shot
        c[idx[("nsA", t)]] = cost.t_sub
        c[idx[("nsT", t)]] = cost.t_sub
        for k in ("a", "y", "nsA", "nsT"):
            integrality[idx[(k, t)]] = 1
        ub[idx[("a", t)]] = 1
        ub[idx[("y", t)]] = 1
        ub[idx[("L", t)]] = Lmax
        for u in reqs[t]:
            integrality[idx[("z", t, u)]] = 1
            ub[idx[("z", t, u)]] = 1
    c[idx[("L", Tb - 1)]] += cost.t_shot
    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        rows.append(coefs)
        lo.append(l)
        hi.append(h)

    for t in range(Tb):
        prev = {idx[("L", t - 1)]: -1.0} if t > 0 else {}
        # L nondecreasing; append job indicator; sub-jobs
        add({idx[("L", t)]: 1.0, **prev}, 0, np.inf)
        add({idx[("L", t)]: 1.0, **prev, idx[("a", t)]: -Lmax}, -np.inf, 0)
        add({idx[("nsA", t)]: 1.0, idx[("L", t)]: -1.0 / E, **{k: -v / E for k, v in prev.items()}}, 0, np.inf)
        add({idx[("nsA", t)]: 1.0, idx[("a", t)]: -1.0}, 0, np.inf)
        xs = {idx[("x", t, g)]: 1.0 for g in range(len(G))}
        add({**xs, idx[("y", t)]: -BigS * len(G)}, -np.inf, 0)
        add({idx[("nsT", t)]: 1.0, **{k: -1.0 / E for k in xs}}, 0, np.inf)
        add({idx[("nsT", t)]: 1.0, idx[("y", t)]: -1.0}, 0, np.inf)
        for u, n in reqs[t].items():
            z = idx[("z", t, u)]
            ls = Lstar[u]
            if ls >= INF:
                ub[z] = 0  # cannot be met by appends within the stream cap
            else:
                add({idx[("L", t)]: 1.0, z: -float(ls)}, 0, np.inf)  # L_t >= Lstar * z
            # targeted shots up to t: sum x * C >= n (1 - z)
            co = {z: float(n)}
            for tp in range(t + 1):
                for g in np.nonzero(C[:, u])[0]:
                    co[idx[("x", tp, int(g))]] = 1.0
            add(co, float(n), np.inf)
    A = lil_matrix((len(rows), nv))
    for i, r in enumerate(rows):
        for k, v in r.items():
            A[i, k] = v
    cons = LinearConstraint(A.tocsr(), np.array(lo), np.array(hi))
    res = milp(c, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "disp": False})
    lp = milp(c, constraints=cons, integrality=np.zeros(nv), bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    out = {"status": int(res.status), "message": res.message, "mip_obj": float(res.fun) if res.x is not None else None,
           "mip_dual_bound": float(getattr(res, "mip_dual_bound", np.nan)) if res.x is not None else None,
           "lp_status": int(lp.status), "lp_message": lp.message,
           # the LP objective is a valid lower bound only if the LP was solved to optimality
           "lp_bound": float(lp.fun) if (lp.x is not None and lp.status == 0) else None,
           "mip_optimal": bool(res.status == 0), "n_vars": nv, "n_cons": len(rows),
           "n_settings": len(G)}
    if res.x is not None:  # decomposition (diagnosis): sub-jobs and shots of the MIP solution
        x = res.x
        out["mip_subjobs"] = float(sum(x[idx[("nsA", t)]] + x[idx[("nsT", t)]] for t in range(Tb)))
        out["mip_append_rows"] = float(x[idx[("L", Tb - 1)]])
        out["mip_target_shots"] = float(sum(x[idx[("x", t, g)]] for t in range(Tb) for g in range(len(G))))
        out["mip_batches_with_job"] = int(sum(1 for t in range(Tb) if x[idx[("a", t)]] > 0.5 or x[idx[("y", t)]] > 0.5))
    return out
