"""Shared intersection plans for a batch (a multiple-query-optimization instance, G18).

The literals (qubit, basis) of all terms are ranked by frequency in the batch (most frequent first); every term's
literals are sorted by that rank and the terms are inserted into a trie. A trie node holds the partial AND of the
compatibility bitmaps and the partial XOR of the outcome bitmaps along its root path, so terms sharing a literal prefix
share the work. Evaluation walks the trie in preorder per word chunk with a stack of partial results.
"""
import numpy as np
from numba import njit, prange

from ._bits import popcnt64


class Trie:
    def __init__(self, terms):
        lit_count = {}
        for t in range(terms.T):
            qq, bb = terms.term(t)
            for qi, bi in zip(qq.tolist(), bb.tolist()):
                lit_count[(qi, bi)] = lit_count.get((qi, bi), 0) + 1
        rank = {lit: r for r, lit in enumerate(sorted(lit_count, key=lambda L: (-lit_count[L], L)))}
        seqs = []
        for t in range(terms.T):
            qq, bb = terms.term(t)
            seq = sorted(zip(qq.tolist(), bb.tolist()), key=lambda L: rank[L])
            seqs.append((tuple(rank[L] for L in seq), tuple(seq), t))
        seqs.sort()
        # build preorder node list
        node_q, node_b, node_depth, node_parent = [], [], [], []
        term_node = np.full(terms.T, -1, np.int64)
        path = []  # stack of (rank tuple prefix element, node id)
        for key, seq, t in seqs:
            # common prefix with current path
            c = 0
            while c < len(path) and c < len(key) and path[c][0] == key[c]:
                c += 1
            path = path[:c]
            for d in range(c, len(key)):
                nid = len(node_q)
                node_q.append(seq[d][0])
                node_b.append(seq[d][1])
                node_depth.append(d)
                node_parent.append(path[-1][1] if path else -1)
                path.append((key[d], nid))
            if len(key) > 0:
                term_node[t] = path[len(key) - 1][1]
        self.node_q = np.array(node_q, np.int64)
        self.node_b = np.array(node_b, np.int64)
        self.node_depth = np.array(node_depth, np.int64)
        self.n_nodes = len(node_q)
        self.max_depth = int(max(node_depth) + 1) if node_depth else 1
        # terminal lists per node (CSR)
        order = np.argsort(term_node, kind="stable")
        order = order[term_node[order] >= 0]
        counts = np.bincount(term_node[order], minlength=self.n_nodes) if self.n_nodes else np.zeros(0, np.int64)
        self.term_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.term_ids = order.astype(np.int64)
        self.T = terms.T
        self.literal_ops = int(terms.weights().sum())


@njit(parallel=True, cache=True, fastmath=True)
def _shared_batch(Z, O, node_q, node_b, node_depth, term_ptr, term_ids, T, max_depth, chunk, select_only):
    NW = Z.shape[2]
    nc = (NW + chunk - 1) // chunk
    nn = node_q.shape[0]
    Mp = np.zeros((nc, T), np.int64)
    Np = np.zeros((nc, T), np.int64)
    for c in prange(nc):
        w0 = c * chunk
        w1 = min(NW, w0 + chunk)
        L = w1 - w0
        A = np.empty((max_depth, L), np.uint64)
        X = np.empty((max_depth, L), np.uint64)
        for v in range(nn):
            d = node_depth[v]
            qv = node_q[v]
            bv = node_b[v]
            if d == 0:
                for w in range(L):
                    A[0, w] = Z[qv, bv, w0 + w]
                if not select_only:
                    for w in range(L):
                        X[0, w] = O[qv, w0 + w]
            else:
                for w in range(L):
                    A[d, w] = A[d - 1, w] & Z[qv, bv, w0 + w]
                if not select_only:
                    for w in range(L):
                        X[d, w] = X[d - 1, w] ^ O[qv, w0 + w]
            k0 = term_ptr[v]
            k1 = term_ptr[v + 1]
            if k1 > k0:
                m = 0
                neg = 0
                for w in range(L):
                    m += popcnt64(A[d, w])
                if not select_only:
                    for w in range(L):
                        neg += popcnt64(A[d, w] & X[d, w])
                for kk in range(k0, k1):
                    Mp[c, term_ids[kk]] = m
                    Np[c, term_ids[kk]] = neg
    M = Mp.sum(axis=0)
    S = M - 2 * Np.sum(axis=0)
    return M, S


class SharedPlanPath:
    def __init__(self, Z, O, N, chunk=2048):
        self.Z, self.O, self.N, self.chunk = Z, O, N, chunk

    def plan(self, terms):
        return Trie(terms)

    def execute(self, tr, select_only=False):
        return _shared_batch(self.Z, self.O, tr.node_q, tr.node_b, tr.node_depth, tr.term_ptr, tr.term_ids,
                             tr.T, tr.max_depth, self.chunk, select_only)

    def run(self, terms, select_only=False):
        return self.execute(self.plan(terms), select_only)
