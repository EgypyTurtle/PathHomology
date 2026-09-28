# -*- coding: utf-8 -*-
"""
Check the theorems of

  X. Tang, S.-T. Yau, "Path Homology of Circulant Digraphs", arXiv:2602.04140.

The circulant digraph C_n^S is the Cayley digraph of Z_n with connection set S:
vertices 0..n-1, edges a -> a+s for s in S.

Checked here:

  Thm 1.1  (C_5^{1,2}):  dim_K Omega_n = 10 for EVERY n >= 1, and
                         H_0 = K, H_1 = K, H_m = 0 for m >= 2.

  Thm 1.2  (n >= 5, S = {1,s}, 1 < s < n/2):
             s = 2   ->  H_0 = K, H_1 = K,   H_m = 0 (m >= 2)
             s != 2  ->  H_0 = K, H_1 = K^2, H_2 = K, H_m = 0 (m >= 3)

Self-contained: depends only on core.py and the standard library.
"""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import path_homology


def circulant(n, S):
    V = list(range(n))
    E = sorted({(a, (a + s) % n) for a in range(n) for s in S})
    return V, E


def allowed_paths(v, E, D):
    out = {x: [] for x in v}
    for (a, b) in E:
        out[a].append(b)
    A = {}
    for p in range(D + 1):
        res = []

        def rec(seq):
            if len(seq) == p + 1:
                res.append(tuple(seq))
                return
            for w in out[seq[-1]]:
                rec(seq + [w])

        for x in v:
            rec([x])
        A[p] = res
    return A


def rank_q(M):
    """Rank over Q by Gaussian elimination with Fractions."""
    if M.size == 0:
        return 0
    A = [[Fraction(int(x)) for x in row] for row in np.asarray(M)]
    rows, cols, r = len(A), len(A[0]), 0
    for c in range(cols):
        pr = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        pv = A[r][c]
        for i in range(r + 1, rows):
            if A[i][c] != 0:
                f = A[i][c] / pv
                for j in range(c, cols):
                    A[i][j] -= f * A[r][j]
        r += 1
        if r == rows:
            break
    return r


def bad_matrix(A, p):
    prev = set(A[p - 1])
    rows = {}
    cols = []
    for s in A[p]:
        d = {}
        for k in range(p + 1):
            f = s[:k] + s[k + 1:]
            d[f] = d.get(f, 0) + (-1) ** k
        cols.append({f: c for f, c in d.items() if c and f not in prev})
        for f in cols[-1]:
            rows.setdefault(f, len(rows))
    M = np.zeros((len(rows), len(A[p])), dtype=np.int64)
    for j, d in enumerate(cols):
        for f, c in d.items():
            M[rows[f], j] = c
    return M


def dim_omega(v, E, D):
    A = allowed_paths(v, E, D)
    out, sizes = {}, {}
    for p in range(D + 1):
        sizes[p] = len(A[p])
        out[p] = len(A[p]) if p == 0 else len(A[p]) - rank_q(bad_matrix(A, p))
    return out, sizes


def betti(v, E, D, field="Q"):
    r = path_homology(vertices=v, edges=E, max_dim=D, field=field)
    return [r.betti.get(p, 0) for p in range(D + 1)]


if __name__ == "__main__":
    print("=" * 96)
    print("THEOREM 1.1 check:  G = C_5^{1,2}   (5 vertices, |E| = 10)")
    print("   the paper claims  dim_K Omega_n = 10 for EVERY n >= 1")
    print("=" * 96)
    v, E = circulant(5, [1, 2])
    dom, sizes = dim_omega(v, E, 7)
    print(f"  {'n':>2} {'|A_n|':>10} {'dim Omega_n':>12}   paper says 10")
    ok11 = True
    for p in range(0, 8):
        good = (dom[p] == 10) or p == 0
        ok11 &= good
        print(f"  {p:>2} {sizes[p]:>10} {dom[p]:>12}   "
              f"{'OK' if good else '*** MISMATCH ***'}")
    print(f"  betti (D=6) = {betti(v, E, 6)}")
    print("  paper: H_0 = K, H_1 = K, H_m = 0 for m >= 2")

    print()
    print("=" * 96)
    print("THEOREM 1.2 check:  C_n^{1,s},  n >= 5, 1 < s < n/2")
    print("   s = 2   -> (H_0,H_1,H_2) = (1,1,0),  H_m = 0 for m >= 2")
    print("   s != 2  -> (H_0,H_1,H_2) = (1,2,1),  H_m = 0 for m >= 3")
    print("=" * 96)
    print(f"  {'n':>3} {'s':>3} | {'dimOmega (0..4)':>26} | {'betti (0..5)':>22} "
          f"| verdict")
    allok = True
    for n in [5, 6, 7, 8, 9, 10, 11]:
        for s in [2, 3, 4]:
            if not (1 < s < n / 2):
                continue
            v, E = circulant(n, [1, s])
            dom, _ = dim_omega(v, E, 5)
            b = betti(v, E, 5)
            exp = [1, 1, 0, 0, 0] if s == 2 else [1, 2, 1, 0, 0]
            # index 5 is the truncation artefact dim ker d_5, so compare 0..4
            good = (b[:5] == exp)
            allok &= good
            print(f"  {n:>3} {s:>3} | {str([dom[p] for p in range(5)]):>26} | "
                  f"{str(b):>22} | {'OK' if good else '*** MISMATCH ***'}")
        print()
    print("Theorem 1.1 holds (dim Omega_n = 10 for n = 1..7):", ok11)
    print("Theorem 1.2 holds in every case above:", allok)
