# -*- coding: utf-8 -*-
"""
Check the torsion conjecture of

  S. Chowdhury, S. Huntsman, M. Yutin,
  "Path homologies of motifs and temporal network representations",
  Applied Network Science 7:4 (2022), doi:10.1007/s41109-021-00441-z

verbatim: "... we conjecture that digraphs in this family with central paths of
length 2n have torsion subgroups Z/nZ in H~_1.
(We have computationally verified this conjecture for n <= 8.)"

The family: a central unidirected closed path of length 2n, with each of its
vertices linked to one of two external "polar" vertices in alternating fashion.
That is  P_n:  vertices 0..2n-1 (the directed 2n-cycle 0->1->...->0) plus two
polar vertices a,b, with a <-> even vertices and b <-> odd vertices.

Uses the NON-REGULAR path complex (Definition 2.3 / (2.2) of arXiv:1207.2834),
which is the convention that paper states it uses.

Self-contained: depends only on snf_transforms.py (generic integer linear
algebra) and the Python standard library.
"""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snf_transforms import snf_transforms


# --------------------------------------------------------------------------
def build_Pn(n):
    """P_n: directed 2n-cycle 0..2n-1 plus polar vertices a=2n, b=2n+1."""
    A, B = 2 * n, 2 * n + 1
    verts = list(range(2 * n)) + [A, B]
    E = [(i, i + 1) for i in range(2 * n - 1)] + [(2 * n - 1, 0)]
    for k in range(n):
        e, o = 2 * k, 2 * k + 1
        E += [(A, e), (e, A), (B, o), (o, B)]
    return verts, E


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


def boundary(p, s):
    d = {}
    for k in range(p + 1):
        f = s[:k] + s[k + 1:]
        d[f] = d.get(f, 0) + (-1) ** k
    return {f: c for f, c in d.items() if c}


def bad_matrix(A, p):
    """Rows = non-allowed (p-1)-sequences that occur; cols = A_p."""
    prev = set(A[p - 1])
    rows, cols = {}, []
    for s in A[p]:
        d = boundary(p, s)
        cols.append({f: c for f, c in d.items() if f not in prev})
        for f in cols[-1]:
            rows.setdefault(f, len(rows))
    M = np.zeros((len(rows), len(A[p])), dtype=np.int64)
    for j, d in enumerate(cols):
        for f, c in d.items():
            M[rows[f], j] = c
    return M


def d_matrix(A, p):
    idx = {s: i for i, s in enumerate(A[p - 1])}
    M = np.zeros((len(A[p - 1]), len(A[p])), dtype=np.int64)
    for j, s in enumerate(A[p]):
        for k in range(p + 1):
            i = idx.get(s[:k] + s[k + 1:])
            if i is not None:
                M[i, j] += (-1) ** k
    return M


def rational_nullspace(M):
    """Columns span ker(M) over Q."""
    if M.size == 0:
        return np.zeros((M.shape[1], 0), dtype=object)
    A = [[Fraction(int(x)) for x in row] for row in np.asarray(M)]
    rows, cols = len(A), len(A[0])
    piv, r = [], 0
    for c in range(cols):
        pr = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [A[i][k] - f * A[r][k] for k in range(cols)]
        piv.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in set(piv)]
    basis = []
    for f in free:
        w = [Fraction(0)] * cols
        w[f] = Fraction(1)
        for i, pc in enumerate(piv):
            w[pc] = -A[i][f]
        basis.append(w)
    return np.array(basis, dtype=object).T if basis else \
        np.zeros((cols, 0), dtype=object)


def integer_kernel(M):
    """Saturated Z-basis of ker(M): columns."""
    K = rational_nullspace(M)
    n, k = K.shape
    if k == 0:
        return np.zeros((M.shape[1], 0), dtype=object)
    cols = []
    from math import gcd
    for j in range(k):
        den = 1
        for i in range(n):
            den = den * K[i, j].denominator // gcd(den, K[i, j].denominator)
        cols.append([int(K[i, j] * den) for i in range(n)])
    Z = [[cols[j][i] for j in range(k)] for i in range(n)]
    D, U, V, r = snf_transforms(Z)
    d = [D[i][i] for i in range(k)]
    ZV = [[sum(Z[i][t] * V[t][j] for t in range(k)) for j in range(k)]
          for i in range(n)]
    out = np.empty((n, k), dtype=object)
    for i in range(n):
        for j in range(k):
            assert ZV[i][j] % d[j] == 0
            out[i, j] = ZV[i][j] // d[j]
    return out


def solve_int(basis, target):
    """basis @ X = target, X integral."""
    R, k = basis.shape
    B = [[int(basis[i, j]) for j in range(k)] for i in range(R)]
    out = np.zeros((k, target.shape[1]), dtype=object)
    for c in range(target.shape[1]):
        A = [[Fraction(B[i][j]) for j in range(k)]
             + [Fraction(int(target[i, c]))] for i in range(R)]
        piv, r = [], 0
        for j in range(k):
            pr = next((i for i in range(r, R) if A[i][j] != 0), None)
            if pr is None:
                continue
            A[r], A[pr] = A[pr], A[r]
            pv = A[r][j]
            A[r] = [x / pv for x in A[r]]
            for i in range(R):
                if i != r and A[i][j] != 0:
                    f = A[i][j]
                    A[i] = [A[i][t] - f * A[r][t] for t in range(k + 1)]
            piv.append(j)
            r += 1
        assert len(piv) == k
        for i, j in enumerate(piv):
            out[j, c] = int(A[i][k])
    return out


def h1_integral(v, E):
    A = allowed_paths(v, E, 2)
    B2 = integer_kernel(bad_matrix(A, 2))
    C1 = np.array(d_matrix(A, 1), dtype=object)
    C2 = np.array(d_matrix(A, 2), dtype=object) @ B2
    Z1 = integer_kernel(C1)
    z = Z1.shape[1]
    X = solve_int(Z1, C2)
    Xm = [[int(X[i, j]) for j in range(X.shape[1])] for i in range(z)]
    D, U, V, r = snf_transforms(Xm)
    return z - r, [D[i][i] for i in range(r) if D[i][i] > 1]


if __name__ == "__main__":
    print("=" * 88)
    print("Chowdhury-Huntsman-Yutin torsion conjecture: H~_1(P_n;Z) has torsion "
          "Z/nZ")
    print("  (central unidirected closed path of length 2n + two polar "
          "vertices linked alternately)")
    print("  paper verified it for n <= 8; this check goes to n <= 12")
    print("=" * 88)
    print(f"{'n':>3} {'|V|':>4} {'|E|':>5} | {'H_1(P_n;Z)':<22} | torsion | "
          f"conjecture Z/n")
    ok = True
    for n in range(1, 13):
        v, E = build_Pn(n)
        free, tors = h1_integral(v, E)
        grp = f"Z^{free}" + "".join(f" + Z/{d}" for d in tors)
        good = (tors == [n]) if n > 1 else (tors == [])
        ok &= good
        print(f"{n:>3} {len(v):>4} {len(E):>5} | {grp:<22} | {str(tors):<7} | "
              f"{'OK' if good else '*** DIFFERS ***'}", flush=True)
    print()
    print("torsion subgroup is exactly Z/nZ for every n tested:", ok)
