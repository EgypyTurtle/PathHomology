# -*- coding: utf-8 -*-
"""
Smith normal form WITH transformation matrices, guaranteed to terminate.

`find_h1_generators.snf_with_transforms` picks the first nonzero entry as pivot
and repairs divisibility by adding a row to the pivot row.  Because the pivot
column has already been cleared, that repair leaves the pivot unchanged, so the
same violation can be found again forever -- the loop that made the earlier
H_1-generator search time out at 600 s.

Fix (same as `snf_robust.snf`): at every step take as pivot the nonzero entry of
MINIMAL ABSOLUTE VALUE in the remaining block.  Each repair then either lowers
that minimum or Euclid-reduces the pivot, and a positive integer cannot decrease
forever.

Returned convention:  U @ A @ V = D,  U and V unimodular, D diagonal
non-negative with d_1 | d_2 | ... | d_r.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple


def snf_transforms(A: Sequence[Sequence[int]], max_steps: int = 400_000):
    """Return (D, U, V, rank) with U @ A @ V = D."""
    A = [[int(x) for x in row] for row in A]
    m = len(A)
    n = len(A[0]) if m else 0
    if m == 0 or n == 0:
        return [], [[1 if i == j else 0 for j in range(m)] for i in range(m)], \
               [[1 if i == j else 0 for j in range(n)] for i in range(n)], 0

    U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def swap_rows(i, j):
        A[i], A[j] = A[j], A[i]
        U[i], U[j] = U[j], U[i]

    def swap_cols(i, j):
        for r in range(m):
            A[r][i], A[r][j] = A[r][j], A[r][i]
        for r in range(n):
            V[r][i], V[r][j] = V[r][j], V[r][i]

    def add_rows(dst, src, q):
        if q == 0:
            return
        Ar, As = A[dst], A[src]
        for j in range(n):
            Ar[j] += q * As[j]
        Ur, Us = U[dst], U[src]
        for j in range(m):
            Ur[j] += q * Us[j]

    def add_cols(dst, src, q):
        if q == 0:
            return
        for i in range(m):
            A[i][dst] += q * A[i][src]
        for i in range(n):
            V[i][dst] += q * V[i][src]

    steps = 0
    t = 0
    while t < min(m, n):
        steps += 1
        if steps > max_steps:
            raise RuntimeError("snf_transforms: step limit exceeded")

        # pivot = nonzero entry of minimal absolute value in the remaining block
        piv, best = None, None
        for i in range(t, m):
            Ai = A[i]
            for j in range(t, n):
                v = Ai[j]
                if v and (best is None or (v if v > 0 else -v) < best):
                    best = v if v > 0 else -v
                    piv = (i, j)
        if piv is None:
            break
        pi, pj = piv
        if pj != t:
            swap_cols(t, pj)
        if pi != t:
            swap_rows(t, pi)

        # clear column t and row t
        while True:
            again = False
            for i in range(t + 1, m):
                if A[i][t]:
                    add_rows(i, t, -(A[i][t] // A[t][t]))
                    if A[i][t]:                 # remainder became the pivot
                        swap_rows(t, i)
                        again = True
                        break
            if again:
                continue
            for j in range(t + 1, n):
                if A[t][j]:
                    add_cols(j, t, -(A[t][j] // A[t][t]))
                    if A[t][j]:
                        swap_cols(t, j)
                        again = True
                        break
            if again:
                continue
            break

        # divisibility of the remaining block
        bad = None
        for i in range(t + 1, m):
            for j in range(t + 1, n):
                if A[i][j] % A[t][t]:
                    bad = (i, j)
                    break
            if bad:
                break
        if bad is not None:
            add_rows(t, bad[0], 1)              # pivot row += violating row
            continue

        if A[t][t] < 0:
            add_rows(t, t, -2)                  # negate row t (row += -2*row)
        t += 1

    rank = sum(1 for i in range(min(m, n)) if A[i][i] != 0)
    return A, U, V, rank


def _det(M):
    """Exact determinant by fraction-free (Bareiss) elimination."""
    n = len(M)
    if n == 0:
        return 1
    A = [[int(x) for x in row] for row in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k]:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def _matmul(X, Y):
    n, k, p = len(X), len(Y), len(Y[0])
    return [[sum(X[i][t] * Y[t][j] for t in range(k)) for j in range(p)]
            for i in range(n)]


if __name__ == "__main__":
    import random
    random.seed(20241207)

    print("consistency checks  (U A V = D, U,V unimodular, d_1 | d_2 | ...)")
    ok = True
    for trial in range(60):
        m = random.randint(1, 7)
        n = random.randint(1, 7)
        A = [[random.randint(-9, 9) for _ in range(n)] for _ in range(m)]
        D, U, V, r = snf_transforms(A)
        prod = _matmul(_matmul(U, A), V)
        same = all(prod[i][j] == D[i][j] for i in range(m) for j in range(n))
        diag = all(D[i][j] == 0 for i in range(m) for j in range(n) if i != j)
        d = [D[i][i] for i in range(r)]
        div = all(d[i + 1] % d[i] == 0 for i in range(len(d) - 1))
        nonneg = all(x >= 0 for x in d)
        uni = abs(_det(U)) == 1 and abs(_det(V)) == 1
        good = same and diag and div and nonneg and uni
        ok &= good
        if not good:
            print("  FAIL", m, n, A, "->", D)
    print("  all random trials pass:", ok)

    print()
    print("known cases:")
    cases = [
        ([[2, 4, 4], [-6, 6, 12], [10, -4, -16]], [2, 6, 12]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 3, 0]),
        ([[0, 0], [0, 0]], [0, 0]),
        ([[5]], [5]),
    ]
    for A, want in cases:
        D, U, V, r = snf_transforms(A)
        got = [D[i][i] for i in range(len(A))]
        print(f"  {A} -> {got}   (expected {want})  "
              f"{'OK' if got == want else 'MISMATCH'}")
