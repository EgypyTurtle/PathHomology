
# Tournaments

A tournament orients every pair of distinct vertices by exactly one arc, so it
has exactly `C(n,2)` arcs and is `(n-1)/2`-regular when `n` is odd.

## 1. Transitive tournament `TT_n`

`TT_n` has vertices `0..n-1` and the arc `i -> j` for every `i < j`. As a
path complex it is the ordered simplex on `n` vertices, so

```
dim Omega_p = C(n, p+1)        betti = (1, 0, 0, ...)
```

| digraph | N | \|E\| | dim Omega | betti_Q | betti_F2 | Q vs F2 |
|---|---|---|---|---|---|---|
| TT_2 | 2 | 1 | [2, 1, 0, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| TT_3 | 3 | 3 | [3, 3, 1, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| TT_4 | 4 | 6 | [4, 6, 4, 1, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| TT_5 | 5 | 10 | [5, 10, 10, 5, 1] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| TT_6 | 6 | 15 | [6, 15, 20, 15, 6] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| TT_7 | 7 | 21 | [7, 21, 35, 35, 21] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |

The path homology of a transitive tournament is that of a point. Note that
`Omega_p` is *non-zero in every degree up to `n`* while the homology vanishes:
the path complex is acyclic but far from trivial.

## 2. Cyclic (regular) tournament `RT_n`, `n` odd

Vertices `Z_n`, arc `i -> j` iff `j - i mod n` lies in `{1, ..., (n-1)/2}`.
Out-degree `d = (n-1)/2`, so `|E| = n*d`.

| digraph | N | \|E\| | dim Omega | betti_Q | betti_F2 | Q vs F2 |
|---|---|---|---|---|---|---|
| RT_3 | 3 | 3 | [3, 3, 0, 0, 0] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| RT_5 | 5 | 10 | [5, 10, 10, 10, 10] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| RT_7 | 7 | 21 | [7, 21, 42, 84, 168] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| RT_9 | 9 | 36 | [9, 36, 108, 324, 972] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |

Two regularities are visible over the computed range `n = 3, 5, 7, 9`:

```
dim Omega_p(RT_n) = |E| * (d - 1)^(p-1)        for p >= 1,   d = (n-1)/2
betti(RT_n)       = (1, 1, 0, 0, ...)
```

So the space of allowed paths grows exponentially in the dimension while the
homology stays that of a circle: `b_1 = 1` in every computed case, in both
characteristics. `RT_3` is the directed 3-cycle and the factor `(d-1) = 0`
correctly collapses `Omega_p` to zero for `p >= 2`, consistent with the
directed cycle result in `CLASSICAL.md`.

*The exponential shape `|E| * (d-1)^(p-1)` is an empirical fit on four values
of `n`; it is not proved here.*

## 3. All tournaments up to isomorphism, `n <= 5`

Complete list of isomorphism classes: 1, 1, 4, 12 classes for `n = 2, 3, 4, 5`.
Canonical form is the minimum adjacency bit-string over all `n!` relabellings,
so the classification is complete, not a heuristic. The score sequence is
listed for orientation but is *not* a complete invariant -- see `T_5#9`,
`T_5#10`, `T_5#11`, which share `[1, 2, 2, 2, 3]` but split into two
isomorphism classes and two different Betti vectors.

| digraph | N | \|E\| | dim Omega | betti_Q | betti_F2 | Q vs F2 |
|---|---|---|---|---|---|---|
| T_2#1 score-seq [0, 1] | 2 | 1 | [2, 1, 0, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_3#1 score-seq [0, 1, 2] | 3 | 3 | [3, 3, 1, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_3#2 score-seq [1, 1, 1] | 3 | 3 | [3, 3, 0, 0, 0] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| T_4#1 score-seq [0, 1, 2, 3] | 4 | 6 | [4, 6, 4, 1, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_4#2 score-seq [1, 1, 1, 3] | 4 | 6 | [4, 6, 3, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_4#3 score-seq [1, 1, 2, 2] | 4 | 6 | [4, 6, 3, 1, 0] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| T_4#4 score-seq [0, 2, 2, 2] | 4 | 6 | [4, 6, 3, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#1 score-seq [0, 1, 2, 3, 4] | 5 | 10 | [5, 10, 10, 5, 1] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#2 score-seq [1, 1, 1, 3, 4] | 5 | 10 | [5, 10, 9, 3, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#3 score-seq [1, 1, 2, 2, 4] | 5 | 10 | [5, 10, 9, 4, 1] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#4 score-seq [0, 2, 2, 2, 4] | 5 | 10 | [5, 10, 9, 3, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#5 score-seq [1, 1, 2, 3, 3] | 5 | 10 | [5, 10, 9, 5, 1] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| T_5#6 score-seq [0, 2, 2, 3, 3] | 5 | 10 | [5, 10, 9, 4, 1] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#7 score-seq [1, 1, 2, 3, 3] | 5 | 10 | [5, 10, 9, 3, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#8 score-seq [0, 1, 3, 3, 3] | 5 | 10 | [5, 10, 9, 3, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#9 score-seq [1, 2, 2, 2, 3] | 5 | 10 | [5, 10, 10, 8, 4] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| T_5#10 score-seq [1, 2, 2, 2, 3] | 5 | 10 | [5, 10, 8, 3, 0] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |
| T_5#11 score-seq [1, 2, 2, 2, 3] | 5 | 10 | [5, 10, 9, 4, 1] | [1, 0, 0, 0] | [1, 0, 0, 0] | = |
| T_5#12 score-seq [2, 2, 2, 2, 2] | 5 | 10 | [5, 10, 10, 10, 10] | [1, 1, 0, 0] | [1, 1, 0, 0] | = |

Observations over these 19 classes:

1. `b_1` is always `0` or `1`, and `b_p = 0` for `p >= 2`.
2. The rational and mod-2 Betti vectors always agree, so no 2-torsion appears.
3. Acyclic (transitive) tournaments have `b_1 = 0`; `b_1 = 1` occurs for
   `T_3#2`, `T_4#3` and four of the twelve 5-vertex classes, i.e. not only for
   the strongly connected ones.
4. `dim Omega_3` can be zero or positive independently of `b_1`: `T_5#10` has
   `dim Omega_3 = 3` and `b_1 = 1`, `T_5#2` has `dim Omega_3 = 3` and
   `b_1 = 0`.
