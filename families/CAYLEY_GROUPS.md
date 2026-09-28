
# Cayley digraphs of finite groups

For a finite group `G` and a generating set `S`, the Cayley digraph has vertex
set `G` and the arc `x -> x*s` for `x in G`, `s in S`. It is `|S|`-regular and
connected. All groups below are built concretely as permutation groups, so
subgroups, products and quotients are all obtained from the same
multiplication.

## 1. Groups of order `<= 10`

`max_dim = 3`. `dim Omega` is `[Omega_0, Omega_1, Omega_2, Omega_3]`,
`betti = [b_0, b_1, b_2]`.

| group and generating set | \|G\| | out-deg | dim Omega | betti_Q | betti_F2 |
|---|---|---|---|---|---|
| C_m with {1}, m = 2..10 | m | 1 | (m, m, 0, 0) | (1, 1, 0) | (1, 1, 0) |
| C_3 with {1,-1} | 3 | 2 | (3, 6, 9, 12) | (1, 0, 0) | (1, 0, 0) |
| C_4 with {1,-1} | 4 | 2 | (4, 8, 8, 8) | (1, 1, 0) | (1, 1, 0) |
| C_m with {1,-1}, m = 5..10 | m | 2 | (m, 2m, m, 0) | (1, 2, 1) | (1, 2, 1) |
| D_3, D_4, D_5 with {r, s} | 2m | 2 | (2m, 4m, 2m, 0) | (1, 2, 1) | (1, 2, 1) |
| D_3 with {r, r^-1, s} | 6 | 3 | (6, 18, 36, 60) | (1, 0, 0) | (1, 0, 0) |
| D_4 with {r, r^-1, s} | 8 | 3 | (8, 24, 40, 56) | (1, 1, 0) | (1, 1, 0) |
| D_5 with {r, r^-1, s} | 10 | 3 | (10, 30, 40, 40) | (1, 2, 1) | (1, 2, 1) |

With the single generator `{1}`, `C_m` *is* the directed cycle, and the table
reproduces the `C_n` result. With `{1,-1}`, `C_3` degenerates to the complete
digraph `K_3` and `C_4` to `K_2 box K_2`.

`D_n` with `{r, s}` gives the clean family `dim Omega = (2n, 4n, 2n, 0)` and
`betti = (1, 2, 1)` for all three computed orders.

The generating sets `{r, r^-1, s}` are exactly the **prism** family
`C_n box K_2`: `n = 3` gives `(1,0,0)` with `dim Omega_p = 3(p+1)(p+2)`
(quadratic in `p`), `n = 4` gives `(1,1,0)` with `dim Omega_p = 16p + 8`
(linear in `p`), and `n >= 5` gives `(1,2,1)` with `dim Omega_p = 8n`
(constant in `p`, verified up to `n = 30`). The two small cases are genuine
exceptions, not part of the pattern.

## 2. Permutation groups `S_n` and `A_n`

Two generators: `S_n` uses the transposition `(0 1)` and the `n`-cycle
`(0 1 ... n-1)`; `A_n` uses a 3-cycle and an even `(n-1)`-cycle.

| group | \|G\| | dim Omega | betti_Q | betti_F2 | time |
|---|---|---|---|---|---|
| S_3 | 6 | [6, 12, 0] | [1, 7] | [1, 7] | 0.00 s |
| S_4 | 24 | [24, 48, 0] | [1, 25] | [1, 25] | 0.00 s |
| S_5 | 120 | [120, 240, 0] | [1, 121] | [1, 121] | 0.01 s |
| S_6 | 720 | [720, 1440, 0] | [1, 721] | [1, 721] | 0.03 s |
| S_7 | 5040 | [5040, 10080, 0] | [1, 5041] | [1, 5041] | 0.47 s |
| S_8 | 40320 | [40320, 80640, 0] | [1, 40321] | [1, 40321] | 20.2 s |
| A_3 | 3 | [3, 3, 0] | [1, 1] | [1, 1] | 0.00 s |
| A_4 | 12 | [12, 24, 0] | [1, 13] | [1, 13] | 0.00 s |
| A_5 | 60 | [60, 120, 0] | [1, 61] | [1, 61] | 0.00 s |
| A_6 | 360 | [360, 720, 0] | [1, 361] | [1, 361] | 0.01 s |
| A_7 | 2520 | [2520, 5040, 0] | [1, 2521] | [1, 2521] | 0.22 s |
| A_8 | 20160 | [20160, 40320, 0] | [1, 20161] | [1, 20161] | 5.30 s |

(`max_dim = 2`, so only `b_0` and `b_1` are read off; with `max_dim = 3` the
same values plus `b_2 = 0` are obtained for `n <= 6`.)

### The pattern, and why it holds

```
b_1(S_n) = |S_n| + 1      b_1(A_n) = |A_n| + 1      (A_3 excepted, see below)
```

The reason is a two-line argument, not a coincidence.

**Step 1: `Omega_2 = 0`.** For the Cayley digraph, the constraint attached to a
non-edge `(x, z)` is `sum_y c_{x,y,z} = 0` over
`mid(x, z) = {y : x -> y -> z}`, and

```
mid(x, z)  <->  {(s, t) in S x S : s*t = x^{-1} z}.
```

So if every element of `S*S` outside `S` has a **unique** decomposition
`s*t` with `s, t in S`, each such equation is single-variable and forces
`Omega_2 = 0`. For `S = {(0 1), (0 1...n-1)}` the elements of `S*S` are
`e, a*b, b*a, b^2`, and each has exactly one decomposition. The same check
passes for the `A_n` generating sets. The test costs `O(|S|^2)` and needs no
linear algebra.

**Step 2: if `Omega_2 = 0` then `H_1 = ker d_1`.** With `Omega_2 = 0` the
image of `d_2` is zero, so `H_1 = ker(d_1 : Omega_1 -> Omega_0)` is the
ordinary cycle space of the underlying graph, and

```
b_1 = |E| - |V| + b_0 = |G| * (|S| - 1) + 1.
```

For `|S| = 2` this is `|G| + 1`, which is the table.

`A_3` is listed as an exception because the two chosen generators coincide
(both are the same 3-cycle), so the distinct generating set has size 1 and
the formula correctly returns `3*0 + 1 = 1`.

### Certification beyond the computed range

Because the criterion is cheap, it certifies groups whose Cayley digraph is
far too large to build:

| group | \|G\| | unique 2-factorisation | certified b_1 | directly computed |
|---|---|---|---|---|
| S_3 .. S_8 | 6 .. 40320 | yes | 7 .. 40321 | matches, all six |
| S_9 | 362880 | yes | 362881 | not computed (`\|A_2\| = 1451520`) |
| A_3 | 3 | no | 1 | matches |
| A_4 .. A_8 | 12 .. 20160 | yes | 13 .. 20161 | matches, all five |
| A_9 | 181440 | yes | 181441 | not computed (`\|A_2\| = 725760`) |
| A_10 | 1814400 | yes | 1814401 | not computed |

This is a certificate for `b_1` only, and it needs the group to be enumerated
(it is `O(|G|)`), but it needs no boundary matrix at all.

## 3. How subgroups, products and quotients appear

### 3.1 Subgroups: no simple relation

| inclusion | \|G\| | dim Omega | betti_Q |
|---|---|---|---|
| `C_4 = <r> <= D_4`, gens `{r}` | 4 | [4, 4, 0, 0] | [1, 1, 0] |
| `D_4` itself, gens `{r, s}` | 8 | [8, 16, 8, 0] | [1, 2, 1] |
| `C_6 = <r> <= D_6`, gens `{r}` | 6 | [6, 6, 0, 0] | [1, 1, 0] |
| `D_6` itself, gens `{r, s}` | 12 | [12, 24, 12, 0] | [1, 2, 1] |
| `A_4 <= S_4` | 12 | [12, 24, 0, 0] | [1, 13, 0] |
| `S_4` itself | 24 | [24, 48, 0, 0] | [1, 25, 0] |

A subgroup carrying the restricted generating set is a different Cayley
digraph with a different path homology; neither restriction nor transfer
relates the two. A subgroup can drop `b_1` (`D_4` to `C_4`: `2 -> 1`) or the
subgroup can have far larger `b_1` than the ambient group (`A_4` inside `S_4`:
`13` versus `25`, but note the different vertex counts, `b_1 = |G| + 1` in both
cases because `Omega_2 = 0` for both).

### 3.2 Products: the Cartesian product and the Kunneth prediction

`C_m x C_n` with the product generating set is the Cartesian product of the
two Cayley digraphs, so GLMY's Kunneth formula applies and predicts

```
H_1 = (H_0(C_m) (x) H_1(C_n))  +  (H_1(C_m) (x) H_0(C_n)) = K^2
H_2 = H_1(C_m) (x) H_1(C_n) = K
```

using `H_1(C_k) = K` from `CLASSICAL.md`.

| product | \|G\| | dim Omega | betti_Q | betti_F2 | matches Kunneth? |
|---|---|---|---|---|---|
| C_2 x C_2 | 4 | [4, 8, 8, 8] | [1, 1, 0] | [1, 1, 0] | **no** |
| C_2 x C_3 | 6 | [6, 12, 6, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_2 x C_4 | 8 | [8, 16, 8, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_2 x C_5 | 10 | [10, 20, 10, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_3 x C_3 | 9 | [9, 18, 9, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_3 x C_4 | 12 | [12, 24, 12, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_3 x C_5 | 15 | [15, 30, 15, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_3 x C_6 | 18 | [18, 36, 18, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_4 x C_4 | 16 | [16, 32, 16, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_4 x C_5 | 20 | [20, 40, 20, 0] | [1, 2, 1] | [1, 2, 1] | yes |
| C_5 x C_5 | 25 | [25, 50, 25, 0] | [1, 2, 1] | [1, 2, 1] | yes |

Ten of the eleven computed products agree exactly with the Kunneth prediction,
with the clean shape `dim Omega = (mn, 2mn, mn, 0)`.

`C_2 x C_2` is the single exception, and it is worth stating precisely rather
than hiding: it is the Klein four group, whose Cayley digraph with
`{(1,0), (0,1)}` is the bidirectional 4-cycle `C_4^{+-1}`. Its Betti vector is
`(1, 1, 0)`, strictly smaller than the Kunneth prediction `(1, 2, 1)`. This is
recorded here as an observed discrepancy; it is **not** claimed that GLMY's
Kunneth theorem is false, only that its hypotheses must be checked before
applying it to this case. (Both factors have order 2, and `C_2` is the only
directed cycle that is also the complete digraph `K_2`.)

### 3.3 Quotients: genuine Cayley digraphs of `G/N` on cosets

Built by right cosets, with the induced generating set (repetitions dropped).

| quotient | \|G\| | \|G/N\| | image gens | dim Omega | betti_Q | betti_F2 |
|---|---|---|---|---|---|---|
| D_4 / <r^2> = Z_2 x Z_2 | 8 | 4 | 2 of 2 | [4, 8, 8, 8] | [1, 1, 0] | [1, 1, 0] |
| D_6 / <r^3> = D_3 | 12 | 6 | 2 of 2 | [6, 12, 6, 0] | [1, 2, 1] | [1, 2, 1] |
| D_6 / <r^2> = Z_2 x Z_2 | 12 | 4 | 2 of 2 | [4, 8, 8, 8] | [1, 1, 0] | [1, 1, 0] |
| Z_6 / <r^2> = Z_2 | 6 | 2 | 1 of 1 | [2, 2, 0, 0] | [1, 1, 0] | [1, 1, 0] |
| Z_6 / <r^3> = Z_3 | 6 | 3 | 1 of 1 | [3, 3, 0, 0] | [1, 1, 0] | [1, 1, 0] |
| S_4 / V_4 = S_3 | 24 | 6 | 2 of 2 | [6, 12, 6, 0] | [1, 2, 1] | [1, 2, 1] |
| S_4 / A_4 = Z_2 | 24 | 2 | 1 of 2 | [2, 2, 0, 0] | [1, 1, 0] | [1, 1, 0] |
| A_4 / V_4 = Z_3 | 12 | 3 | 2 of 2 | [3, 6, 9, 12] | [1, 0, 0] | [1, 0, 0] |

Two internal consistency checks come out of this table for free:

* `D_6 / <r^3> = D_3`, and `S_4 / V_4 = S_3 = D_3` as abstract groups. Both
  were computed through cosets and both return `dim Omega = [6, 12, 6, 0]`,
  `betti = (1, 2, 1)` -- identical to `D_3` computed directly in section 1.
  The path homology of a Cayley digraph really does depend only on the
  abstract group and the image generating set, as it must.
* `A_4 / V_4 = Z_3` with two image generators returns `[3, 6, 9, 12]`,
  `(1, 0, 0)`, matching `C_3` with `{1,-1}`, which is `K_3`. Both are `Z_3`
  with the full non-zero generating set.

The quotient can strictly *increase* the number of generators that matter:
`S_4 / A_4` has two images but only one distinct permutation, so the quotient
Cayley digraph has out-degree 1 (a directed 2-cycle); and `A_4 / V_4` has two
distinct images on 3 vertices, which is the complete digraph.

**Summary of the three operations.** Products of cyclic groups follow the
Kunneth prediction with a single small exception; quotients behave exactly
like the abstract quotient group recomputed from scratch; subgroups have no
functorial relation to the ambient group at all.
