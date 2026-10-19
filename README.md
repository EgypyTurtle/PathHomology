# PathHomology

An implementation of GLMY path homology, a record of the optimisation rounds it went
through, and a point-by-point comparison against the worked examples of the public
literature.

The reference implementation follows:

> A. Grigor'yan, Y. Lin, Y. Muranov, S.-T. Yau,
> *Homologies of path complexes and digraphs*, arXiv:1207.2834v4.

## Update: gen6 and algebraic Morse reduction (2026-10-19)

The existing implementation, optimisation history, literature checks and digraph
families below are retained. The root `core.py` is now **gen6**. New features include
streaming dimension-only calculations, stronger Omega-support pruning and exact
rank certificates, and an optional **algebraic Morse backend** for explicit
representatives. The default remains `homology_method="linear"`; ordinary path
homology is not silently replaced by a persistent or Laplacian calculation.

- [Algebraic Morse technique, proof, and application to Omega_*](docs/ALGEBRAIC_MORSE_PATH_HOMOLOGY_20261019.md)
- [Current measurements and their limitations](docs/MORSE_PERFORMANCE_20261019.md)
- [Current version index](VERSIONS.md) and [repository update details](docs/REPOSITORY_UPDATE.md)

Tested setup: Python 3.12 and NumPy 2.3.5 (`python -m pip install -r requirements.txt`).
Run `python -B examples.py`, `python -B verify_release.py`, and
`python -B verify_repository.py`. The last two commands cover 10 release checks
and four existing repository validation entry points; all 14 passed locally.
GitHub Actions is configured for Windows/Linux; cloud results are separate from
the local verification record.

---

## 1. The original implementation

`original_core.py` is the first version (54,458 bytes, 2026-09-18), written directly
from the paper's definitions:

- **Allowed paths** `A_p` (Example 3.3): sequences whose consecutive pairs are edges.
  **Vertices may repeat** -- Definition 2.1 of the paper says verbatim that
  *"a priori the vertices in the path do not have to be distinct"*.
- **Boundary operator** (2.2):
  `d e_{i_0...i_p} = sum_{q=0}^{p} (-1)^q e_{i_0...i_q^...i_p}`.
- **The boundary does not preserve `A_*`**: deleting an interior vertex merges two
  edges into one, and the resulting face need not be allowed. One therefore restricts
  to `Omega_p = { v in A_p : d v in A_{p-1} }` (3.8), and defines
  `H_p = ker(d on Omega_p) / d(Omega_{p+1})` (3.11).
- **Coefficients**: `Q` (exact, via `Fraction`), `R` (float64), `F_p` (a `GF` class).
- **Three interchangeable algorithms for `Omega_p`** (`kernel`, `lemma41`, `prop42`)
  so that they can be cross-checked against one another.
- A `regular=True/False` switch selects the regular or the non-regular boundary
  convention (Section 2.3 and (2.10) of the paper). The default is `regular=False`.

The current version is `core.py` (generation 6, 116,557 bytes). Its default linear
backend preserves the existing usage. The new Morse backend additionally requires
`morse.py` and can return different, valid bases of cycles, boundaries and homology,
all lifted to the original allowed-path coordinates. The original version remains
available as `original_core.py`; it does not implement the newly added options.

## 2. What each optimisation round changed

| Round | Tag | Bytes | What changed |
|---|---|---:|---|
| -- | gen0 | 54458 | Original. Matrices are `dtype=object` arrays of `Fraction` / `GF`; all arithmetic is Python-level. |
| 1 | gen1 | 65728 | (a) Every prime field is converted to an integer NumPy array for elimination, nullspace, rank, pivots and matrix products, then converted back to `GF` objects. (b) Matrices over `Q` that are integral and provably free of int64 overflow use a native int64 kernel; genuine fractions still fall back to `Fraction`. (c) The canonical basis produced by the nullspace routine has an identity block on its free coordinates, so change of basis reads those rows directly instead of running a new elimination. (d) Expanding cycles, boundaries and homology representatives from Omega coordinates back to paths is done in one batch per dimension instead of one matrix product per column. (e) The constraint matrix keeps only the rows that a boundary can actually hit. |
| 2 | gen2 | 82143 | (a) Rank and nullspace over `Q` are computed fraction-free, by clearing denominators row by row and eliminating with primitive integer rows; `Fraction` is used only in the final back-substitution. Still exact. (b) Each differential is reduced once: the rank comes from the rank-nullity theorem instead of a second elimination. (c) The case `Omega_p = A_p` is detected explicitly and reuses existing coordinates. (d) A new **`generators=False` dimension-only mode**: it returns only `dim Omega_p = (number of p-paths) - rank(C_p)` and `dim Z_p = (number of p-paths) - rank(F_p)`, where `C_p` is the non-allowed-face constraint matrix and `F_p` the full boundary on allowed p-paths. No Omega basis, no change of basis, no representatives. Boundary rows are stored as sparse dictionaries, and over `F_2` they are packed into Python big-integer bit rows. |
| 3 | gen3 | 87483 | (a) For an integer matrix, the rank mod 2 is computed first. It is a strict lower bound for the rational rank; if it reaches `min(number of non-zero rows, number of columns)` the rational rank is certified unconditionally and no integer elimination runs. (b) The chain condition `im(d_p) is contained in ker(d_{p-1})` supplies a tighter upper bound; when the two bounds meet, that is again a certificate. Otherwise the code falls back to exact elimination, so the algorithm is never probabilistic. (c) The rank of the first differential is taken from the incidence-matrix theorem: it equals the number of vertices minus the number of weak components. (d) Sparse row elimination merges non-zero entries in place. (e) All free variables of an exact nullspace are back-substituted in one batch. (f) Over `F_2`, small and medium widths build big-integer bit rows directly; very wide matrices fall back to sparse dictionaries and are packed once. |
| 4 | gen4 | 97942 | Sparse no-fill singleton peeling, certified zero Omega tails, Omega_2 support and higher-prefix pruning, sparse columns, and `store_paths=False` streaming dimensions with exact formal path counts. |
| 5 | gen5 | 110690 | Two-sided, multilayer support pruning and forced-zero propagation; canonical Omega bases computed on retained candidates and lifted back; stronger chain-condition rank bounds, certified multi-prime ranks, component splitting and sparse ordering. |
| 6 | gen6 | 116557 | Optional exact `homology_method="morse"`: construct the Omega differential sparsely, cancel invertible pairs on the chain complex, and lift Z/B/H bases to original paths. Allows `store_matrices=False` while returning generators. The default linear and streaming rank routes remain available. |

### Historical measured speed-up (gen0 through gen3)

Same machine; the Betti numbers agree between all four algorithms on every case.
"dim-only" is the `generators=False` mode of gen3.

| Case | Field | gen0 | gen3 | Speed-up | gen3 dim-only | vs gen0 |
|---|---|---:|---:|---:|---:|---:|
| complete DAG, N=10, max_dim=3 | F2 | 23.89 s | 0.06 s | 402x | 0.001 s | 32,700x |
| (same) | Q | 35.31 s | 0.13 s | 281x | 0.001 s | 26,200x |
| (same) | R | 0.04 s | 0.03 s | 1.2x | 0.005 s | 8.9x |
| complete DAG, N=10, max_dim=5 | F2 | 91.25 s | 0.18 s | 496x | 0.004 s | 20,700x |
| (same) | Q | 151.98 s | 0.43 s | 352x | 0.004 s | 34,300x |
| (same) | R | 0.23 s | 0.16 s | 1.5x | 0.034 s | 6.6x |
| complete DAG, N=15, max_dim=6 | F2 | not measured | 81.4 s | -- | 0.111 s | -- |

Overall, from the original version to the current dimension-only mode, the speed-up
is about 2x10^4 on `F_2` and 3x10^4 on `Q`.

These are the original gen0/gen3 measurements, not a new measurement of gen6.

### Current gen5 versus gen6 comparison

Three-run medians on the same machine, with explicit generators and dense
allowed-path boundary matrices retained in **both** modes:

| Case and target | Field | gen5 linear | gen6 Morse |
|---|---|---:|---:|
| sparse digraph, 300 vertices / 900 arcs, through H_2 | Q | 10.200 s | 0.297 s |
| three-dimensional periodic grid, 125 vertices / 375 arcs, through H_3 | Q | 16.914 s | 0.697 s |

The grid has Betti numbers `(1,3,3,1)`. Both timings include Omega construction.
The gain combines chain cancellation with sparse engineering and less redundant
boundary output; it is **not** the effect of an abstract theorem in isolation.
The [performance report](docs/MORSE_PERFORMANCE_20261019.md) includes an empty-column
matrix-product ablation, memory measurements, raw records, and censored cases.
Morse does not remove the cost of constructing Omega or guarantee a speed-up on
every graph. Its representatives need not agree term-by-term with the old basis.

## 3. Agreement with the public literature

### 3.1 GLMY, arXiv:1207.2834v4

`check_paper_examples.py` recomputes every worked example of the paper.
**All 94 assertions pass**, covering Definition 2.2/3.1/3.8, Proposition 4.2/4.7,
Theorem 4.3, Examples 3.3/3.9/3.14/6.17, Fig. 8 (semi-edges) and Fig. 24 (the
octahedron, with `dim Omega_2 = 8`, `H = (1,0,1)` and Euler characteristic 2).
Example 3.14 (the digraph `0 <-> 1`) is tested under **both** the regular and the
non-regular convention. The check also confirms that the square of the boundary
vanishes on `Omega_*` (it does not on `A_*`, which is precisely why `Omega` is
needed).

Output: `results/paper_examples_glmy.txt`.

### 3.2 Tang-Yau, circulant digraphs, arXiv:2602.04140

The circulant digraph `C_n^S` is the Cayley digraph of `Z_n` with connection set `S`:
vertices `0..n-1`, edges `a -> a+s` for `s` in `S`. `check_circulant.py` checks:

- **Theorem 1.1** (`C_5^{1,2}`): the paper states that `dim Omega_n = 10` for
  **every** `n >= 1`, and that `H_0 = H_1 = K`, `H_m = 0` for `m >= 2`.
  Measured for `n = 0..7`: `dim Omega_n = (5, 10, 10, 10, 10, 10, 10, 10)` and
  `H = (1, 1, 0, ...)`. **Agrees.**
- **Theorem 1.2** (`S = {1, s}` with `1 < s < n/2`): `s = 2` gives `(1,1,0,...)`,
  and `s != 2` gives `(1,2,1,0,...)`. Measured for `n = 5..11` and `s = 2,3,4`:
  **all agree.** When the hypothesis `s < n/2` fails (for instance `n = 5, s = 3`)
  the answer is indeed different, so the hypothesis is necessary.

Output: `results/circulant_tang_yau.txt`.

### 3.3 The torsion conjecture of Chowdhury-Huntsman-Yutin

> S. Chowdhury, S. Huntsman, M. Yutin,
> *Path homologies of motifs and temporal network representations*,
> Applied Network Science **7**:4 (2022), doi:10.1007/s41109-021-00441-z.
>
> Verbatim: *"... we conjecture that digraphs in this family with central paths of
> length 2n have torsion subgroups Z/nZ in H~_1. (We have computationally verified
> this conjecture for n <= 8.)"*

The family consists of a central unidirected closed path of length `2n`, each of whose
vertices is linked to one of two external "polar" vertices in alternating fashion.
That is exactly the graph written `P_n` here: the directed `2n`-cycle
`0 -> 1 -> ... -> 2n-1 -> 0` together with two polar vertices `a` and `b`, where `a`
is linked to the even vertices and `b` to the odd ones.

`check_torsion.py` recomputes `H_1(P_n; Z)` exactly, by Smith normal form
(`Z_k` below means `Z` direct sum `Z/k`):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H_1(P_n; Z) | Z | Z_2 | Z_3 | Z_4 | Z_5 | Z_6 | Z_7 | Z_8 | Z_9 | Z_10 | Z_11 | Z_12 |

The torsion subgroup is exactly `Z/nZ` for every `n` tested, **up to `n = 12`, four
cases beyond the paper's `n <= 8`.**

There is also an explicit generator:

```
tau_k = -e_{a,0} + e_{a,2k} + e_{b,1} - e_{b,2k+1} - e_{0,1} + e_{2k,2k+1}
ord[tau_k] = n / gcd(k, n)
```

Taking `k = 1` gives order `n`, hence `Z/n` is contained in the torsion subgroup; the
other inclusion follows from the free rank together with the Smith-form upper bound.
The two together give `Tor H_1(P_n) = Z/nZ`.

**Convention.** That paper states explicitly (footnote 3) that it uses **non-regular**
path homology, so that a directed 2-cycle counts as a hole. This implementation
defaults to `regular=False`, which is the same convention. The two conventions differ
precisely on directed 2-cycles: for `0 -> 1 -> 0`, the regular version has
`d e_010 = e_10 + e_01` in `A_1` (because `e_00` is zero in the quotient), so the hole
is filled and `H_1 = 0`; the non-regular version has
`d e_010 = e_10 - e_00 + e_01` with `e_00` not an edge, so the hole survives and
`H_1 = F`.

Output: `results/torsion_conjecture.txt`.

---

## Usage

Requires Python and NumPy; Python 3.12 / NumPy 2.3.5 is the verified setup.
`core.py` alone is enough for default or dimensions-only calculations;
the Morse backend additionally needs `morse.py`.

```python
from core import path_homology
r = path_homology(vertices=[0, 1, 2], edges=[(0, 1), (1, 2)], max_dim=3, field="Q")
print(r.dim_omega, r.betti)
```

When only Betti numbers are needed, use `generators=False` (the historical
gen3 measurements above showed an additional 45x to 731x speed-up):

```python
r = path_homology(vertices=v, edges=E, max_dim=6, field="Q", generators=False)
```

```bash
python check_paper_examples.py    # GLMY worked examples
python check_circulant.py         # Tang-Yau Theorems 1.1 and 1.2
python check_torsion.py           # Chowdhury-Huntsman-Yutin torsion conjecture
python test_path_homology.py      # self-tests
```

### Current usage and coefficients (gen6)

Current minimal usage and coefficient conventions:

```python
from core import path_homology
v = [0, 1, 2, 3]
E = [(0, 1), (1, 2), (2, 3), (3, 0)]
k = 1

# H_0,...,H_k dimensions, without explicit Omega bases or representatives.
dims = path_homology(v, E, max_dim=k+1, field="Q", loops=False,
                     generators=False, store_paths=False)
print({p: dims.betti[p] for p in range(k+1)})  # {0: 1, 1: 1}

# Explicit representatives in ORIGINAL path coordinates; no additional
# dense allowed-path boundary matrices, but Omega bases are still constructed.
r = path_homology(v, E, max_dim=k+1, field="Q", loops=False,
                  homology_method="morse", store_matrices=False)
print(r.homology_basis[1])
print(r.morse_stats)
```

- To compute the original graph's H_k, enumerate through **at least k+1**, unless
  Omega_(k+1)=0 has been separately certified. The top truncation's homology is not
  automatically the original graph's homology in that degree.
- Morse supports exact **Q and prime fields F_p**, e.g. `field=2`, `3`, `5`, `101`;
  it is not restricted to F_2. It does **not** implement arbitrary coefficient
  rings, integral torsion, or arbitrary extension fields. Floating-point `R`
  continues to use the linear backend. The independent Smith-form calculation in
  `check_torsion.py` is retained; it is not an integral Morse backend.
- Morse requires `generators=True` (the default). Omega's canonical basis stays
  unchanged; its Z/B/H bases may differ from the linear route. Its boundary output
  is an independent basis rather than possibly redundant columns.
- `regular=False` and `loops=True` remain defaults. The example explicitly removes
  loops; use `regular=True` only when that boundary convention is intended.
- Chain homotopy equivalence here does not by itself preserve a filtration or
  the nonzero spectrum of the original Laplacian. Neither separate project is
  replaced by this ordinary path-homology reduction.

```bash
python cli.py run --edges "0-1,1-2,2-3,3-0" --max-dim 2 --stream --json
python cli.py run --edges "0-1,1-2,2-3,3-0" --max-dim 2 --homology-method morse --no-boundary-matrices --json
```

## Layout

```
core.py                        current implementation (gen6)
morse.py                       exact algebraic Morse cancellation and lifts
original_core.py               first version (gen0)
docs/OPTIMIZATION_REPORT.md    full optimisation report
VERSIONS.md                    current version index and hashes
docs/VERSIONS.md               annotated historical gen0--gen3 record
docs/ALGEBRAIC_MORSE_PATH_HOMOLOGY_20261019.md  technique and proof
docs/MORSE_PERFORMANCE_20261019.md             current performance report
performance_morse_20261019/    raw measurements and verification summaries
reference/                     fixed gen0/gen3 sources and historical oracle
_baseline_gen4_20261019/        fixed gen4 regression source
_baseline_gen5_20261019/        fixed gen5 comparison source
verify_release.py              10 portable release checks
verify_repository.py           four retained repository validation checks
.local-results/                ignored fresh local verification outputs
check_paper_examples.py        GLMY worked examples
check_circulant.py             Tang-Yau theorems
check_torsion.py               torsion conjecture
results/                       raw output of the three checks and the self-tests
```

## References

1. A. Grigor'yan, Y. Lin, Y. Muranov, S.-T. Yau, *Homologies of path complexes and
   digraphs*, arXiv:1207.2834.
2. X. Tang, S.-T. Yau, *Path Homology of Circulant Digraphs*, arXiv:2602.04140.
3. S. Chowdhury, S. Huntsman, M. Yutin, *Path homologies of motifs and temporal
   network representations*, Applied Network Science **7**:4 (2022),
   doi:10.1007/s41109-021-00441-z.

4. E. Sköldberg, *Algebraic Morse theory and homological perturbation theory*,
   [arXiv:1311.5803](https://arxiv.org/abs/1311.5803).
5. L. Lampret, *Chain complex reduction via fast digraph traversal*,
   [arXiv:1903.00783](https://arxiv.org/abs/1903.00783). This implementation uses
   sequential elementary cancellations, not that paper's batch matching/traversal
   algorithm.

## Digraph families

`families/` collects path-homology computations for classical digraph
families and for Cayley digraphs of finite groups: transitive and regular
tournaments, every tournament up to isomorphism on at most five vertices,
directed and bidirectional cycles, complete and complete bipartite digraphs,
de Bruijn digraphs, all groups of order at most 10, and the permutation
groups S_n and A_n, together with how subgroups, Cartesian products and
quotients appear. See `families/README.md` for the index and
`families/COMPLEXITY.md` for the cost model and the estimated reach of the
engine inside a three-hour budget.

Those studies and their original results are retained unchanged; this update does
not claim that all family scans have been rerun with gen6. Some historical family
scripts rely on research-engine files not included in the original repository;
see [the update notes](docs/REPOSITORY_UPDATE.md) for the verification boundary.
