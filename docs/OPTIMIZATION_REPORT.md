# Path homology: code review and optimisation report

## 1. Minimal set of files needed to compute

The actual GLMY path homology computation lives in `core.py`; at run time it depends
only on NumPy and the Python standard library.

- Smallest single-file usage: `core.py` plus NumPy, via `from core import path_homology`.
- Standard package usage: add `__init__.py` and use `from path_homology import path_homology`.
  This has been corrected to a package-relative import.
- Add `cli.py` only if a command line is wanted.
- `test_path_homology.py`, `test_fields.py` and the newly added `test_optimization.py`
  are verification code, not run-time dependencies of the engine.
- The `pn_*`, `dm_*`, `dmn_*`, `snf_*`, `check_*`, `verify_*`, `bench_*` and `profile_*`
  scripts are studies of particular graph families, integral-coefficient homology,
  audits, benchmarks or reporting scripts. They are not part of the minimal general
  implementation.

## 2. Combinatorial optimisation of Omega_n

Layer-by-layer expansion in `allowed_paths` is essentially an enumeration of directed
walks. If the API is to return the bases of all `A_p` together with generators, the
output size itself can be exponential, and no lossless pruning works for every graph.
One important safety optimisation was made here: input edges are de-duplicated before
expansion, so that duplicated edges are not amplified exponentially in higher
dimensions.

The extra BFS performed by the original default `kernel` method is not necessary. A
constraint row can only appear if it really is a non-allowed boundary face of some
allowed p-path. The current implementation scans the p+1 faces of each allowed p-path
directly and groups the faces that lie outside `A_{p-1}` into the constraint matrix.
The cost of building the constraints is therefore

`Theta((p+1)|A_p|)`,

and no longer depends on the number of candidate walks in the joint relation between
edges and semi-edges. `lemma41` was changed to the same direct grouping over actual
boundary faces, and `prop42` now performs a single scan of `A_2` instead of rescanning
all 2-paths for every semi-edge.

A dedicated stress test shows the effect of this pruning clearly. For a one-way chain
on 40 vertices, computing `Omega_15` involves only 25 allowed 15-paths. The old
algorithm built a `311270 x 25` constraint matrix, the overwhelming majority of whose
rows could never be hit by a boundary, and took 1.982 s; the new algorithm keeps only
the 350 real constraints and takes 0.013 s, about 154 times faster, with both giving
the same `dim Omega_15 = 0`.

## 3. Linear algebra optimisation

The original implementation stored `Fraction` and `GF` elements in `dtype=object`
NumPy matrices. NumPy matrix multiplication on such matrices falls back to Python
object arithmetic, producing large numbers of `GF.__mul__`, `GF.__add__` and
`Fraction` calls. Now:

1. Every prime field `GF(p)` is converted to an integer NumPy array for modular
   elimination, nullspace, rank, pivot columns and matrix multiplication, and then
   converted back to `GF` objects so the original API is preserved.
2. Matrix products whose coefficients lie in `Q` but are integral, and which can be
   proved not to overflow int64, use a native int64 kernel; genuinely fractional data
   still falls back to exact `Fraction`.
3. The canonical basis produced by `_nullspace` contains an identity block on its free
   coordinates. Representing a vector in the `Omega_{p-1}` basis can therefore read off
   the corresponding rows, usually without any new Gaussian elimination.
4. Expanding cycles, boundaries and homology representatives from Omega coordinates to
   allowed-path coordinates was changed from one matrix product per column to one
   batched product per dimension.
5. An all-zero matrix over a finite field now retains its field information, so the
   fast path is not missed when the first element happens to be a plain integer zero.

End-to-end measurement on the same machine, complete DAG with `N=9, max_dim=5`:

| Coefficients | Before | After | Speed-up |
|---|---:|---:|---:|
| `GF(2)` | 8.633 s | 0.081 s | 107x |
| `Q` | 10.906 s | 0.710 s | 15x |

In this benchmark `dim Omega_p = (9, 36, 84, 126, 126, 84)`. After optimisation,
`GF(3)` and `GF(5)` take about 0.090 s and 0.093 s on the same task.

### 3.1 Second round: linear algebra and sparsity

A further profiling pass produced the following:

1. Rank and nullspace over the rationals first clear denominators row by row and then
   perform fraction-free elimination with primitive integer rows; `Fraction` is used
   only when back-substituting the nullspace basis at the end. The algorithm remains
   strictly exact.
2. Each differential `d_p` is reduced only once, and its rank is obtained from the
   rank-nullity theorem, removing a second elimination of the same matrix.
3. The case `Omega_p = A_p` is detected explicitly, reusing the boundary matrix and
   existing coordinates instead of repeatedly scanning an identity matrix.
4. A new `generators=False` dimension-only mode was added. Let `C_p` be the
   non-allowed-boundary-face constraints and `F_p` the full boundary on allowed
   p-paths. Then

   `dim Omega_p = |A_p| - rank(C_p)`,

   `dim Z_p = |A_p| - rank(F_p)`,

   `dim B_p = dim Omega_{p+1} - dim Z_{p+1}`.

   When only dimensions and Betti numbers are wanted, no Omega basis, no change of
   basis and no representatives are needed at all.
5. This mode no longer constructs dense boundary matrices. Each boundary row is stored
   as a dictionary of its non-zero columns; over `F_2` rows are further packed into
   Python big-integer bitsets and eliminated by big-integer XOR; over `Q` a sparse
   primitive-integer elimination is used.

Second-round results on the same complete DAG with `N=9, max_dim=5`:

| Mode | `GF(2)` | `Q` | Relative to the original |
|---|---:|---:|---:|
| Original | 8.633 s | 10.906 s | 1x |
| Full generator output | 0.037 s | 0.106 s | 235x / 103x |
| Sparse dimension mode | 0.0010 s | 0.0063 s | 8200x / 1700x |

On a complete DAG with `N=15, max_dim=6`, where there are 6435 allowed 6-paths, the
sparse dimension mode needs about 0.067 s over `GF(2)` and about 5.62 s over exact
`Q`, and does not need to retain any dense matrix of shape
(number of (p-1)-paths) by (number of p-paths).

### 3.2 Third round: certifiable modular rank and batched exact back-substitution

Profiling of the third round showed that the exact `Q` sparse elimination of round two
still generated tens of millions of dictionary lookups. This round adds:

1. For an integer matrix the rank mod 2 is computed first. The mod 2 rank is a strict
   lower bound for the rational rank; if it reaches `min(non-zero rows, columns)`, the
   rational rank is proved unconditionally and integer elimination is skipped.
2. When all boundary faces are still allowed, the chain condition that the image of
   `partial_p` lies in the kernel of `partial_(p-1)` gives a tighter upper bound for
   the rational rank. When the mod 2 lower bound reaches that upper bound, a rigorous
   certificate is again obtained. If it does not, the code still falls back to the
   original exact fraction-free integer elimination, so this is not a probabilistic
   algorithm.
3. The one-dimensional boundary matrix uses the graph incidence-matrix theorem
   directly: its rank equals the number of vertices minus the number of weakly
   connected components.
4. Sparse row elimination now merges non-zero entries in place, removing repeated
   construction of key sets and multiple dictionary lookups.
5. All free variables of an exact nullspace are back-substituted simultaneously,
   skipping the many necessarily-zero checks in the free-variable identity block. The
   output is still the same canonical free-variable basis.
6. Over `GF(2)`, small and medium widths generate big-integer bit rows directly; very
   wide matrices automatically switch back to sparse dictionaries and are packed in a
   single pass, avoiding repeated copying of ever-larger Python integers.

Re-measured on the same machine:

| Task | Round 2 | Round 3 | Speed-up |
|---|---:|---:|---:|
| `N=15, max_dim=6, Q`, dimension mode | 3.65 s | 0.092 s | 39.6x |
| `N=11, max_dim=6, Q`, full generators | 3.57 s | 1.21 s | 2.9x |
| Random graph with cycles (949 four-paths), `Q`, dimension mode | 0.422 s | 0.221 s | 1.9x |

On the complete DAG with `N=15`, the third-round exact `Q` dimension mode is now in
the same order of magnitude as `GF(2)`. Every fast return carries a mathematical
certificate in which the lower and upper bounds coincide, and when this cannot be
proved the code automatically takes the exact fallback path.

## 4. Actual support for coefficient fields

The core does not implement only `F_2`. It supports:

- `field="Q"`: exact rationals, the default;
- `field=2`, `"Z2"`, `"GF2"`, `"F2"`: `F_2`;
- `field=p`: the prime field `F_p` for any prime p;
- `field="R"`: floating-point approximate real linear algebra.

Primality validation was added in this round: the original code accepted `field=4`,
but `Z/4Z` is not a field and a Fermat-style "inversion" returns wrong results.
Composite moduli are now rejected explicitly.

"Arbitrary coefficients", if it means arbitrary fields, still does not include
arbitrary field extensions, algebraic number fields or user-defined fields; if it
means arbitrary rings, the core API does not support them either. In particular,
torsion with integer coefficients `Z` cannot be obtained from rank and nullspace over
an ordinary field; the scripts `pn_integral.py`, `h1_integral.py`, `snf_robust.py` and
others in the directory are separate Smith-normal-form research scripts and have not
been merged into the general core.

The characteristic of the coefficient field genuinely changes the answer. For example,
for a member of the `P_2` graph family appearing in the code:

- `dim H_1(P_2; Q) = 1`;
- `dim H_1(P_2; F_2) = 2`;
- `dim H_1(P_2; F_3) = dim H_1(P_2; F_5) = 1`.

This is consistent with the 2-torsion in the integral homology, and shows that `F_2`
is not merely an interchangeable speed-up option.

## 5. Verification and limits of use

- All 21 pre-existing mathematical regression tests pass.
- On 12 fixed random graphs, `Q`, `F_2`, `F_3` and `F_5` were compared one by one
  against the pre-optimisation implementation; `dim Omega`, cycle dimensions, boundary
  dimensions and Betti numbers all agree.
- Native modular matrix multiplication for `GF(2)`, `GF(3)`, `GF(5)` and `GF(101)`,
  de-duplication of repeated edges, rejection of composite moduli, and the
  characteristic-dependent examples above were additionally verified.

Finally, note the truncation. For digraphs with directed cycles, allowed paths may
exist in every dimension. `max_dim=D` computes the truncated complex, and the
top-dimensional `H_D` ignores boundaries coming from `Omega_{D+1}`. To obtain `H_k`
one must therefore construct at least up to `Omega_{k+1}`, and judge from the problem
itself whether still higher dimensions continue to affect the result.
