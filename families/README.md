
# Digraph families: path homology computations

This folder collects computations of the GLMY path homology
(Grigor'yan–Lin–Muranov–Yau, *Homologies of path complexes and digraphs*,
[arXiv:1207.2834](https://arxiv.org/abs/1207.2834)) for digraph families that
are either classical or arise as Cayley digraphs of finite groups.

Everything here is reproducible from the scripts in `scripts/` using the
engine in the repository root. Every number below was produced by that engine;
nothing is quoted from a paper.

## Contents

| file | what it covers |
|---|---|
| `TOURNAMENTS.md` | transitive tournaments `TT_n`, cyclic (regular) tournaments `RT_n`, and all tournaments up to isomorphism for `n <= 5` |
| `CLASSICAL.md` | directed cycles, bidirectional cycles, complete digraphs, complete bipartite digraphs, de Bruijn digraphs |
| `CAYLEY_GROUPS.md` | Cayley digraphs of finite groups: all groups of order `<= 10`, dihedral groups, the prism family, `S_n`, `A_n`, and how subgroups, products and quotients appear |
| `COMPLEXITY.md` | cost model of the engine and the largest instances reachable inside a 3-hour budget in Betti-number-only mode |

## Conventions used throughout

* `dim Omega = [dim Omega_0, dim Omega_1, ...]` and
  `betti = [b_0, b_1, ...]`.
* Unless stated otherwise the computation ran with `max_dim = 4`, over the
  rationals and over `F_2`.
* **Truncation rule.** With `max_dim = D` the top entry of `betti` is
  `dim ker(d_D)`, which is *not* a homology dimension. A Betti number `b_p` is
  only read off when `max_dim >= p + 1`. All tables below obey this rule.
* `non-regular` convention (vertices of an allowed path need not be distinct),
  matching Definition 2.1 of the GLMY paper.
* An entry `Q vs F2` of `=` means the rational and mod-2 Betti vectors agree;
  `DIFFER` would indicate torsion.

## Verification scope

* The engine reproduces every worked example in GLMY (arXiv:1207.2834)
  (94 assertions), Theorem 1.1 and 1.2 of Tang–Yau for the circulant digraph
  `C_5^{1,2}`, and the torsion conjecture of Chowdhury–Huntsman–Yutin
  (*Applied Network Science* **7**:4, 2022) for `n <= 12` (that paper reports
  `n <= 8`).
* The family tables in this folder are computations with this engine, not
  quotations from the literature. Where a table agrees with a published
  statement, or fails to, this is said explicitly in the relevant section --
  see in particular the `C_2 x C_2` entry in `CAYLEY_GROUPS.md`.
* Every table states the range over which it was computed. Fitted formulas
  (`RT_n`, `K_{n,n}`, `b_1 = |G|(|S|-1) + 1`) are labelled as fits or as
  arguments; none is presented as a theorem unless the argument is given.
* `Q vs F2` agreement is a numerical observation over the stated range, not a
  proof that the integral homology is torsion-free.

## Reproducing

```
python scripts/families_scan.py            # tournaments + classical families
python scripts/cayley_extend.py            # S_n / A_n, Omega_2 criterion
python scripts/cayley_certify.py           # closed form certificate
python scripts/cayley_quotient.py          # subgroup / product / quotient
python scripts/complexity_extrapolate.py   # cost model and 3-hour reach
```

`cayley_extend.py` needs `core_v4.py` next to it; the other scripts only need
the public engine.
