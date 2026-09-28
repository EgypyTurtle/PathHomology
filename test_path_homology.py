# -*- coding: utf-8 -*-
"""
test_path_homology.py -- checks of every construction against the paper.

Run with ``python test_path_homology.py`` (no pytest required) or
``pytest test_path_homology.py``.

Each test cites the statement in arXiv:1207.2834v4 that it verifies.
"""

from __future__ import annotations

import numpy as np

from core import (
    _rank,
    allowed_paths,
    boundary_matrix,
    boundary_of_path,
    bridges,
    is_semi_allowed,
    omega_basis,
    path_homology,
    semi_edges,
)

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# eq. (2.2)-(2.3): the boundary operator
# ---------------------------------------------------------------------------


def test_boundary_operator_formulas():
    """eq. (2.3): partial e_i = e, partial e_ij = e_j - e_i, partial e_ijk = e_jk - e_ik + e_ij."""
    check("partial e_i", boundary_of_path((0,)), [((), 1)])
    check("partial e_ij", boundary_of_path((0, 1)), [((1,), 1), ((0,), -1)])
    check(
        "partial e_ijk",
        boundary_of_path((0, 1, 2)),
        [((1, 2), 1), ((0, 2), -1), ((0, 1), 1)],
    )


def test_partial_squared_zero_on_lambda():
    """Lemma 2.4: partial^2 = 0 on the full space Lambda_*."""
    verts = list(range(4))
    import itertools

    for p in (1, 2, 3):
        paths_p = list(itertools.product(verts, repeat=p + 1))
        paths_prev = list(itertools.product(verts, repeat=p))
        paths_prev2 = list(itertools.product(verts, repeat=p - 1))
        d_p = boundary_matrix(paths_p, paths_prev)
        d_prev = boundary_matrix(paths_prev, paths_prev2)
        prod = d_prev @ d_p
        check(
            f"partial^2=0 on Lambda_{p}",
            all(c == 0 for c in prod.ravel()) if prod.size else True,
            True,
        )


def test_regular_boundary_drops_nonregular_faces():
    """Section 2.3 / Example 3.14: partial^reg e_010 = e_10 + e_01 (e_00 dropped)."""
    check(
        "partial^reg e_010",
        boundary_of_path((0, 1, 0), regular=True),
        [((1, 0), 1), ((0, 1), 1)],
    )
    check(
        "partial e_010 (non-regular keeps e_00)",
        boundary_of_path((0, 1, 0), regular=False),
        [((1, 0), 1), ((0, 0), -1), ((0, 1), 1)],
    )


# ---------------------------------------------------------------------------
# Example 3.3: allowed paths
# ---------------------------------------------------------------------------


def test_allowed_paths_are_directed_walks():
    """Example 3.3: allowed n-paths are exactly the directed walks of length n."""
    E = [(0, 1), (1, 2), (0, 2)]
    paths = allowed_paths(None, E, max_dim=4)
    check("A_0", paths[0], [(0,), (1,), (2,)])
    check("A_1", paths[1], [(0, 1), (0, 2), (1, 2)])
    check("A_2", paths[2], [(0, 1, 2)])
    check("A_3 empty", 3 in paths, False)


def test_allowed_paths_truncation_property():
    """
    Property (3.1): every truncated path of an allowed path is allowed.

    Note that we must exclude the *top* dimension: enumeration stops at
    ``max_dim``, so the truncations of the top-dimensional paths are present but
    their own extensions were never generated.  The property is a statement
    about the untruncated path complex.
    """
    import itertools

    E = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    paths = allowed_paths(None, E, max_dim=5)
    top = max(paths)
    ok = True
    for p in paths:
        if p in (0, top):
            # dim 0 truncates to the (-1)-path e; the top dimension's own
            # extensions were never generated because enumeration was truncated
            continue
        # truncating a p-path gives a (p-1)-path, so compare against paths[p-1]
        smaller = set(paths.get(p - 1, []))
        for path in paths[p]:
            if path[:-1] not in smaller:        # drop last vertex
                ok = False
            if path[1:] not in smaller:         # drop first vertex
                ok = False
    check("(3.1) truncation property (truncated digraph)", ok, True)

    # An independent check that does not depend on truncation at all: take a
    # transitive tournament (i->j for every i<j).  The longest directed path
    # uses all n vertices, so enumeration terminates naturally at n-1 and no
    # path is cut off.  The truncation property must then hold everywhere.
    n = 7
    complete_dag = [(i, j) for i in range(n) for j in range(i + 1, n)]
    cpaths = allowed_paths(None, complete_dag, max_dim=6)
    check("complete DAG enumerates all dimensions", max(cpaths), n - 1)
    ok_dag = True
    for p in cpaths:
        if p == 0:
            # truncating a vertex yields the (-1)-path e, i.e. P_{-1} = {e},
            # which is not stored in the dictionary of positive dimensions
            continue
        # truncating a p-path gives a (p-1)-path
        smaller = set(cpaths.get(p - 1, []))
        for path in cpaths[p]:
            if path[:-1] not in smaller or path[1:] not in smaller:
                ok_dag = False
    check("(3.1) truncation property (complete DAG, untruncated)", ok_dag, True)


# ---------------------------------------------------------------------------
# Section 4.1: semi-edges and bridges
# ---------------------------------------------------------------------------


def test_semi_edges_and_bridges():
    """Section 4.1: i >-> j iff i->j is not an edge but i->k->j is allowed."""
    E = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]
    check("semi-edges of Fig.8", semi_edges(E), [(0, 3), (0, 4)])
    check("bridges of 0>->3", bridges(E)[(0, 3)], [1, 2])
    check("bridges of 0>->4", bridges(E)[(0, 4)], [1, 2])
    check("an edge is not a semi-edge", (0, 1) in semi_edges(E), False)


def test_is_semi_allowed():
    """Section 4.1: exactly one semi-edge among the consecutive pairs."""
    E = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]
    es, se = set(E), set(semi_edges(E))
    check("0,3 is semi-allowed", is_semi_allowed((0, 3), es, se), True)
    check("0,1 is not (it is an edge)", is_semi_allowed((0, 1), es, se), False)
    check("0,5 is not (neither)", is_semi_allowed((0, 5), es, se), False)


# ---------------------------------------------------------------------------
# Section 3.3: Omega_p
# ---------------------------------------------------------------------------


def test_omega_0_and_1_equal_A():
    """After (3.8): Omega_0 = A_0 and Omega_1 = A_1 always."""
    E = [(0, 1), (1, 2), (2, 0), (0, 2)]
    r = path_homology(edges=E, max_dim=5)
    check("dim Omega_0 = dim A_0", r.dim_omega[0], len(r.allowed[0]))
    check("dim Omega_1 = dim A_1", r.dim_omega[1], len(r.allowed[1]))


def test_definition_3_8_by_hand():
    """Example 3.9: for 0->1->2 the allowed 2-path e_012 is NOT partial-invariant."""
    r = path_homology(edges=[(0, 1), (1, 2)], max_dim=4)
    check("A_2 = span{e_012}", r.allowed[2], [(0, 1, 2)])
    check("dim Omega_2 = 0", r.dim_omega[2], 0)


def test_omega_methods_agree():
    """Three implementations of Omega_p must agree (definition 3.8, Lemma 4.1, Prop 4.2)."""
    for name, E in [
        ("Fig8", [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]),
        ("square", [(0, 1), (1, 2), (0, 3), (3, 2)]),
        ("triangle", [(0, 1), (1, 2), (0, 2)]),
        ("C4", [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ]:
        paths = allowed_paths(None, E, max_dim=6)
        dims = {}
        for m in ("kernel", "lemma41", "prop42"):
            dims[m] = tuple(
                len(omega_basis(paths, E, p, method=m)[0]) for p in range(0, 5)
            )
        vals = set(dims.values())
        check(f"{name}: kernel == lemma41 == prop42", len(vals), 1)


# ---------------------------------------------------------------------------
# Proposition 4.2 and Theorem 4.3
# ---------------------------------------------------------------------------


def test_proposition_4_2():
    """(4.4): dim Omega_2 = |P_2| - |S|, with S the set of semi-edges."""
    for name, E in [
        ("Fig8", [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]),
        ("square", [(0, 1), (1, 2), (0, 3), (3, 2)]),
        ("triangle", [(0, 1), (1, 2), (0, 2)]),
    ]:
        r = path_homology(edges=E, max_dim=4)
        check(
            f"{name}: dim Omega_2 = |P_2| - |S|",
            r.dim_omega[2],
            len(r.allowed.get(2, [])) - len(semi_edges(E)),
        )


def test_theorem_4_3_no_squares():
    """Theorem 4.3: with no squares, dim Omega_2 = #triangles and Omega_p = 0 for p>2."""
    # a triangle plus a pendant edge: no square
    E = [(0, 1), (1, 2), (0, 2), (2, 3)]
    r = path_homology(edges=E, max_dim=4)
    check("dim Omega_2 = 1 (one triangle)", r.dim_omega[2], 1)
    check("dim Omega_p = 0 for p >= 3",
          all(r.dim_omega.get(p, 0) == 0 for p in range(3, 6)), True)


# ---------------------------------------------------------------------------
# Proposition 4.7 and Section 4.6: full homology computations
# ---------------------------------------------------------------------------


def test_figure_8_example():
    """Section 4.6, Figure 8: dim Omega = (6,8,2), H = (1,1,0), chi = 0."""
    E = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]
    r = path_homology(edges=E, max_dim=6)
    check("dim Omega_0", r.dim_omega[0], 6)
    check("dim Omega_1", r.dim_omega[1], 8)
    check("dim Omega_2", r.dim_omega[2], 2)
    check("dim H_0", r.betti[0], 1)
    check("dim H_1", r.betti[1], 1)
    check("dim H_2", r.betti[2], 0)
    check("chi = 0", r.euler_characteristic, 0)
    # the paper's spanning element of H_1
    check("H_1 generator", r._fmt(r.homology_basis[1][0]), "e13 -e14 -e53 +e54")
    # the basis of Omega_2 given in the paper (up to sign)
    got = {r._fmt(v) for v in r.omega_paths[2]}
    check(
        "Omega_2 basis up to sign",
        got in ({"-e013 +e023", "-e014 +e024"}, {"e013 -e023", "e014 -e024"}),
        True,
    )


def test_proposition_4_7_cycle_graphs():
    """Proposition 4.7: for a cycle-graph, dim H_0 = 1 and dim H_p = 0 for p >= 2."""
    cases = [
        ("C3 directed", [(0, 1), (1, 2), (2, 0)], False),
        ("C4 directed", [(0, 1), (1, 2), (2, 3), (3, 0)], False),
        ("C5 directed", [(i, (i + 1) % 5) for i in range(5)], False),
        ("C6 directed", [(i, (i + 1) % 6) for i in range(6)], False),
        ("C4 mixed", [(0, 1), (2, 1), (2, 3), (0, 3)], False),
        ("triangle (transitive)", [(0, 1), (1, 2), (0, 2)], True),
        ("square (transitive)", [(0, 1), (1, 2), (0, 3), (3, 2)], True),
    ]
    for name, E, tri_or_sq in cases:
        r = path_homology(edges=E, max_dim=6)
        check(f"{name}: dim H_0 = 1", r.betti.get(0), 1)
        check(f"{name}: dim H_1", r.betti.get(1), 0 if tri_or_sq else 1)
        check(f"{name}: dim Omega_2", r.dim_omega[2], 1 if tri_or_sq else 0)
        check(f"{name}: dim Omega_p = 0 for p >= 3",
              all(r.dim_omega.get(p, 0) == 0 for p in range(3, 7)), True)
        check(f"{name}: dim H_p = 0 for p >= 2",
              all(r.betti.get(p, 0) == 0 for p in range(2, 7)), True)
        check(f"{name}: chi", r.euler_characteristic, 1 if tri_or_sq else 0)


def test_example_3_14_regular_vs_nonregular():
    """Example 3.14: 0<->1; non-regular sees the 'hole', regular does not."""
    # non-regular: H_0 = 1, H_1 = 1, chi = 0
    r = path_homology(edges=[(0, 1), (1, 0)], max_dim=4)
    check("nonreg dim Omega_2", r.dim_omega[2], 0)
    check("nonreg dim H_0", r.betti[0], 1)
    check("nonreg dim H_1", r.betti[1], 1)
    check("nonreg H_1 generator", r._fmt(r.homology_basis[1][0]), "e01 +e10")
    check("nonreg chi", r.euler_characteristic, 0)
    # regular: Omega^reg_n = A_n, and H_n = 0 for n >= 1
    rr = path_homology(edges=[(0, 1), (1, 0)], max_dim=4, regular=True)
    check("reg dim Omega_2 = dim A_2", rr.dim_omega[2], len(rr.allowed[2]))
    check("reg dim H_0", rr.betti[0], 1)
    check("reg dim H_1", rr.betti[1], 0)
    check("reg dim H_2", rr.betti[2], 0)


def test_octahedron():
    """
    Example 6.17, Fig.24: the octahedron is a 2-sphere graph.

    The paper states |V| = 6, |E| = 12, an EMPTY semi-edge set,
    A_2 = span{e_024,e_025,e_034,e_035,e_124,e_125,e_134,e_135},
    dim Omega = (6,12,8), H = (1,0,1) and chi = 2.  The 8 A_2 triples are the
    8 faces of the octahedron; the 8 directed edges they induce are
    (0,2),(0,3),(1,2),(1,3) and (2,4),(2,5),(3,4),(3,5).  Completing these to
    |E| = 12 while emptying the semi-edge set forces the remaining four edges
    (0,4),(0,5),(1,4),(1,5), which is the digraph used here.
    """
    edges = [
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (2, 5), (3, 4), (3, 5),
    ]
    r = path_homology(edges=edges, max_dim=6)
    check("|E| = 12", len(r.edges), 12)
    check("semi-edges empty", semi_edges(edges), [])
    check(
        "A_2 equals the paper's",
        sorted(r.allowed[2]),
        [(0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
         (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5)],
    )
    check("dim Omega_0", r.dim_omega[0], 6)
    check("dim Omega_1", r.dim_omega[1], 12)
    check("dim Omega_2", r.dim_omega[2], 8)
    check("Omega_2 = A_2 (no semi-edges, Prop 4.2)", r.dim_omega[2], len(r.allowed[2]))
    check("no allowed 3-paths", 3 in r.allowed, False)
    check("dim H_0", r.betti[0], 1)
    check("dim H_1", r.betti[1], 0)
    check("dim H_2", r.betti[2], 1)
    check("chi = 2", r.euler_characteristic, 2)


# ---------------------------------------------------------------------------
# internal consistency
# ---------------------------------------------------------------------------


def test_partial_squared_zero_on_omega():
    """partial(Omega_n) is contained in Omega_{n-1}, so partial^2 = 0 on Omega_*."""
    for name, E in [
        ("Fig8", [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]),
        ("C5", [(i, (i + 1) % 5) for i in range(5)]),
        ("octahedron", [(0, 2), (0, 3), (0, 4), (0, 5),
                        (1, 2), (1, 3), (1, 4), (1, 5),
                        (2, 4), (2, 5), (3, 4), (3, 5)]),
    ]:
        r = path_homology(edges=E, max_dim=6)
        ok = True
        for p in sorted(r.allowed):
            om = r.omega_matrix.get(p)
            if p >= 2 and om is not None and om.shape[1]:
                d_p = r.boundary_matrices[p] @ om
                prod = r.boundary_matrices[p - 1] @ d_p
                if prod.size and any(c != 0 for c in prod.ravel()):
                    ok = False
        check(f"{name}: partial^2 = 0 on Omega_*", ok, True)


def test_euler_characteristic_consistency():
    """(3.13)/(3.16): chi = sum (-1)^p dim Omega_p = sum (-1)^p dim H_p."""
    for name, E in [
        ("Fig8", [(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)]),
        ("C4", [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ("triangle", [(0, 1), (1, 2), (0, 2)]),
    ]:
        r = path_homology(edges=E, max_dim=6)
        alt_betti = sum((-1) ** p * r.betti.get(p, 0) for p in r.betti)
        check(f"{name}: chi = alternating sum of Betti", r.euler_characteristic, alt_betti)


def test_betti_0_is_weak_component_count():
    """dim H_0 equals the number of weakly connected components."""
    for name, n, E in [
        ("2 comps", 6, [(0, 1), (1, 2), (3, 4)]),
        ("3 comps", 9, [(0, 1), (2, 3), (4, 5), (6, 7)]),
        ("connected", 5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ]:
        r = path_homology(vertices=list(range(n)), edges=E, max_dim=4)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in E:
            a, b = find(u), find(v)
            if a != b:
                parent[a] = b
        comps = len({find(i) for i in range(n)})
        check(f"{name}: dim H_0 = #components", r.betti.get(0), comps)


def test_betti_numbers_are_nonnegative():
    """Sanity: dim H_p = dim Z_p - dim B_p must never be negative."""
    import random

    random.seed(7)
    worst = 0
    for trial in range(12):
        n = random.randint(6, 11)
        E = [(i, j) for i in range(n) for j in range(i + 1, n)
             if random.random() < 0.3]
        r = path_homology(vertices=list(range(n)), edges=E, max_dim=6)
        for p in r.betti:
            worst = min(worst, r.betti[p])
            if r.betti[p] < 0:
                check(f"trial{trial} dim H_{p} >= 0", r.betti[p], ">= 0")
    check("all Betti numbers non-negative", worst >= 0, True)


def test_boundary_matrix_orientation():
    """The matrix acts on column vectors as partial: (d v)_i = sum_j d[i, j] v_j."""
    E = [(0, 1), (1, 2), (0, 2)]
    paths = allowed_paths(None, E, max_dim=4)
    d2 = boundary_matrix(paths[2], paths[1])
    check("d_2 shape (A_1 rows, A_2 cols)", d2.shape, (3, 1))
    # partial e_012 = e_12 - e_02 + e_01, so the column is the coefficients of
    # e_01, e_02, e_12 in that order
    idx = {p: i for i, p in enumerate(paths[1])}
    col = {p: d2[idx[p], 0] for p in paths[1]}
    check("partial e_012 coefficients", col, {(0, 1): 1, (0, 2): -1, (1, 2): 1})


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} test functions\n")
    for t in tests:
        print(f"--- {t.__name__} ---")
        t()
        print("")
    print("=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print("   ", f)
    else:
        print("ALL TESTS PASSED")
    print("=" * 70)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
