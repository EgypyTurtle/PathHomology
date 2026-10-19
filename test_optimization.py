# -*- coding: utf-8 -*-
"""Regression checks for the optimized path-homology implementation."""

from __future__ import annotations

import importlib.util
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

import core


_HERE = Path(__file__).resolve().parent
ORIGINAL = _HERE / "reference" / "gen0_core.py"


def load_original():
    spec = importlib.util.spec_from_file_location("original_core", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_pn(n):
    vertices = list(range(2 * n)) + [-2, -1]
    edges = [(i, i + 1) for i in range(2 * n - 1)] + [(2 * n - 1, 0)]
    for k in range(n):
        edges += [(-2, 2 * k), (2 * k, -2), (-1, 2 * k + 1), (2 * k + 1, -1)]
    return vertices, edges


def test_random_crosscheck():
    original = load_original()
    rng = random.Random(20260926)
    for _ in range(12):
        vertices = list(range(4))
        edges = [
            (u, v)
            for u in vertices
            for v in vertices
            if u != v and rng.random() < 0.35
        ]
        for field in ("Q", 2, 3, 5):
            old = original.path_homology(vertices, edges, max_dim=3, field=field)
            new = core.path_homology(vertices, edges, max_dim=3, field=field)
            fast = core.path_homology(
                vertices, edges, max_dim=3, field=field, generators=False
            )
            assert new.dim_omega == old.dim_omega
            assert new.dim_cycles == old.dim_cycles
            assert new.dim_boundaries == old.dim_boundaries
            assert new.betti == old.betti
            assert fast.dim_omega == new.dim_omega
            assert fast.dim_cycles == new.dim_cycles
            assert fast.dim_boundaries == new.dim_boundaries
            assert fast.betti == new.betti
            assert not fast.generators_materialized


def test_native_modular_matmul():
    rng = random.Random(7)
    for p in (2, 3, 5, 101):
        a0 = np.array([[rng.randrange(p) for _ in range(7)] for _ in range(5)])
        b0 = np.array([[rng.randrange(p) for _ in range(4)] for _ in range(7)])
        a = np.array([[core.GF(x, p) for x in row] for row in a0], dtype=object)
        b = np.array([[core.GF(x, p) for x in row] for row in b0], dtype=object)
        got = core._field_matmul(a, b)
        got_int = np.array([[int(x) for x in row] for row in got])
        assert np.array_equal(got_int, (a0 @ b0) % p)


def test_duplicate_edges_do_not_duplicate_paths():
    simple = [(0, 1), (1, 2)]
    duplicate = [(0, 1), (0, 1), (1, 2), (1, 2)]
    assert core.allowed_paths(edges=simple, max_dim=4) == core.allowed_paths(
        edges=duplicate, max_dim=4
    )


def test_coefficients_and_prime_validation():
    vertices, edges = build_pn(2)
    expected = {"Q": 1, 2: 2, 3: 1, 5: 1}
    for field, h1 in expected.items():
        result = core.path_homology(vertices, edges, max_dim=2, field=field)
        assert result.betti[1] == h1
    for composite in (4, 6, 9, "15"):
        try:
            core.resolve_field(composite)
        except ValueError:
            pass
        else:
            raise AssertionError(f"composite modulus accepted: {composite}")


def test_sparse_dimension_modes():
    cases = [
        ([(0, 1), (1, 0)], False, "kernel"),
        ([(0, 1), (1, 0)], True, "kernel"),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], False, "lemma41"),
        ([(0, 1), (1, 2), (0, 2)], False, "prop42"),
    ]
    for edges, regular, method in cases:
        for field in ("Q", 2, 3, "R"):
            full = core.path_homology(
                edges=edges, max_dim=3, regular=regular,
                method=method, field=field,
            )
            sparse = core.path_homology(
                edges=edges, max_dim=3, regular=regular,
                method=method, field=field, generators=False,
            )
            dense = core.path_homology(
                edges=edges, max_dim=3, regular=regular,
                method=method, field=field, generators=False,
                store_matrices=True,
            )
            stream = core.path_homology(
                edges=edges, max_dim=3, regular=regular,
                method=method, field=field, generators=False, store_paths=False,
            )
            for result in (sparse, dense, stream):
                assert result.dim_omega == full.dim_omega
                assert result.dim_cycles == full.dim_cycles
                assert result.dim_boundaries == full.dim_boundaries
                assert result.betti == full.betti
            assert not sparse.boundary_matrices
            assert dense.boundary_matrices


def test_fraction_free_and_sparse_ranks():
    original = load_original()
    rng = random.Random(314159)
    for _ in range(40):
        m, n = rng.randrange(1, 7), rng.randrange(1, 8)
        rational = np.array([
            [Fraction(rng.randrange(-5, 6), rng.randrange(1, 6)) for _ in range(n)]
            for _ in range(m)
        ], dtype=object)
        assert core._rank(rational) == original._rank(rational)
        nullspace = core._nullspace(rational)
        original_nullspace = original._nullspace(rational)
        assert nullspace.shape[1] == n - core._rank(rational)
        assert all(value == 0 for value in (rational @ nullspace).ravel())
        assert np.array_equal(nullspace, original_nullspace)

        integer_rows = []
        integer_dense = np.zeros((m, n), dtype=object)
        for i in range(m):
            row = {}
            for j in range(n):
                value = rng.randrange(-2, 3)
                integer_dense[i, j] = Fraction(value)
                if value:
                    row[j] = value
            integer_rows.append(row)
        expected_q = core._rank(integer_dense)
        assert core._sparse_rank(integer_rows, n, "Q") == expected_q
        for p in (2, 3, 5):
            finite = np.array([
                [core.GF(int(integer_dense[i, j]), p) for j in range(n)]
                for i in range(m)
            ], dtype=object)
            assert core._sparse_rank(integer_rows, n, "GF", p) == core._rank(finite)
        real = np.array(integer_dense, dtype=float)
        assert core._sparse_rank(integer_rows, n, "R") == core._rank(real)

    # Rank drops modulo two here, so the certified route must fall back to Q.
    bad_prime_rows = [{0: 1, 1: 1}, {0: 1, 1: -1}]
    assert core._sparse_rank(bad_prime_rows, 2, "GF", 2) == 1
    assert core._q_sparse_rank_certified(bad_prime_rows, 2) == 2


def test_direct_gf2_boundary_rows():
    rng = random.Random(271828)
    for _ in range(30):
        vertices = list(range(6))
        edges = [
            (u, v) for u in vertices for v in vertices
            if rng.random() < 0.22
        ]
        paths = core.allowed_paths(vertices, edges, max_dim=4)
        for p in range(1, max(paths) + 1):
            sparse = core._sparse_boundary_rows(paths[p])
            packed = core._gf2_boundary_bitrows(paths[p])
            for face in set(sparse) | set(packed):
                expected = 0
                for column, value in sparse.get(face, {}).items():
                    if value & 1:
                        expected ^= 1 << column
                assert packed.get(face, 0) == expected
            assert core._sparse_rank(sparse.values(), len(paths[p]), "GF", 2) == (
                core._sparse_rank(packed.values(), len(paths[p]), "GF", 2)
            )


def test_sparse_peeling_and_streaming():
    rng = random.Random(20261018)
    for _ in range(25):
        rows = [{j: rng.choice((-2, -1, 1, 2))
                 for j in rng.sample(range(45), rng.randrange(1, 7))}
                for _ in range(55)]
        for kind, prime in (("Q", None), ("GF", 2), ("GF", 3), ("GF", 101)):
            offset, residual, width = core._peel_sparse_rows(rows, kind, prime)
            expected = core._sparse_rank_raw(rows, 45, kind, prime)
            assert offset + core._sparse_rank_raw(residual, width, kind, prime) == expected

    for _ in range(70):
        n = rng.randrange(2, 7)
        edges = [(u, v) for u in range(n) for v in range(n) if rng.random() < .22]
        for field in ("Q", 2, 3, "R"):
            for regular in (False, True):
                retained = core.path_homology(
                    range(n), edges, max_dim=4, field=field, regular=regular,
                    generators=False,
                )
                stream = core.path_homology(
                    iter(range(n)), iter(edges), max_dim=4, field=field,
                    regular=regular, generators=False, store_paths=False,
                )
                assert stream.dim_allowed == retained.dim_allowed
                assert stream.dim_omega == retained.dim_omega
                assert stream.dim_cycles == retained.dim_cycles
                assert stream.dim_boundaries == retained.dim_boundaries
                assert stream.betti == retained.betti
                assert stream.euler_characteristic == retained.euler_characteristic
                assert stream.allowed == {} and not stream.paths_materialized
                assert "streaming mode" in stream.summary()

    # Omega_2=0 for a long directed cycle. A huge formal path count must not
    # force materialisation of later degrees; the Betti values stay exact.
    edges = [(i, (i + 1) % 100) for i in range(100)]
    stream = core.path_homology(edges=edges, max_dim=30, field="Q",
                                generators=False, store_paths=False)
    assert stream.dim_allowed[30] == 100
    assert stream.betti[0] == stream.betti[1] == 1
    assert all(stream.betti[p] == 0 for p in range(2, 31))

    # Boundary-closed spaces can still have higher homology (octahedron).
    edges = [(i, j) for lower, upper in ((range(2), range(2, 4)),
                                        (range(2, 4), range(4, 6)),
                                        (range(2), range(4, 6)))
             for i in lower for j in upper]
    result = core.path_homology(range(6), edges, max_dim=3, field="Q",
                               generators=False, store_paths=False)
    assert result.betti[2] == 1

    for n in (2, 3, 4):
        vertices, edges = build_pn(n)
        for field in ("Q", 2, 3, 5):
            full = core.path_homology(vertices, edges, max_dim=3, field=field)
            stream = core.path_homology(vertices, edges, max_dim=3, field=field,
                                        generators=False, store_paths=False)
            assert stream.betti == full.betti
            assert stream.dim_omega == full.dim_omega

    # Acyclic and boundary-closed, but Omega spaces continue to grow. The
    # absence of candidate pruning must preserve the sparse-column fast path.
    for field in ("Q", 2, 3):
        edges = [(i, j) for i in range(9) for j in range(i + 1, 9)]
        full = core.path_homology(range(9), edges, max_dim=5, field=field,
                                  generators=False)
        stream = core.path_homology(range(9), edges, max_dim=5, field=field,
                                    generators=False, store_paths=False)
        assert stream.dim_omega == full.dim_omega
        assert stream.betti == full.betti


if __name__ == "__main__":
    tests = [
        test_random_crosscheck,
        test_native_modular_matmul,
        test_duplicate_edges_do_not_duplicate_paths,
        test_coefficients_and_prime_validation,
        test_sparse_dimension_modes,
        test_fraction_free_and_sparse_ranks,
        test_direct_gf2_boundary_rows,
        test_sparse_peeling_and_streaming,
    ]
    for test in tests:
        test()
        print("OK", test.__name__)
    print("ALL OPTIMIZATION TESTS PASSED")
