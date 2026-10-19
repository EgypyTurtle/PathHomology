"""Deterministic gen5 rank certificates and conservative-support regressions."""
import importlib.util
import random
from pathlib import Path
from unittest.mock import patch
import numpy as np

import core

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gen4_core", HERE / "_baseline_gen4_20261019" / "core.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def test_modular_certificates():
    for prime in (2, 3, 65521, 65519):
        assert core._is_prime(prime)
    # Rank drops modulo two, while the odd-prime certificate is full rank.
    rows = [{0: 1, 1: 1}, {0: 1, 1: -1}]
    with patch.object(core, "_sparse_rank_raw", wraps=core._sparse_rank_raw) as traced:
        assert core._q_sparse_rank_certified(rows, 2) == 2
        assert not any(call.args[2] == "Q" for call in traced.call_args_list)
    # Agreement across prime fields does NOT prove equality over Q.
    factor = 2 * 65521
    rows = [{0: factor, 1: factor}, {0: factor, 1: -factor}]
    assert core._q_sparse_rank_certified(rows, 2) == 2
    # Deficient ranks must fall back, even if all modular trials agree.
    rows = [{0: 1, 1: 2}, {0: 2, 1: 4}]
    with patch.object(core, "_sparse_rank_raw", wraps=core._sparse_rank_raw) as traced:
        assert core._q_sparse_rank_certified(rows, 2) == 1
        assert any(call.args[2] == "Q" for call in traced.call_args_list)
    rng = random.Random(20261019)
    for trial in range(180):
        n, m = rng.randrange(1, 42), rng.randrange(1, 55)
        rows = [{c: rng.choice((-3, -2, -1, 1, 2, 3))
                 for c in rng.sample(range(n), rng.randrange(min(n, 9) + 1))} for _ in range(m)]
        if trial % 3 == 0:
            rows += [dict(row) for row in rows[:m // 3]]
        expected = old._sparse_rank_raw(rows, n, "Q")
        assert core._q_sparse_rank_certified(rows, n) == expected
        assert core._q_sparse_rank_certified(rows, n, expected) == expected
        for p in (2, 3, 5, 101):
            assert core._sparse_rank(rows, n, "GF", p) == old._sparse_rank_raw(rows, n, "GF", p)


def test_independent_blocks():
    rows = []
    for b in range(80):
        rows.extend(({3*b: 1, 3*b+1: 1, 3*b+2: 1},
                     {3*b: 1, 3*b+1: -1, 3*b+2: 2}))
    blocks = core._sparse_row_blocks(rows, 240)
    assert len(blocks) == 80
    assert core._q_sparse_rank_certified(rows, 240) == 160
    assert core._q_sparse_rank_certified(rows, 240, 160) == 160
    for p in (2, 3, 5):
        assert core._sparse_rank(rows * 2, 240, "GF", p) == old._sparse_rank_raw(rows, 240, "GF", p)
    # Connected, large, nontrivial Q residual exercises the ordering heuristic.
    rng = random.Random(20261020)
    rows = [{c: rng.choice((-2, -1, 1, 2)) for c in rng.sample(range(85), 15)} for _ in range(70)]
    assert core._sparse_rank_raw(rows, 85, "Q") == old._sparse_rank_raw(rows, 85, "Q")
    assert core._gf2_sparse_column_rank(rows, 85) == old._sparse_rank_raw(rows, 85, "GF", 2)
    rows = [{c: rng.choice((-1, 1)) for c in rng.sample(range(200), 24)} for _ in range(220)]
    assert core._gf2_sparse_column_rank(rows, 200) == old._sparse_rank_raw(rows, 200, "GF", 2)


def test_materialized_bases():
    rng = random.Random(20261022)
    for _ in range(40):
        n = rng.randrange(3, 7)
        edges = [(u, v) for u in range(n) for v in range(n)
                 if u != v and rng.random() < .3]
        for field in ("Q", 2, 3, 5):
            for regular in (False, True):
                args = dict(max_dim=3, loops=False, regular=regular, field=field)
                got = core.path_homology(range(n), edges, **args)
                expected = old.path_homology(range(n), edges, **args)
                assert got.betti == expected.betti
                assert got.dim_omega == expected.dim_omega
                assert got.dim_cycles == expected.dim_cycles
                assert got.allowed == expected.allowed
                for p in got.omega_matrix:
                    assert np.array_equal(got.omega_matrix[p], expected.omega_matrix[p])
                assert got.cycles == expected.cycles
                assert got.homology_basis == expected.homology_basis


def test_support_and_streaming():
    rng = random.Random(20261021)
    graphs = []
    for mask in range(64):
        arcs = [(u, v) for u in range(3) for v in range(3) if u != v]
        graphs.append((3, [e for j, e in enumerate(arcs) if mask >> j & 1]))
    for _ in range(80):
        n = rng.randrange(4, 7)
        graphs.append((n, [(u, v) for u in range(n) for v in range(n)
                           if u != v and rng.random() < .3]))
    singleton_seen = suffix_seen = False
    for n, edges in graphs:
        for regular in (False, True):
            for field in ("Q", 2, 3, 5):
                kind, modulus = core.resolve_field(field)
                # Independently compare each support exclusion with an exact
                # canonical nullspace, not just another pruning implementation.
                paths = old.allowed_paths(range(n), edges, max_dim=3, loops=False)
                for p in (2, 3):
                    before = paths.get(p, [])
                    if not before:
                        continue
                    kept, rows, removed = core._prune_omega_singletons(before, set(edges), regular, kind, modulus)
                    basis, _ = old.omega_basis(paths, edges, p, regular=regular, field=field)
                    support = {before[i] for vector in basis for i, value in enumerate(vector) if value != 0}
                    assert support <= set(kept), (edges, p, regular, field)
                    assert rows == dict(core._sparse_boundary_rows(kept, regular=regular)) or not kept
                args = dict(max_dim=4, loops=False, regular=regular, field=field, generators=False)
                expected = old.path_homology(range(n), edges, **args)
                got = core.path_homology(range(n), edges, store_paths=False, **args)
                for name in ("dim_allowed", "dim_omega", "dim_cycles", "dim_boundaries", "betti"):
                    assert getattr(got, name) == getattr(expected, name), (n, edges, field, regular, name)
                singleton_seen |= any(s["singleton_removed"] for s in got.support_pruning.values())
                suffix_seen |= any(s["suffix_rejected"] for s in got.support_pruning.values())
    assert singleton_seen and suffix_seen
    # Unsupported shortcuts must remain disabled for self-loops and R.
    for regular in (False, True):
        edges = [(0, 0), (0, 1), (1, 0), (1, 2)]
        for field in ("Q", 2, 3, "R"):
            args = dict(max_dim=4, loops=True, regular=regular, field=field, generators=False)
            got = core.path_homology(range(3), edges, store_paths=False, **args)
            expected = old.path_homology(range(3), edges, **args)
            assert got.betti == expected.betti and got.dim_omega == expected.dim_omega


if __name__ == "__main__":
    for test in (test_modular_certificates, test_independent_blocks,
                 test_support_and_streaming, test_materialized_bases):
        test()
        print("PASS", test.__name__, flush=True)
