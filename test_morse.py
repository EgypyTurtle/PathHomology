"""Exact regression tests for the chain-level Morse backend (no pytest needed)."""
from fractions import Fraction
import random
import time
import numpy as np
import core
from morse import Arithmetic, reduce_complex


def matrix(vectors, n, field):
    kind, prime = core.resolve_field(field)
    out = core._field_zeros((n, len(vectors)), kind, prime)
    for j, vector in enumerate(vectors):
        for i, value in vector.items():
            out[i, j] = core.make_scalar(value, kind, prime)
    return out


def boundary(chain, field, regular=False):
    arithmetic = Arithmetic(None if field == "Q" else field)
    out = {}
    for path, value in chain.items():
        for face, sign in core.boundary_of_path(path, regular=regular):
            arithmetic.axpy(out, {face: sign}, arithmetic.scalar(value))
    return out


def check_path(vertices, edges, field, regular, loops=False, degree=3):
    options = dict(max_dim=degree, field=field, regular=regular, loops=loops)
    old = core.path_homology(vertices, edges, **options)
    new = core.path_homology(vertices, edges, homology_method="morse",
                             store_matrices=False, **options)
    for key in ("betti", "dim_omega", "dim_cycles", "dim_boundaries", "dim_allowed"):
        assert getattr(old, key) == getattr(new, key), (key, field, regular, edges)
    assert not new.boundary_matrices and not new.matrices_materialized
    assert new.generators_materialized
    for p, paths in new.allowed.items():
        assert old.omega_paths[p] == new.omega_paths[p]
        index = {path: j for j, path in enumerate(paths)}
        for name in ("cycles", "boundaries", "homology_basis"):
            vectors = getattr(new, name)[p]
            for chain in vectors:
                assert chain and chain.keys() <= index.keys()
                if p:
                    assert not boundary(chain, field, regular), (name, p, chain)
            coords = [{index[path]: c for path, c in v.items()} for v in vectors]
            assert core._rank(matrix(coords, len(paths), field)) == len(vectors)
        b = [{index[path]: c for path, c in v.items()} for v in new.boundaries[p]]
        h = [{index[path]: c for path, c in v.items()} for v in new.homology_basis[p]]
        assert core._rank(matrix(b + h, len(paths), field)) == new.dim_cycles[p]
        actual_b = [boundary(chain, field, regular)
                    for chain in new.omega_paths.get(p + 1, [])]
        actual_b = [{index[path]: c for path, c in v.items()} for v in actual_b]
        assert core._rank(matrix(b + actual_b, len(paths), field)) == len(b)
    assert sum((-1)**p * n for p, n in new.betti.items()) == new.euler_characteristic


def test_contractions():
    """Verify I,P,h identities for EACH step, not just final Betti numbers."""
    rng = random.Random(20261019)
    count = 0
    for field in ("Q", 2, 3, 5, 101):
        kind, prime = core.resolve_field(field)
        arithmetic = Arithmetic(prime)
        for _ in range(10):
            dims = {0: 3, 1: 4, 2: 3}
            d1 = matrix([{r: rng.randrange(-3, 4) for r in range(3)}
                         for _ in range(4)], 3, field)
            kernel = core._nullspace(d1)
            d2 = core._field_matmul(kernel, matrix(
                [{r: rng.randrange(-3, 4) for r in range(kernel.shape[1])}
                 for _ in range(3)], kernel.shape[1], field))
            differentials = {p: [{r: arithmetic.scalar(m[r, j])
                                   for r in range(m.shape[0]) if m[r, j] != 0}
                                  for j in range(m.shape[1])]
                             for p, m in ((1, d1), (2, d2))}
            reduced = reduce_complex(dims, differentials, modulus=prime,
                                     check=True, record_steps=True)
            labels = [(p, j) for p, n in dims.items() for j in range(n)]
            original_labels = list(labels)
            index = {label: i for i, label in enumerate(labels)}
            d = core._field_zeros((len(labels), len(labels)), kind, prime)
            for p, cols in differentials.items():
                for j, col in enumerate(cols):
                    for r, value in col.items():
                        d[index[p-1, r], index[p, j]] = core.make_scalar(value, kind, prime)
            inclusion = core._field_eye(len(labels), kind, prime)
            for step in reduced["steps"]:
                p, a, b = step["degree"], step["upper"], step["lower"]
                ia, ib = labels.index((p, a)), labels.index((p-1, b))
                keep = [i for i in range(len(labels)) if i not in (ia, ib)]
                eye = core._field_eye(len(labels), kind, prime)
                h = core._field_zeros(d.shape, kind, prime)
                h[ia, ib] = core.make_scalar(arithmetic.ratio(1, step["pivot"]), kind, prime)
                j = eye[:, keep]
                i_map = (eye - h @ d) @ j
                p_map = j.T @ (eye - d @ h)
                new_d = p_map @ d @ i_map
                assert np.all(p_map @ i_map == core._field_eye(len(keep), kind, prime))
                assert np.all(eye - i_map @ p_map == d @ h + h @ d)
                assert np.all(d @ i_map == i_map @ new_d)
                assert np.all(p_map @ d == new_d @ p_map)
                assert np.all(new_d @ new_d == 0)
                inclusion = inclusion @ i_map
                d, labels = new_d, [labels[i] for i in keep]
            assert np.all(d == 0)
            for col, (p, _) in enumerate(labels):
                critical_ids = [label for label in labels if label[0] == p]
                vector = reduced["homology"][p][critical_ids.index(labels[col])]
                lifted = {index[p, j]: c for j, c in vector.items()}
                expected = matrix([lifted], len(original_labels), field)[:, 0]
                assert np.all(expected == inclusion[:, col])
            for p, n in dims.items():
                b, reps = reduced["boundaries"][p], reduced["homology"][p]
                assert len(reps) == n - reduced["ranks"][p] - reduced["ranks"].get(p+1, 0)
                assert core._rank(matrix(b + reps, n, field)) == len(b) + len(reps)
            count += 1
    # A nonunit over Z is not a valid integral cancellation; its field image matters.
    assert reduce_complex({0: 1, 1: 1}, {1: [{0: 2}]})["stats"]["pairs"] == 1
    assert reduce_complex({0: 1, 1: 1}, {1: [{0: 2}]}, modulus=2)["stats"]["pairs"] == 0
    try:
        reduce_complex({0: 1, 1: 1, 2: 1}, {1: [{0: 1}], 2: [{0: 1}]}, check=True)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid chain complex accepted")
    print(f"contraction identities: {count} exact complexes; characteristic test passed", flush=True)


def test_graphs():
    count = 0
    possible = [(u, v) for u in range(3) for v in range(3) if u != v]
    for mask in range(64):
        edges = [e for i, e in enumerate(possible) if mask & (1 << i)]
        for field in ("Q", 2, 3, 5):
            for regular in (False, True):
                check_path(list(range(3)), edges, field, regular)
                count += 1
    rng = random.Random(90123)
    for case in range(60):
        n = rng.randrange(2, 7)
        loops = case % 5 == 0
        edges = [(u, v) for u in range(n) for v in range(n)
                 if (loops or u != v) and rng.random() < .27]
        field = ("Q", 2, 3, 5, 101)[case % 5]
        check_path(list(range(n)), edges, field, bool(case % 2), loops, degree=4)
        count += 1
    for vertices, edges in (([], []), ([0], []), (list(range(5)), [])):
        check_path(vertices, edges, "Q", False)
        count += 1
    for method in ("auto", "kernel", "lemma41", "prop42"):
        options = dict(edges=[(0,1),(1,2),(0,2)], max_dim=3, method=method)
        old = core.path_homology(**options)
        new = core.path_homology(**options, homology_method="morse")
        assert old.betti == new.betti
        assert all(np.array_equal(m, new.boundary_matrices[p])
                   for p, m in old.boundary_matrices.items())
    for options in (dict(field="R"), dict(generators=False), dict(store_paths=False)):
        try:
            core.path_homology(edges=[(0,1)], homology_method="morse", **options)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported mode accepted")
    print(f"path complexes: {count} cross-checks; original-path representatives verified", flush=True)


if __name__ == "__main__":
    start = time.perf_counter()
    test_contractions()
    test_graphs()
    print(f"All Morse tests passed ({time.perf_counter()-start:.3f}s)")
