"""Small reproducible examples; run from this repository's root."""
from core import path_homology


def main():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for field in ("Q", 2, 3):
        options = dict(edges=edges, max_dim=2, loops=False, field=field)
        dims = path_homology(**options, generators=False, store_paths=False)
        morse = path_homology(**options, homology_method="morse", store_matrices=False)
        assert dims.betti[0] == morse.betti[0] == 1
        assert dims.betti[1] == morse.betti[1] == 1
        print(f"cycle, {field}: H0={morse.betti[0]}, H1={morse.betti[1]}")
    triangle = path_homology(edges=[(0,1), (1,2), (0,2)], max_dim=2,
                             homology_method="morse", store_matrices=False)
    assert triangle.betti[1] == 0
    print("transitive triangle: H1=0")


if __name__ == "__main__":
    main()
