# -*- coding: utf-8 -*-
"""
path_homology -- path homology of digraphs (GLMY theory).

Reference
---------
A. Grigor'yan, Y. Lin, Yu. Muranov, S.-T. Yau,
"Homologies of path complexes and digraphs", arXiv:1207.2834v4.

Quick start
-----------
>>> from path_homology import path_homology
>>> res = path_homology(edges=[(0, 1), (1, 2), (2, 3), (3, 0)])
>>> print(res.betti)          # {0: 1, 1: 1, 2: 0, ...}
>>> print(res.summary())      # full report: A_p, boundary matrices, Omega_p, H_p

The computation follows the paper step by step:

1. ``allowed_paths``  -- the path complex ``P(G)`` of the digraph (Example 3.3):
   ``A_p`` is spanned by all sequences ``i0 -> i1 -> ... -> ip``.
2. ``boundary_matrix`` -- the operator ``partial`` of eq. (2.2) as a matrix.
3. ``omega_basis``    -- ``Omega_p = {v in A_p : partial v in A_{p-1}}`` (3.8),
   with three interchangeable algorithms (direct kernel, Lemma 4.1 via
   semi-edges, Proposition 4.2 for p = 2).
4. ``path_homology``  -- assembles the chain complex ``Omega_*`` (3.9)/(3.10)
   and returns its closed chains ``Z_p = ker partial``, exact chains
   ``B_p = partial Omega_{p+1}`` and homology ``H_p = Z_p / B_p`` (3.11).
"""

from core import (
    DEFAULT_MAX_DIM,
    PathHomologyResult,
    allowed_paths,
    boundary_matrix,
    boundary_of_path,
    bridges,
    is_regular_path,
    is_semi_allowed,
    omega_basis,
    path_homology,
    semi_edges,
)

__version__ = "1.0.0"

__all__ = [
    "path_homology",
    "allowed_paths",
    "boundary_matrix",
    "boundary_of_path",
    "omega_basis",
    "semi_edges",
    "bridges",
    "is_semi_allowed",
    "is_regular_path",
    "PathHomologyResult",
    "DEFAULT_MAX_DIM",
    "__version__",
]
