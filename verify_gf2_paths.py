# -*- coding: utf-8 -*-
"""Cross-check the GF(2) fast paths against the generic object-dtype path."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (
    GF, _gf2_nullspace, _gf2_pivot_columns, _gf2_rank, _nullspace,
    _pivot_columns, _rank,
)

rng = np.random.default_rng(0)
print("Cross-checking GF(2) fast paths vs generic object path:")
ok = True
for shape in [(20, 20), (40, 25), (30, 60), (50, 50), (5, 80)]:
    ints = rng.integers(0, 2, size=shape)
    obj = np.array([[GF(int(x), 2) for x in row] for row in ints], dtype=object)

    r1, r2 = _rank(obj), _gf2_rank(ints)
    ns1, ns2 = _nullspace(obj), _gf2_nullspace(ints)
    p1, p2 = _pivot_columns(obj), _gf2_pivot_columns(ints)

    good = (r1 == r2) and (ns1.shape[1] == ns2.shape[1]) and (p1 == p2)
    ok &= good
    print(f"  {str(shape):>9}: rank {r1}=={r2}  nullity {ns1.shape[1]}=={ns2.shape[1]}"
          f"  pivots equal={p1 == p2}  {'OK' if good else 'MISMATCH'}")

print()
print("ALL GF(2) FAST PATHS AGREE:", ok)
