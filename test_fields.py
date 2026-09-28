# -*- coding: utf-8 -*-
"""Check that Q and Z2 agree with the paper on all the worked examples."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import path_homology, resolve_field, GF

print("field resolution:")
for f in ["Q", "Z2", "GF2", 2, 3, "R", None]:
    print("  ", repr(f), "->", resolve_field(f))

# known paper cases: name, edges, expected (dimOmega, betti) over Q
CASES = [
    ("Fig8", [(0,1),(0,2),(1,3),(1,4),(2,3),(2,4),(5,3),(5,4)], 6,
     {0:6,1:8,2:2}, {0:1,1:1,2:0}),
    ("triangle", [(0,1),(1,2),(0,2)], 4, {0:3,1:3,2:1}, {0:1,1:0,2:0}),
    ("square", [(0,1),(1,2),(0,3),(3,2)], 4, {0:4,1:4,2:1}, {0:1,1:0,2:0}),
    ("C3 directed", [(0,1),(1,2),(2,0)], 4, {0:3,1:3}, {0:1,1:1}),
    ("C4 directed", [(0,1),(1,2),(2,3),(3,0)], 4, {0:4,1:4}, {0:1,1:1}),
    ("C5 directed", [(i,(i+1)%5) for i in range(5)], 4, {0:5,1:5}, {0:1,1:1}),
    ("Ex3.14", [(0,1),(1,0)], 4, {0:2,1:2}, {0:1,1:1}),
    ("octahedron", [(0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),
                    (2,4),(2,5),(3,4),(3,5)], 6, {0:6,1:12,2:8}, {0:1,1:0,2:1}),
]

bad = 0
print("\n%-12s %-22s %-22s" % ("case", "Q (dimOmega, betti)", "Z2 (dimOmega, betti)"))
for name, E, md, wantO, wantB in CASES:
    rQ = path_homology(edges=E, max_dim=md, field="Q")
    r2 = path_homology(edges=E, max_dim=md, field="Z2")
    oQ = {k:v for k,v in sorted(rQ.dim_omega.items()) if v}
    o2 = {k:v for k,v in sorted(r2.dim_omega.items()) if v}
    # compare only the nonzero entries: `r.betti` records a 0 for every
    # dimension, including the padded ones, so filter both sides alike
    bQ = {k:v for k,v in sorted(rQ.betti.items()) if v}
    b2 = {k:v for k,v in sorted(r2.betti.items()) if v}
    wantO = {k:v for k,v in wantO.items() if v}
    wantB = {k:v for k,v in wantB.items() if v}
    okO = oQ == wantO and o2 == wantO
    okB = bQ == wantB and b2 == wantB
    status = "OK" if (okO and okB) else "MISMATCH"
    if not (okO and okB):
        bad += 1
        print(f"  {name}: Q dimO={oQ} want={wantO}")
        print(f"  {' '*len(name)}  Q betti={bQ} want={wantB}")
        print(f"  {' '*len(name)}  Z2 dimO={o2} betti={b2}")
    print(f"{name:<12} {str(bQ):<22} {str(b2):<22} {status}")

print("\n" + ("ALL AGREE" if bad == 0 else f"{bad} MISMATCHES"))
