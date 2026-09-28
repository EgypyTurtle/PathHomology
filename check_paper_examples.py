# -*- coding: utf-8 -*-
"""
validate_paper_examples.py -- check the implementation against the worked
examples of arXiv:1207.2834v4.

Run:  python validate_paper_examples.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import path_homology, allowed_paths, semi_edges, omega_basis
fails = []
def check(name, got, want):
    ok = got == want
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got={got} want={want}")
    if not ok: fails.append(name)

print("=== Example 3.9 / Sec 4.6-style: 0->1->2 (single 2-path, non-allowed boundary) ===")
r = path_homology(edges=[(0,1),(1,2)])
print(" A_2 =", r.allowed[2], " dim Omega_2 =", r.dim_omega[2])
check("Ex3.9 dimOmega2=0", r.dim_omega[2], 0)

print("\n=== Sec 4.6, Figure 8: V={0,1,2,3,5}, E={01,02,13,14,23,24,53,54} ===")
E = [(0,1),(0,2),(1,3),(1,4),(2,3),(2,4),(5,3),(5,4)]
r = path_homology(edges=E)
print(" semi-edges:", semi_edges(E))
check("Fig8 dimOmega0", r.dim_omega[0], 6)
check("Fig8 dimOmega1", r.dim_omega[1], 8)
check("Fig8 dimA2", len(r.allowed[2]), 4)
check("Fig8 dimOmega2", r.dim_omega[2], 2)
check("Fig8 dimH2", r.betti[2], 0)
check("Fig8 dimH1", r.betti[1], 1)
check("Fig8 chi", r.euler_characteristic, 0)
print(" H_1 generator:", r._fmt(r.homology_basis[1][0]) if r.homology_basis[1] else None)
print(" Omega_2 basis:", [r._fmt(v) for v in r.omega_paths[2]])

print("\n=== Triangle a->b->c, a->c : dimH_2=0, dimH_1=0 ===")
r = path_homology(edges=[(0,1),(1,2),(0,2)])
check("triangle dimA2", len(r.allowed[2]), 1)
check("triangle dimOmega2", r.dim_omega[2], 1)
check("triangle dimH2", r.betti[2], 0)
check("triangle dimH1", r.betti[1], 0)
check("triangle chi", r.euler_characteristic, 1)

print("\n=== Square a->b->c, a->b'->c : Omega_2 = span{e_abc - e_ab'c} ===")
E = [(0,1),(1,2),(0,3),(3,2)]
r = path_homology(edges=E)
print(" semi-edges:", semi_edges(E))
check("square dimA2", len(r.allowed[2]), 2)
check("square dimOmega2", r.dim_omega[2], 1)
print(" Omega_2 basis:", [r._fmt(v) for v in r.omega_paths[2]])
check("square dimH2", r.betti[2], 0)
check("square dimH1", r.betti[1], 0)
check("square chi", r.euler_characteristic, 1)

print("\n=== Example 3.14: 0<->1 (E={01,10}) -- regular vs non-regular versions ===")
print("    paper, NON-regular: Omega_0=A_0, Omega_1=A_1, Omega_n=0 for n>=2;")
print("                        dim H_0=1, dim H_1=1, chi=0")
print("    paper, REGULAR    : Omega^reg_n = A_n for all n; dim H_0=1, dim H_n=0 for n>=1, chi^reg=1")
r = path_homology(edges=[(0,1),(1,0)], max_dim=4)
print("  allowed 1-paths:", r.allowed[1], " allowed 2-paths:", r.allowed[2])
check("Ex3.14 nonreg dimOmega2", r.dim_omega[2], 0)
check("Ex3.14 nonreg dimH0", r.betti[0], 1)
check("Ex3.14 nonreg dimH1", r.betti[1], 1)
# H_0 is 1-dimensional and spanned by any single vertex class; the paper writes
# the equivalent representative e_0 + e_1.  We only check the dimension and
# that our representative is nonzero.
check("Ex3.14 nonreg H0 nonzero", bool(r.homology_basis[0]) and r._fmt(r.homology_basis[0][0]) != "0", True)
print("       H_0 representative:", r._fmt(r.homology_basis[0][0]), "  (paper writes e0 +e1)")
check("Ex3.14 nonreg H1 = span{e01+e10}", r._fmt(r.homology_basis[1][0]), "e01 +e10")
check("Ex3.14 nonreg chi", r.euler_characteristic, 0)

r_reg = path_homology(edges=[(0,1),(1,0)], max_dim=4, regular=True)
print("  regular: dimOmega2 =", r_reg.dim_omega[2])
check("Ex3.14 reg dimH0", r_reg.betti[0], 1)
check("Ex3.14 reg dimH1", r_reg.betti[1], 0)
check("Ex3.14 reg dimH2", r_reg.betti[2], 0)

print("\n=== Proposition 4.7: cycle-graphs (undirected cycle, arbitrary orientations) ===")
print("    paper: dim H_0 = 1, dim Omega_0 = dim Omega_1 = |V|, dim Omega_p = 0 for p >= 3,")
print("           dim H_p = 0 for p >= 2; and")
print("             * a 'triangle' a->b,a->c,b->c or 'square'  -> dim Omega_2 = 1, dim H_1 = 0, chi = 1")
print("             * otherwise                               -> dim Omega_2 = 0, dim H_1 = 1, chi = 0")
print("    NB the paper's triangle/square are the TRANSITIVE ones, not the directed")
print("       cycles 0->1->2->0 / 0->1->2->3->0, which fall in the 'otherwise' case.")
cases = [
    # (name, edges, is_triangle_or_square)
    ("C3 directed 0->1->2->0", [(0,1),(1,2),(2,0)], False),
    ("C4 directed", [(0,1),(1,2),(2,3),(3,0)], False),
    ("C5 directed", [(i,(i+1)%5) for i in range(5)], False),
    ("C6 directed", [(i,(i+1)%6) for i in range(6)], False),
    ("C7 directed", [(i,(i+1)%7) for i in range(7)], False),
    ("C4 mixed orient", [(0,1),(2,1),(2,3),(0,3)], False),
    ("triangle transitive", [(0,1),(1,2),(0,2)], True),
    ("square transitive", [(0,1),(1,2),(0,3),(3,2)], True),
]
for name, E, tri_or_sq in cases:
    r = path_homology(edges=E)
    h = {p: r.betti[p] for p in sorted(r.betti)}
    print(f"  {name:22s}: dimOmega={ {p:r.dim_omega[p] for p in sorted(r.dim_omega) if r.dim_omega[p]} } betti={ {k:v for k,v in h.items() if v} } chi={r.euler_characteristic}")
    check(f"{name} H0=1", h.get(0), 1)
    check(f"{name} H1", h.get(1), 0 if tri_or_sq else 1)
    check(f"{name} dimOmega2", r.dim_omega[2], 1 if tri_or_sq else 0)
    check(f"{name} dimOmega_p=0 for p>=3",
          all(r.dim_omega[p]==0 for p in r.dim_omega if p>=3), True)
    check(f"{name} H_p=0 for p>=2", all(h.get(p,0)==0 for p in h if p>=2), True)
    check(f"{name} chi", r.euler_characteristic, 1 if tri_or_sq else 0)
    if not tri_or_sq and r.homology_basis.get(1):
        print(f"       H_1 generator: {r._fmt(r.homology_basis[1][0])}")

print("\n=== Octahedron (Fig.24): dimOmega0=6, dimOmega1=12, dimOmega2=8, H=(1,0,1), chi=2 ===")
# The paper states |E| = 12, empty semi-edge set and A_2 = {024,025,034,035,124,125,134,135}.
E = [(0,2),(0,3),(0,4),(0,5),
     (1,2),(1,3),(1,4),(1,5),
     (2,4),(2,5),(3,4),(3,5)]
print("  |E| =", len(E), " semi-edges:", semi_edges(E))
r = path_homology(edges=E, max_dim=6)
print("  betti:", {k:v for k,v in sorted(r.betti.items()) if v}, " dimOmega:", {k:v for k,v in sorted(r.dim_omega.items()) if v})
check("octa |V|", len(r.vertices), 6)
check("octa |E|", len(r.edges), 12)
check("octa semi-edges empty", semi_edges(E), [])
check("octa A_2 equals paper", sorted(r.allowed[2]),
      [(0,2,4),(0,2,5),(0,3,4),(0,3,5),(1,2,4),(1,2,5),(1,3,4),(1,3,5)])
check("octa dimOmega0", r.dim_omega[0], 6)
check("octa dimOmega1", r.dim_omega[1], 12)
check("octa dimOmega2", r.dim_omega[2], 8)
check("octa H0", r.betti[0], 1)
check("octa H1", r.betti[1], 0)
check("octa H2", r.betti[2], 1)
check("octa chi", r.euler_characteristic, 2)
if r.homology_basis.get(2):
    print("  H_2 generator:", r._fmt(r.homology_basis[2][0]))

print("\n=== method cross-check: kernel vs lemma41 vs prop42 ===")
for name, EE in [("Fig8", [(0,1),(0,2),(1,3),(1,4),(2,3),(2,4),(5,3),(5,4)]),
                 ("square", [(0,1),(1,2),(0,3),(3,2)]),
                 ("triangle", [(0,1),(1,2),(0,2)])]:
    paths = allowed_paths(None, EE, max_dim=6)
    dims = {}
    for m in ["kernel","lemma41","prop42"]:
        dims[m] = {p: len(omega_basis(paths, EE, p, method=m)[0]) for p in (2,3,4)}
    print(f"  {name}: {dims}")
    vals = list(dims.values())
    check(f"{name} methods agree", all(v==vals[0] for v in vals), True)

print("\n=== chain-complex sanity ===")
print("  NB partial^2 = 0 (Lemma 2.4) holds on Lambda_*, NOT on A_*: partial of an")
print("     allowed path may contain a NON-allowed face (e.g. partial e_013 = e_13 - e_03 + e_01")
print("     with e_03 non-allowed), which the restriction to A_* drops.  That is exactly")
print("     why Omega_* is introduced.  We therefore check partial^2 = 0 on Omega_*.")
for name, EE in [("Fig8", [(0,1),(0,2),(1,3),(1,4),(2,3),(2,4),(5,3),(5,4)]),
                 ("C5", [(i,(i+1)%5) for i in range(5)]),
                 ("C3", [(i,(i+1)%3) for i in range(3)]),
                 ("tri", [(0,1),(1,2),(0,2)]),
                 ("square", [(0,1),(1,2),(0,3),(3,2)])]:
    r = path_homology(edges=EE)
    ok = True
    for p in sorted(r.allowed):
        if p >= 2 and r.omega_matrix.get(p) is not None and r.omega_matrix[p].shape[1]:
            om = r.omega_matrix[p]
            d_p = r.boundary_matrices[p] @ om          # partial : Omega_p -> A_{p-1}
            # partial of a *partial-invariant* path is allowed, so the next
            # application is well defined on the same matrices
            prod = r.boundary_matrices[p - 1] @ d_p
            if prod.size and any(c != 0 for c in prod.ravel()):
                ok = False
    check(f"{name} partial^2=0 on Omega_*", ok, True)

print("\n" + ("ALL PASS" if not fails else f"FAILURES: {fails}"))
