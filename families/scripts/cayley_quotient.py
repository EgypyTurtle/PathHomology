# -*- coding: utf-8 -*-
"""
How quotients show up in path homology: genuine Cayley digraphs of G/N built
on the cosets of N.  The quotient generating set is the image of S, so it can
be smaller than S (and repetitions are dropped).
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kern = load("kern", os.path.join(HERE, "core_v4.py"))


def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def closure(n, gens):
    idp = tuple(range(n))
    seen = {idp}
    frontier = [idp]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = compose(x, g)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    return sorted(seen)


def cycle_perm(n, k):
    p = list(range(n))
    for i in range(k):
        p[i] = (i + 1) % k
    return tuple(p)


def transposition(n, i, j):
    p = list(range(n))
    p[i], p[j] = j, i
    return tuple(p)


def sngens(n):
    return [transposition(n, 0, 1), cycle_perm(n, n)]


def angens(n):
    if n % 2 == 1:
        return [cycle_perm(n, 3), cycle_perm(n, n)]
    p = list(range(n))
    for i in range(1, n):
        p[i] = 1 + (i % (n - 1))
    return [cycle_perm(n, 3), tuple(p)]


def dihedral_gens(m):
    r = list(range(2 * m))
    for i in range(m):
        r[i] = (i + 1) % m
    for i in range(m):
        r[m + i] = m + ((i + 1) % m)
    s = list(range(2 * m))
    for i in range(m):
        s[i] = m + i
        s[m + i] = i
    return tuple(r), tuple(s)


def hom_of(N, edges, D=3):
    v = list(range(N))
    r = kern.path_homology(vertices=v, edges=edges, max_dim=D, field="Q",
                           generators=False)
    r2 = kern.path_homology(vertices=v, edges=edges, max_dim=D, field="Z2",
                            generators=False)
    bq = [r.betti.get(p, 0) for p in range(D)]
    b2 = [r2.betti.get(p, 0) for p in range(D)]
    dom = [r.dim_omega.get(p, 0) for p in range(D + 1)]
    return dom, bq, b2, sorted({a for a, b in edges}), \
        len({b for a, b in edges})


def cayley_of(elems_order, gens):
    """elems_order: group elements (perm tuples) or plain indices."""
    idx = {e: i for i, e in enumerate(elems_order)}
    E = set()
    for x in elems_order:
        for g in gens:
            y = compose(x, g) if isinstance(x, tuple) else g[x]
            if y in idx:
                E.add((idx[x], idx[y]))
    return sorted(E)


def as_perms(n, elems):
    """Express group elements as permutations of the index set."""
    idx = {e: i for i, e in enumerate(elems)}
    return [tuple(idx[compose(x, g)] for x in elems) for g in elems]


def quotient(label, n, gens, subgens, D=3):
    G = closure(n, gens)
    N = closure(n, subgens)
    Nset = set(N)
    reps, seen = [], set()
    for x in G:
        if x in seen:
            continue
        c = frozenset(compose(h, x) for h in N)
        seen |= set(c)
        reps.append(x)
    cos = [frozenset(compose(h, x) for h in N) for x in reps]
    cidx = {}
    for i, c in enumerate(cos):
        for x in c:
            cidx[x] = i
    qgens = []
    for g in gens:
        qgens.append(tuple(cidx[compose(x, g)] for x in reps))
    # drop repeated images
    uniq = []
    for p in qgens:
        if p not in uniq:
            uniq.append(p)
    k = len(cos)
    print(f"  {label}")
    print(f"    |G| = {len(G)},  |N| = {len(N)},  |G/N| = {k},  "
          f"image generators = {len(uniq)} (from {len(gens)})", flush=True)
    if k > 3000:
        print("    skipped (too large)")
        return
    E = cayley_of(list(range(k)), uniq)
    dom, bq, b2, _, _ = hom_of(k, E, D)
    print(f"    dim Omega = {dom}, betti_Q = {bq}, betti_F2 = {b2}",
          flush=True)


print("=" * 104)
print("QUOTIENT Cayley digraphs (genuine, built on cosets)")
print("=" * 104)
r4, s4 = dihedral_gens(4)
r6, s6 = dihedral_gens(6)
print("[source groups, for comparison]")
for lab, n, g in (("D_4 = <r,s>", 8, [r4, s4]), ("D_6 = <r,s>", 12, [r6, s6]),
                  ("Z_6 = <r>", 6, [cycle_perm(6, 6)]),
                  ("S_4", 4, sngens(4)), ("A_4", 4, angens(4))):
    G = closure(n, g)
    E = cayley_of(G, g)
    dom, bq, b2, _, _ = hom_of(len(G), E, 3)
    print(f"  {lab:<14} |G|={len(G):>3}  dim Omega = {dom}, "
          f"betti_Q = {bq}, betti_F2 = {b2}", flush=True)
print()
print("[quotients]")
v1 = compose(transposition(4, 0, 1), transposition(4, 2, 3))
v2 = compose(transposition(4, 0, 2), transposition(4, 1, 3))
r42 = compose(r4, r4)
r63 = compose(compose(r6, r6), r6)
r62 = compose(r6, r6)
quotient("D_4 / <r^2>   (N = Z_2, quotient Z_2 x Z_2, order 4)", 8,
         [r4, s4], [r42])
quotient("D_6 / <r^3>   (N = centre Z_2, quotient D_3, order 6)", 12,
         [r6, s6], [r63])
quotient("D_6 / <r^2>   (N = Z_3, quotient Z_2 x Z_2, order 4)", 12,
         [r6, s6], [r62])
quotient("Z_6 / <r^2>   (N = Z_3, quotient Z_2, order 2)", 6,
         [cycle_perm(6, 6)], [compose(cycle_perm(6, 6), cycle_perm(6, 6))])
quotient("Z_6 / <r^3>   (N = Z_2, quotient Z_3, order 3)", 6,
         [cycle_perm(6, 6)], [r63])
quotient("S_4 / V_4     (quotient S_3, order 6)", 4, sngens(4), [v1, v2])
quotient("S_4 / A_4     (quotient Z_2, order 2)", 4, sngens(4), angens(4))
quotient("A_4 / V_4     (quotient Z_3, order 3)", 4, angens(4), [v1, v2])
print()
print("=" * 104)
print("SUBGROUP contrast")
print("=" * 104)
for lab, n, g in (("C_4 = <r> <= D_4, gens {r}", 8, [r4]),
                  ("D_4 itself, gens {r,s}", 8, [r4, s4]),
                  ("C_6 = <r> <= D_6, gens {r}", 12, [r6]),
                  ("D_6 itself, gens {r,s}", 12, [r6, s6]),
                  ("A_4 <= S_4", 4, angens(4)),
                  ("S_4 itself", 4, sngens(4))):
    G = closure(n, g)
    E = cayley_of(G, g)
    dom, bq, b2, _, _ = hom_of(len(G), E, 3)
    print(f"  {lab:<28} |G|={len(G):>3}  dim Omega = {dom}, "
          f"betti_Q = {bq}, betti_F2 = {b2}", flush=True)
print()
print("=" * 104)
print("PRODUCT contrast: Cartesian products of Cayley digraphs")
print("=" * 104)
import itertools  # noqa: E402
for (m, nn) in [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4),
                (4, 5), (5, 5), (2, 5), (3, 6)]:
    k = m * nn
    g1 = tuple([(i + 1) % m + m * (i // m) for i in range(k)])
    g2 = tuple([(i % m) + m * ((i // m + 1) % nn) for i in range(k)])
    E = sorted({(x, g1[x]) for x in range(k)} | {(x, g2[x]) for x in range(k)})
    dom, bq, b2, _, _ = hom_of(k, E, 3)
    tag = "both factors >= 3" if min(m, nn) >= 3 else "some factor = 2"
    print(f"  C_{m} x C_{nn}  |G|={k:>3}  dim Omega = {dom}, "
          f"betti_Q = {bq}, F2 = {b2}   [{tag}]", flush=True)
