# -*- coding: utf-8 -*-
"""
Extended Cayley runs, three parts.

(A) S_n / A_n pushed as far as a 60-second-per-case budget allows.
(B) The Omega_2 = 0 criterion.  For a Cayley digraph, the fibre of the
    constraint attached to a non-edge (x,z) is
        mid(x,z) = {y : x->y->z}  <->  {(s,t) in S x S : s*t = x^-1 z},
    so if every element of S*S \ S has a UNIQUE two-factor decomposition the
    equation is single-variable and forces Omega_2 = 0.  Then
        H_1 = ker d_1 = the graph cycle space,
        b_1 = |E| - |V| + b_0 = |G|*(|S|-1) + 1.
    The criterion costs O(|S|^2), so it certifies arbitrarily large groups.
(C) Genuine QUOTIENT Cayley digraphs, built on cosets.
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

BUDGET = 60.0          # seconds per case
CAP = 600_000          # memory-safe cap on |A_D| (see complexity notes)


# --------------------------------------------------------------- group utils
def compose(p, q):
    return tuple(p[q[i]] for i in range(len(p)))


def identity(n):
    return tuple(range(n))


def closure(n, gens):
    idp = identity(n)
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


def cayley(n, gens):
    elems = closure(n, gens)
    idx = {e: i for i, e in enumerate(elems)}
    E = sorted({(idx[x], idx[compose(x, g)]) for x in elems for g in gens})
    return list(range(len(elems))), E, elems


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


# --------------------------------------------------------------- Omega_2 test
def unique_two_factorisation(gens):
    """True iff every element of S*S \ S has a unique st-decomposition."""
    cnt = {}
    for s in gens:
        for t in gens:
            g = compose(s, t)
            cnt[g] = cnt.get(g, 0) + 1
    bad = {k: v for k, v in cnt.items() if v > 1 and k not in set(gens)
           and k != identity(len(gens[0]))}
    return (not bad), cnt, bad


# --------------------------------------------------------------- reporting
def run(label, n, gens, D):
    v, E, elems = cayley(n, gens)
    A = kern.allowed_paths(v, E, D)
    sizes = [len(A.get(p, [])) for p in range(D + 1)]
    if sizes[D] > CAP:
        return (f"{label:<40} |G|={len(v):>6} SKIPPED |A_{D}|={sizes[D]:,}",
                None, "skip")
    t = time.time()
    try:
        r = kern.path_homology(vertices=v, edges=E, max_dim=D, field="Q",
                               generators=False)
        r2 = kern.path_homology(vertices=v, edges=E, max_dim=D, field="Z2",
                                generators=False)
    except MemoryError:
        return (f"{label:<40} |G|={len(v):>6} MemoryError after "
                f"{time.time() - t:.1f}s (|A_{D}|={sizes[D]:,})", None, "mem")
    dt = time.time() - t
    bq = [r.betti.get(p, 0) for p in range(D)]
    b2 = [r2.betti.get(p, 0) for p in range(D)]
    dom = [r.dim_omega.get(p, 0) for p in range(D + 1)]
    txt = (f"{label:<40} |G|={len(v):>6} Bet={bq} F2={b2} "
           f"dimOm={dom} {dt:>7.2f}s")
    return txt, (bq, dom, dt, sizes), "ok"


print("=" * 118)
print("(A) S_n / A_n, max_dim = 2, 60 s per case")
print("=" * 118)
rowsA = []
for tag, fn in (("S", sngens), ("A", angens)):
    for n in range(3, 11):
        g = fn(n)
        ordv = len(closure(n, g))
        D = 2
        t = time.time()
        txt, dat, code = run(f"{tag}_{n} (order {ordv})", n, g, D)
        print(txt, flush=True)
        rowsA.append((tag, n, ordv, txt, dat))
        if code != "ok":
            print(f"    -> {tag}_{n}: {code}, stopping this family", flush=True)
            break
        if dat[2] > BUDGET:
            print(f"    -> {tag}_{n} took {dat[2]:.1f}s > budget, stopping "
                  f"this family", flush=True)
            break
print()

print("=" * 118)
print("(B) Omega_2 = 0 criterion (unique two-factorisation) and the closed form")
print("    b_1 = |G|*(|S|-1) + 1 ,  certified without any linear algebra")
print("=" * 118)
hdr = f"{'case':<40} {'|G|':>12} {'pred b_1':>12} {'computed':>10} {'ok':>4}"
print(hdr)
for tag, fn, nmax in (("S", sngens, 12), ("A", angens, 12)):
    for n in range(3, nmax + 1):
        g = fn(n)
        ordv = len(closure(n, g))
        ok_u, cnt, bad = unique_two_factorisation(g)
        pred = ordv * (len(g) - 1) + 1
        comp = None
        for (t2, n2, o2, txt, dat) in rowsA:
            if t2 == tag and n2 == n and dat is not None:
                comp = dat[0][1]
        mark = "-" if comp is None else ("OK" if comp == pred else "FAIL")
        print(f"{tag}_{n:<37} {ordv:>12,} {pred:>12,} "
              f"{('-' if comp is None else format(comp, ',')):>10} "
              f"{('crit=' + ('U' if ok_u else 'x')):>4} {mark}", flush=True)
print()

print("   where the criterion FAILS (S*S has a repeated product outside S):")
for tag, fn, rng in (("S", sngens, range(3, 9)), ("A", angens, range(3, 9))):
    for n in rng:
        g = fn(n)
        ok_u, cnt, bad = unique_two_factorisation(g)
        if not ok_u:
            print(f"      {tag}_{n}: {len(bad)} repeated product(s)", flush=True)
print()

print("=" * 118)
print("(C) QUOTIENT Cayley digraphs, built on cosets")
print("=" * 118)


def coset_quotient(n, gens, subgens):
    """G = <gens>, N = <subgens>; returns (perms of G/N, induced generators)."""
    G = closure(n, gens)
    N = set(closure(n, subgens))
    idx_of = {x: i for i, x in enumerate(G)}
    # right cosets N*x
    reps, seen = [], set()
    for x in G:
        if x in seen:
            continue
        c = frozenset(compose(h, x) for h in N)
        seen |= set(c)
        reps.append((x, c))
    cos = [c for (_, c) in reps]
    cidx = {}
    for i, c in enumerate(cos):
        for x in c:
            cidx[x] = i
    out_gens = []
    for g in gens:
        out_gens.append(tuple(cidx[compose(x, g)] for (x, _) in reps))
    return cos, out_gens


def show_quotient(label, n, gens, subgens, D=3):
    try:
        cos, qgens = coset_quotient(n, gens, subgens)
    except Exception as exc:                                  # noqa: BLE001
        print(f"{label:<40} FAILED {type(exc).__name__}: {exc}", flush=True)
        return
    k = len(cos)
    # degree of the quotient generating set
    txt, dat, code = run(f"{label} (order {k})", k, qgens, D)
    print(txt, flush=True)


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


r4, s4 = dihedral_gens(4)
r42 = compose(r4, r4)
print("  D_4 = <r,s>, order 8;  N = <r^2> = Z_2, order 2;  D_4/N = Z_2 x Z_2")
show_quotient("D_4 / <r^2>", 8, [r4, s4], [r42])
print("  D_4/<r^2> generators are the images of r and s, so the quotient")
print("  Cayley digraph has out-degree 2 (not 3).")
print()
r6, s6 = dihedral_gens(6)
print("  D_6 = <r,s>, order 12;  N = <r^3> (centre), order 2;  D_6/N = D_3")
show_quotient("D_6 / <r^3>", 12, [r6, s6], [compose(compose(r6, r6), r6)])
print()
r3, s3 = dihedral_gens(3)
r32 = compose(r3, r3)
print("  D_6 / <r^2>  (N = <r^2> = Z_3, order 3; quotient = Z_2 x Z_2)")
show_quotient("D_6 / <r^2>", 12, [r6, s6],
              [compose(r6, r6)])
print()
print("  Z_6 = <r>, N = <r^2> = Z_3, quotient Z_2")
show_quotient("Z_6 / <r^2>", 6, [cycle_perm(6, 6)],
              [compose(cycle_perm(6, 6), cycle_perm(6, 6))])
print()
print("  S_4, N = V_4 = <(0 1)(2 3), (0 2)(1 3)>, S_4/V_4 = S_3, order 6")
v1 = compose(transposition(4, 0, 1), transposition(4, 2, 3))
v2 = compose(transposition(4, 0, 2), transposition(4, 1, 3))
show_quotient("S_4 / V_4", 4, sngens(4), [v1, v2])
print()
print("  S_4, N = A_4, S_4/A_4 = Z_2, order 2")
show_quotient("S_4 / A_4", 4, sngens(4), angens(4))
print()
print("  A_4, N = V_4, A_4/V_4 = Z_3, order 3")
show_quotient("A_4 / V_4", 4, angens(4), [v1, v2])
print()
print("  reference: the groups themselves, same generating sets")
for lab, nn, gg in (("D_4, gens {r,s}", 8, [r4, s4]),
                    ("D_6, gens {r,s}", 12, [r6, s6]),
                    ("Z_6, gens {r}", 6, [cycle_perm(6, 6)]),
                    ("S_4, gens {(0 1),(0..3)}", 4, sngens(4)),
                    ("A_4, gens {(0 1 2),(1..3)}", 4, angens(4))):
    t, d, c = run(lab, nn, gg, 3)
    print("  " + t, flush=True)
