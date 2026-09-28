# -*- coding: utf-8 -*-
"""Cheap certification: distinct generating set + out-degree + Omega_2 test."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


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


def unique_two_factorisation(gens, outdeg):
    """True iff every element of S*S outside S has a unique st-decomposition."""
    S = set(gens)
    cnt = {}
    for s in gens:
        for t in gens:
            g = compose(s, t)
            cnt[g] = cnt.get(g, 0) + 1
    e = tuple(range(len(gens[0])))
    bad = {k: v for k, v in cnt.items() if v > 1 and k not in S and k != e}
    return (not bad), bad


print("=" * 96)
print("Certification by criterion + cycle-rank formula (distinct generators)")
print("=" * 96)
print(f"{'case':<8} {'|G|':>12} {'|S|':>4} {'Omega2=0':>9} {'pred b_1':>13} "
      f"{'computed':>10} {'ok':>5}")
KNOWN = {("S", 3): 7, ("S", 4): 25, ("S", 5): 121, ("S", 6): 721,
         ("S", 7): 5041, ("S", 8): 40321,
         ("A", 3): 1, ("A", 4): 13, ("A", 5): 61, ("A", 6): 361,
         ("A", 7): 2521, ("A", 8): 20161}
for tag, fn in (("S", sngens), ("A", angens)):
    for n in range(3, 12):
        g = fn(n)
        G = closure(n, g)
        ordv = len(G)
        if ordv > 3_000_000:
            print(f"{tag}_{n:<5} {'-':>12} (closure of order "
                  f"{ordv if ordv < 10**9 else '>1e9'} not enumerated)")
            break
        idx = {e: i for i, e in enumerate(G)}
        perms = {tuple(idx[compose(x, s)] for x in G) for s in g}
        outdeg = len(perms)
        ok_u, bad = unique_two_factorisation(g, outdeg)
        pred = ordv * (outdeg - 1) + 1
        comp = KNOWN.get((tag, n))
        mark = "-" if comp is None else ("OK" if comp == pred else "FAIL")
        print(f"{tag}_{n:<5} {ordv:>12,} {outdeg:>4} "
              f"{('yes' if ok_u else 'NO'):>9} {pred:>13,} "
              f"{('-' if comp is None else format(comp, ',')):>10} "
              f"{mark:>5}", flush=True)
print()
print("  The criterion costs O(|G|) enumeration (it needs the group itself),")
print("  but no linear algebra: it certifies b_1 for groups whose Cayley")
print("  digraph is far too large to build a boundary matrix for.")
