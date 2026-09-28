# -*- coding: utf-8 -*-
"""
Verified digraph families: tournaments and other classical families.
Writes Markdown tables (English) ready for the public repository.
"""
import importlib.util
import itertools
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
OUT = []


def emit(s=""):
    print(s, flush=True)
    OUT.append(s)


def hom(N, E, D=4):
    v = list(range(N))
    A = kern.allowed_paths(v, E, D)
    sizes = [len(A.get(p, [])) for p in range(D + 1)]
    rq = kern.path_homology(vertices=v, edges=E, max_dim=D, field="Q",
                            generators=False)
    rf = kern.path_homology(vertices=v, edges=E, max_dim=D, field="Z2",
                            generators=False)
    bq = [rq.betti.get(p, 0) for p in range(D)]
    bf = [rf.betti.get(p, 0) for p in range(D)]
    dom = [rq.dim_omega.get(p, 0) for p in range(D + 1)]
    return sizes, dom, bq, bf


def line(label, N, E, D=4):
    sizes, dom, bq, bf = hom(N, E, D)
    same = "=" if bq == bf else "DIFFER"
    emit(f"| {label} | {N} | {len(E)} | {dom} | {bq} | {bf} | {same} |")
    return dom, bq, bf


# =============================================================== tournaments
emit("# Tournaments and other classical digraph families")
emit()
emit("All values computed with the gen3 core, `max_dim = 4`, over Q and over")
emit("F2. `dim Omega` is listed for p = 0,1,2,3,4; `betti` for p = 0,1,2,3.")
emit("A betti entry is only read where `max_dim >= p+1`.")
emit()
emit("## 1. Tournaments")
emit()
emit("A tournament orients every pair of distinct vertices by exactly one arc.")
emit()
emit("### 1.1 Transitive tournament TT_n")
emit()
emit("Vertices 0..n-1, arc i->j for every i<j. This is the ordered simplex")
emit("on n vertices, so `dim Omega_p = C(n, p+1)` and the homology is that of a")
emit("point: `(1, 0, 0, ...)`.")
emit()
emit("| digraph | N | |E| | dim Omega | betti_Q | betti_F2 | Q vs F2 |")
emit("|---|---|---|---|---|---|---|")
for n in range(2, 8):
    E = [(i, j) for i in range(n) for j in range(n) if i < j]
    line(f"TT_{n}", n, E)
emit()
emit("### 1.2 Cyclic (regular) tournament RT_n, n odd")
emit()
emit("Vertices Z_n, arc i->j iff `j-i mod n` is in `{1,...,(n-1)/2}`. Every")
emit("vertex has out-degree `(n-1)/2`, so the digraph is regular.")
emit()
emit("| digraph | N | |E| | dim Omega | betti_Q | betti_F2 | Q vs F2 |")
emit("|---|---|---|---|---|---|---|")
for n in (3, 5, 7, 9):
    h = (n - 1) // 2
    E = [(i, (i + d) % n) for i in range(n) for d in range(1, h + 1)]
    line(f"RT_{n}", n, E)
emit()
emit("### 1.3 All tournaments up to isomorphism, n <= 5")
emit()
emit("Complete list of isomorphism classes (1, 1, 4, 12 classes for")
emit("n = 2, 3, 4, 5). Canonical form by brute force over all relabellings.")
emit()


def canon(n, adj):
    best = None
    for p in itertools.permutations(range(n)):
        bits = 0
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[p[i]][p[j]]:
                    bits |= 1 << k
                k += 1
        if best is None or bits < best[0]:
            best = (bits, p)
    return best


def tourn_edges(n, bits):
    E, k = [], 0
    for i in range(n):
        for j in range(i + 1, n):
            if (bits >> k) & 1:
                E.append((i, j))
            else:
                E.append((j, i))
            k += 1
    return E


emit("| digraph | N | |E| | dim Omega | betti_Q | betti_F2 | Q vs F2 |")
emit("|---|---|---|---|---|---|---|")
for n in (2, 3, 4, 5):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    seen = {}
    for mask in range(1 << len(pairs)):
        adj = [[False] * n for _ in range(n)]
        for k, (i, j) in enumerate(pairs):
            if (mask >> k) & 1:
                adj[i][j] = True
            else:
                adj[j][i] = True
        cb, cp = canon(n, adj)
        if cb not in seen:
            seen[cb] = mask
    emit(f"*n = {n}: {len(seen)} isomorphism class(es)*")
    emit("")
    for i, (cb, mask) in enumerate(sorted(seen.items())):
        E = tourn_edges(n, mask)
        scores = sorted(sum(1 for (a, b) in E if a == x) for x in range(n))
        line(f"T_{n}#{i + 1} score-seq {scores}", n, E)
emit()

# =============================================================== classics
emit("## 2. Classical families")
emit()
emit("| digraph | N | |E| | dim Omega | betti_Q | betti_F2 | Q vs F2 |")
emit("|---|---|---|---|---|---|---|")
for n in range(2, 11):
    E = [(i, (i + 1) % n) for i in range(n)]
    line(f"directed cycle C_{n}", n, E)
emit()
for n in range(3, 11):
    E = sorted({(i, (i + d) % n) for i in range(n) for d in (1, -1)})
    line(f"bidirectional cycle C_{n} (+-1)", n, E)
emit()
for n in range(2, 7):
    E = [(i, j) for i in range(n) for j in range(n) if i != j]
    line(f"complete digraph K_{n}", n, E)
emit()
for n in range(1, 6):
    E = [(x, y) for x in range(n) for y in range(n, 2 * n)]
    E += [(y, x) for x in range(n) for y in range(n, 2 * n)]
    line(f"complete bipartite K_{{{n},{n}}} both ways", 2 * n, E)
emit()
emit("Hypercube-like: the n-cube with all edges doubled is the Cayley digraph")
emit("of (Z_2)^n with the standard basis, covered in the Cayley tables below.")
emit()
emit("de Bruijn digraph B(d, n): vertices are words of length n over d letters,")
emit("arc `a1..an -> a2..an b`. It is d-regular with d^n vertices.")
emit()
emit("| digraph | N | |E| | dim Omega | betti_Q | betti_F2 | Q vs F2 |")
emit("|---|---|---|---|---|---|---|")
for (d, n) in [(2, 2), (2, 3), (3, 2)]:
    words = list(itertools.product(range(d), repeat=n))
    idx = {w: i for i, w in enumerate(words)}
    E = sorted({(idx[w], idx[w[1:] + (b,)]) for w in words for b in range(d)})
    line(f"de Bruijn B({d},{n})", len(words), E)
emit()
emit("Directed complete bipartite/tournament hybrids and the directed grid")
emit("P_m x P_n (the D_{m,n} family) are documented in the accompanying")
emit("group and lattice notes.")
emit()

path = os.path.join(HERE, "families_scan.md")
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
print(f"\n[written] {path}")
