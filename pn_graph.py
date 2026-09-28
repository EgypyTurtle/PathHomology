# -*- coding: utf-8 -*-
"""
pn_graph.py -- the digraph P_n and its path homology over Q and Z/2.

The digraph
-----------
    vertices : 0, 1, ..., 2n-1, a, b
    edges    :  i -> i+1          for i = 0, ..., 2n-2
                 2n-1 -> 0                       (the cycle closes)
                 a  <-> 2k        for k = 0, ..., n-1  (a joined both ways to evens)
                 b  <-> 2k+1      for k = 0, ..., n-1  (b joined both ways to odds)

So the even cycle vertices 0,2,4,... form one independent set joined to ``a``,
the odd ones 1,3,5,... form another joined to ``b``, and the 2n cycle vertices
carry the directed cycle 0->1->...->2n-1->0.

Vertex labels are integers: ``a`` is ``-2`` and ``b`` is ``-1``, because the
enumeration routines sort the vertex set and mixed str/int labels are not
mutually comparable.  :func:`name_of` renders them back as ``a`` / ``b``.

Usage
-----
    python pn_graph.py                 # P_1 .. P_6, both fields
    python pn_graph.py 4 --max-dim 2   # just P_4
    python pn_graph.py --table         # the summary table
    python pn_graph.py 4 --plot p4.png # draw the digraph
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import boundary_of_path, path_homology, semi_edges

#: integer labels for the two special vertices (a < b so sorting is stable)
A_LABEL = -2
B_LABEL = -1

_LABEL_NAMES = {A_LABEL: "a", B_LABEL: "b"}


def name_of(v) -> str:
    """Render a vertex label: -2 -> 'a', -1 -> 'b', others as themselves."""
    return _LABEL_NAMES.get(v, str(v))


def build_P(n: int):
    """
    Build ``P_n``.  Returns ``(vertices, edges)``.

    Raises ``ValueError`` for ``n < 1``.
    """
    if n < 1:
        raise ValueError("P_n requires n >= 1")
    even = [2 * k for k in range(n)]
    odd = [2 * k + 1 for k in range(n)]
    core = list(range(2 * n))

    edges = [(i, i + 1) for i in range(2 * n - 1)]
    edges.append((2 * n - 1, 0))                 # close the directed cycle

    for v in even:                                # a <-> even vertices
        edges.append((A_LABEL, v))
        edges.append((v, A_LABEL))
    for v in odd:                                 # b <-> odd vertices
        edges.append((B_LABEL, v))
        edges.append((v, B_LABEL))

    return core + [A_LABEL, B_LABEL], edges


def fmt_chain(res, vec) -> str:
    """Render a chain as a readable linear combination of elementary paths."""
    if not vec:
        return "0"
    parts = []
    for path in sorted(vec, key=lambda t: (len(t), t)):
        c = vec[path]
        if c == 0:
            continue
        name = "e" + "".join(name_of(x) for x in path) if path else "e"
        if c == 1:
            parts.append(f"+{name}" if parts else name)
        elif c == -1:
            parts.append(f"-{name}")
        else:
            parts.append(f"{'+' if c > 0 and parts else ''}{c}*{name}")
    return " ".join(parts) if parts else "0"


def compute(n: int, max_dim: int = 2, fields=("Q", "Z2")):
    """Run the path-homology computation for ``P_n`` over each requested field."""
    verts, edges = build_P(n)
    return {
        f: path_homology(vertices=verts, edges=edges, max_dim=max_dim, field=f)
        for f in fields
    }


def summarize(n: int, max_dim: int = 2) -> dict:
    """Print a full report for ``P_n`` and return the two results."""
    verts, edges = build_P(n)
    print("=" * 74)
    print(f"P_{n}    |V| = {len(verts)}    |E| = {len(edges)}    "
          f"max_dim = {max_dim}")
    print("=" * 74)
    print("  vertices          :", [name_of(v) for v in verts])
    print("  directed cycle    :",
          " -> ".join(name_of(i) for i in range(2 * n)) + f" -> {name_of(0)}")
    print("  a  <-> even       :", [name_of(v) for v in range(0, 2 * n, 2)])
    print("  b  <-> odd        :", [name_of(v) for v in range(1, 2 * n, 2)])
    se = semi_edges(edges)
    print(f"  semi-edges        : {len(se)}  "
          f"({', '.join(name_of(u) + '>->' + name_of(v) for u, v in se[:8])}"
          f"{', ...' if len(se) > 8 else ''})")
    print()

    res = {}
    for field, label in (("Q", "Q   (exact rationals)"), ("Z2", "Z/2 (two-element field)")):
        r = path_homology(vertices=verts, edges=edges, max_dim=max_dim, field=field)
        res[field] = r
        dims = range(max_dim + 1)
        print(f"  --- K = {label} ---")
        print(f"      |A_p|      : "
              f"{ {p: len(r.allowed.get(p, [])) for p in dims} }")
        print(f"      dim Omega_p: "
              f"{ {p: r.dim_omega.get(p, 0) for p in dims} }")
        print(f"      dim Z_p    : "
              f"{ {p: r.dim_cycles.get(p, 0) for p in dims} }")
        print(f"      dim B_p    : "
              f"{ {p: r.dim_boundaries.get(p, 0) for p in dims} }")
        print(f"      dim H_p    : "
              f"{ {p: r.betti.get(p, 0) for p in dims} }")
        print(f"      chi        : {r.euler_characteristic}")
        for p in dims:
            if r.homology_basis.get(p):
                for g in r.homology_basis[p]:
                    print(f"      H_{p} generator: {fmt_chain(r, g)}")
        print()

    print("  --- comparison ---")
    doQ, do2 = res["Q"], res["Z2"]
    for p in range(max_dim + 1):
        a, b = doQ.betti.get(p, 0), do2.betti.get(p, 0)
        flag = "" if a == b else "   <-- differs"
        print(f"      dim H_{p} :  Q -> {a}    Z/2 -> {b}{flag}")
    print()
    return res


def table(nmax: int = 10, max_dim: int = 2):
    """Print the summary table of dim H_1 over both fields for n = 1..nmax."""
    print(f"{'n':>3} {'|V|':>5} {'|E|':>5} {'dim A_2':>8} {'dim O_2':>8} "
          f"{'dim H_1(Q)':>11} {'dim H_1(Z2)':>12} {'dim H_2(Q)':>11} "
          f"{'dim H_2(Z2)':>12}  {'Q==Z2':>6}")
    print("-" * 96)
    for n in range(1, nmax + 1):
        verts, edges = build_P(n)
        rq = path_homology(vertices=verts, edges=edges, max_dim=max_dim, field="Q")
        r2 = path_homology(vertices=verts, edges=edges, max_dim=max_dim, field="Z2")
        same = (rq.betti == r2.betti)
        print(f"{n:>3} {len(verts):>5} {len(edges):>5} "
              f"{len(rq.allowed.get(2, [])):>8} {rq.dim_omega.get(2, 0):>8} "
              f"{rq.betti.get(1, 0):>11} {r2.betti.get(1, 0):>12} "
              f"{rq.betti.get(2, 0):>11} {r2.betti.get(2, 0):>12}  "
              f"{'yes' if same else 'NO':>6}")


def draw(n: int, path: str | None = None):
    """Draw ``P_n``: the directed cycle in the middle, a and b outside."""
    import math

    from visualize import draw_digraph

    verts, edges = build_P(n)
    # custom layout: cycle on a circle, a outside near the evens, b outside near
    # the odds
    pos = {}
    R = 1.0
    for i in range(2 * n):
        ang = math.pi / 2 - 2 * math.pi * i / (2 * n)
        pos[i] = (R * math.cos(ang), R * math.sin(ang))
    pos[A_LABEL] = (-2.15, 0.0)
    pos[B_LABEL] = (2.15, 0.0)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    draw_digraph(
        verts, edges, ax=ax, pos=pos, label_of=name_of,
        title=f"$P_{{{n}}}$ : directed $2n$-cycle with $a\\leftrightarrow$even "
              f"and $b\\leftrightarrow$odd  (|V|={len(verts)}, |E|={len(edges)})",
    )
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-1.45, 1.45)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"figure saved to {path}")
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Path homology of the digraph P_n over Q and Z/2."
    )
    ap.add_argument("n", nargs="*", type=int, default=None,
                    help="which P_n to analyse (default 1..4)")
    ap.add_argument("--max-dim", type=int, default=2,
                    help="truncate the chain complex at this dimension (default 2)")
    ap.add_argument("--table", action="store_true",
                    help="print the summary table of dim H_1 / H_2 over both fields")
    ap.add_argument("--nmax", type=int, default=10,
                    help="upper limit for --table (default 10)")
    ap.add_argument("--plot", default=None,
                    help="save a drawing of P_n (single n only)")
    args = ap.parse_args(argv)

    if args.table:
        table(args.nmax, args.max_dim)
        return 0

    ns = args.n or [1, 2, 3, 4]
    for n in ns:
        summarize(n, args.max_dim)

    if args.plot:
        if len(ns) != 1:
            print("--plot expects exactly one n", file=sys.stderr)
            return 2
        draw(ns[0], args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
