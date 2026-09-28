# -*- coding: utf-8 -*-
"""
cli.py -- command line interface and worked examples.

Run ``python cli.py --help`` for usage.  ``python cli.py examples`` reproduces
the worked examples of arXiv:1207.2834v4 and checks the numbers against the
values stated in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from core import (
    PathHomologyResult,
    allowed_paths,
    boundary_matrix,
    bridges,
    path_homology,
    semi_edges,
)

__all__ = ["main", "build_parser", "EXAMPLES"]


# ---------------------------------------------------------------------------
# the paper's worked examples
# ---------------------------------------------------------------------------

#: name -> (vertices, edges, expected dict of paper values)
EXAMPLES: Dict[str, dict] = {
    "fig8": dict(
        vertices=[0, 1, 2, 3, 4, 5],
        edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 4), (5, 3), (5, 4)],
        max_dim=6,
        expect=dict(
            dim_omega={0: 6, 1: 8, 2: 2},
            betti={0: 1, 1: 1, 2: 0},
            euler=0,
            semi_edges=[(0, 3), (0, 4)],
            h1="e13 -e14 -e53 +e54",
        ),
        note="Section 4.6, Figure 8 (6 vertices, 8 edges)",
    ),
    "triangle": dict(
        vertices=[0, 1, 2],
        edges=[(0, 1), (1, 2), (0, 2)],
        max_dim=4,
        expect=dict(dim_omega={0: 3, 1: 3, 2: 1}, betti={0: 1, 1: 0, 2: 0}, euler=1),
        note="Section 4.5: the (transitive) triangle a->b, b->c, a->c",
    ),
    "square": dict(
        vertices=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (0, 3), (3, 2)],
        max_dim=4,
        expect=dict(dim_omega={0: 4, 1: 4, 2: 1}, betti={0: 1, 1: 0, 2: 0}, euler=1),
        note="Section 4.5: the square a->b,b'->c with a->b', b->c",
    ),
    "cycle_c4": dict(
        vertices=[0, 1, 2, 3],
        edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
        max_dim=4,
        expect=dict(dim_omega={0: 4, 1: 4, 2: 0}, betti={0: 1, 1: 1, 2: 0}, euler=0),
        note="Proposition 4.7: a cycle-graph that is neither triangle nor square",
    ),
    "example_3_14": dict(
        vertices=[0, 1],
        edges=[(0, 1), (1, 0)],
        max_dim=4,
        expect=dict(dim_omega={0: 2, 1: 2, 2: 0}, betti={0: 1, 1: 1, 2: 0}, euler=0),
        note="Example 3.14: 0<->1, non-regular chain complex",
    ),
    "octahedron": dict(
        vertices=[0, 1, 2, 3, 4, 5],
        # The paper (Example 6.17) states |E| = 12, an empty semi-edge set and
        # A_2 = span{e_024,e_025,e_034,e_035,e_124,e_125,e_134,e_135}.  The 8
        # faces of the octahedron give the 8 directed edges
        # (0,2),(0,3),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5); completing to 12
        # edges with no semi-edges forces (0,4),(0,5),(1,4),(1,5).
        edges=[
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4), (2, 5), (3, 4), (3, 5),
        ],
        max_dim=6,
        expect=dict(
            dim_omega={0: 6, 1: 12, 2: 8},
            betti={0: 1, 1: 0, 2: 1},
            euler=2,
            semi_edges=[],
            a2=[(0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
                (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5)],
        ),
        note="Example 6.17, Figure 24: octahedron, a 2-sphere graph",
    ),
}


def _parse_edges(spec: str) -> List[Tuple[str, str]]:
    """
    Parse an edge list such as ``0-1,1-2,2-0`` (also accepts ``0>1`` and
    ``0->1``, and ``;`` or whitespace as separators).
    """
    if not spec:
        return []
    norm = spec.replace("->", "-").replace(">", "-")
    for sep in (";", " ", "\n", "\t"):
        norm = norm.replace(sep, ",")
    out: List[Tuple[str, str]] = []
    for chunk in norm.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(
                f"cannot parse edge {chunk!r}; expected the form u-v or u->v"
            )
        u, _, v = chunk.partition("-")
        out.append((u.strip(), v.strip()))
    return out


def _coerce_vertices(edges: Sequence[Tuple[str, str]]):
    """Turn string labels into ints when every label looks like an integer."""
    labels = {x for e in edges for x in e}
    if labels and all(s.lstrip("-").isdigit() for s in labels):
        return [(int(u), int(v)) for u, v in edges]
    return list(edges)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="path_homology",
        description=(
            "Path homology of a digraph (GLMY theory, arXiv:1207.2834v4). "
            "Computes allowed paths A_p, the boundary matrices, the "
            "partial-invariant spaces Omega_p, and the chain groups "
            "Z_p = ker(partial|Omega_p) and B_p = partial(Omega_{p+1}) with "
            "H_p = Z_p / B_p."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--edges", "-e", default="",
        help="comma-separated edges, e.g. '0-1,1-2,2-0' (or 0->1,1->2,2->0)",
    )
    common.add_argument(
        "--vertices", "-v", default="",
        help="comma-separated vertex labels (default: inferred from the edges)",
    )
    common.add_argument(
        "--max-dim", type=int, default=None,
        help="truncate allowed-path enumeration at this dimension",
    )
    common.add_argument(
        "--regular", action="store_true",
        help="use the regular chain complex (partial^reg drops non-regular faces)",
    )
    common.add_argument(
        "--no-loops", action="store_true",
        help="ignore self-loops present in the edge list",
    )
    common.add_argument(
        "--method", choices=["auto", "kernel", "lemma41", "prop42"], default="auto",
        help="how Omega_p is computed (default: auto)",
    )
    common.add_argument(
        "--float", action="store_true",
        help="use floating point instead of exact rational arithmetic",
    )
    common.add_argument(
        "--field", default=None,
        help="coefficient field: 'Q' (exact rationals, default), 'Z2'/'GF2'/2 "
             "(two-element field), a prime p for GF(p), or 'R'",
    )
    common.add_argument("--json", action="store_true", help="emit JSON")

    run = sub.add_parser("run", parents=[common], help="analyse one digraph")
    run.add_argument(
        "--plot", default=None,
        help="save a figure (digraph + homology generators + dimension profile) "
             "to this path, e.g. out.png",
    )

    ex = sub.add_parser(
        "examples", help="run the worked examples from the paper and check them"
    )
    ex.add_argument(
        "--plot-dir", default=None,
        help="also save figures for each example into this directory",
    )
    return p


def _load_digraph(args) -> Tuple[Optional[List], List[Tuple]]:
    edges = _parse_edges(args.edges)
    edges = _coerce_vertices(edges)
    if args.vertices.strip():
        raw = [x.strip() for x in args.vertices.split(",") if x.strip()]
        if edges and all(isinstance(u, int) for u, _ in edges):
            verts = [int(x) for x in raw]
        else:
            verts = raw
    else:
        verts = None
    return verts, edges


def _result_to_json(res: PathHomologyResult) -> dict:
    def fmt(vec):
        return {("".join(str(x) for x in k) if k else "e"): str(v) for k, v in vec.items()}

    return {
        "vertices": [str(v) for v in res.vertices],
        "edges": [[str(u), str(v)] for u, v in res.edges],
        "allowed_paths": {
            str(p): ["".join(str(x) for x in t) for t in res.allowed[p]]
            for p in sorted(res.allowed)
        },
        "boundary_matrices": {
            str(p): [[str(c) for c in row] for row in res.boundary_matrices[p].tolist()]
            for p in sorted(res.boundary_matrices)
        },
        "dim_allowed": {str(p): len(res.allowed[p]) for p in sorted(res.allowed)},
        "dim_omega": {str(p): res.dim_omega[p] for p in sorted(res.dim_omega)},
        "dim_cycles": {str(p): res.dim_cycles.get(p, 0) for p in sorted(res.dim_omega)},
        "dim_boundaries": {
            str(p): res.dim_boundaries.get(p, 0) for p in sorted(res.dim_omega)
        },
        "betti": {str(p): res.betti.get(p, 0) for p in sorted(res.dim_omega)},
        "omega_basis": {
            str(p): [fmt(v) for v in res.omega_paths.get(p, [])]
            for p in sorted(res.dim_omega)
        },
        "cycles": {
            str(p): [fmt(v) for v in res.cycles.get(p, [])] for p in sorted(res.dim_omega)
        },
        "boundaries": {
            str(p): [fmt(v) for v in res.boundaries.get(p, [])]
            for p in sorted(res.dim_omega)
        },
        "homology_basis": {
            str(p): [fmt(v) for v in res.homology_basis.get(p, [])]
            for p in sorted(res.dim_omega)
        },
        "euler_characteristic": res.euler_characteristic,
    }


def _cmd_run(args) -> int:
    verts, edges = _load_digraph(args)
    if not edges and not verts:
        print("error: nothing to do; pass --edges (and/or --vertices)", file=sys.stderr)
        return 2

    res = path_homology(
        vertices=verts,
        edges=edges,
        max_dim=args.max_dim,
        loops=not args.no_loops,
        regular=args.regular,
        exact=not args.float,
        field=args.field,
        method=args.method,
    )

    if args.json:
        print(json.dumps(_result_to_json(res), indent=2, ensure_ascii=False))
    else:
        print(res.summary())
        se = semi_edges(res.edges)
        print("")
        print("-- Semi-edges (Section 4.1) --")
        if se:
            for (u, v) in se:
                br = bridges(res.edges).get((u, v), [])
                mids = ", ".join(str(k) for k in br)
                print(f"  {u} >-> {v}   via bridges: {mids}")
        else:
            print("  (none)")

    if args.plot:
        import matplotlib.pyplot as plt
        from visualize import (
            draw_digraph,
            draw_homology_generators,
            plot_dimension_profile,
        )

        fig = plt.figure(figsize=(15, 5.0))
        ax_graph = fig.add_subplot(1, 2, 1)
        draw_digraph(res.vertices, res.edges, ax=ax_graph, title="Digraph")
        ax_prof = fig.add_subplot(1, 2, 2)
        _profile_on_axes(ax_prof, res, plot_dimension_profile)
        fig.suptitle("Path complex of the digraph", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # a separate figure for the homology generators
        fig2 = draw_homology_generators(res, title="Path homology generators")
        gen_path = args.plot.rsplit(".", 1)
        gen_out = (
            f"{gen_path[0]}_generators.{gen_path[1]}"
            if len(gen_path) == 2
            else f"{args.plot}_generators.png"
        )
        fig2.savefig(gen_out, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"\nfigures saved to {args.plot} and {gen_out}")
    return 0


def _profile_on_axes(ax, res, plot_dimension_profile):
    """Draw the dimension profile onto an existing Axes."""
    import matplotlib.pyplot as plt

    tmp = plot_dimension_profile(res)
    # copy the bars over by re-plotting on the target axes
    plt.close(tmp)
    dims = sorted(set(res.dim_omega) | set(res.allowed))
    x = range(len(dims))
    w = 0.27
    ax.bar(
        [i - w for i in x],
        [len(res.allowed.get(p, [])) for p in dims], w,
        label=r"$\dim A_p$", color="#cbd5e1",
    )
    ax.bar(
        list(x), [res.dim_omega.get(p, 0) for p in dims], w,
        label=r"$\dim \Omega_p$", color="#60a5fa",
    )
    ax.bar(
        [i + w for i in x], [res.betti.get(p, 0) for p in dims], w,
        label=r"$\dim H_p$", color="#f97316",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"p={p}" for p in dims])
    ax.set_ylabel("dimension")
    ax.set_title("Dimensions per degree", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)


def _check(name: str, got, want) -> bool:
    ok = got == want
    mark = "OK  " if ok else "FAIL"
    print(f"  {mark} {name}: got={got} want={want}")
    return ok


def _cmd_examples(args) -> int:
    all_ok = True
    for name, spec in EXAMPLES.items():
        print("=" * 74)
        print(f"{name}  --  {spec['note']}")
        print("=" * 74)
        res = path_homology(
            vertices=spec.get("vertices"),
            edges=spec["edges"],
            max_dim=spec.get("max_dim"),
        )
        exp = spec["expect"]

        for p, want in exp.get("dim_omega", {}).items():
            all_ok &= _check(f"dim Omega_{p}", res.dim_omega[p], want)
        for p, want in exp.get("betti", {}).items():
            all_ok &= _check(f"dim H_{p}", res.betti.get(p, 0), want)
        if "euler" in exp:
            all_ok &= _check("Euler characteristic", res.euler_characteristic, exp["euler"])
        if "semi_edges" in exp:
            all_ok &= _check("semi-edges", semi_edges(res.edges), exp["semi_edges"])
        if "a2" in exp:
            all_ok &= _check("A_2", sorted(res.allowed.get(2, [])), exp["a2"])
        if "h1" in exp and res.homology_basis.get(1):
            all_ok &= _check(
                "H_1 generator", res._fmt(res.homology_basis[1][0]), exp["h1"]
            )

        print("  Omega_* basis and homology:")
        for p in sorted(res.dim_omega):
            if res.dim_omega[p] == 0 and res.betti.get(p, 0) == 0:
                continue
            print(
                f"    p={p}: dim Omega={res.dim_omega[p]}  "
                f"dim Z={res.dim_cycles[p]}  dim B={res.dim_boundaries[p]}  "
                f"dim H={res.betti.get(p, 0)}"
            )
            for v in res.homology_basis.get(p, []):
                print(f"          H_{p} generator: {res._fmt(v)}")
        print("")

        if args.plot_dir:
            import os
            from visualize import draw_homology_generators, plot_dimension_profile
            os.makedirs(args.plot_dir, exist_ok=True)
            fig = draw_homology_generators(res, title=spec["note"])
            fig.savefig(
                os.path.join(args.plot_dir, f"{name}_homology.png"),
                dpi=150, bbox_inches="tight",
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            fig = plot_dimension_profile(res, title=spec["note"])
            fig.savefig(
                os.path.join(args.plot_dir, f"{name}_profile.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

    print("=" * 74)
    print("ALL EXAMPLES MATCH THE PAPER" if all_ok else "SOME EXAMPLES DISAGREE")
    print("=" * 74)
    return 0 if all_ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "examples":
        return _cmd_examples(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
