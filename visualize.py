# -*- coding: utf-8 -*-
"""
visualize.py -- drawing helpers for the path complex of a digraph.

The drawings are intentionally dependency-light: matplotlib is imported lazily
so that the numerical core (``core.py``) keeps working without it.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "circular_layout",
    "draw_digraph",
    "draw_homology_generators",
    "plot_dimension_profile",
]


def circular_layout(vertices: Sequence) -> Dict[object, Tuple[float, float]]:
    """Place the vertices evenly on the unit circle (deterministic layout)."""
    n = len(vertices)
    if n == 0:
        return {}
    if n == 1:
        return {vertices[0]: (0.0, 0.0)}
    return {
        v: (
            math.cos(2 * math.pi * i / n),
            math.sin(2 * math.pi * i / n),
        )
        for i, v in enumerate(vertices)
    }


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting; install it with "
            "`pip install matplotlib`"
        ) from exc
    import matplotlib.pyplot as plt

    return plt


def draw_digraph(
    vertices: Iterable,
    edges: Iterable[Tuple],
    *,
    title: str = "Digraph",
    highlight: Optional[Iterable[Tuple]] = None,
    ax=None,
    pos: Optional[Dict[object, Tuple[float, float]]] = None,
    node_color: str = "#dbeafe",
    highlight_color: str = "#f97316",
    label_of=None,
):
    """
    Draw the digraph with curved, clearly-directed arrows.

    Parameters
    ----------
    vertices, edges : the digraph.
    highlight : iterable of pairs, optional
        Edges drawn in ``highlight_color`` (e.g. the support of a homology
        generator).
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created when omitted.
    pos : dict, optional
        Vertex positions; defaults to :func:`circular_layout`.
    label_of : callable, optional
        Maps a vertex to the text drawn inside its marker.  Defaults to
        ``str``; pass e.g. ``name_of`` to render integer labels as letters.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5))

    if label_of is None:
        label_of = str
    verts = list(vertices)
    edge_list = list(edges)
    hi = set(tuple(e) for e in (highlight or ()))
    if pos is None:
        pos = circular_layout(verts)

    # --- arrows ------------------------------------------------------------
    import matplotlib.patches as mpatches

    for (u, v) in edge_list:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy) or 1.0
        # shrink so the arrow does not run under the node marker
        shrink = 0.14
        sx, sy = x0 + dx * shrink, y0 + dy * shrink
        ex, ey = x1 - dx * shrink, y1 - dy * shrink
        is_hi = (u, v) in hi
        # curve the arrow to make the two directions of a 2-cycle distinguishable
        rad = 0.18 if (v, u) in set(edge_list) else 0.0
        arrow = mpatches.FancyArrowPatch(
            (sx, sy),
            (ex, ey),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>" if not is_hi else "-|>",
            mutation_scale=14 if not is_hi else 20,
            linewidth=2.4 if is_hi else 1.1,
            color=highlight_color if is_hi else "#64748b",
            zorder=2 if not is_hi else 4,
            shrinkA=0,
            shrinkB=0,
        )
        ax.add_patch(arrow)

    # --- nodes ------------------------------------------------------------
    for v in verts:
        if v not in pos:
            continue
        x, y = pos[v]
        ax.scatter(
            [x], [y], s=520, c=node_color, edgecolors="#1e293b",
            linewidths=1.4, zorder=5,
        )
        ax.text(
            x, y, label_of(v), ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=6, color="#0f172a",
        )

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def _support(path_dict: Dict, ordered_basis: Sequence) -> List[Tuple]:
    """
    Reduce a chain (``{path: coefficient}``) to the list of directed edges in
    its support, for highlighting.  Only 0- and 1-paths map directly to edges;
    for higher-dimensional generators we highlight their 1-skeleton faces.
    """
    out: List[Tuple] = []
    for path in path_dict:
        if path_dict[path] == 0:
            continue
        if len(path) == 2:                     # a 1-path is an edge
            out.append((path[0], path[1]))
        elif len(path) > 2:                    # consecutive pairs of a p-path
            for k in range(len(path) - 1):
                out.append((path[k], path[k + 1]))
    return out


def draw_homology_generators(
    result,
    *,
    max_generators: int = 4,
    title: Optional[str] = None,
):
    """
    One subplot per dimension, showing the digraph with the edges in the
    support of a generator of ``H_p`` highlighted.
    """
    plt = _require_matplotlib()

    dims = [p for p in sorted(result.homology_basis) if result.betti.get(p, 0) > 0]
    if not dims:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        draw_digraph(
            result.vertices, result.edges, ax=ax,
            title=(title or "No non-trivial path homology"),
        )
        return fig

    dims = dims[:max_generators]
    n = len(dims)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.4), squeeze=False)
    pos = circular_layout(result.vertices)
    for ax, p in zip(axes[0], dims):
        gen = result.homology_basis[p][0]
        hi = _support(gen, result.allowed.get(p, []))
        draw_digraph(
            result.vertices, result.edges, ax=ax, pos=pos,
            highlight=hi,
            title=(
                f"$\\dim H_{p} = {result.betti[p]}$\n"
                f"generator: {result._fmt(gen)[:46]}"
                + ("..." if len(result._fmt(gen)) > 46 else "")
            ),
        )
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_dimension_profile(result, *, title: Optional[str] = None):
    """Bar chart of ``dim A_p``, ``dim Omega_p`` and ``dim H_p`` per dimension."""
    plt = _require_matplotlib()

    dims = sorted(set(result.dim_omega) | set(result.allowed))
    counts = getattr(result, "dim_allowed", None) or {
        p: len(paths) for p, paths in result.allowed.items()
    }
    a_vals = [counts.get(p, 0) for p in dims]
    o_vals = [result.dim_omega.get(p, 0) for p in dims]
    h_vals = [result.betti.get(p, 0) for p in dims]

    x = range(len(dims))
    fig, ax = plt.subplots(figsize=(1.15 * len(dims) + 4.0, 4.6))
    w = 0.27
    ax.bar([i - w for i in x], a_vals, w, label=r"$\dim A_p$", color="#cbd5e1")
    ax.bar(list(x), o_vals, w, label=r"$\dim \Omega_p$", color="#60a5fa")
    ax.bar([i + w for i in x], h_vals, w, label=r"$\dim H_p$", color="#f97316")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"p={p}" for p in dims])
    ax.set_ylabel("dimension")
    ax.set_title(title or "Path complex dimensions", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
