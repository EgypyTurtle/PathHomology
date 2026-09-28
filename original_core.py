# -*- coding: utf-8 -*-
"""
core.py -- Path homology of digraphs (GLMY theory).

Reference
---------
A. Grigor'yan, Y. Lin, Yu. Muranov, S.-T. Yau,
"Homologies of path complexes and digraphs", arXiv:1207.2834v4.

Notation used below follows the paper exactly:

    V                     finite vertex set
    e_{i0...ip}           elementary p-path (a sequence of p+1 vertices)
    A_p = A_p(P)          span of the ALLOWED p-paths          (Sec. 3.2)
    partial               boundary operator, eq. (2.2)
    Omega_p = Omega_p(P)  {v in A_p : partial v in A_{p-1}}     (3.8)
    H_p(P)                ker(partial|Omega_p) / partial(Omega_{p+1})  (3.11)

Everything is computed over a field K; the default is the rationals
(fractions.Fraction) so that rank computations are exact.

Main public entry point: :func:`path_homology`.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "PathHomologyResult",
    "chain_group_ranks",
    "boundary_matrix",
    "path_homology",
]

# ---------------------------------------------------------------------------
# basic linear algebra over an arbitrary field, exact where possible
# ---------------------------------------------------------------------------

Scalar = Union[int, Fraction, float]

#: an elementary p-path is stored as a tuple of p+1 vertices
Path = Tuple[int, ...]


class GF:
    """
    A prime field ``GF(p)``, i.e. the integers modulo a prime ``p``.

    Instances are immutable field elements.  ``GF(0, p)`` is the zero element
    and arithmetic wraps modulo ``p``.  Division uses the modular inverse, so
    the class is a genuine field whenever ``p`` is prime (the caller supplies a
    prime; ``field="Z2"`` and ``field=2`` are the cases used here).

    This exists so that path homology can be computed over ``Z/2`` alongside the
    default exact rationals: the chain complex ``Omega_*`` is defined over any
    field ``K`` (Section 2), and the Betti numbers can in principle depend on
    the characteristic.
    """

    __slots__ = ("v", "p")

    def __init__(self, v, p):
        self.p = p
        self.v = int(v) % p

    # -- arithmetic ---------------------------------------------------------
    def _coerce(self, other):
        if isinstance(other, GF):
            if other.p != self.p:
                raise ValueError(
                    f"cannot mix GF({self.p}) with GF({other.p})"
                )
            return other.v
        if isinstance(other, Fraction):
            if other.denominator == 1:
                return int(other.numerator) % self.p
            return (
                int(other.numerator) % self.p
                * pow(int(other.denominator) % self.p, self.p - 2, self.p)
            ) % self.p
        return int(other) % self.p

    # NumPy may hand us a whole array as the operand (e.g. in
    # ``a[row, col] * a[rank, :]``).  Broadcasting the scalar multiply over an
    # object array would otherwise call ``_coerce`` on an ndarray, so we handle
    # arrays explicitly and vectorise elementwise.
    @staticmethod
    def _is_array(x):
        return hasattr(x, "shape") and hasattr(x, "ravel") and x.shape != ()

    def _elementwise(self, other, op):
        out = np.empty(other.shape, dtype=object)
        flat = other.ravel()
        res = out.ravel()
        for i, x in enumerate(flat):
            res[i] = op(x)
        return out

    def __add__(self, o):
        if self._is_array(o):
            return self._elementwise(o, lambda x: self + x)
        return GF(self.v + self._coerce(o), self.p)

    __radd__ = __add__

    def __sub__(self, o):
        if self._is_array(o):
            return self._elementwise(o, lambda x: self - x)
        return GF(self.v - self._coerce(o), self.p)

    def __rsub__(self, o):
        if self._is_array(o):
            return self._elementwise(o, lambda x: x - self)
        return GF(self._coerce(o) - self.v, self.p)

    def __mul__(self, o):
        if self._is_array(o):
            return self._elementwise(o, lambda x: self * x)
        return GF(self.v * self._coerce(o), self.p)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if self._is_array(o):
            return self._elementwise(o, lambda x: self / x)
        c = self._coerce(o)
        if c == 0:
            raise ZeroDivisionError("division by zero in GF(%d)" % self.p)
        return GF(self.v * pow(c, self.p - 2, self.p), self.p)

    def __rtruediv__(self, o):
        if self._is_array(o):
            return self._elementwise(o, lambda x: x / self)
        if self.v == 0:
            raise ZeroDivisionError("division by zero in GF(%d)" % self.p)
        return GF(self._coerce(o) * pow(self.v, self.p - 2, self.p), self.p)

    def __neg__(self):
        return GF(-self.v, self.p)

    def __eq__(self, o):
        if isinstance(o, GF):
            return self.p == o.p and self.v == o.v
        try:
            return self.v == self._coerce(o)
        except Exception:
            return NotImplemented

    def __ne__(self, o):
        r = self.__eq__(o)
        return r if r is NotImplemented else not r

    def __hash__(self):
        return hash((self.v, self.p))

    def __bool__(self):
        return self.v != 0

    def __int__(self):
        return self.v

    def __repr__(self):
        return f"GF({self.v}, {self.p})"

    def __str__(self):
        return str(self.v)


#: named fields accepted by the ``field=`` argument
FIELD_ALIASES = {
    "q": "Q", "qq": "Q", "rational": "Q", "rationals": "Q", "fraction": "Q",
    "z2": "Z2", "gf2": "Z2", "f2": "Z2", "z/2": "Z2", "z_2": "Z2",
    "float": "R", "r": "R", "real": "R",
}


def resolve_field(field):
    """
    Turn a ``field=`` argument into a concrete arithmetic kind.

    Accepts ``"Q"`` (exact rationals, the default), ``"Z2"``/``"GF2"``/``2``
    (the two-element field), any prime ``int`` (giving ``GF(p)``), or ``"R"``
    (floating point).  Returns ``(kind, modulus)`` with ``kind`` in
    ``{"Q", "GF", "R"}``.
    """
    if field is None:
        return "Q", None
    if isinstance(field, bool):
        raise TypeError("field must be 'Q', 'Z2', a prime, or 'R'")
    if isinstance(field, int):
        if field < 2:
            raise ValueError("a prime field GF(p) needs p >= 2")
        return "GF", field
    if isinstance(field, str):
        key = field.strip().lower()
        if key in FIELD_ALIASES:
            k = FIELD_ALIASES[key]
            return (k, None) if k in ("Q", "R") else ("GF", 2)
        if key.isdigit():
            return "GF", int(key)
        raise ValueError(f"unknown field {field!r}; use 'Q', 'Z2', a prime, or 'R'")
    raise TypeError(f"unsupported field specification: {field!r}")


def make_scalar(value, kind, modulus=None):
    """Build a scalar of the requested field from an integer/rational value."""
    if kind == "GF":
        return GF(value, modulus)
    if kind == "Q":
        return value if isinstance(value, Fraction) else Fraction(value)
    return float(value)


def dtype_for(kind):
    """NumPy dtype used to store matrices over the given field."""
    return float if kind == "R" else object

#: default truncation dimension when the digraph has directed cycles and the
#: caller did not specify ``max_dim`` (allowed paths then exist in every
#: dimension, cf. the remark after (3.15) in the paper).
DEFAULT_MAX_DIM = 6


def _to_scalar(x) -> Scalar:
    """Coerce a user-supplied field element to the working scalar type."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, bool):
        return Fraction(int(x))
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return x  # floating point field (e.g. for large sparse problems)
    return Fraction(x)


def _is_gf2_array(mat: np.ndarray) -> bool:
    """Whether ``mat`` holds GF(2) elements (all of them, if any)."""
    if mat.dtype != object or mat.size == 0:
        return False
    flat = mat.ravel()
    x = flat[0]
    return isinstance(x, GF) and x.p == 2


def _as_bits(mat: np.ndarray) -> np.ndarray:
    """Object array of GF(2) elements -> uint8 0/1 array."""
    out = np.empty(mat.shape, dtype=np.uint8)
    flat_in = mat.ravel()
    flat_out = out.ravel()
    for i, x in enumerate(flat_in):
        flat_out[i] = x.v if isinstance(x, GF) else (int(x) & 1)
    return out


def _gf2_rank(bits: np.ndarray) -> int:
    """
    Rank over GF(2) by XOR elimination on a packed uint8 matrix.

    This replaces the generic object-dtype elimination when the field is GF(2);
    it is the same algorithm, but the row operation ``r_i -= c * r_k`` becomes a
    single vectorised XOR, which is ~2 orders of magnitude faster than looping
    over Python ``GF`` objects.
    """
    A = bits.copy()
    m, n = A.shape
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        rows = np.nonzero(A[:, c])[0]
        rows = rows[rows != r]
        if rows.size:
            A[rows] ^= A[r]
        r += 1
        if r == m:
            break
    return r


def _gf2_nullspace(bits: np.ndarray) -> np.ndarray:
    """Nullspace over GF(2) as a uint8 matrix whose COLUMNS span the kernel."""
    n_rows, n_cols = bits.shape
    A = bits.copy()
    pivots: List[int] = []
    r = 0
    for c in range(n_cols):
        piv = None
        for i in range(r, n_rows):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        rows = np.nonzero(A[:, c])[0]
        rows = rows[rows != r]
        if rows.size:
            A[rows] ^= A[r]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    free = [c for c in range(n_cols) if c not in pivots]
    B = np.zeros((n_cols, len(free)), dtype=np.uint8)
    for j, f in enumerate(free):
        B[f, j] = 1
        for i, p in enumerate(pivots):
            B[p, j] = A[i, f]
    return B


def _rank(mat: np.ndarray) -> int:
    """
    Rank of a matrix over a field, by Gaussian elimination.

    Exact when the entries are :class:`fractions.Fraction`; numerically
    robust (complete pivoting + tolerance) when they are floats.  For GF(2) the
    elimination is dispatched to a vectorised uint8/XOR routine, which is far
    faster than the generic object-dtype path.
    """
    if mat.size == 0:
        return 0
    if _is_gf2_array(mat):
        return _gf2_rank(_as_bits(mat))
    a = np.array(mat, dtype=object if mat.dtype == object else float)
    n_rows, n_cols = a.shape
    rank = 0

    if a.dtype == object:
        for col in range(n_cols):
            pivot = None
            for row in range(rank, n_rows):
                if a[row, col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            if pivot != rank:
                a[[rank, pivot], :] = a[[pivot, rank], :]
            a[rank, :] = a[rank, :] / a[rank, col]
            for row in range(n_rows):
                if row != rank and a[row, col] != 0:
                    a[row, :] = a[row, :] - a[row, col] * a[rank, :]
            rank += 1
            if rank == n_rows:
                break
        return rank

    atol = 1e-9 * max(1.0, float(np.max(np.abs(a))))
    for col in range(n_cols):
        pivot = None
        best = 0.0
        for row in range(rank, n_rows):
            val = abs(a[row, col])
            if val > best:
                best, pivot = val, row
        if pivot is None or best <= atol:
            continue
        if pivot != rank:
            a[[rank, pivot], :] = a[[pivot, rank], :]
        a[rank, :] = a[rank, :] / a[rank, col]
        for row in range(rank + 1, n_rows):
            if abs(a[row, col]) > atol:
                a[row, :] = a[row, :] - a[row, col] * a[rank, :]
        rank += 1
        if rank == n_rows:
            break
    return rank


def _nullspace(mat: np.ndarray) -> np.ndarray:
    """
    Basis of the kernel of ``mat`` (columns are the unknowns).

    Returns a matrix whose **columns** span ``ker(mat)``; shape
    ``(n_cols, nullity)``.  Over :class:`~fractions.Fraction` the result is
    an exact basis.
    """
    n_rows, n_cols = mat.shape
    if n_cols == 0:
        return np.zeros((0, 0), dtype=mat.dtype)
    if _is_gf2_array(mat):
        bits = _gf2_nullspace(_as_bits(mat))
        # return in the same object/GF form the callers expect
        out = np.empty(bits.shape, dtype=object)
        flat = out.ravel()
        src = bits.ravel()
        for i in range(src.size):
            flat[i] = GF(int(src[i]), 2)
        return out
    exact = mat.dtype == object
    if exact:
        # Coerce any plain ints (which can appear in an object array built by
        # hand) to Fractions, so the elimination stays exact rather than
        # silently producing floats on division.
        a = np.empty(mat.shape, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                x = mat[i, j]
                a[i, j] = x if isinstance(x, (GF, Fraction)) else Fraction(x)
    else:
        a = np.array(mat, dtype=float)
    pivots: List[int] = []
    row = 0
    for col in range(n_cols):
        pivot = None
        if exact:
            for r in range(row, n_rows):
                if a[r, col] != 0:
                    pivot = r
                    break
        else:
            best, atol = 0.0, 1e-9 * max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)
            for r in range(row, n_rows):
                val = abs(a[r, col])
                if val > best:
                    best, pivot = val, r
            if pivot is not None and best <= atol:
                pivot = None
        if pivot is None:
            continue
        if pivot != row:
            a[[row, pivot], :] = a[[pivot, row], :]
        a[row, :] = a[row, :] / a[row, col]
        for r in range(n_rows):
            if r != row and a[r, col] != 0:
                a[r, :] = a[r, :] - a[r, col] * a[row, :]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break

    free = [c for c in range(n_cols) if c not in pivots]
    basis = np.zeros((n_cols, len(free)), dtype=object if exact else float)
    # Build 0 and 1 in the SAME field as the entries.  We cannot hardcode
    # Fraction here, because the matrix may live over GF(p).
    zero = _zero_like(a)
    one = _one_like(a)
    for j, f in enumerate(free):
        basis[f, j] = one
        for i, p in enumerate(pivots):
            basis[p, j] = -a[i, f]
    return basis


def _pivot_columns(mat: np.ndarray) -> List[int]:
    """
    Indices of the pivot (linearly independent) COLUMNS of ``mat``.

    One Gaussian elimination replaces a whole sequence of rank computations:
    the returned columns form a maximal independent subset, so the columns of
    ``mat`` are exactly the ones that increase the rank when appended from left
    to right.  This is what the basis-of-the-quotient selection needs.
    """
    if mat.size == 0:
        return []
    if _is_gf2_array(mat):
        return _gf2_pivot_columns(_as_bits(mat))
    a = np.array(mat, dtype=object if mat.dtype == object else float)
    n_rows, n_cols = a.shape
    exact = a.dtype == object
    if exact:
        b = np.empty(a.shape, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                v = a[i, j]
                b[i, j] = v if isinstance(v, (GF, Fraction)) else Fraction(v)
        a = b
    atol = 1e-9 * max(1.0, float(np.max(np.abs(a.astype(float))))) if not exact else 0.0
    pivots: List[int] = []
    row = 0
    for col in range(n_cols):
        pivot = None
        if exact:
            for r in range(row, n_rows):
                if a[r, col] != 0:
                    pivot = r
                    break
        else:
            best = 0.0
            for r in range(row, n_rows):
                v = abs(a[r, col])
                if v > best:
                    best, pivot = v, r
            if pivot is not None and best <= atol:
                pivot = None
        if pivot is None:
            continue
        if pivot != row:
            a[[row, pivot], :] = a[[pivot, row], :]
        a[row, :] = a[row, :] / a[row, col]
        for r in range(row + 1, n_rows):
            if a[r, col] != 0:
                a[r, :] = a[r, :] - a[r, col] * a[row, :]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    return pivots


def _gf2_pivot_columns(bits: np.ndarray) -> List[int]:
    """Pivot columns over GF(2) via uint8 XOR elimination."""
    A = bits.copy()
    m, n = A.shape
    pivots: List[int] = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        rows = np.nonzero(A[:, c])[0]
        rows = rows[rows != r]
        if rows.size:
            A[rows] ^= A[r]
        pivots.append(c)
        r += 1
        if r == m:
            break
    return pivots


def _zero_like(a: np.ndarray):
    """Zero element of the field that the entries of ``a`` live in."""
    for x in a.ravel():
        if isinstance(x, GF):
            return GF(0, x.p)
        if isinstance(x, Fraction):
            return Fraction(0)
        if isinstance(x, (int, np.integer)):
            # an object array holding plain ints is still exact: keep it
            # rational so that the elimination below does not fall back to
            # floating point
            return Fraction(0)
        return 0.0
    return Fraction(0)


def _one_like(a: np.ndarray):
    """One element of the field that the entries of ``a`` live in."""
    for x in a.ravel():
        if isinstance(x, GF):
            return GF(1, x.p)
        if isinstance(x, Fraction):
            return Fraction(1)
        if isinstance(x, (int, np.integer)):
            return Fraction(1)
        return 1.0
    return Fraction(1)


def _columns(mat: np.ndarray) -> List[np.ndarray]:
    return [mat[:, j] for j in range(mat.shape[1])]


def _column_space_contains(mat: np.ndarray, vec: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether ``vec`` lies in the column space of ``mat``."""
    if mat.shape[1] == 0:
        return not any(abs(v) > tol for v in vec)
    return _rank(mat) == _rank(np.column_stack([mat, vec]))


# ---------------------------------------------------------------------------
# the path complex of a digraph
# ---------------------------------------------------------------------------


def allowed_paths(
    vertices: Optional[Iterable] = None,
    edges: Optional[Iterable[Tuple]] = None,
    max_dim: Optional[int] = None,
    *,
    loops: bool = True,
) -> Dict[int, List[Path]]:
    """
    Path complex ``P(G)`` of a digraph, as in Example 3.3.

    An elementary n-path ``i0...in`` on ``V`` is **allowed** iff
    ``i_{k-1} -> i_k`` for every ``k = 1..n``.  The family ``{P_n}`` then
    satisfies the truncation property (3.1), i.e. it is a path complex.

    .. warning::
       If the digraph contains a directed cycle then allowed paths exist in
       *every* dimension (just go around the cycle repeatedly).  In that case
       ``max_dim=None`` would enumerate forever, so it is automatically capped
       at :data:`DEFAULT_MAX_DIM`.  Pass ``max_dim`` explicitly to choose the
       truncation.  The induced chain complex is the one of (3.10) truncated at
       that dimension, which is what makes the Betti numbers finite.

    Parameters
    ----------
    vertices : iterable, optional
        Explicit vertex labels.  Defaults to all labels occurring in ``edges``.
    edges : iterable of pairs
        Directed edges ``(u, v)`` meaning ``u -> v``.
    max_dim : int, optional
        Stop enumerating at this dimension.
    loops : bool, default True
        Whether self-loops ``(v, v)`` are kept.  When ``False`` the digraph is
        made loopless, which makes the path complex *regular* in the sense of
        Definition 3.11.

    Returns
    -------
    dict
        ``{p: sorted list of allowed elementary p-paths}`` for
        ``p = 0, 1, ..., d`` where ``d`` is the largest non-empty dimension.
    """
    adj: Dict[object, List[object]] = defaultdict(list)
    labels: List[object] = []
    seen = set()

    def _touch(x):
        if x not in seen:
            seen.add(x)
            labels.append(x)

    if vertices is not None:
        for v in vertices:
            _touch(v)
    for e in edges or ():
        u, v = e
        _touch(u)
        _touch(v)
        if loops or u != v:
            adj[u].append(v)

    if max_dim is None:
        max_dim = DEFAULT_MAX_DIM

    paths: Dict[int, List[Path]] = {0: [(v,) for v in labels]}
    dim = 0
    while paths[dim] and dim < max_dim:
        nxt: List[Path] = []
        for p in paths[dim]:
            for w in adj[p[-1]]:
                nxt.append(p + (w,))
        if not nxt:
            break
        dim += 1
        paths[dim] = nxt
    paths = {p: v for p, v in paths.items() if v}
    for p in paths:
        paths[p] = sorted(set(paths[p]), key=lambda t: (len(t), t))
    return paths


# ---------------------------------------------------------------------------
# boundary operator
# ---------------------------------------------------------------------------


def is_regular_path(path: Path) -> bool:
    """Definition 2.7: regular means ``i_{k-1} != i_k`` for all ``k``."""
    return all(path[k - 1] != path[k] for k in range(1, len(path)))


def boundary_of_path(
    path: Path, *, regular: bool = False
) -> List[Tuple[Path, int]]:
    """
    ``partial e_{i0...ip} = sum_q (-1)^q e_{i0...ihat_q...ip}``   -- eq. (2.2).

    Parameters
    ----------
    path : tuple
        The elementary p-path ``(i0, ..., ip)``.
    regular : bool, default False
        ``False`` -- the plain boundary operator (2.2) on ``Lambda_p``.  Its
        result may contain non-regular faces such as ``e_{ii}``, as in
        Example 3.14.
        ``True``  -- the *regular* boundary operator ``partial^reg`` of
        Section 2.3, obtained by discarding all non-regular components of the
        result (cf. (2.10)).  This is the operator used for the regular
        version of the chain complex ``Omega_*``.

    Returns
    -------
    list of (path, coefficient)
        The terms of the boundary, with the empty path ``()`` representing the
        (-1)-path ``e`` when ``p = 0``.
    """
    p = len(path) - 1
    out: List[Tuple[Path, int]] = []
    for q in range(p + 1):
        face = path[:q] + path[q + 1:]
        if regular and not is_regular_path(face):
            continue
        out.append((face, (-1) ** q))
    return out


def boundary_matrix(
    paths_p: Sequence[Path],
    paths_prev: Sequence[Path],
    *,
    exact: bool = True,
    regular: bool = False,
    field=None,
) -> np.ndarray:
    """
    Matrix of ``partial : A_p -> A_{p-1}`` in the elementary-path bases.

    Rows are indexed by ``paths_prev`` (the target basis) and columns by
    ``paths_p`` (the source basis), so the (i, j) entry is the coefficient of
    the i-th (p-1)-path in ``partial`` of the j-th p-path.  This is the
    standard matrix of a linear map acting on column vectors::

        coordinates(partial v) = boundary_matrix @ (coordinates of v)

    For ``p = 0`` the matrix has no rows (``A_{-1} = K``) and we return the
    ``1 x |A_0|`` matrix of the augmentation ``sum_i v_i`` -- see (2.5).  Set
    ``paths_prev = [()]`` to get that.

    ``regular=True`` uses the regular boundary operator ``partial^reg``
    (Section 2.3), which drops non-regular faces.

    ``field`` selects the coefficient field: ``"Q"`` (exact rationals, the
    default), ``"Z2"`` or a prime ``p`` for ``GF(p)``, or ``"R"`` for floating
    point.  ``exact=False`` is a shortcut for ``field="R"``.
    """
    kind, modulus = resolve_field(field) if field is not None else (
        ("Q", None) if exact else ("R", None)
    )
    index = {p: i for i, p in enumerate(paths_prev)}
    mat = np.zeros((len(paths_prev), len(paths_p)), dtype=dtype_for(kind))
    for j, p in enumerate(paths_p):
        for face, coef in boundary_of_path(p, regular=regular):
            if face in index:
                mat[index[face], j] += make_scalar(coef, kind, modulus)
    return mat


# ---------------------------------------------------------------------------
# Omega_p  (partial-invariant paths)
# ---------------------------------------------------------------------------


def semi_edges(edges: Iterable[Tuple]) -> List[Tuple]:
    """
    Semi-edges of the digraph (Section 4.1).

    A pair ``i j`` is a **semi-edge** (written ``i >-> j``) if it is *not* an
    edge but there exists a vertex ``k`` with ``i -> k`` and ``k -> j``; the
    2-path ``i k j`` is then a *bridge*.

    Returns the sorted list of semi-edges.
    """
    edge_set = set()
    succ: Dict[object, set] = defaultdict(set)
    pred: Dict[object, set] = defaultdict(set)
    for u, v in edges:
        edge_set.add((u, v))
        succ[u].add(v)
        pred[v].add(u)
    out = []
    # the vertex list is materialised first so that the defaultdict look-ups
    # below cannot mutate a container that is being iterated
    verts = sorted({x for e in edge_set for x in e})
    for u, v in itertools.permutations(verts, 2):
        if (u, v) in edge_set:
            continue
        if succ[u] & pred[v]:
            out.append((u, v))
    return out


def bridges(edges: Iterable[Tuple]) -> Dict[Tuple, List[object]]:
    """
    For each semi-edge ``ij``, the list of middles ``k`` with ``i k j`` allowed.

    The returned mapping is keyed by the semi-edge ``(i, j)`` and its value is
    the sorted list of bridges ``k``; summing ``v_{...i k j...}`` over these
    ``k`` is the deficiency (4.1) of Lemma 4.1.
    """
    edge_set = set()
    succ: Dict[object, set] = {}
    for u, v in edges:
        edge_set.add((u, v))
        succ.setdefault(u, set()).add(v)

    out: Dict[Tuple, List[object]] = {}
    # snapshot: look-ups below must not mutate the adjacency while iterating
    for u, us in list(succ.items()):
        for v in list(us):
            for w in succ.get(v, ()):
                if w != u and (u, w) not in edge_set:
                    out.setdefault((u, w), []).append(v)
    return {k: sorted(set(v)) for k, v in out.items()}


def is_semi_allowed(path: Path, edge_set: set, semi: set) -> bool:
    """
    Whether an elementary path is *semi-allowed* (Section 4.1): among the
    consecutive pairs exactly one is a semi-edge and all the others are edges.
    """
    n_semi = 0
    for k in range(len(path) - 1):
        pair = (path[k], path[k + 1])
        if pair in edge_set:
            continue
        if pair in semi:
            n_semi += 1
            if n_semi > 1:
                return False
        else:
            return False
    return n_semi == 1


def omega_basis(
    paths: Dict[int, List[Path]],
    edges: Iterable[Tuple],
    p: int,
    *,
    exact: bool = True,
    method: str = "auto",
    regular: bool = False,
    field=None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Basis of ``Omega_p = {v in A_p : partial v in A_{p-1}}``  -- definition (3.8).

    Three equivalent implementations are provided so that they can be
    cross-checked against each other:

    ``"kernel"``
        brute force: the kernel of the composed map
        ``A_p --partial--> Lambda_{p-1} --project--> Lambda_{p-1}/A_{p-1}``.
        This is definition (3.8) read literally, and works for any path complex.
    ``"lemma41"``
        the linear conditions of Lemma 4.1: ``[v]_{i0...ip} = 0`` for every
        semi-allowed path, where the *deficiency* (4.1) is

            ``[v]_{i0...ip} = sum_k v_{i0...i_{q-1} k i_q...i_p}``

        summed over the bridges ``i_{q-1} k i_q`` that make the path allowed.
    ``"prop42"``
        for ``p = 2`` only: ``dim Omega_2 = |P_2| - |S|`` (Proposition 4.2).
        The constraints are ``sum_{abc : ac semi-edge} v_{abc} = 0``, one per
        semi-edge, which are linearly independent.

    ``"auto"`` picks ``"kernel"`` for robustness.

    ``regular=True`` uses the regular boundary operator ``partial^reg``, giving
    the regular spaces ``Omega^reg_p`` of Section 3.3.

    Returns
    -------
    (basis_vectors, condition_matrix)
        ``basis_vectors`` is a list of coefficient vectors in ``A_p``;
        ``condition_matrix`` is the matrix whose kernel is ``Omega_p``.
    """
    kind, modulus = resolve_field(field) if field is not None else (
        ("Q", None) if exact else ("R", None)
    )
    if p < 0:
        return [], np.zeros((0, 0), dtype=object)
    if p == 0:
        n = len(paths[0])
        eye = np.eye(n, dtype=dtype_for(kind))
        return [eye[:, i] for i in range(n)], np.zeros((0, n), dtype=object)
    if not paths.get(p):
        # A_p = 0, hence Omega_p = 0.
        return [], np.zeros((0, 0), dtype=dtype_for(kind))

    key = method if method != "auto" else "kernel"

    if key == "prop42" and p == 2 and paths.get(2):
        cond = _prop42_conditions(paths, edges, kind, modulus)
    elif key == "lemma41":
        cond = _lemma41_conditions(paths, edges, p, kind, modulus)
    else:
        cond = _kernel_conditions(
            paths, p, kind, modulus, edges=edges, regular=regular
        )

    ns = _nullspace(cond)
    basis = _columns(ns)
    return basis, cond


def _kernel_conditions(paths, p, kind="Q", modulus=None, *, edges=None, regular=False):
    """
    Matrix whose kernel is ``Omega_p``, read straight off definition (3.8).

    ``v in A_p`` lies in ``Omega_p`` iff ``partial v in A_{p-1}``.  Writing
    ``Lambda_{p-1} = A_{p-1} (+) C`` for a complement ``C``, this holds iff the
    ``C``-component of ``partial v`` vanishes.  We take ``C`` to be spanned by
    the *non-allowed* relevant elementary (p-1)-paths, so the constraint matrix
    is ``proj_C @ partial_p``.

    Rather than forming the dense projection and multiplying, we evaluate
    ``proj_C @ partial_p`` column by column: for each allowed p-path we compute
    its boundary, keep only the faces that are **not** allowed, and record
    their coefficients.  This is the identical matrix, built in time
    proportional to the number of boundary terms.
    """
    all_prev = _all_elementary_paths(paths, p - 1, edges, regular=regular)
    allowed_p = paths.get(p, [])
    dtype = dtype_for(kind)
    if not allowed_p:
        # No allowed p-paths => A_p = 0 => Omega_p = 0.
        return np.zeros((0, 0), dtype=dtype)
    allowed_prev = set(paths.get(p - 1, []))
    bad = {q for q in all_prev if q not in allowed_prev}
    one = make_scalar(1, kind, modulus)

    if not bad:
        # every relevant elementary (p-1)-path is allowed => partial always
        # lands in A_{p-1} => Omega_p = A_p.
        return np.zeros((0, len(allowed_p)), dtype=dtype)

    rows = {q: i for i, q in enumerate(sorted(bad))}
    mat = np.zeros((len(rows), len(allowed_p)), dtype=dtype)
    for j, path in enumerate(allowed_p):
        for face, coef in boundary_of_path(path, regular=regular):
            r = rows.get(face)
            if r is not None:
                mat[r, j] += coef * one
    return mat

def _all_elementary_paths(
    paths: Dict[int, List[Path]],
    p: int,
    edges: Optional[Iterable[Tuple]] = None,
    *,
    regular: bool = False,
) -> List[Path]:
    """
    The elementary p-paths on the vertex set that are **relevant** to
    definition (3.8) and to Lemma 4.1.

    By Lemma 4.1 a (p-1)-path ``i0...i_{p-1}`` can carry a non-trivial
    constraint only if it is *semi-allowed*: among its consecutive pairs exactly
    one is a non-edge, and that non-edge must be a **semi-edge**.  So the
    relevant paths are those built from the relation

        pair is usable  <=>  it is an edge, or it is a semi-edge,

    which is exactly the set enumerated here.  Any path with two or more
    non-edge pairs yields no constraint, since inserting a single vertex can
    eliminate at most one non-edge.

    For the *non-regular* chain complex a non-edge pair may also be a **loop**
    ``ii`` that is a face of an allowed path (e.g. ``e_{00}`` from
    ``partial e_{010}`` in Example 3.14).  Those pairs are added to the usable
    relation as well, so that the resulting constraint set is complete.
    """
    if edges is not None:
        edge_list = [(u, v) for u, v in edges]
        allowed_pairs = set(edge_list)
        usable = allowed_pairs | set(semi_edges(edge_list))
        if not regular:
            # Loops that occur as boundary faces must be constrained too.
            for path in paths.get(p + 1, []):
                for q in range(len(path)):
                    face = path[:q] + path[q + 1:]
                    for k in range(len(face) - 1):
                        if face[k] == face[k + 1]:
                            usable.add((face[k], face[k + 1]))
    else:
        allowed_pairs = set(paths.get(1, []))
        semi_pairs = set()
        for (a, b, c) in paths.get(2, []):
            if (a, c) not in allowed_pairs:
                semi_pairs.add((a, c))
        usable = allowed_pairs | semi_pairs

    verts = [v for (v,) in paths[0]]
    adj: Dict[object, List[object]] = defaultdict(list)
    for (u, v) in usable:
        adj[u].append(v)

    out: List[Path] = []

    def walk(prefix: Path):
        if len(prefix) == p + 1:
            out.append(prefix)
            return
        for w in adj[prefix[-1]]:
            walk(prefix + (w,))

    for v in verts:
        walk((v,))
    return out


def _lemma41_conditions(paths, edges, p, kind="Q", modulus=None):
    """
    The linear conditions of Lemma 4.1, for ``Omega_p``.

    Lemma 4.1 states: ``v in A_{p+1}`` belongs to ``Omega_{p+1}`` iff
    ``[v]_{i0...ip} = 0`` for every **semi-allowed p-path** ``i0...ip``.  So to
    constrain ``Omega_p`` we must range over semi-allowed paths of dimension
    ``p-1`` (a path with ``p`` vertices), whose unique semi-edge is
    ``i_{q-1} i_q``, and impose the deficiency (4.1)

        sum_{k : i_{q-1} k i_q allowed} v_{i0...i_{q-1} k i_q...ip-1} = 0.
    """
    edge_list = list(edges)
    edge_set = {(u, v) for u, v in edge_list}
    semi = set(semi_edges(edge_list))
    bridge_map = bridges(edge_list)
    index = {q: j for j, q in enumerate(paths[p])}
    dtype = dtype_for(kind)
    one = make_scalar(1, kind, modulus)

    # dimension of the semi-allowed paths carrying the conditions
    q_dim = p - 1

    rows: List[np.ndarray] = []
    for path in _all_elementary_paths(paths, q_dim, edge_list):
        if not is_semi_allowed(path, edge_set, semi):
            continue
        q = next(
            k for k in range(q_dim) if (path[k], path[k + 1]) not in edge_set
        )
        row = np.zeros(len(paths[p]), dtype=dtype)
        for k in bridge_map.get((path[q], path[q + 1]), ()):
            ext = path[: q + 1] + (k,) + path[q + 1:]
            j = index.get(ext)
            if j is not None:
                row[j] += one
        if any(_is_nonzero(x) for x in row):
            rows.append(row)
    if not rows:
        return np.zeros((0, len(paths[p])), dtype=dtype)
    return np.array(rows, dtype=dtype)


def _is_nonzero(x) -> bool:
    """``x != 0`` that works for Fractions, floats and GF elements alike."""
    return bool(x != 0)


def _prop42_conditions(paths, edges, kind="Q", modulus=None):
    """Proposition 4.2: one condition ``sum_{abc} v_abc = 0`` per semi-edge."""
    semi = semi_edges(edges)
    dtype = dtype_for(kind)
    one = make_scalar(1, kind, modulus)
    rows = []
    for (a, c) in semi:
        row = np.zeros(len(paths[2]), dtype=dtype)
        for j, (x, y, z) in enumerate(paths[2]):
            if (x, z) == (a, c):
                row[j] += one
        rows.append(row)
    if not rows:
        return np.zeros((0, len(paths[2])), dtype=dtype)
    return np.array(rows, dtype=dtype)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


class PathHomologyResult:
    """
    Container for everything the computation produces.

    Attributes
    ----------
    vertices : list
    edges : list of pairs
    allowed : dict {p: [allowed p-paths]}
    boundary_matrices : dict {p: ndarray}   matrix of ``partial : A_p -> A_{p-1}``
    omega_basis : dict {p: [ndarray]}       basis vectors of ``Omega_p`` in ``A_p``
    omega_paths : dict {p: [dict]}          the same vectors expanded as
                                            ``{path: coefficient}``
    omega_matrix : dict {p: ndarray}        basis of Omega_p as the columns of a
                                            matrix in the A_p basis
    cycles : dict {p: [dict]}               basis of ``ker partial|_{Omega_p}``
                                            (closed paths)
    boundaries : dict {p: [dict]}           basis of ``partial(Omega_{p+1})``
                                            (exact paths) inside ``Omega_p``
    betti : dict {p: int}                   dim H_p
    homology_basis : dict {p: [dict]}       representative closed paths modulo
                                            boundaries
    dim_omega : dict {p: int}
    dim_cycles, dim_boundaries : dict {p: int}
    euler_characteristic : int or None
    """

    def __init__(self, vertices, edges):
        self.vertices = list(vertices)
        self.edges = list(edges)
        #: name of the coefficient field, e.g. "Q" or "GF(2)"
        self.field: str = "Q"
        self.allowed: Dict[int, List[Path]] = {}
        self.boundary_matrices: Dict[int, np.ndarray] = {}
        self.omega_basis: Dict[int, List[np.ndarray]] = {}
        self.omega_matrix: Dict[int, np.ndarray] = {}
        self.omega_paths: Dict[int, List[Dict[Path, Scalar]]] = {}
        self.cycles: Dict[int, List[Dict[Path, Scalar]]] = {}
        self.boundaries: Dict[int, List[Dict[Path, Scalar]]] = {}
        self.homology_basis: Dict[int, List[Dict[Path, Scalar]]] = {}
        self.betti: Dict[int, int] = {}
        self.dim_omega: Dict[int, int] = {}
        self.dim_cycles: Dict[int, int] = {}
        self.dim_boundaries: Dict[int, int] = {}
        self.euler_characteristic: Optional[int] = None

    # -- pretty printing ---------------------------------------------------

    def _fmt(self, vec: Dict[Path, Scalar]) -> str:
        if not vec:
            return "0"
        parts = []
        for path in sorted(vec, key=lambda t: (len(t), t)):
            c = vec[path]
            if c == 0:
                continue
            name = "e" + "".join(str(x) for x in path) if path else "e"
            if c == 1:
                parts.append(f"+{name}" if parts else name)
            elif c == -1:
                parts.append(f"-{name}")
            else:
                parts.append(f"{'+' if c > 0 and parts else ''}{c}*{name}")
        return " ".join(parts) if parts else "0"

    def summary(self) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("Path complex P(G) of the digraph")
        lines.append("=" * 72)
        lines.append(f"  |V| = {len(self.vertices)}, |E| = {len(self.edges)}")
        lines.append(f"  coefficient field K = {self.field}")
        dim_max = max(self.allowed) if self.allowed else -1
        lines.append(f"  max dimension with A_p != 0 : p = {dim_max}")
        lines.append("")
        lines.append("-- Allowed paths  A_p (Section 3.2 / Example 3.3) --")
        for p in sorted(self.allowed):
            dim_a = len(self.allowed[p])
            lines.append(f"  A_{p} : dim = {dim_a}")
            shown = self.allowed[p][:12]
            txt = ", ".join("".join(str(x) for x in t) for t in shown)
            if len(self.allowed[p]) > len(shown):
                txt += f", ... (+{len(self.allowed[p]) - len(shown)} more)"
            lines.append(f"        {txt}")
        lines.append("")
        lines.append("-- Boundary matrices  partial : A_p -> A_(p-1) --")
        for p in sorted(self.boundary_matrices):
            m = self.boundary_matrices[p]
            lines.append(
                f"  partial_{p} : A_{p} -> A_{p - 1}"
                f"   shape = {m.shape[0]} x {m.shape[1]}"
                f"   rank = {_rank(m)}"
            )
        lines.append("")
        lines.append("-- Omega_p = {v in A_p : partial v in A_(p-1)}  (3.8) --")
        for p in sorted(self.dim_omega):
            lines.append(
                f"  Omega_{p} : dim A_{p} = {len(self.allowed.get(p, []))}"
                f"   dim Omega_{p} = {self.dim_omega[p]}"
            )
        lines.append("")
        lines.append("-- Chain complex Omega_* and its homology (3.9)-(3.11) --")
        lines.append(
            f"  {'p':>3} {'dim Omega_p':>12} {'dim ker d_p':>12}"
            f" {'dim im d_(p+1)':>16} {'dim H_p':>8}"
        )
        for p in sorted(self.dim_omega):
            lines.append(
                f"  {p:>3} {self.dim_omega[p]:>12} {self.dim_cycles.get(p, 0):>12}"
                f" {self.dim_boundaries.get(p, 0):>16} {self.betti.get(p, 0):>8}"
            )
        lines.append("")
        lines.append("-- Generators --")
        for p in sorted(self.dim_omega):
            if self.dim_cycles.get(p):
                lines.append(f"  Closed p-paths Z_{p} = ker(partial|Omega_{p}), dim = {self.dim_cycles[p]}:")
                for v in self.cycles[p]:
                    lines.append(f"      {self._fmt(v)}")
            if self.dim_boundaries.get(p):
                lines.append(
                    f"  Exact p-paths B_{p} = partial(Omega_{p + 1}), dim = {self.dim_boundaries[p]}:"
                )
                for v in self.boundaries[p]:
                    lines.append(f"      {self._fmt(v)}")
            if self.betti.get(p):
                lines.append(f"  H_{p} representatives, dim = {self.betti[p]}:")
                for v in self.homology_basis[p]:
                    lines.append(f"      {self._fmt(v)}")
            lines.append("")
        if self.euler_characteristic is not None:
            lines.append(
                f"  Euler characteristic  chi = sum_p (-1)^p dim H_p = "
                f"{self.euler_characteristic}"
            )
        return "\n".join(lines)

    def __repr__(self):  # pragma: no cover
        return (
            f"<PathHomologyResult |V|={len(self.vertices)} |E|={len(self.edges)} "
            f"betti={self.betti}>"
        )


def _solve_in_basis(basis: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Express each column of ``target`` in the basis given by the columns of
    ``basis``, i.e. find ``X`` with ``basis @ X = target``.

    ``basis`` is assumed to have full column rank (it comes from
    :func:`_nullspace`, so its columns are linearly independent and each column
    of ``target`` is assumed to lie in their span).  Returns ``X`` of shape
    ``(basis.shape[1], target.shape[1])``.
    """
    n_cols = basis.shape[1]
    # The arithmetic is dictated by the entries themselves: object dtype means
    # an exact field (Fraction or GF), float means floating point.
    exact = basis.dtype == object or target.dtype == object
    dtype = object if exact else float
    if target.size == 0:
        return np.zeros((n_cols, 0), dtype=dtype)
    if n_cols == 0:
        return np.zeros((0, target.shape[1]), dtype=dtype)

    # Solve the system by Gaussian elimination on the augmented matrix
    # [basis | target].  With an exact field and a consistent system this yields
    # the exact coordinates.
    aug = np.column_stack([np.array(basis, dtype=dtype), np.array(target, dtype=dtype)])
    n_rows = aug.shape[0]
    pivots: List[int] = []
    row = 0
    for col in range(n_cols):
        pivot = None
        if exact:
            for r in range(row, n_rows):
                if aug[r, col] != 0:
                    pivot = r
                    break
        else:
            best = 0.0
            for r in range(row, n_rows):
                v = abs(aug[r, col])
                if v > best:
                    best, pivot = v, r
            if pivot is not None and best <= 1e-9:
                pivot = None
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot], :] = aug[[pivot, row], :]
        aug[row, :] = aug[row, :] / aug[row, col]
        for r in range(n_rows):
            if r != row and aug[r, col] != 0:
                aug[r, :] = aug[r, :] - aug[r, col] * aug[row, :]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break
    return aug[:n_cols, n_cols:]


def _vec_to_dict(vec: np.ndarray, basis: Sequence[Path]) -> Dict[Path, Scalar]:
    out: Dict[Path, Scalar] = {}
    for c, path in zip(vec, basis):
        if c != 0:
            out[path] = c
    return out


def path_homology(
    vertices: Optional[Iterable] = None,
    edges: Optional[Iterable[Tuple]] = None,
    *,
    max_dim: Optional[int] = None,
    loops: bool = True,
    regular: bool = False,
    exact: bool = True,
    field=None,
    method: str = "auto",
    verbose: bool = False,
) -> PathHomologyResult:
    """
    Compute allowed paths, boundary matrices, ``Omega_p``, and path homology.

    This is the single entry point implementing the paper's construction:

    1. **Allowed paths** ``A_p`` -- Example 3.3.
    2. **Boundary operator** ``partial`` -- eq. (2.2), assembled into the
       matrices ``partial_p : A_p -> A_{p-1}``.
    3. **Omega_p** -- (3.8), the ``partial``-invariant p-paths.
    4. **Chain complex** ``Omega_*`` -- (3.9)/(3.10); its cycles
       ``Z_p = ker(partial|Omega_p)`` (closed paths) and boundaries
       ``B_p = partial(Omega_{p+1})`` (exact paths).
    5. **Path homology** ``H_p = Z_p / B_p`` -- (3.11).

    Parameters
    ----------
    vertices, edges : as in :func:`allowed_paths`.
    max_dim : int, optional
        Truncate path enumeration at this dimension.  Required for digraphs
        with directed cycles, where allowed paths exist in every dimension.
    loops : bool, default True
        Whether self-loops ``(v, v)`` present in ``edges`` are treated as
        edges of the digraph.
    regular : bool, default False
        ``False`` -- the plain (non-regular) chain complex (3.9)/(3.10).
        ``True``  -- the *regular* chain complex of Section 3.3, using
        ``partial^reg``, which drops non-regular faces such as ``e_{ii}``.
        For a strictly regular path complex (Definition 3.15) the two agree.
    exact : bool, default True
        Use exact rational arithmetic (``fractions.Fraction``).  Ignored when
        ``field`` is given.
    field : optional
        Coefficient field ``K``.  ``"Q"`` (exact rationals, default), ``"Z2"``
        / ``"GF2"`` / the integer ``2`` for the two-element field, any prime
        ``p`` for ``GF(p)``, or ``"R"`` for floating point.  The chain complex
        ``Omega_*`` is defined over an arbitrary field (Section 2), so the
        Betti numbers can in principle depend on the characteristic.
    method : {"auto", "kernel", "lemma41", "prop42"}
        How ``Omega_p`` is obtained; see :func:`omega_basis`.
    verbose : bool
        Print progress.

    Returns
    -------
    PathHomologyResult

    Examples
    --------
    The cycle-graph ``0->1->2->3->0`` (Proposition 4.7).  A square is neither a
    triangle nor has it a hole in this sense, so ``dim H_1 = 1``::

        >>> r = path_homology(edges=[(0,1),(1,2),(2,3),(3,0)])
        >>> r.betti[0], r.betti[1]
        (1, 1)

    The triangle ``0->1->2`` plus ``0->2`` instead has ``dim H_1 = 0``::

        >>> r = path_homology(edges=[(0,1),(1,2),(0,2)])
        >>> r.betti[0], r.betti[1], r.betti[2]
        (1, 0, 0)

    Over the two-element field::

        >>> r = path_homology(edges=[(0,1),(1,2),(2,3),(3,0)], field="Z2")
        >>> r.betti[1]
        1
    """
    kind, modulus = resolve_field(field) if field is not None else (
        ("Q", None) if exact else ("R", None)
    )
    res = PathHomologyResult(vertices or [], edges or [])
    res.field = "Q" if kind == "Q" else (f"GF({modulus})" if kind == "GF" else "R")
    # When loops are disabled the self-loops are removed from the edge relation
    # itself, so every downstream step (semi-edges, bridges, ambient paths) sees
    # the loopless digraph.
    edge_list = [(u, v) for (u, v) in (edges or ()) if loops or u != v]
    res.edges = edge_list

    paths = allowed_paths(vertices, edge_list, max_dim=max_dim, loops=loops)
    res.allowed = paths
    if vertices is None:
        res.vertices = [v for (v,) in paths[0]]
    else:
        res.vertices = list(vertices)

    dim_max = max(paths) if paths else 0

    # ---- boundary matrices -------------------------------------------------
    # A_{-1} = K is one-dimensional, spanned by the empty path e = ().
    for p in range(0, dim_max + 1):
        prev = [()] if p == 0 else paths.get(p - 1, [])
        res.boundary_matrices[p] = boundary_matrix(
            paths[p], prev, regular=regular, field=field if field is not None
            else ("Q" if exact else "R")
        )
    if verbose:
        print(f"[path_homology] allowed paths up to dimension {dim_max}")

    # ---- Omega_p -----------------------------------------------------------
    dtype = dtype_for(kind)
    for p in range(0, dim_max + 1):
        basis, _ = omega_basis(
            paths, edge_list, p, method=method, regular=regular,
            field=field if field is not None else ("Q" if exact else "R"),
        )
        res.omega_basis[p] = basis
        n = len(paths[p])
        mat = (
            np.column_stack(basis)
            if basis
            else np.zeros((n, 0), dtype=dtype)
        )
        res.omega_matrix[p] = mat
        res.omega_paths[p] = [_vec_to_dict(b, paths[p]) for b in basis]
        res.dim_omega[p] = len(basis)
    # Omega_p = 0 beyond dim_max (there are no allowed paths there)
    res.dim_omega[dim_max + 1] = 0
    res.omega_matrix[dim_max + 1] = np.zeros((0, 0), dtype=dtype)

    # ---- differentials between the Omega bases ----------------------------
    # d_p : Omega_p -> Omega_{p-1}, expressed in the Omega bases.  We obtain it
    # by mapping Omega_p into A_{p-1} via partial_p, then expressing the result
    # in the Omega_{p-1} basis.
    #
    # partial_p has len(paths[p-1]) rows (one per allowed (p-1)-path) while the
    # Omega_{p-1} basis has dim_omega[p-1] columns spanning a subspace of that
    # A_{p-1}; Omega_{p-1} = A_{p-1} for p-1 <= 1, and in general the image of a
    # partial-invariant path is itself partial-invariant, so the change of basis
    # is well defined.  We solve for the coordinates in the Omega basis.
    d: Dict[int, np.ndarray] = {}
    for p in range(1, dim_max + 1):
        img = res.boundary_matrices[p] @ res.omega_matrix[p]   # in A_{p-1} coords
        om_prev = res.omega_matrix.get(p - 1)
        if om_prev is None or om_prev.shape[1] == 0 or img.size == 0:
            d[p] = np.zeros((0, img.shape[1] if img.size else 0), dtype=dtype)
            continue
        d[p] = _solve_in_basis(om_prev, img)
    if dim_max >= 1:
        d[0] = np.zeros((0, res.dim_omega[0]), dtype=dtype)

    # ---- cycles, boundaries, homology -------------------------------------
    for p in range(0, dim_max + 1):
        dp = d.get(p, np.zeros((0, res.dim_omega[p]), dtype=dtype))
        if dp.shape[0] == 0:
            dp = np.zeros((0, res.dim_omega[p]), dtype=dtype)
        z = _nullspace(dp)
        res.dim_cycles[p] = z.shape[1]
        res.cycles[p] = [
            _vec_to_dict(res.omega_matrix[p] @ z[:, j], paths[p])
            for j in range(z.shape[1])
        ]

        # B_p = im(d_{p+1}) inside Omega_p
        if p + 1 <= dim_max and (p + 1) in d:
            b = d[p + 1]
        else:
            b = np.zeros((res.dim_omega[p], 0), dtype=dtype)
        res.dim_boundaries[p] = _rank(b) if b.size else 0
        res.boundaries[p] = [
            _vec_to_dict(res.omega_matrix[p] @ b[:, j], paths[p])
            for j in range(b.shape[1])
            if any(_is_nonzero(c) for c in b[:, j])
        ]
        res.betti[p] = res.dim_cycles[p] - res.dim_boundaries[p]

        # Representatives of H_p = Z_p / B_p.
        #
        # We are given generators z (columns) of Z_p and generators b (columns)
        # of B_p, all in the same coordinates (the Omega_p basis).  A set of
        # classes generates Z_p/B_p iff adjoining their representatives to b
        # raises the rank of the span.
        #
        # Doing that one column at a time costs O(nullity) eliminations, each
        # O(n^3).  Instead we eliminate ONCE on the stacked matrix [b | z]: the
        # pivot columns falling inside the z block are exactly the independent
        # representatives modulo colspace(b).
        reps: List[Dict[Path, Scalar]] = []
        if res.betti[p] > 0 and z.shape[1] > 0:
            nb = b.shape[1] if b.size else 0
            stacked = np.column_stack([b, z]) if nb else z
            for c in _pivot_columns(stacked):
                if c >= nb:
                    reps.append(
                        _vec_to_dict(res.omega_matrix[p] @ z[:, c - nb], paths[p])
                    )
                if len(reps) == res.betti[p]:
                    break
        res.homology_basis[p] = reps

    # ---- Euler characteristic (3.14)/(3.16) -------------------------------
    # By (3.16) chi = sum_p (-1)^p dim Omega_p, valid because dim Omega_p = 0
    # for p > dim_max.  We record it unconditionally; the alternating sum of
    # the Betti numbers equals it by (3.13).
    if res.dim_omega:
        res.euler_characteristic = sum(
            (-1) ** p * res.dim_omega[p]
            for p in sorted(res.dim_omega)
            if p <= dim_max
        )
    if verbose:
        print(f"[path_homology] betti numbers: {res.betti}")
    return res
