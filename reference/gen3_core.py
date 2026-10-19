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
        if isinstance(o, (int, np.integer, bool)):
            return self.v == int(o) % self.p
        try:
            return self.v == self._coerce(o)
        except Exception:
            return NotImplemented

    def __ne__(self, o):
        if isinstance(o, GF):
            return self.p != o.p or self.v != o.v
        if isinstance(o, (int, np.integer, bool)):
            return self.v != int(o) % self.p
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
        if not _is_prime(field):
            raise ValueError(f"GF({field}) is not a field because {field} is not prime")
        return "GF", field
    if isinstance(field, str):
        key = field.strip().lower()
        if key in FIELD_ALIASES:
            k = FIELD_ALIASES[key]
            return (k, None) if k in ("Q", "R") else ("GF", 2)
        if key.isdigit():
            modulus = int(key)
            if not _is_prime(modulus):
                raise ValueError(
                    f"GF({modulus}) is not a field because {modulus} is not prime"
                )
            return "GF", modulus
        raise ValueError(f"unknown field {field!r}; use 'Q', 'Z2', a prime, or 'R'")
    raise TypeError(f"unsupported field specification: {field!r}")


def _is_prime(n: int) -> bool:
    """Deterministic Miller--Rabin primality test for machine-size integers."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small:
        return True
    if any(n % p == 0 for p in small):
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2
    # This base set is deterministic for every unsigned 64-bit integer.  For
    # larger Python integers it remains an excellent probable-prime test.
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


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


def _field_zeros(shape, kind, modulus=None):
    """A zero matrix whose entries retain the requested coefficient field."""
    if kind == "GF":
        out = np.empty(shape, dtype=object)
        out.fill(GF(0, modulus))
        return out
    return np.zeros(shape, dtype=dtype_for(kind))


def _field_eye(n, kind, modulus=None):
    """An identity matrix whose object entries are genuine field elements."""
    out = _field_zeros((n, n), kind, modulus)
    one = make_scalar(1, kind, modulus)
    for i in range(n):
        out[i, i] = one
    return out

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


def _gf_modulus(mat: np.ndarray):
    """Return the modulus carried by a GF object array, if it has one."""
    if mat.dtype != object or mat.size == 0:
        return None
    # Matrices constructed here are homogeneous and field-filled, including
    # their zero entries.  Looking at one element avoids repeatedly scanning a
    # whole rational matrix merely to establish that it is not finite-field.
    first = mat.flat[0]
    return first.p if isinstance(first, GF) else None


def _is_gf2_array(mat: np.ndarray) -> bool:
    """Whether ``mat`` holds elements of GF(2)."""
    return _gf_modulus(mat) == 2


def _as_bits(mat: np.ndarray) -> np.ndarray:
    """Object array of GF(2) elements -> uint8 0/1 array."""
    out = np.empty(mat.shape, dtype=np.uint8)
    flat_in = mat.ravel()
    flat_out = out.ravel()
    for i, x in enumerate(flat_in):
        flat_out[i] = x.v if isinstance(x, GF) else (int(x) & 1)
    return out


def _scalar_mod(x, p: int) -> int:
    """Convert an integer/rational/GF scalar to its residue modulo ``p``."""
    if isinstance(x, GF):
        if x.p != p:
            raise ValueError(f"cannot mix GF({x.p}) with GF({p})")
        return x.v
    if isinstance(x, Fraction):
        den = x.denominator % p
        if den == 0:
            raise ZeroDivisionError(f"denominator is zero in GF({p})")
        return (x.numerator % p) * pow(den, p - 2, p) % p
    return int(x) % p


def _as_mod_array(mat: np.ndarray, p: int, *, inner: int = 1) -> np.ndarray:
    """Convert a homogeneous GF(p) matrix to fast integer residues."""
    # Native int64 arithmetic is exact provided no dot-product accumulation can
    # overflow.  Fall back to Python integers for very large primes/matrices.
    bound = max(1, inner) * max(1, p - 1) * max(1, p - 1)
    dtype = np.int64 if bound <= np.iinfo(np.int64).max else object
    out = np.empty(mat.shape, dtype=dtype)
    src, dst = mat.ravel(), out.ravel()
    for i, x in enumerate(src):
        dst[i] = _scalar_mod(x, p)
    return out


def _from_mod_array(values: np.ndarray, p: int) -> np.ndarray:
    """Convert an integer residue matrix back to the public GF object form."""
    out = np.empty(values.shape, dtype=object)
    src, dst = values.ravel(), out.ravel()
    for i, x in enumerate(src):
        dst[i] = GF(int(x), p)
    return out


def _is_identity_matrix(mat: np.ndarray) -> bool:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return False
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if mat[i, j] != (1 if i == j else 0):
                return False
    return True


def _integral_int64(mat: np.ndarray):
    """Return an int64 copy when every exact entry is an integral scalar."""
    if mat.dtype != object:
        return None
    out = np.empty(mat.shape, dtype=np.int64)
    src, dst = mat.ravel(), out.ravel()
    limit = np.iinfo(np.int64).max
    for i, x in enumerate(src):
        if isinstance(x, GF):
            return None
        if isinstance(x, Fraction):
            if x.denominator != 1:
                return None
            value = x.numerator
        elif isinstance(x, (int, np.integer, bool)):
            value = int(x)
        else:
            return None
        if not -limit <= value <= limit:
            return None
        dst[i] = value
    return out


def _from_integer_array(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.shape, dtype=object)
    src, dst = values.ravel(), out.ravel()
    for i, x in enumerate(src):
        dst[i] = Fraction(int(x))
    return out


def _field_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Matrix product with C-level fast paths for GF(p) and integral Q data."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("incompatible matrix dimensions")
    if left.shape[1] == 0:
        dtype = object if left.dtype == object or right.dtype == object else float
        return np.zeros((left.shape[0], right.shape[1]), dtype=dtype)
    if _is_identity_matrix(left):
        return np.array(right, copy=True)
    if _is_identity_matrix(right):
        return np.array(left, copy=True)

    p_left, p_right = _gf_modulus(left), _gf_modulus(right)
    if p_left is not None or p_right is not None:
        p = p_left if p_left is not None else p_right
        if p_left is not None and p_right is not None and p_left != p_right:
            raise ValueError(f"cannot multiply GF({p_left}) by GF({p_right})")
        inner = left.shape[1]
        a = _as_mod_array(left, p, inner=inner)
        b = _as_mod_array(right, p, inner=inner)
        product = (a @ b) % p
        return _from_mod_array(product, p)

    if left.dtype != object and right.dtype != object:
        return left @ right

    # Path-complex matrices and the canonical nullspace bases are very often
    # integral even over Q.  NumPy can multiply them natively, avoiding millions
    # of Fraction method calls.  Check an overflow bound before doing so.
    a, b = _integral_int64(left), _integral_int64(right)
    if a is not None and b is not None:
        max_a = int(np.max(np.abs(a))) if a.size else 0
        max_b = int(np.max(np.abs(b))) if b.size else 0
        bound = left.shape[1] * max_a * max_b
        if bound <= np.iinfo(np.int64).max:
            return _from_integer_array(a @ b)
    return left @ right


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


def _primitive_integer_row(row: List[int], start: int = 0) -> List[int]:
    """Divide an integer row by its content to control coefficient growth."""
    import math

    content = 0
    for value in row[start:]:
        content = math.gcd(content, abs(value))
        if content == 1:
            return row
    if content > 1:
        for j in range(start, len(row)):
            row[j] //= content
    return row


def _integer_rows(mat: np.ndarray):
    """Clear row denominators of a Q-matrix, or return None if not rational."""
    import math

    rows: List[List[int]] = []
    for source in mat:
        denominator = 1
        for x in source:
            if isinstance(x, Fraction):
                denominator = math.lcm(denominator, x.denominator)
            elif isinstance(x, (int, np.integer, bool)):
                continue
            else:
                return None
        if denominator == 1:
            row = [
                x.numerator if isinstance(x, Fraction) else int(x)
                for x in source
            ]
        else:
            row = [
                x.numerator * (denominator // x.denominator)
                if isinstance(x, Fraction)
                else int(x) * denominator
                for x in source
            ]
        rows.append(_primitive_integer_row(row))
    return rows


def _fraction_free_echelon(rows: List[List[int]], n_cols: int):
    """Exact row echelon form over Q using primitive integer row operations."""
    import math

    A = [row[:] for row in rows]
    n_rows = len(A)
    pivots: List[int] = []
    rank = 0
    for col in range(n_cols):
        pivot = min(
            (i for i in range(rank, n_rows) if A[i][col]),
            key=lambda i: abs(A[i][col]),
            default=None,
        )
        if pivot is None:
            continue
        if pivot != rank:
            A[rank], A[pivot] = A[pivot], A[rank]
        pivot_row = A[rank]
        pivot_value = pivot_row[col]
        for i in range(rank + 1, n_rows):
            value = A[i][col]
            if not value:
                continue
            common = math.gcd(abs(pivot_value), abs(value))
            pivot_scale = pivot_value // common
            row_scale = value // common
            row = A[i]
            for j in range(col + 1, n_cols):
                row[j] = pivot_scale * row[j] - row_scale * pivot_row[j]
            row[col] = 0
            _primitive_integer_row(row, col + 1)
        pivots.append(col)
        rank += 1
        if rank == n_rows:
            break
    return A, pivots


def _q_rank(mat: np.ndarray):
    rows = _integer_rows(mat)
    if rows is None:
        return None
    return len(_fraction_free_echelon(rows, mat.shape[1])[1])


def _q_nullspace(mat: np.ndarray):
    """Canonical exact Q-nullspace via integer echelon + rational backsolve."""
    rows = _integer_rows(mat)
    if rows is None:
        return None
    n_cols = mat.shape[1]
    echelon, pivots = _fraction_free_echelon(rows, n_cols)
    pivot_set = set(pivots)
    free = [col for col in range(n_cols) if col not in pivot_set]
    zero = Fraction(0)
    basis = np.empty((n_cols, len(free)), dtype=object)
    basis.fill(zero)
    one = Fraction(1)
    for out_col, free_col in enumerate(free):
        basis[free_col, out_col] = one

    # Solve all free-variable right-hand sides together.  The former scalar
    # loop scanned every free column once for every basis vector, even though
    # the free-coordinate block is the identity.  Initialising that block
    # directly and accumulating already-solved pivot rows removes those
    # quadratic zero tests while preserving the same canonical basis.
    for i in range(len(pivots) - 1, -1, -1):
        pivot = pivots[i]
        row = echelon[i]
        totals = [
            Fraction(row[column]) if row[column] else zero
            for column in free
        ]
        for solved_pivot in pivots[i + 1:]:
            coefficient = row[solved_pivot]
            if not coefficient:
                continue
            solved = basis[solved_pivot, :]
            for out_col, value in enumerate(solved):
                if value:
                    totals[out_col] += coefficient * value
        pivot_value = row[pivot]
        for out_col, total in enumerate(totals):
            if total:
                basis[pivot, out_col] = -total / pivot_value
    return basis


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
    modulus = _gf_modulus(mat)
    if modulus == 2:
        return _gf2_rank(_as_bits(mat))
    if modulus is not None:
        return _gfp_rank(_as_mod_array(mat, modulus), modulus)
    if mat.dtype == object:
        rational_rank = _q_rank(mat)
        if rational_rank is not None:
            return rational_rank
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
    modulus = _gf_modulus(mat)
    if modulus == 2:
        bits = _gf2_nullspace(_as_bits(mat))
        return _from_mod_array(bits, 2)
    if modulus is not None:
        return _from_mod_array(
            _gfp_nullspace(_as_mod_array(mat, modulus), modulus), modulus
        )
    if mat.dtype == object:
        rational_basis = _q_nullspace(mat)
        if rational_basis is not None:
            return rational_basis
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
    modulus = _gf_modulus(mat)
    if modulus == 2:
        return _gf2_pivot_columns(_as_bits(mat))
    if modulus is not None:
        return _gfp_pivot_columns(_as_mod_array(mat, modulus), modulus)
    if mat.dtype == object:
        rows = _integer_rows(mat)
        if rows is not None:
            return _fraction_free_echelon(rows, mat.shape[1])[1]
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


def _gfp_rref(values: np.ndarray, p: int):
    """Reduced row echelon form over GF(p), stored as integer residues."""
    A = np.array(values, copy=True)
    m, n = A.shape
    pivots: List[int] = []
    row = 0
    for col in range(n):
        candidates = np.nonzero(A[row:, col])[0]
        if not candidates.size:
            continue
        pivot = row + int(candidates[0])
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        inv = pow(int(A[row, col]), p - 2, p)
        A[row] = (A[row] * inv) % p
        rows = np.nonzero(A[:, col])[0]
        rows = rows[rows != row]
        if rows.size:
            factors = A[rows, col].copy()
            A[rows] = (A[rows] - factors[:, None] * A[row]) % p
        pivots.append(col)
        row += 1
        if row == m:
            break
    return A, pivots


def _gfp_rank(values: np.ndarray, p: int) -> int:
    return len(_gfp_rref(values, p)[1])


def _gfp_nullspace(values: np.ndarray, p: int) -> np.ndarray:
    n_rows, n_cols = values.shape
    A, pivots = _gfp_rref(values, p)
    pivot_set = set(pivots)
    free = [c for c in range(n_cols) if c not in pivot_set]
    dtype = A.dtype
    basis = np.zeros((n_cols, len(free)), dtype=dtype)
    for j, f in enumerate(free):
        basis[f, j] = 1
        for i, pivot in enumerate(pivots):
            basis[pivot, j] = (-A[i, f]) % p
    return basis


def _gfp_pivot_columns(values: np.ndarray, p: int) -> List[int]:
    return _gfp_rref(values, p)[1]


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
    # Deduplicate the edge relation before enumeration.  Repeated input edges
    # used to duplicate every descendant path and were removed only after the
    # full breadth-first expansion, which can amplify one duplicate
    # exponentially with ``max_dim``.
    adj_sets: Dict[object, set] = defaultdict(set)
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
            adj_sets[u].add(v)

    adj = {u: tuple(vs) for u, vs in adj_sets.items()}

    if max_dim is None:
        max_dim = DEFAULT_MAX_DIM

    paths: Dict[int, List[Path]] = {0: [(v,) for v in labels]}
    dim = 0
    while paths[dim] and dim < max_dim:
        nxt: List[Path] = []
        for p in paths[dim]:
            for w in adj.get(p[-1], ()):
                nxt.append(p + (w,))
        if not nxt:
            break
        dim += 1
        paths[dim] = nxt
    paths = {p: v for p, v in paths.items() if v}
    for p in paths:
        paths[p] = sorted(paths[p], key=lambda t: (len(t), t))
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
    mat = _field_zeros((len(paths_prev), len(paths_p)), kind, modulus)
    for j, p in enumerate(paths_p):
        for face, coef in boundary_of_path(p, regular=regular):
            if face in index:
                mat[index[face], j] += make_scalar(coef, kind, modulus)
    return mat


def _sparse_boundary_rows(paths_p: Sequence[Path], *, regular: bool = False):
    """Full boundary as ``face -> {column: integer coefficient}`` rows."""
    rows: Dict[Path, Dict[int, int]] = defaultdict(dict)
    for column, path in enumerate(paths_p):
        for face, coefficient in boundary_of_path(path, regular=regular):
            row = rows[face]
            value = row.get(column, 0) + coefficient
            if value:
                row[column] = value
            else:
                row.pop(column, None)
    return rows


def _gf2_boundary_bitrows(paths_p: Sequence[Path], *, regular: bool = False):
    """GF(2) boundary as ``face -> packed integer row``.

    Building the packed rows directly avoids first allocating one dictionary
    entry for every nonzero matrix coefficient.  Repeated equal faces cancel
    by XOR, exactly as their signed integer coefficients do modulo two.
    """
    rows: Dict[Path, int] = {}
    for column, path in enumerate(paths_p):
        bit = 1 << column
        for q in range(len(path)):
            face = path[:q] + path[q + 1:]
            if regular and not is_regular_path(face):
                continue
            value = rows.get(face, 0) ^ bit
            if value:
                rows[face] = value
            else:
                rows.pop(face, None)
    return rows


def _primitive_sparse_row(row: Dict[int, int]) -> Dict[int, int]:
    import math

    content = 0
    for value in row.values():
        content = math.gcd(content, abs(value))
        if content == 1:
            return row
    if content > 1:
        for column in row:
            row[column] //= content
    return row


def _sparse_rank(rows, n_cols: int, kind: str, modulus=None) -> int:
    """Rank from sparse integer rows over Q, GF(p), or floating point."""
    if kind == "GF" and modulus == 2:
        pivots: Dict[int, int] = {}
        for source in rows:
            if isinstance(source, int):
                bits = source
            else:
                bits = 0
                for column, value in source.items():
                    if value & 1:
                        bits ^= 1 << column
            while bits:
                pivot = bits.bit_length() - 1
                previous = pivots.get(pivot)
                if previous is None:
                    pivots[pivot] = bits
                    break
                bits ^= previous
        return len(pivots)

    if kind == "GF":
        p = modulus
        pivots: Dict[int, Dict[int, int]] = {}
        for source in rows:
            row = {c: v % p for c, v in source.items() if v % p}
            while row:
                pivot = min(row)
                previous = pivots.get(pivot)
                if previous is None:
                    inverse = pow(row[pivot], p - 2, p)
                    pivots[pivot] = {
                        c: (v * inverse) % p for c, v in row.items()
                    }
                    break
                factor = row[pivot]
                for column, value in previous.items():
                    updated = (row.get(column, 0) - factor * value) % p
                    if updated:
                        row[column] = updated
                    else:
                        row.pop(column, None)
        return len(pivots)

    if kind == "Q":
        import math

        pivots: Dict[int, Dict[int, int]] = {}
        for source in rows:
            row = _primitive_sparse_row(dict(source))
            while row:
                pivot = min(row)
                previous = pivots.get(pivot)
                if previous is None:
                    if row[pivot] < 0:
                        row = {c: -v for c, v in row.items()}
                    pivots[pivot] = row
                    break
                value, pivot_value = row[pivot], previous[pivot]
                common = math.gcd(abs(value), abs(pivot_value))
                row_scale = pivot_value // common
                pivot_scale = value // common
                for column in row:
                    row[column] *= row_scale
                for column, previous_value in previous.items():
                    updated = row.get(column, 0) - pivot_scale * previous_value
                    if updated:
                        row[column] = updated
                    else:
                        row.pop(column, None)
                _primitive_sparse_row(row)
        return len(pivots)

    # Sparse numerical elimination for field="R".
    pivots: Dict[int, Dict[int, float]] = {}
    tolerance = 1e-10
    for source in rows:
        row = {c: float(v) for c, v in source.items() if v}
        while row:
            pivot = min(row)
            value = row[pivot]
            if abs(value) <= tolerance:
                del row[pivot]
                continue
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = {c: v / value for c, v in row.items()}
                break
            columns = set(row) | set(previous)
            row = {
                c: row.get(c, 0.0) - value * previous.get(c, 0.0)
                for c in columns
            }
            row = {c: v for c, v in row.items() if abs(v) > tolerance}
    return len(pivots)


def _q_sparse_rank_certified(rows, n_cols: int, upper_bound=None) -> int:
    """Exact sparse rank over Q, with a cheap characteristic-two certificate.

    Reduction modulo two can only decrease the rank of an integer matrix.  If
    that lower bound reaches a proven upper bound, the rational rank is known
    exactly and expensive integer elimination is unnecessary.  Otherwise we
    fall back to the fraction-free sparse algorithm, so this optimisation is
    deterministic and never turns the result into a probabilistic claim.
    """
    materialized = [row for row in rows if row]
    if not materialized or n_cols == 0:
        return 0
    upper = min(len(materialized), n_cols)
    if upper_bound is not None:
        upper = min(upper, int(upper_bound))
    lower = _sparse_rank(materialized, n_cols, "GF", 2)
    if lower == upper:
        return lower
    return _sparse_rank(materialized, n_cols, "Q")


def _incidence_rank(paths_0: Sequence[Path], paths_1: Sequence[Path]) -> int:
    """Exact rank of the directed incidence matrix over any characteristic 0 field."""
    vertices = [path[0] for path in paths_0]
    parent = {vertex: vertex for vertex in vertices}

    def find(vertex):
        root = vertex
        while parent[root] != root:
            root = parent[root]
        while parent[vertex] != vertex:
            following = parent[vertex]
            parent[vertex] = root
            vertex = following
        return root

    for path in paths_1:
        source, target = path
        if source == target:
            continue
        root_source, root_target = find(source), find(target)
        if root_source != root_target:
            parent[root_target] = root_source
    components = len({find(vertex) for vertex in vertices})
    return len(vertices) - components


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
    # Enumerate endpoints of actual two-step paths instead of testing every
    # ordered pair of vertices.  The keys of bridges are precisely semi-edges.
    return sorted(bridges(edges))


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
    if p < 0 or not paths.get(p):
        return [], _field_zeros((0, 0), kind, modulus)
    if p == 0:
        n = len(paths[0])
        eye = _field_eye(n, kind, modulus)
        return [eye[:, i] for i in range(n)], _field_zeros(
            (0, n), kind, modulus
        )

    cond = _omega_conditions(
        paths, edges, p, kind, modulus, method=method, regular=regular
    )

    ns = _nullspace(cond)
    basis = _columns(ns)
    return basis, cond


def _omega_conditions(
    paths, edges, p, kind="Q", modulus=None, *, method="auto", regular=False
):
    """Build only the constraint matrix defining Omega_p, without its kernel."""
    if p <= 0:
        return _field_zeros((0, len(paths.get(p, []))), kind, modulus)
    if not paths.get(p):
        return _field_zeros((0, 0), kind, modulus)
    key = method if method != "auto" else "kernel"
    if key == "prop42" and p == 2 and paths.get(2):
        return _prop42_conditions(paths, edges, kind, modulus)
    if key == "lemma41":
        return _lemma41_conditions(paths, edges, p, kind, modulus)
    return _kernel_conditions(
        paths, p, kind, modulus, edges=edges, regular=regular
    )


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
    allowed_p = paths.get(p, [])
    if not allowed_p:
        # No allowed p-paths => A_p = 0 => Omega_p = 0.
        return _field_zeros((0, 0), kind, modulus)
    allowed_prev = set(paths.get(p - 1, []))
    one = make_scalar(1, kind, modulus)

    # A constraint row can occur only as an actual boundary face of an allowed
    # p-path.  Enumerating all paths made from edges and semi-edges first (the
    # old implementation) explores many candidates with two or more bad pairs,
    # although such candidates can never be a face here.  Collecting the faces
    # directly is complete by definition and costs O((p+1)|A_p|).
    bad = set()
    for path in allowed_p:
        for face, _ in boundary_of_path(path, regular=regular):
            if face not in allowed_prev:
                bad.add(face)

    if not bad:
        # every relevant elementary (p-1)-path is allowed => partial always
        # lands in A_{p-1} => Omega_p = A_p.
        return _field_zeros((0, len(allowed_p)), kind, modulus)

    rows = {q: i for i, q in enumerate(sorted(bad))}
    mat = _field_zeros((len(rows), len(allowed_p)), kind, modulus)
    for j, path in enumerate(allowed_p):
        for face, coef in boundary_of_path(path, regular=regular):
            r = rows.get(face)
            if r is not None:
                mat[r, j] = mat[r, j] + coef * one
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
    dtype = dtype_for(kind)
    one = make_scalar(1, kind, modulus)

    # A semi-allowed (p-1)-path contributes iff it is a face of an allowed
    # p-path.  Group those actual faces directly; this is the same deficiency
    # sum as Lemma 4.1 without enumerating the much larger edge|semi-edge walk
    # space and filtering it afterwards.
    grouped: Dict[Path, List[int]] = defaultdict(list)
    for j, allowed in enumerate(paths[p]):
        for face, _ in boundary_of_path(allowed):
            if is_semi_allowed(face, edge_set, semi):
                grouped[face].append(j)
    if not grouped:
        return _field_zeros((0, len(paths[p])), kind, modulus)
    mat = _field_zeros((len(grouped), len(paths[p])), kind, modulus)
    for i, face in enumerate(sorted(grouped)):
        for j in grouped[face]:
            mat[i, j] = mat[i, j] + one
    return np.array(mat, dtype=dtype)


def _is_nonzero(x) -> bool:
    """``x != 0`` that works for Fractions, floats and GF elements alike."""
    if isinstance(x, GF):
        return x.v != 0
    return bool(x != 0)


def _prop42_conditions(paths, edges, kind="Q", modulus=None):
    """Proposition 4.2: one condition ``sum_{abc} v_abc = 0`` per semi-edge."""
    semi = semi_edges(edges)
    one = make_scalar(1, kind, modulus)
    if not semi:
        return _field_zeros((0, len(paths[2])), kind, modulus)
    row_index = {pair: i for i, pair in enumerate(semi)}
    mat = _field_zeros((len(semi), len(paths[2])), kind, modulus)
    # Group paths by their skipped edge in one pass instead of scanning all
    # 2-paths once for every semi-edge.
    for j, (a, _, c) in enumerate(paths[2]):
        i = row_index.get((a, c))
        if i is not None:
            mat[i, j] = mat[i, j] + one
    return mat


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
        #: whether explicit Omega/cycle/boundary/homology bases were built
        self.generators_materialized: bool = True
        #: whether dense boundary matrices are present in boundary_matrices
        self.matrices_materialized: bool = True
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
            elif isinstance(c, GF):
                parts.append(f"{'+' if parts else ''}{c}*{name}")
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
        if not self.generators_materialized:
            lines.append("  not materialized (dimensions-only fast mode)")
            lines.append("")
            if self.euler_characteristic is not None:
                lines.append(
                    f"  Euler characteristic  chi = sum_p (-1)^p dim H_p = "
                    f"{self.euler_characteristic}"
                )
            return "\n".join(lines)
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


def _coordinate_rows(basis: np.ndarray):
    """Rows on which a canonical nullspace basis restricts to the identity."""
    n_cols = basis.shape[1]
    if n_cols == 0:
        return []
    rows = [None] * n_cols
    for r in range(basis.shape[0]):
        unit_col = None
        for c in range(n_cols):
            value = basis[r, c]
            if value == 0:
                continue
            if value != 1 or unit_col is not None:
                unit_col = None
                break
            unit_col = c
        if unit_col is not None and rows[unit_col] is None:
            rows[unit_col] = r
    return None if any(r is None for r in rows) else rows


def _gfp_solve_in_basis(basis: np.ndarray, target: np.ndarray, p: int) -> np.ndarray:
    """Solve ``basis @ X = target`` by vectorised elimination over GF(p)."""
    n_cols = basis.shape[1]
    a = _as_mod_array(basis, p)
    b = _as_mod_array(target, p)
    aug = np.column_stack([a, b])
    row = 0
    for col in range(n_cols):
        candidates = np.nonzero(aug[row:, col])[0]
        if not candidates.size:
            raise ValueError("basis does not have full column rank")
        pivot = row + int(candidates[0])
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        inv = pow(int(aug[row, col]), p - 2, p)
        aug[row] = (aug[row] * inv) % p
        rows = np.nonzero(aug[:, col])[0]
        rows = rows[rows != row]
        if rows.size:
            factors = aug[rows, col].copy()
            aug[rows] = (aug[rows] - factors[:, None] * aug[row]) % p
        row += 1
    return _from_mod_array(aug[:n_cols, n_cols:] % p, p)


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

    # Every basis produced by _nullspace has an identity block on its free
    # coordinates.  Reading those target rows gives the coordinates directly,
    # eliminating the cubic change-of-basis solve in the common path.
    coordinate_rows = _coordinate_rows(basis)
    if coordinate_rows is not None:
        return np.array(target[coordinate_rows, :], copy=True)

    modulus = _gf_modulus(basis) or _gf_modulus(target)
    if modulus is not None:
        return _gfp_solve_in_basis(basis, target, modulus)

    if not exact:
        solution, _, _, _ = np.linalg.lstsq(basis, target, rcond=None)
        return solution

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
    if vec.size and isinstance(vec.flat[0], GF):
        for c, path in zip(vec, basis):
            if c.v:
                out[path] = c
        return out
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
    generators: bool = True,
    store_matrices: Optional[bool] = None,
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
    generators : bool, default True
        When ``True``, materialise bases and representative chains for Omega,
        cycles, boundaries and homology.  When ``False``, compute only their
        dimensions and Betti numbers directly from ranks of the constraint and
        full-boundary matrices.  The latter is much faster and uses much less
        memory; the corresponding basis dictionaries in the result stay empty.
    store_matrices : bool, optional
        Retain dense boundary matrices in the result.  The default follows
        ``generators``: full computations keep them, while dimensions-only
        computations use sparse rows and do not materialise them.  Pass
        ``True`` with ``generators=False`` when matrix output is still wanted.
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
    if store_matrices is None:
        store_matrices = generators
    if generators and not store_matrices:
        raise ValueError("generators=True requires store_matrices=True")
    res = PathHomologyResult(vertices or [], edges or [])
    res.field = "Q" if kind == "Q" else (f"GF({modulus})" if kind == "GF" else "R")
    res.generators_materialized = generators
    res.matrices_materialized = store_matrices
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
    if store_matrices:
        for p in range(0, dim_max + 1):
            prev = [()] if p == 0 else paths.get(p - 1, [])
            res.boundary_matrices[p] = boundary_matrix(
                paths[p], prev, regular=regular, field=field if field is not None
                else ("Q" if exact else "R")
            )
    if verbose:
        print(f"[path_homology] allowed paths up to dimension {dim_max}")

    if not generators and not store_matrices:
        # Fully sparse dimensions-only route.  Each row is a dictionary of its
        # nonzero columns; over GF(2) it is immediately packed into one Python
        # integer and eliminated with machine-level big-integer XOR.
        for p in range(0, dim_max + 1):
            n_paths = len(paths[p])
            if p == 0:
                condition_rank = 0
                full_rank = 0
            else:
                # Direct packed-row construction wins while each row is a
                # modest-size integer.  For very wide matrices, repeatedly
                # XOR-growing those integers copies more memory than building
                # sparse dictionaries once and packing them at elimination.
                if kind == "GF" and modulus == 2 and n_paths <= 10_000:
                    full_rows = _gf2_boundary_bitrows(
                        paths[p], regular=regular
                    )
                else:
                    full_rows = _sparse_boundary_rows(
                        paths[p], regular=regular
                    )
                allowed_previous = set(paths.get(p - 1, []))
                boundary_closed = all(
                    face in allowed_previous for face in full_rows
                )
                if method in ("auto", "kernel"):
                    condition_rows = [
                        row for face, row in full_rows.items()
                        if face not in allowed_previous
                    ]
                    if kind == "Q":
                        condition_rank = _q_sparse_rank_certified(
                            condition_rows, n_paths
                        )
                    else:
                        condition_rank = _sparse_rank(
                            condition_rows, n_paths, kind, modulus
                        )
                else:
                    conditions = _omega_conditions(
                        paths, edge_list, p, kind, modulus,
                        method=method, regular=regular,
                    )
                    condition_rank = _rank(conditions) if conditions.size else 0
                if kind == "Q":
                    if p == 1:
                        # A directed incidence matrix has rank |V|-c over Q,
                        # where c is the number of weak components.
                        full_rank = _incidence_rank(paths[0], paths[1])
                    else:
                        # When every face is allowed, this is an honest chain
                        # map A_p -> A_{p-1}.  Hence rank(d_p) is at most the
                        # already-known nullity of d_{p-1}; reaching this bound
                        # modulo two certifies the exact rational rank.
                        chain_upper = (
                            res.dim_cycles[p - 1] if boundary_closed else None
                        )
                        full_rank = _q_sparse_rank_certified(
                            full_rows.values(), n_paths, chain_upper
                        )
                else:
                    full_rank = _sparse_rank(
                        full_rows.values(), n_paths, kind, modulus
                    )
            res.dim_omega[p] = n_paths - condition_rank
            res.dim_cycles[p] = n_paths if p == 0 else n_paths - full_rank
            res.omega_basis[p] = []
            res.omega_paths[p] = []
            res.omega_matrix[p] = _field_zeros((n_paths, 0), kind, modulus)
            res.cycles[p] = []
            res.boundaries[p] = []
            res.homology_basis[p] = []

        res.dim_omega[dim_max + 1] = 0
        res.omega_matrix[dim_max + 1] = _field_zeros((0, 0), kind, modulus)
        for p in range(0, dim_max + 1):
            res.dim_boundaries[p] = (
                res.dim_omega[p + 1] - res.dim_cycles[p + 1]
                if p < dim_max else 0
            )
            res.betti[p] = res.dim_cycles[p] - res.dim_boundaries[p]
        res.euler_characteristic = sum(
            (-1) ** p * res.dim_omega[p] for p in range(dim_max + 1)
        )
        if verbose:
            print(f"[path_homology] betti numbers: {res.betti}")
        return res

    if not generators:
        # Dimension-only route.  Let C_p be the non-allowed-face constraint
        # matrix and F_p=[partial_p; C_p] the full boundary of allowed p-paths.
        # Then Omega_p=ker(C_p), Z_p=ker(F_p) (p>0), and
        # rank(d_{p+1})=dim(Omega_{p+1})-dim(Z_{p+1}).  Thus no Omega basis,
        # change of basis, or representative expansion is needed.
        for p in range(0, dim_max + 1):
            n_paths = len(paths[p])
            conditions = _omega_conditions(
                paths, edge_list, p, kind, modulus,
                method=method, regular=regular,
            )
            condition_rank = _rank(conditions) if conditions.size else 0
            res.dim_omega[p] = n_paths - condition_rank
            if p == 0:
                res.dim_cycles[p] = n_paths
            else:
                full_boundary = (
                    np.vstack([res.boundary_matrices[p], conditions])
                    if conditions.shape[0]
                    else res.boundary_matrices[p]
                )
                res.dim_cycles[p] = n_paths - _rank(full_boundary)
            res.omega_basis[p] = []
            res.omega_paths[p] = []
            res.omega_matrix[p] = _field_zeros((n_paths, 0), kind, modulus)
            res.cycles[p] = []
            res.boundaries[p] = []
            res.homology_basis[p] = []

        res.dim_omega[dim_max + 1] = 0
        res.omega_matrix[dim_max + 1] = _field_zeros((0, 0), kind, modulus)
        for p in range(0, dim_max + 1):
            res.dim_boundaries[p] = (
                res.dim_omega[p + 1] - res.dim_cycles[p + 1]
                if p < dim_max else 0
            )
            res.betti[p] = res.dim_cycles[p] - res.dim_boundaries[p]
        res.euler_characteristic = sum(
            (-1) ** p * res.dim_omega[p] for p in range(dim_max + 1)
        )
        if verbose:
            print(f"[path_homology] betti numbers: {res.betti}")
        return res

    # ---- Omega_p -----------------------------------------------------------
    dtype = dtype_for(kind)
    omega_is_full: Dict[int, bool] = {}
    for p in range(0, dim_max + 1):
        basis, conditions = omega_basis(
            paths, edge_list, p, method=method, regular=regular,
            field=field if field is not None else ("Q" if exact else "R"),
        )
        omega_is_full[p] = conditions.shape[0] == 0
        res.omega_basis[p] = basis
        n = len(paths[p])
        mat = (
            np.column_stack(basis)
            if basis
            else _field_zeros((n, 0), kind, modulus)
        )
        res.omega_matrix[p] = mat
        res.omega_paths[p] = [_vec_to_dict(b, paths[p]) for b in basis]
        res.dim_omega[p] = len(basis)
    # Omega_p = 0 beyond dim_max (there are no allowed paths there)
    res.dim_omega[dim_max + 1] = 0
    res.omega_matrix[dim_max + 1] = _field_zeros((0, 0), kind, modulus)

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
        img = (
            np.array(res.boundary_matrices[p], copy=True)
            if omega_is_full[p]
            else _field_matmul(res.boundary_matrices[p], res.omega_matrix[p])
        )  # in A_{p-1} coords
        om_prev = res.omega_matrix.get(p - 1)
        if om_prev is None or om_prev.shape[1] == 0:
            d[p] = _field_zeros(
                (0, img.shape[1]), kind, modulus
            )
            continue
        if img.shape[1] == 0:
            d[p] = _field_zeros((om_prev.shape[1], 0), kind, modulus)
            continue
        d[p] = (
            np.array(img, copy=True)
            if omega_is_full[p - 1]
            else _solve_in_basis(om_prev, img)
        )
    if dim_max >= 1:
        d[0] = _field_zeros((0, res.dim_omega[0]), kind, modulus)

    # Compute each differential nullspace once.  Its nullity simultaneously
    # gives rank(d_p) by rank-nullity, so the boundary dimensions below do not
    # need a second elimination of the same matrix.
    cycle_coordinates: Dict[int, np.ndarray] = {}
    differential_rank: Dict[int, int] = {}
    for p in range(0, dim_max + 1):
        dp = d.get(p, _field_zeros((0, res.dim_omega[p]), kind, modulus))
        if dp.shape[0] == 0:
            dp = _field_zeros((0, res.dim_omega[p]), kind, modulus)
        z = _nullspace(dp)
        cycle_coordinates[p] = z
        differential_rank[p] = res.dim_omega[p] - z.shape[1]

    # ---- cycles, boundaries, homology -------------------------------------
    for p in range(0, dim_max + 1):
        z = cycle_coordinates[p]
        res.dim_cycles[p] = z.shape[1]
        cycles_in_a = (
            np.array(z, copy=True)
            if omega_is_full[p]
            else _field_matmul(res.omega_matrix[p], z)
        )
        res.cycles[p] = [
            _vec_to_dict(cycles_in_a[:, j], paths[p])
            for j in range(z.shape[1])
        ]

        # B_p = im(d_{p+1}) inside Omega_p
        if p + 1 <= dim_max and (p + 1) in d:
            b = d[p + 1]
        else:
            b = _field_zeros((res.dim_omega[p], 0), kind, modulus)
        res.dim_boundaries[p] = differential_rank.get(p + 1, 0)
        boundaries_in_a = (
            np.array(b, copy=True)
            if omega_is_full[p]
            else _field_matmul(res.omega_matrix[p], b)
        )
        res.boundaries[p] = [
            _vec_to_dict(boundaries_in_a[:, j], paths[p])
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
                        _vec_to_dict(cycles_in_a[:, c - nb], paths[p])
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
