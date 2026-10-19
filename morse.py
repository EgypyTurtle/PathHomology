"""Exact, sequential algebraic Morse cancellation of a finite based complex.

No graph-specific deletion rules: the input must already satisfy d*d = 0.
Columns and inclusion maps are sparse dictionaries.  Scalars are integers or
Fractions over Q, and ordinary reduced integers over the prime field F_p.
The caller must supply a prime modulus (or None for Q).
"""
from fractions import Fraction
import heapq


class Arithmetic:
    def __init__(self, modulus=None):
        self.modulus = modulus

    def scalar(self, value):
        if self.modulus is not None:
            if isinstance(value, Fraction):
                return (value.numerator * pow(value.denominator, -1, self.modulus)) % self.modulus
            return int(value) % self.modulus
        if isinstance(value, float):
            raise TypeError("Morse cancellation requires exact scalars, not floats")
        return value if isinstance(value, (int, Fraction)) else Fraction(value)

    def ratio(self, numerator, denominator):
        if self.modulus is not None:
            return numerator * pow(denominator, -1, self.modulus) % self.modulus
        if denominator == 1:
            return numerator
        if denominator == -1:
            return -numerator
        return Fraction(numerator) / denominator

    def axpy(self, target, source, factor):
        """target += factor * source; source and target must be distinct."""
        if self.modulus is not None:
            factor %= self.modulus
        if not factor:
            return
        p = self.modulus
        if p == 2:
            for key in source:
                if key in target:
                    del target[key]
                else:
                    target[key] = 1
            return
        for key, value in source.items():
            value = target.get(key, 0) + factor * value
            if p is not None:
                value %= p
            if value:
                target[key] = value
            else:
                target.pop(key, None)


def reduce_complex(dimensions, differentials, *, modulus=None,
                   check=False, record_steps=False):
    """Cancel all nonzero pivots, returning lifted H and independent B bases.

    dimensions[p] is dim C_p; differentials[p][j] is column j of d_p,
    a dict of {row: coefficient}.  This is ordinary (not augmented) homology:
    d_0 = 0.  Differential columns are copied; input is never mutated.

    Degrees are processed bottom-up. Short columns are considered first;
    within one column, prefer a shortest incident row to limit fill-in.
    Every cancellation acts on the UPDATED complex, so no unverified global
    matching/acyclicity assumption is made. Over a field all remaining maps
    eventually vanish. There is no claim of a worst-case linear-time bound.

    Inclusion maps to the ORIGINAL C are updated with each basis change.
    Boundaries saved on cancellation and final inclusions give bases of B
    and H; their concatenation gives a basis of Z. Full P/H contraction
    matrices are deliberately not materialized. record_steps is for tests.
    """
    if not dimensions or sorted(dimensions) != list(range(max(dimensions) + 1)):
        raise ValueError("dimensions must have consecutive degrees starting at 0")
    if any(not isinstance(n, int) or n < 0 for n in dimensions.values()):
        raise ValueError("dimensions must be nonnegative integers")
    if modulus is not None and (not isinstance(modulus, int) or modulus < 2):
        raise ValueError("modulus must be prime")
    arithmetic = Arithmetic(modulus)
    top = max(dimensions)
    columns, rows = {}, {}
    for p in range(1, top + 1):
        source = differentials.get(p)
        if source is None:
            source = [{} for _ in range(dimensions[p])]
        if len(source) != dimensions[p]:
            raise ValueError("wrong number of differential columns")
        columns[p], rows[p] = {}, {}
        for j, vector in enumerate(source):
            col = {}
            for r, value in vector.items():
                if not isinstance(r, int) or not 0 <= r < dimensions[p - 1]:
                    raise ValueError("differential row out of range")
                value = arithmetic.scalar(value)
                if value:
                    col[r] = value
                    rows[p].setdefault(r, set()).add(j)
            columns[p][j] = col
    if any(differentials.get(0, [])):
        raise ValueError("d_0 must be zero (ordinary homology)")
    if check:
        for p in range(2, top + 1):
            for col in columns[p].values():
                square = {}
                for r, value in col.items():
                    arithmetic.axpy(square, columns[p - 1][r], value)
                if square:
                    raise ValueError("input is not a chain complex: d*d != 0")
    lifts = {p: {j: {j: 1} for j in range(n)} for p, n in dimensions.items()}
    boundaries = {p: [] for p in dimensions}
    ranks = {p: 0 for p in dimensions}
    stats = dict(pairs=0, unit_pivots=0, schur_updates=0, new_entries=0,
                 initial_nnz=sum(len(c) for cols in columns.values() for c in cols.values()))
    steps = []
    for p in range(1, top + 1):
        cols, incident = columns[p], rows[p]
        versions = {j: 0 for j in cols}
        heap = [(len(c), j, 0) for j, c in cols.items() if c]
        heapq.heapify(heap)
        while heap:
            _, a, version = heapq.heappop(heap)
            if a not in cols or version != versions[a] or not cols[a]:
                continue
            da = cols[a]
            b = min(da, key=lambda r: (len(incident[r]), r))
            pivot = da[b]
            affected = [(x, arithmetic.ratio(cols[x][b], pivot))
                        for x in sorted(incident[b]) if x != a]
            if record_steps:
                steps.append(dict(degree=p, upper=a, lower=b, pivot=pivot,
                                  boundary=dict(da), affected=dict(affected)))
            # d(I_old a) is a cycle in the ORIGINAL lower chain group.
            boundary = {}
            for r, value in da.items():
                arithmetic.axpy(boundary, lifts[p - 1][r], value)
            boundaries[p - 1].append(boundary)
            # In the new upper basis x' = x - ([b]d x / pivot) a.
            for x, factor in affected:
                col = cols[x]
                old_keys = set(col)
                arithmetic.axpy(col, da, -factor)
                new_keys = set(col)
                for r in old_keys - new_keys:
                    incident[r].remove(x)
                for r in new_keys - old_keys:
                    incident.setdefault(r, set()).add(x)
                stats["new_entries"] += len(new_keys - old_keys)
                arithmetic.axpy(lifts[p][x], lifts[p][a], -factor)
                versions[x] += 1
                if col:
                    heapq.heappush(heap, (len(col), x, versions[x]))
                stats["schur_updates"] += 1
            # In d_(p+1), the transformed a-coordinate is zero by d*d=0.
            if p < top:
                if check:
                    for col in columns[p + 1].values():
                        coefficient = col.get(a, 0)
                        coefficient += sum(f * col.get(x, 0) for x, f in affected)
                        if arithmetic.scalar(coefficient):
                            raise AssertionError("adjacent differential violates chain condition")
                for j in sorted(rows[p + 1].get(a, ())):
                    columns[p + 1][j].pop(a)
                rows[p + 1].pop(a, None)
            for r in da:
                incident[r].remove(a)
                if not incident[r]:
                    del incident[r]
            del cols[a]
            # d_(p-1) was reduced to zero already, so its b-column is empty.
            if p > 1:
                assert not columns[p - 1][b]
                del columns[p - 1][b]
            del lifts[p][a]
            del lifts[p - 1][b]
            ranks[p] += 1
            stats["pairs"] += 1
            stats["unit_pivots"] += int(pivot in (1, -1 if modulus is None else modulus - 1))
        if any(cols.values()):
            raise AssertionError("cancellation heap lost a nonzero column")
    homology = {p: [lifts[p][j] for j in sorted(lifts[p])] for p in dimensions}
    stats["critical_dimensions"] = {p: len(homology[p]) for p in dimensions}
    stats["lift_nnz"] = sum(len(v) for vs in homology.values() for v in vs)
    stats["boundary_nnz"] = sum(len(v) for vs in boundaries.values() for v in vs)
    return dict(homology=homology, boundaries=boundaries, ranks=ranks,
                stats=stats, steps=steps)
