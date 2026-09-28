# -*- coding: utf-8 -*-
"""
Complexity audit and 3-hour-reach extrapolation for the gen3 core.

Nothing here runs for hours: a small, fast grid of random k-out digraphs is
measured, a cost model is fitted to it, and the model is then used to PREDICT
the largest instance that would finish inside a 3-hour budget in
dimension-only ("betti only") mode.
"""
import importlib.util
import math
import os
import random
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

rng = random.Random(20260927)


def kout(N, k):
    """Random digraph, out-degree exactly k, no loops, no repeated arcs."""
    E = set()
    for u in range(N):
        pool = [w for w in range(N) if w != u]
        for w in rng.sample(pool, min(k, len(pool))):
            E.add((u, w))
    return list(range(N)), sorted(E)


print("=" * 118)
print("PART 1   measured grid  (gen3 core, random k-out digraphs, no loops)")
print("=" * 118)
print(f"{'N':>6} {'k':>2} {'D':>2} {'|A_0|':>8} {'|A_1|':>9} {'|A_2|':>10} "
      f"{'|A_3|':>11} {'t_Q':>9} {'t_F2':>9} {'t_Q(gen=True)':>14}")
GRID = []
for D in (2, 3):
    for k in (2, 3, 4, 5, 6):
        for N in (20, 40, 80, 160, 320, 640, 1280):
            v, E = kout(N, k)
            A = kern.allowed_paths(v, E, D)
            sizes = [len(A.get(p, [])) for p in range(D + 1)]
            if D == 3 and sizes[3] > 400_000:
                continue
            if sizes[D] > 400_000:
                continue
            t = time.time()
            rq = kern.path_homology(vertices=v, edges=E, max_dim=D, field="Q",
                                    generators=False)
            tq = time.time() - t
            t = time.time()
            rf = kern.path_homology(vertices=v, edges=E, max_dim=D, field="Z2",
                                    generators=False)
            tf = time.time() - t
            tg = float("nan")
            if sizes[D] <= 200_000:
                t = time.time()
                kern.path_homology(vertices=v, edges=E, max_dim=D, field="Z2",
                                   generators=True)
                tg = time.time() - t
            print(f"{N:>6} {k:>2} {D:>2} {sizes[0]:>8,} {sizes[1]:>9,} "
                  f"{sizes[2]:>10,} {sizes[3] if D == 3 else 0:>11,} "
                  f"{tq:>8.3f}s {tf:>8.3f}s {tg:>13.3f}s", flush=True)
            GRID.append(dict(N=N, k=k, D=D, sizes=sizes, tq=tq, tf=tf, tg=tg))
            if tq > 60:
                break
print()

print("=" * 118)
print("PART 2   cost model")
print("=" * 118)


def fit(pairs):
    """log-log least squares  log t = log c + alpha log m."""
    pts = [(math.log(m), math.log(t)) for (m, t) in pairs
           if m > 0 and t > 1e-4]
    if len(pts) < 3:
        return None
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    n = len(pts)
    alpha = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = math.exp((sy - alpha * sx) / n)
    return c, alpha, n


for D in (2, 3):
    for fld, key in (("Q", "tq"), ("F2", "tf")):
        sub = [g for g in GRID if g["D"] == D and g[key] == g[key]]
        if not sub:
            continue
        m = [max(g["sizes"][D], 1) for g in sub]
        f = fit(list(zip(m, [g[key] for g in sub])))
        if f:
            c, a, n = f
            print(f"  D={D}, field={fld:<2}:  t ~= {c:.3e} * |A_{D}|^{a:.3f}"
                  f"   (fit on {n} points)")
print()
for D in (2, 3):
    sub = [g for g in GRID if g["D"] == D and g["tq"] == g["tq"]]
    m = [max(g["sizes"][D], 1) for g in sub]
    f = fit(list(zip(m, [g["tq"] for g in sub])))
    if f:
        c, a, n = f
        print(f"  D={D}, TOTAL (all dims, Q): t ~= {c:.3e} * |A_{D}|^{a:.3f}")
print()

print("=" * 118)
print("PART 3   PREDICTED reach inside a 3-hour budget (10 800 s), betti only")
print("         generator-free mode; model t = c * |A_D|^alpha")
print("=" * 118)
for D in (2, 3):
    sub = [g for g in GRID if g["D"] == D and g["tq"] == g["tq"]]
    m = [max(g["sizes"][D], 1) for g in sub]
    f = fit(list(zip(m, [g["tq"] for g in sub])))
    if not f:
        continue
    c, a, _ = f
    tmax = 10800.0
    Amax = (tmax / c) ** (1.0 / a)
    print(f"  D={D}:  |A_{D}| at 3 h  ~= {Amax:,.0f}")
    print(f"         {'k':>3} {'|A_1|/N':>8} {'N at 3h':>12} "
          f"{'|E| at 3h':>12} {'A_D memory':>12}")
    for k in (2, 3, 4, 5, 6, 8, 10):
        # |A_D| ~= N * k^D  (D=2) ; |A_3| ~= N * k^3
        Nmax = Amax / (k ** D)
        mem = Amax * (72 + 8 * (D + 1)) / 2 ** 30
        print(f"         {k:>3} {k**D:>8} {Nmax:>12,.0f} {Nmax*k:>12,.0f} "
              f"{mem:>9.2f} GiB")
    print()

print("=" * 118)
print("PART 4   sanity: the extrapolation reproduces the measured points")
print("=" * 118)
for D in (2, 3):
    sub = [g for g in GRID if g["D"] == D and g["tq"] == g["tq"]]
    m = [max(g["sizes"][D], 1) for g in sub]
    f = fit(list(zip(m, [g["tq"] for g in sub])))
    if not f:
        continue
    c, a, _ = f
    print(f"  D={D}")
    print(f"    {'N':>6} {'k':>3} {'|A_D|':>11} {'measured':>10} {'model':>10}")
    for g in sorted(sub, key=lambda z: z["sizes"][D])[:14]:
        pred = c * max(g["sizes"][D], 1) ** a
        print(f"    {g['N']:>6} {g['k']:>3} {g['sizes'][D]:>11,} "
              f"{g['tq']:>9.3f}s {pred:>9.3f}s")
