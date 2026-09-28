
# Cost model and the largest reachable instances

This note describes how the engine's cost scales and estimates -- **without
running anything for hours** -- the largest instance that would finish inside
a 3-hour budget when only Betti numbers (not generators) are requested.

## 1. Where the cost sits

For `max_dim = D` the engine performs, in order:

1. **Allowed-path enumeration.** `|A_p|` paths of length `p`, built level by
   level from `A_{p-1}`. Cost and memory are both `Theta(|A_p|)`, and for a
   digraph with `N` vertices and out-degree `k`,

   ```
   |A_0| = N        |A_p| ~= N * k^p
   ```

   exactly for `k`-out digraphs, and `N * k^p` in general up to collisions.
   This is the term that dominates everything else.
2. **`Omega_p` extraction.** `Omega_p` is the kernel of the restriction map
   `A_p -> A_{p-1}/Omega_{p-1}`, i.e. the kernel of a sparse matrix with one
   row per "forbidden face" and `|A_p|` columns. Three interchangeable
   implementations are available (`kernel`, `lemma41`, `prop42`).
3. **Boundary ranks.** `dim H_p = dim ker(d_p) - rank(d_{p+1})`, each rank by
   sparse elimination.

Steps 2 and 3 are the ones that actually bite, and they bite through memory
before they bite through time.

## 2. Measured grid

Random `k`-out digraphs, no loops, vertices `N`, `max_dim = 2`, rationals,
`generators = False`. `t_Q` is the wall time for the whole computation.

| N | k | \|A_1\| | \|A_2\| | t_Q |
|---|---|---|---|---|
| 20 | 2 | 40 | 80 | 0.000 s |
| 320 | 2 | 640 | 1,280 | 0.005 s |
| 1280 | 2 | 2,560 | 5,120 | 0.055 s |
| 20 | 3 | 60 | 180 | 0.005 s |
| 320 | 3 | 960 | 2,880 | 0.022 s |
| 1280 | 3 | 3,840 | 11,520 | 0.233 s |
| 20 | 4 | 80 | 320 | 0.013 s |
| 320 | 4 | 1,280 | 5,120 | 0.666 s |
| 640 | 4 | 2,560 | 10,240 | 1.993 s |
| 1280 | 4 | 5,120 | 20,480 | 0.718 s |
| 80 | 5 | 400 | 2,000 | 0.448 s |
| 320 | 5 | 1,600 | 8,000 | 2.527 s |
| 640 | 5 | 3,200 | 16,000 | 6.531 s |

Two things stand out.

* The exponent is **not** a function of `|A_2|` alone. At `|A_2| ~ 10^4`,
  `k = 3` costs `0.23 s` while `k = 4` costs `1.99 s` and `k = 5` costs
  `6.53 s` -- roughly an order of magnitude per extra out-arc. Out-degree
  raises the *density of the `Omega` constraints*, and step 2 is where the
  time goes.
* The row `N = 1280, k = 4` is **faster** than `N = 640, k = 4`
  (`0.718 s` against `1.993 s`) at twice the size. This is not noise: the
  engine first computes a cheap mod-2 rank estimate and only falls back to
  exact rational elimination when that certificate fails. Runtime is
  therefore **data-dependent with large variance**, and a single power law is
  only an order-of-magnitude summary.

Fitting `t ~= c * |A_2|^alpha` on the `k >= 4` rows gives `alpha ~= 1.4 - 1.6`;
the dilute `k <= 3` rows are far below the same curve because the certificate
succeeds immediately.

## 3. The wall is memory, not time

`|A_2| = 1,451,520` (S_9, out-degree 2, `max_dim = 2`) does not complete: the
process dies with

```
MemoryError
  in _q_sparse_rank_certified -> _sparse_rank -> bits ^= 1 << column
```

The mod-2 rank certificate represents each row of the constraint matrix as a
single Python integer of width `|A_p|` bits. One row therefore costs
`|A_p| / 8` bytes, and there is one row per constraint, so the certificate
alone wants

```
memory ~ (#rows) * |A_p| / 8  bytes
```

which is quadratic. At `|A_p| ~ 10^6` the bigints are ~125 kB each and the
allocation blows up. `S_8` at `|A_2| = 80,640` completes in 20 s; `S_9` at
`|A_2| = 1,451,520` does not complete at all. The practical ceiling of the
current code is

```
|A_p|  <~  10^6        (graph-size ceiling: N <~ 10^6 / k^p)
```

**This is a limitation of the certificate, not of the algorithm.** Replacing
the bigint bit-rows with a sparse bitset or a `k x |A_p|` block rank would
remove the quadratic term and make the time budget, not the memory, the
binding constraint.

## 4. Predicted 3-hour reach (Betti numbers only)

Calibration: `k = 4`, `|A_2| = 2.05 x 10^4`, `t ~= 1 s`, with `alpha = 1.5`.
Extrapolating `t = c |A|^alpha` to `10800 s` gives `|A| ~= 4.7 x 10^6`; the
memory ceiling of section 3 caps this at `~10^6` first. The table reports the
**memory-limited** figure, which is the honest one.

| mode | \|A_p\| at 3 h (time) | \|A_p\| at 3 h (memory) | binding limit |
|---|---|---|---|
| `max_dim = 2`, Q, no generators | ~4.7 x 10^6 | ~10^6 | memory |
| `max_dim = 3`, Q, no generators | ~5.7 x 10^5 | ~10^6 | time |
| `max_dim = 3`, F_2, no generators | ~2 x 10^6 | ~10^6 | memory |

Translating into graph sizes `N ~= |A_D| / k^D`:

| `max_dim` | out-degree `k` | `N` at 3 h | `\|E\|` at 3 h |
|---|---|---|---|
| 2 | 2 | ~250,000 | ~500,000 |
| 2 | 4 | ~62,000 | ~250,000 |
| 2 | 6 | ~28,000 | ~170,000 |
| 2 | 10 | ~10,000 | ~100,000 |
| 3 | 2 | ~60,000 | ~120,000 |
| 3 | 4 | ~8,000 | ~32,000 |
| 3 | 6 | ~2,600 | ~15,000 |
| 3 | 10 | ~570 | ~5,700 |

These are **order-of-magnitude predictions from a fitted model, not
measurements.** The largest instances actually completed so far are:

| instance | N | out-degree | \|A_D\| | mode | time |
|---|---|---|---|---|---|
| karate club | 34 | 5 | 7,344 | `max_dim = 3`, Q | 11.6 s |
| Les Miserables | 77 | 6 | 16,632 | `max_dim = 3`, Q | 54.1 s |
| S_8 Cayley digraph | 40,320 | 2 | 80,640 | `max_dim = 2`, Q | 20.2 s |
| random 4-out | 1,280 | 4 | 20,480 | `max_dim = 2`, Q | 0.7 s |

So the model's `max_dim = 3` row is anchored on two real measurements and the
`max_dim = 2` rows on a size ladder that stops about a factor of 50 short of
the predicted ceiling.

## 5. Levers, in order of effect

1. **`generators = False`** (Betti numbers only). Measured speed-ups over the
   unoptimised first version range from `2 x 10^4` to `3 x 10^4`; over the
   current version it is still the single biggest lever, because generator
   extraction is a second exact rational elimination per degree.
2. **Fix the memory wall** in the mod-2 certificate (section 3). This changes
   the binding constraint from memory to time and multiplies the reachable
   `|A_p|` by roughly `5x` at `max_dim = 2`.
3. **Lower `max_dim`.** `|A_p|` grows like `k^p`; dropping from `max_dim = 3`
   to `max_dim = 2` buys a factor of `k` in `N` for free.
4. **Use `F_2` when only Betti numbers over a field are wanted.** Bit-level
   rows are much cheaper than rationals; the measured gap is roughly `5x` to
   `20x` on the rows tabulated above.
5. **Exploit sparsity of the digraph**, not just its size: the runtime is
   driven by the density of the `Omega` constraints, so two digraphs with the
   same `|A_p|` and different out-degree can differ by an order of magnitude.
