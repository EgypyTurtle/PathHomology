# Version record for `core.py`

The four generations of `core.py`, their hashes, and which generation produced which
result. The purpose is to prevent conclusions from being drawn against a stale copy.

## The generations

| Tag | Bytes | Date | SHA256 (first 12) | Location | Main changes |
|---|---:|---|---|---|---|
| **gen0** | 54458 | 2026-09-18 | `76F2BE7EE555` | `backups/_backup_before_optimization_20260926/core.py` | Original: `dtype=object` arrays of `Fraction` / `GF`, pure Python object arithmetic |
| **gen1** | 65728 | 2026-09-26 21:21 | `10E64353730C` | `backups/_backup_before_optimization_round2_20260927/core.py` | Round 1: all prime fields converted to integer arrays for modular elimination; integral `Q` matrices that provably cannot overflow use an int64 kernel; change of basis reads rows off the identity block of the canonical nullspace basis; representatives expanded in one batch per dimension |
| **gen2** | 82143 | 2026-09-27 01:54 | `B3DAB319B412` | `backups/_backup_before_optimization_round3_20260927/core.py` | Round 2: fraction-free elimination with primitive integer rows; one reduction per differential (rank via rank-nullity); explicit fast path for `Omega_p = A_p`; new **`generators=False` sparse dimension-only mode** (boundary rows as sparse dictionaries, packed into big-integer bit rows over `F_2`) |
| **gen3** | 87483 | 2026-09-27 11:12 | `A6ACF4C90B78` | `core.py` (root, **current**) | Round 3: rank mod 2 as a strict lower bound together with the chain-condition upper bound gives a **certifiable rank** (otherwise fall back to exact elimination); rank of the first differential from the incidence-matrix theorem; in-place sparse elimination; batched back-substitution |

> Note on naming: the backup directories are snapshots taken *before* a round, so
> `_backup_before_optimization_round3_20260927` contains **gen2**, not gen3.

## Which generation produced which result

| Result | Where | Computed with | Re-run on gen3? |
|---|---|---|---|
| GLMY paper examples (94 assertions) | section 3.1 | gen0 | **yes** (gen3 passes the full `verify_all.py` suite, 22/22) |
| Tang-Yau Theorems 1.1 and 1.2 | section 3.2 | gen1 | **yes** -- output byte-identical |
| Torsion conjecture, `n <= 12` | section 3.3 | gen1 + local Smith form | **yes** -- output byte-identical |
| Optimisation speed-up table | section 2 | gen0 / gen1 / gen3 | -- |

## Usage notes

- **Betti numbers only**: use `path_homology(..., generators=False)`. This mode was
  added in round 2 and is another 45x to 731x faster than the full mode;
  `N = 15, max_dim = 6` over `Q` takes about 0.1 s.
- **Generators or representatives required**: use the default mode. Linear algebra is
  then the dominant cost again.
- **Reproducing an old script**: most scripts do `from core import ...`. To pin a
  generation, copy that generation's file to `core.py` next to the script.
