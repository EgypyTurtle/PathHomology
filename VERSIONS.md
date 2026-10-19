# 当前版本与历史版本

当前根目录 `core.py` 为 **gen6**（2026-10-19），并保留默认 linear 后端。

| 版本 | SHA256 前 12 位 | 本仓库位置 |
|---|---|---|
| gen0 | `76f2be7ee555` | `original_core.py`，另有 `reference/gen0_core.py` |
| gen3 | `a6acf4c90b78` | `reference/gen3_core.py`（原 main 核心） |
| gen4 | `3d7e315c2dde` | `_baseline_gen4_20261019/core.py` |
| gen5 | `dd247bdcfa5a` | `_baseline_gen5_20261019/core.py` |
| gen6 | `ffe155a05112` | `core.py`（当前） |
| Morse 模块 | `bb946f1275c2` | `morse.py` |

gen1/gen2 的说明与当时数据保留在 [历史版本记录](docs/VERSIONS_gen0_gen3.md)，本次没有额外复制它们的完整源码。

gen4/gen5 增加 Ω 支撑剪枝、稀疏秩与精确秩认证；gen6 增加可选 `homology_method="morse"`，在真正的 Ω 链复形上作精确配对消去，保留回到原始路径坐标的代表元提升。完整核心哈希与基准依赖见 `SHA256SUMS.txt`。

- 只求维数：`generators=False, store_paths=False`，默认 linear。
- 要显式代表元：可选 `homology_method="morse", store_matrices=False`。
- 要旧规范代表元：继续使用默认 linear，Morse 的 Z/B/H 基可能不同。
- 要原图 H_k：至少计算到 `max_dim=k+1`，或另有上一维为零的严格依据。

历史结果依然属于当时记录的源码版本；不能自动把旧图族、文献与速度结论归到 gen6。新测量见 [Morse 性能报告](docs/MORSE_PERFORMANCE_20261019.md)。
