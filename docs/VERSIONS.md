# 代码版本记录（`core.py` 各代）

本文件记录 `core.py` 的四个世代、各自的哈希，以及**哪些结论是在哪一代上算出来的**。
目的是避免再次出现"对着过期版本下结论"的情况。

## 各代一览

| 代号 | 字节 | 日期 | SHA256 前 12 位 | 存放位置 | 主要变化 |
|---|---:|---|---|---|---|
| **gen0** | 54458 | 2026-09-18 | `76F2BE7EE555` | `backups/_backup_before_optimization_20260926/core.py` | 最初版：`dtype=object` 存 `Fraction`/`GF`，纯对象算术 |
| **gen1** | 65728 | 2026-09-26 21:21 | `10E64353730C` | `backups/_backup_before_optimization_round2_20260927/core.py`、`path homology计算/core.py` | 第 1 轮：全素数域转整数数组做模 $p$ 消元；可证明不溢出的整值 $\Q$ 矩阵走 int64 内核；`_nullspace` 规范基按单位块抽行；同调代表元整批展开 |
| **gen2** | 82143 | 2026-09-27 01:54 | `B3DAB319B412` | `backups/_backup_before_optimization_round3_20260927/core.py` | 第 2 轮：本原整数行无分数消元；每个 $d_p$ 只求一次零空间（秩由秩–零化度定理得）；识别 $\Omega_p=A_p$ 快路；新增 **`generators=False` 稀疏维数模式**（边界行存稀疏字典，$\F_2$ 压成大整数位集） |
| **gen3** | 87483 | 2026-09-27 11:12 | `A6ACF4C90B78` | `core.py`（根目录，**当前**） | 第 3 轮：模 2 秩作有理秩的严格下界 + 链条件作上界，上下界相等即给出**可证明的秩证书**（否则回退精确消元）；$\rank\partial_1=\|V\|-$ 弱连通分支数；原地稀疏消元；自由变量批量回代 |

> 注：备份目录的命名是"轮次**之前**的快照"，故 `_round3_` 里装的是 gen2 而非 gen3。

## 哪些结论在哪一代上算的

| 结论 | 章节 | 计算所用版本 | 是否已在 gen3 上复算 |
|---|---|---|---|
| $P_n$、$D_m$、$D_{m,n}$ 的同调与群结构 | §2–§4 | gen0（`verify_all.py` 全套） | **是**（gen3 通过 22/22） |
| $H_1$/$H_2$ 生成元 | §5 | gen0 | 是（间接，经 `verify_all`） |
| $\Omega_3$ 结构（两风味、条件图） | §6 | gen1 | **是**（`omega3_gamma.py` 已复算） |
| 二面体 Cayley 图 | §7 | gen1 | **是**（`prism_ndep.py` 已复算） |
| 循环有向图（Tang–Yau 定理） | §8 | gen1 | **是**（`circulant_check.py` 已复算） |
| 挠元猜想复算（$n\le12$） | §9 | gen1 + 自写 SNF | **是**（`torsion_conjecture.py` 已复算） |
| 复杂度实测 §11.2–11.5 | §11 | **gen0** | 已在 §11.6 用 gen3 重测并替换结论 |
| 复杂度实测 §11.6 | §11 | gen3 | — |

## 使用建议

- **只要 Betti 数/维数**：用 `path_homology(..., generators=False)`。这是 gen2 起新增的模式，
  比完整模式再快 45--731 倍；$N{=}15,\maxdim{=}6$ 在 $\Q$ 上约 $0.1$ s。
- **要生成元/代表元**：用默认模式。线性代数此时仍是主要成本。
- **复现旧脚本**：本仓库大量 `*.py` 脚本以 `from core import ...` 引入；
  若要固定某一代，把该代文件复制为脚本同目录下的 `core.py` 即可
  （`_work/verify_v4/` 就是这么做的）。
- **不要再引用 "`path homology计算/core.py` 是最新版"**：它是 gen1。

## 待办

- `path homology计算/core.py` 目前是 gen1，应替换为 gen3 或删除以免混淆。
