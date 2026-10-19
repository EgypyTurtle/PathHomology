# 原仓库的 gen6 更新

目标仓库：[EgypyTurtle/PathHomology](https://github.com/EgypyTurtle/PathHomology)。整理基线为 main 的 `2a226be`（Add path-homology computations for digraph families），不是之前仅有首个提交的本机整理目录。

## 保留的内容

本次在独立克隆中合并，没有改动之前那份包含未提交 README/families 的旧本地整理目录。按项目所有者要求，推送前还将在其指定的本机计算目录下保存完整的新版仓库快照；这份本机备份本身不加入 Git。

- `families/` 全部保留，不删除、不重算历史图族/Cayley 图结论。
- `results/` 历史结果保持原字节。
- `original_core.py`、`pn_graph.py`、`snf_transforms.py` 及四个旧校验脚本保持原字节。
- 原 README 原样归档为 `README_before_gen6.md`，原版本记录归档为 `VERSIONS_gen0_gen3.md`。
- 更新前的根目录 gen3 核心另保存为 `reference/gen3_core.py`，其 SHA256 为 `a6acf4c90b78b048514d16cf4d40cda56c70ca7d2d37a30bbcd3d52dbfe80009`。
- 本地 Git 历史与 origin 保留；此次按项目所有者授权正常提交和快进推送，不改写既有历史。

## 合并的更新

1. 根目录 `core.py` 更新为 gen6；增加 `morse.py`，默认 linear 及纯维数模式保留。
2. 主 README 以更新前版本为基础修改：保留原实现、历代优化表、文献核对、挠元例子与图族章节，增加 gen4–gen6、Morse 使用方式与当前性能；旧数据明确标注为历史数据。
3. 增加独立 Morse 理论说明、性能记录、gen4/gen5 回归参照、可移植测试和 GitHub CI。
4. 当前版本记录在 `VERSIONS.md`，而非继续把 gen3 称为 current。
5. `GITHUB_UPLOAD.md` 明确使用原仓库，不再要求初始化新仓库或添加新的远程地址。

## 验证边界

`verify_release.py` 检查 10 项新发布验证；`verify_repository.py` 检查四个旧验证入口，并检查其最终成功标记，避免旧脚本退出码始终为零而漏报错误。此独立克隆上的 14 项检查均已在本机通过。新产生的日志只写入 `.local-results/`，不覆盖历史结果。GitHub CI 已配置同时运行这两个入口，实际云端结果仍需在上传后查看。

这些检查不能代表重新运行 families 的全部研究扫描。原有一些 families 脚本引用未随原仓库提供的 `core_v4.py` 等依赖，本轮没有擅自替换它们所用的历史研究引擎；它们的整理和可复现性审计应另行进行。

Morse 后端支持 Q 和素数域，不代表通用整系数算法。原仓库的 `check_torsion.py` 是独立的整系数 Smith 标准形验证，仍原样保留；不能把它误读为 Morse 后端也可直接处理 Z 上非可逆枢轴。
