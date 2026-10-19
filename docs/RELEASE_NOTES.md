# 发布目录与适配说明

本目录是对 `https://github.com/EgypyTurtle/PathHomology` 的独立本地克隆更新。保留原仓库历史、origin 和现有研究文件。项目所有者已授权将完整新版先备份到指定本机目录，再正常提交和推送；旧 Git 整理目录不动。具体合入说明见 `REPOSITORY_UPDATE.md`。

## 范围

```text
PathHomology_github/
  README.md, GITHUB_UPLOAD.md       用法与上传步骤
  requirements.txt                 已验证的 NumPy 版本
  core.py, morse.py                gen6 核心与精确 Morse 消去
  cli.py, visualize.py, __init__.py 兼容入口与可选绘图工具
  examples.py                     可运行的小例子
  test_*.py, verify_release.py      独立、可移植测试入口
  docs/                            独立理论说明、性能报告和本说明
  performance_morse_20261019/      68 次正式测量、历史回归摘要及秩核对
  _baseline_gen4_20261019/          第五轮测试的固定参考源码
  _baseline_gen5_20261019/          Morse 性能比较的固定参考源码
  reference/                      gen0 与历史秩验证脚本
  benchmark_*.py, audit_morse.py    测速及历史结果审计
  .github/workflows/tests.yml      上传后的 Windows/Linux CI
  SHA256SUMS.txt                   固定源码清单
```

不新增 persistent、laplacian、本机备份、研究草稿、缓存、虚拟环境，以及探索阶段 `initial.jsonl`。本地克隆保留 Git 历史，网页上传 ZIP 排除 `.git`；原仓库已有的研究文件继续保留。不应把本目录覆盖到 persistent 仓库后误认为普通 Morse 消去自动保留持久结构。

## 源码版本

| 名称 | SHA256 前 12 位 | 位置 |
|---|---|---|
| gen0 | `76f2be7ee555` | `reference/gen0_core.py` |
| gen4 | `3d7e315c2dde` | `_baseline_gen4_20261019/core.py` |
| gen5 | `dd247bdcfa5a` | `_baseline_gen5_20261019/core.py` |
| gen6 | `ffe155a05112` | `core.py` |
| Morse 模块 | `bb946f1275c2` | `morse.py` |

完整哈希见 `SHA256SUMS.txt`。这些参考版本是测试输入和可复现证据，不是无用的本机备份；不应被 `.gitignore` 排除。核心及历史基准依赖均保持原字节，数学算法没有在整理时再次改动。

## 为公开发布做的适配

1. 在原 README 基础上保留并扩充内容，补充依赖、快速示例、GitHub 更新说明和只读权限的正确性 CI。
2. `test_optimization.py` 改用仓库内 gen0 参考文件，不再查找本机绝对路径。
3. `test_fields.py` 在出现不一致时返回非零退出码，避免 CI 误报通过。
4. `audit_morse.py` 能读取历史 Windows 路径分隔符。历史日志中的 `laplacian/core.py` 与 gen5 快照完全同哈希，因此以该快照作档案映射核验；这不代表本包包含或重新测试了 Laplacian。
5. 历史 `verification.json` 只公开源码哈希、退出码及耗时摘要；移除含本机路径的 stdout/stderr，不改变测试结果和计时数值。原始完整日志仍在原工作目录。
6. 新验证和秩核对输出到被 Git 忽略的 `.local-results/`，不覆盖发布的历史记录。历史秩核对脚本原字节保存在 `reference/`，用于核验当时记录的脚本哈希。
7. `.gitattributes` 保留文件字节，避免 Git 的换行转换破坏历史源码哈希。

历史 11 组回归包含单独的 persistent/Laplacian 工程；本包的 `verify_release.py` 实际运行 5 组核心测试、示例、独立秩核对、历史审计和 2 个 CLI 检查，共 10 项，不能把两者混为同一套新测试。

## 运行与平台

推荐使用已验证的 Python 3.12 和 `requirements.txt` 中的 NumPy。核心没有系统专用接口；本机发布验证在 Windows 上执行。CI 配置会在上传后执行 Windows/Linux 检查，不能在上传前宣称云端测试已通过。

历史资源隔离测速使用 Windows 的进程内存接口，不承诺在 Linux 上直接运行该测速入口。`verify_morse_dimensions.py` 只使用图生成函数，不调用 Windows 内存接口。

原始核心和参考源码被哈希固定。若以后修改算法，请记录新版本、更新相应基准证据及审计安排，而不要保留旧测量数字却悄悄把它们归给新源码。

## 发布前待项目所有者决定

尚未指定 LICENSE，本次不替作者选择。本次已明确更新 `EgypyTurtle/PathHomology`；请按 `GITHUB_UPLOAD.md` 审阅、提交并推送到原仓库。
