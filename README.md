# PathHomology

GLMY 道路同调（path homology）的一个实现，以及三轮优化的记录与对公开文献例子的吻合度核对。

参考实现来自：

> A. Grigor'yan, Y. Lin, Y. Muranov, S.-T. Yau,
> *Homologies of path complexes and digraphs*, arXiv:1207.2834v4.

---

## 仓库内容与取材原则

本仓库**只收录与公开资料有关的内容**：即"原始论文实现 + 优化记录 + 与公开文献中例子的逐项吻合度核对"。
作者本人的其他图族研究结果（自造图族、其上的群结构与生成元等）不在此列。

---

## 1. 最原始的论文实现

`original_core.py` —— 严格按论文定义实现的初版（54458 字节，2026-09-18）：

- **允许路径** $A_p$（Example 3.3）：相邻性判定的序列，**顶点允许重复**
  （论文 Definition 2.1 原文：*"a priori the vertices in the path do not have to be distinct"*）。
- **边界算子** $\partial$（(2.2)）：$\partial e_{i_0\dots i_p}=\sum_{q=0}^{p}(-1)^q e_{i_0\dots\hat i_q\dots i_p}$。
- **$\partial$ 不保持 $A_\bullet$**：删去中间顶点会把两条边并成一条，该面可能非允许，
  故须取 $\Omega_p=\{v\in A_p:\partial v\in A_{p-1}\}$（(3.8)），
  $H_p=\ker(\partial|_{\Omega_p})/\partial(\Omega_{p+1})$（(3.11)）。
- 系数域：$\Q$（`Fraction`，精确）、$\R$（float64）、$\mathbb F_p$（`GF` 类）。
- 提供三种等价的 $\Omega_p$ 算法（`kernel` / `lemma41` / `prop42`）以便互拍。
- `regular=True/False` 切换正则/非正则两种边界约定（论文 §2.3 与 (2.10)）。
  默认 `regular=False`。

当前版本为 `core.py`（第三代，87483 字节）。两者接口兼容，可直接替换。

## 2. 三轮优化分别做了什么

| 轮次 | 代号 | 字节 | 主要变化 |
|---|---|---:|---|
| — | gen0 | 54458 | 最初版。矩阵以 `dtype=object` 存放 `Fraction`/`GF`，纯 Python 对象算术 |
| 第 1 轮 | gen1 | 65728 | ① 所有素数域 $\mathbb F_p$ 转为整数 NumPy 数组做模 $p$ 消元/零空间/秩/pivot/矩阵乘，再转回 `GF` 对象；② 系数在 $\Q$ 上但可证明不溢出 int64 的矩阵乘走原生 int64 内核，真正的分数仍回退 `Fraction`；③ 利用 `_nullspace` 规范基的自由坐标单位块，换基时直接抽行，免去高斯消元；④ 圈/边界/代表元从 $\Omega$ 坐标展开改为**每维整批一次**矩阵乘；⑤ 约束矩阵只保留真正会被边界命中的行 |
| 第 2 轮 | gen2 | 82143 | ① $\Q$ 的秩与零空间改为**逐行清分母 + 本原整数行**的无分数精确消元；② 每个 $d_p$ 只求一次零空间，秩由**秩–零化度定理**得到，删掉第二次消元；③ 显式识别 $\Omega_p=A_p$ 的情形并复用已有坐标；④ 新增 **`generators=False` 维数快速模式**：只求 $\dim\Omega_p=|A_p|-\operatorname{rank}(C_p)$、$\dim Z_p=|A_p|-\operatorname{rank}(F_p)$，不构造 $\Omega$ 基、不做基变换、不生成代表元；边界行以稀疏字典保存，$\mathbb F_2$ 进一步压成 Python 大整数位集 |
| 第 3 轮 | gen3 | 87483 | ① 先算**模 2 秩**作有理秩的严格下界，达到 $\min(\text{非零行数},\text{列数})$ 即无条件给出**可证明的秩证书**，不再跑整数消元；② 利用链条件 $\operatorname{im}\partial_p\subseteq\ker\partial_{p-1}$ 给出更紧的上界，上下界相等同样成证书；达不到则**回退精确消元**（非概率算法）；③ $\operatorname{rank}\partial_1=|V|-$ 弱连通分支数（图关联矩阵定理）；④ 稀疏行消元改为原地合并非零项；⑤ 精确零空间的自由变量批量回代；⑥ $\mathbb F_2$ 中小宽度直接生成大整数位行，超宽自动切回稀疏字典后一次性打包 |

### 加速实测（同一台机器，四种算法结果的 Betti 数逐例一致）

| 算例 | 系数 | gen0 | gen3 | 加速 |
|---|---|---:|---:|---:|
| 完全 DAG $N{=}10,\maxdim{=}3$ | $\mathbb F_2$ | 23.89 s | 0.06 s | 402× |
| | $\Q$ | 35.31 s | 0.13 s | 281× |
| | $\R$ | 0.04 s | 0.03 s | 1.2× |
| 完全 DAG $N{=}10,\maxdim{=}5$ | $\mathbb F_2$ | 91.25 s | 0.18 s | 496× |
| | $\Q$ | 151.98 s | 0.43 s | 352× |

`generators=False` 维数模式（gen3 内部）：

| 算例 | 系数 | 完整模式 | 维数模式 | 再加速 |
|---|---|---:|---:|---:|
| 完全 DAG $N{=}10,\maxdim{=}5$ | $\Q$ | 0.432 s | 0.004 s | 107× |
| 完全 DAG $N{=}15,\maxdim{=}6$ | $\mathbb F_2$ | 81.4 s | 0.111 s | 731× |

两段合计：从最初版到当前维数模式，$\mathbb F_2$ 约 $2\times10^4$ 倍、$\Q$ 约 $3\times10^4$ 倍。

## 3. 与公开文献中例子的吻合度

### 3.1 GLMY 论文（arXiv:1207.2834v4）

`check_paper_examples.py` 复算论文的各个算例，**94 条断言全部通过**，覆盖
Definition 2.2/3.1/3.8、Proposition 4.2/4.7、Theorem 4.3、Example 3.3/3.9/3.14/6.17、
Fig. 8（半边形）、Fig. 24（八面体，$\dim\Omega_2=8$、$H=(1,0,1)$、$\chi=2$）等。
其中 Example 3.14（$0\leftrightarrow1$）**正则与非正则两种约定都做了自测**。
另核对 $\partial^2=0$ 在 $\Omega_\bullet$ 上成立（在 $A_\bullet$ 上不成立，正因如此才需要 $\Omega$）。

### 3.2 Tang–Yau，循环有向图（arXiv:2602.04140）

`check_circulant.py` 逐项核对：

- **定理 1.1**（$\vec C_5^{1,2}$）：论文断言 $\dim_{\mathbb K}\Omega_n=10$ 对**一切** $n\ge1$ 成立，
  且 $H_0=H_1=\mathbb K$、$H_m=0\ (m\ge2)$。实测 $n=0,\dots,7$ 的
  $\dim\Omega_n=(5,10,10,10,10,10,10,10)$，$H=(1,1,0,\dots)$ —— **吻合**。
- **定理 1.2**（$S=\{1,s\}$，$1<s<n/2$）：$s=2\Rightarrow(1,1,0,\dots)$，
  $s\ne2\Rightarrow(1,2,1,0,\dots)$。实测 $n=5,\dots,11$、$s=2,3,4$ —— **全部吻合**；
  且当 $s<n/2$ 失效时（如 $n=5,s=3$）结果确实不同，说明该假设必要。

### 3.3 Chowdhury–Huntsman–Yutin 挠元猜想

> S. Chowdhury, S. Huntsman, M. Yutin,
> *Path homologies of motifs and temporal network representations*,
> Applied Network Science **7**:4 (2022), doi:10.1007/s41109-021-00441-z.
>
> 原文：*"… we conjecture that digraphs in this family with central paths of length $2n$
> have torsion subgroups $\mathbb Z/n\mathbb Z$ in $\tilde H_1$.
> (We have computationally verified this conjecture for $n\le 8$.)"*

该族即"有向 $2n$-圈 + 两个极点交替相连"（本仓库记作 $P_n$）。
`check_torsion.py` 用精确 Smith 标准形复算：

| $n$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $H_1(P_n;\mathbb Z)$ | $\mathbb Z$ | $\mathbb Z_2$ | $\mathbb Z_3$ | $\mathbb Z_4$ | $\mathbb Z_5$ | $\mathbb Z_6$ | $\mathbb Z_7$ | $\mathbb Z_8$ | $\mathbb Z_9$ | $\mathbb Z_{10}$ | $\mathbb Z_{11}$ | $\mathbb Z_{12}$ |

（$\mathbb Z_k$ 表示 $\mathbb Z\oplus\mathbb Z/k$。）挠子群恰为 $\mathbb Z/n\mathbb Z$，
**验到 $n\le12$，比原文的 $n\le8$ 多四例**。

此外有显式生成元
$$\tau_k=-e_{a,0}+e_{a,2k}+e_{b,1}-e_{b,2k+1}-e_{0,1}+e_{2k,2k+1},
\qquad \operatorname{ord}[\tau_k]=\frac{n}{\gcd(k,n)},$$
取 $k=1$ 得阶为 $n$，故 $(\mathbb Z/n)\subseteq\operatorname{Tor}H_1$；
另一侧由自由秩与 Smith 标准形的上界夹住，两侧相合即得
$\operatorname{Tor}H_1(P_n)=\mathbb Z/n\mathbb Z$。

**约定说明。** 该文明确使用**非正则**道路同调（其脚注 3），本实现的默认
`regular=False` 与之相同。两种约定恰在有向 2-圈上分歧：对 $0\to1\to0$，
正则版 $\partial e_{010}=e_{10}+e_{01}\in A_1$（$e_{00}$ 在商里为 0）故洞被填掉、
$H_1=0$；非正则版 $\partial e_{010}=e_{10}-e_{00}+e_{01}$ 而 $e_{00}$ 不是边，
故洞保留、$H_1=\mathbb F$。

---

## 使用

环境：Python 3 + NumPy。仅需 `core.py` 即可计算。

```python
from core import path_homology
r = path_homology(vertices=[0,1,2], edges=[(0,1),(1,2)], max_dim=3, field="Q")
print(r.dim_omega, r.betti)
```

只要 Betti 数/维数时用 `generators=False`（快 45–731 倍）。

```bash
python check_paper_examples.py     # GLMY 论文算例
python check_circulant.py          # Tang-Yau 循环有向图定理 1.1 / 1.2
python check_torsion.py            # Chowdhury-Huntsman-Yutin 挠元猜想
python verify_all.py               # 全套自检
```

## 文件

```
core.py                  当前实现（gen3）
original_core.py         最初版（gen0）
docs/OPTIMIZATION_REPORT.md   优化报告全文
docs/VERSIONS.md              四代版本对照（字节/日期/哈希/变化/哪些结论出自哪一代）
check_paper_examples.py  GLMY 论文算例核对
check_circulant.py       Tang-Yau 定理核对
check_torsion.py         挠元猜想复算
results/                 上述三项的原始输出
```

## 引用

1. A. Grigor'yan, Y. Lin, Y. Muranov, S.-T. Yau, *Homologies of path complexes and digraphs*, arXiv:1207.2834.
2. X. Tang, S.-T. Yau, *Path Homology of Circulant Digraphs*, arXiv:2602.04140.
3. S. Chowdhury, S. Huntsman, M. Yutin, *Path homologies of motifs and temporal network representations*, Applied Network Science **7**:4 (2022).
