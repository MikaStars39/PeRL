# GASD (Geometry-Aware Steepest Descent) 优化器调研文档

## 1. 概述

GASD 是在 Megatron-LM 中实现的一种面向 RLVR（Reinforcement Learning with Verifiable Rewards）训练的优化器。它在 Muon 正交化的基础上，额外引入了基于当前权重矩阵谱结构的预条件（preconditioning），使更新方向更好地匹配 RLVR 训练动力学的需求。

**核心思想**：先用 Muon（Newton-Schulz 正交化）对梯度做正交归一化，再通过求解线性系统 `(WW^T + εI) Δ = Φ` 引入权重谱结构信息，放大 off-principal 方向的更新、抑制 principal 方向的更新。

**代码位置**：`modules/Megatron-LM/megatron/core/optimizer/gasd.py`

**使用方式**：在 Megatron-LM 中通过 `--optimizer gasd` 启用。

## 2. 算法流程

```
输入：梯度 G_t, 当前权重 W, 动量系数 β, 学习率 η, 权重衰减 λ

Step 1 — 动量累积（EMA）：
    M_t = β * M_{t-1} + (1 - β) * G_t

Step 2 — Nesterov 前瞻（可选）：
    G_nesterov = lerp(G_t, M_t, β)

Step 3 — Muon 正交化：
    Φ = NewtonSchulz(G_nesterov) * scale_factor

Step 4 — GASD 预条件（Conjugate Gradient）：
    解线性系统 (WW^T + ε·I) Δ = Φ，使用 CG 法求解

Step 5 — RMS 归一化：
    Δ = Δ / RMS(Δ) * rms_scale

Step 6 — 权重更新（decoupled weight decay）：
    W_t = (1 - η·λ) * W_{t-1} - η * Δ
```

### 2.1 与 Muon 的关系

GASD 在 Muon 的基础上增加了 Step 4（GASD 预条件）。Muon 的核心是 Newton-Schulz 正交化（近似矩阵符号函数 msign），使梯度在奇异值谱上均匀分布。GASD 进一步利用权重矩阵 W 的谱结构对更新进行非均匀调制。

### 2.2 与 Roo 的关系

Roo（Matrix Natural Gradient）使用 Gram 矩阵求逆 `(M^T M + ε²I)^{-1}` 实现谱变换 `f(σ) = σ/(σ²+ε²)`，基于梯度动量的谱结构。

GASD 则使用**当前权重** W 的谱结构 `(WW^T + εI)^{-1}` 进行预条件，每步重新计算。关键区别：

| 维度 | Roo | GASD |
|------|-----|------|
| 预条件矩阵 | M^T M + ε²I（基于梯度动量） | WW^T + εI（基于当前权重） |
| 正交化 | 无（直接谱变换） | 先 Muon 正交化，再谱变换 |
| 求逆方法 | Newton-Schulz 逆矩阵迭代 | Conjugate Gradient |
| 矩阵维度 | m×m（较小维度） | n×n（行数维度） |
| 每步更新 | Gram 矩阵由动量累积 | W 每步重新计算 |

## 3. GASD 预条件的数学原理

### 3.1 为什么用 `(WW^T + εI)^{-1}`

权重矩阵 W 的 SVD 为 `W = U Σ V^T`，则：

```
WW^T = U Σ² U^T

(WW^T + εI)^{-1} = U diag(1/(σ_i² + ε)) U^T
```

对 Muon 正交化后的更新 Φ 施加此预条件：
- **大奇异值方向**（σ_i 大 → 1/(σ_i²+ε) 小）：更新被**抑制**
- **小奇异值方向**（σ_i 小 → 1/(σ_i²+ε) 大）：更新被**放大**

这正好符合 Three-Gate Theory 的观察：RLVR 训练中有效的参数更新主要发生在 off-principal 子空间（小奇异值方向）。GASD 的预条件自动实现了这种偏向。

### 3.2 自适应 ε

```python
eps = alpha * ||W||_F^2 / min(n, m)
```

- `||W||_F^2 = Σ σ_i²`，即所有奇异值的平方和
- 除以 `min(n, m)` 得到**平均奇异值平方**
- `alpha` 控制阻尼强度

当 `alpha = 1.0` 时，ε 约等于平均 σ²，此时：
- 大于平均值的奇异值方向被抑制
- 小于平均值的奇异值方向被放大
- `alpha` 更大 → 更像 SGD（均匀更新）
- `alpha` 更小 → 更激进地放大 off-principal 方向

### 3.3 Conjugate Gradient 求解

GASD 没有显式构建 n×n 的 `WW^T` 矩阵，而是使用 CG 迭代隐式求解：

```python
# 每步 CG 只需两次矩阵乘法：
v = W^T @ P        # [m, m_cols] → [m_cols]，不存 WW^T
AP = W @ v + eps * P   # 隐式 (WW^T + εI) @ P
```

这避免了 O(n²) 的内存开销，且 CG 的每步操作都是 GEMM，可利用 Tensor Cores 加速。

**CG vs Newton-Schulz**（与 Roo 的对比）：
- Roo 用 NS 需要 O(m²) 存储 Gram 矩阵 + 逆矩阵
- GASD 用 CG 只需 O(n×m) 存储 W 本身（已有）+ O(n) 的 CG 向量
- CG 的收敛速度取决于条件数，但 ε 正则化保证了条件数有界

## 4. 超参数

### 4.1 完整超参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gasd-momentum` | 0.95 | EMA 动量系数 β |
| `--gasd-no-nesterov` | (默认启用 Nesterov) | 禁用 Nesterov 动量 |
| `--gasd-epsilon-alpha` | 1.0 | 自适应 ε 系数：`eps = α * ‖W‖²_F / min(n,m)` |
| `--gasd-cg-iters` | 10 | CG 迭代次数 |
| `--gasd-rms-scale` | 1.0 | RMS 归一化后的缩放因子 |
| `--gasd-split-qkv` / `--gasd-no-split-qkv` | True | 是否拆分 QKV 权重分别处理 |
| `--gasd-num-ns-steps` | 5 | Muon Newton-Schulz 迭代步数 |
| `--gasd-scale-mode` | "spectral" | Muon 缩放模式（spectral / unit_rms_norm / shape_scaling） |
| `--gasd-extra-scale-factor` | 1.0 | Muon 更新的额外缩放因子 |
| `--gasd-fp32-matmul-prec` | "medium" | NS 迭代中的 FP32 matmul 精度（low/medium/high） |
| `--gasd-tp-mode` | "blockwise" | Tensor Parallel 模式（blockwise / duplicated / distributed） |

### 4.2 关键超参数分析

**`epsilon_alpha`**（最关键的超参数）：
- 控制 off-principal 放大的强度
- `alpha → ∞`：ε 很大，`(WW^T + εI)^{-1} ≈ I/ε`，退化为 Muon
- `alpha → 0`：ε 很小，强烈放大 off-principal 方向，可能数值不稳定
- 默认 1.0 是一个保守的起点

**`cg_iters`**：
- CG 迭代次数，越大求解越精确
- 10 步通常已足够收敛
- 每步额外开销 = 2 次 GEMM（W^T @ P + W @ v）

**`rms_scale`**：
- CG 输出后的 RMS 归一化缩放
- 类似于 Roo 中的 scale_factor，控制更新幅度

## 5. 参数分组策略

GASD 采用与 Muon/Roo 相同的参数分组策略：

- **2D 线性层权重**（非 embedding）→ 使用 GASD（Muon + CG 预条件）
- **非 2D 参数**（embedding, LayerNorm, bias 等）→ 使用 Adam
- 通过 `ChainedOptimizer` 将两组优化器组合

QKV 权重特殊处理：当 `split_qkv=True` 时，将 fused QKV 权重拆分为 Q/K/V 三部分，分别独立做 GASD 变换后拼回。

## 6. 计算开销分析

设权重矩阵 W 形状为 [n, m]：

| 操作 | FLOPs | 说明 |
|------|-------|------|
| Newton-Schulz (Muon) | ~15 × 2nm × min(n,m) | 5步 NS，每步 3 次 GEMM |
| CG 每步：W^T @ P | 2nm | GEMM |
| CG 每步：W @ v | 2nm | GEMM |
| CG 共 K 步 | 4K × nm | K=10 时 = 40nm |
| RMS 归一化 | O(nm) | 可忽略 |
| **GASD 总计** | Muon + 40nm | |

对比：
- **Muon**: ~30nm × min(n,m) FLOPs
- **Roo**: ~4mn² + 20n³ FLOPs（Gram 矩阵 + NS 求逆）
- **GASD**: Muon + 40nm FLOPs（Muon + CG）

GASD 的 CG 部分开销 = 40nm，与 Muon 的 NS 部分（~30nm²）相比，CG 部分更廉价（线性于 nm 而非 nm²）。总体开销 ≈ Muon 的 1.x 倍。

## 7. 限制

代码中明确的约束：
- **不支持 Distributed Optimizer**（`use_distributed_optimizer` 不能为 True）
- **不支持 FSDP**（Torch-FSDP2 和 Megatron-FSDP 均不支持）
- **不支持 FP16**（仅支持 BF16 和 FP32）
- **Checkpoint 格式**：仅支持 `torch` 和 `torch_dist`
- **依赖 `emerging_optimizers`**：需要安装 NVIDIA 的 Emerging-Optimizers 包（用于 Muon 的 NS 计算）

## 8. Slime 集成情况

目前 slime 代码库中**没有找到 GASD 的集成代码**。GASD 仅在 Megatron-LM 层面实现：
- `megatron/core/optimizer/gasd.py`：核心优化器类 + 工厂函数
- `megatron/core/optimizer/optimizer_config.py`：配置字段
- `megatron/training/arguments.py`：CLI 参数
- `megatron/training/training.py`：分发路径

如需在 slime RL 训练中使用 GASD，需要参照 Roo 的集成方式，在 `slime/backends/megatron_utils/model.py` 中添加 GASD 的分发路径。

## 9. 与同族优化器的设计哲学对比

这三个优化器（Muon、Roo、GASD）共享 Three-Gate Theory 的核心动机：**RLVR 训练中有效更新主要发生在 off-principal 子空间**。

```
梯度 G
    │
    ├── Muon:  G → NS 正交化 → 各奇异值方向等权重更新
    │          （不区分 principal vs off-principal）
    │
    ├── Roo:   G → 动量 → Gram 求逆 → 放大小奇异值梯度方向
    │          （基于梯度动量的谱结构）
    │
    └── GASD:  G → 动量 → Muon NS → (WW^T+εI)^{-1} → 放大 W 的 off-principal 方向
               （先正交化，再基于权重的谱结构做二次调制）
```

GASD 可以理解为 Muon + Roo 的融合：
- 继承 Muon 的梯度正交化（消除梯度自身的奇异值偏差）
- 额外引入类似 Roo 的谱变换，但基于权重矩阵 W 而非梯度动量 M

## 10. 关键设计决策总结

| 决策 | 选择 | 理由 |
|------|------|------|
| 预条件基于权重 W 而非梯度 M | WW^T | RLVR 中 W 的谱结构（principal 方向）才是需要感知的对象 |
| 用 CG 而非 NS 求逆 | CG | 避免显式构建 n×n 矩阵，内存更友好 |
| 先 Muon 再 GASD | 两步串联 | Muon 先消除梯度内部的奇异值偏差，GASD 再根据 W 的谱做选择性放大 |
| 自适应 ε | α·‖W‖²_F/min(n,m) | 自动适配不同层的权重尺度，无需手动调节 |
| RMS 归一化 | CG 输出后归一化 | 稳定更新幅度，使学习率跨层一致 |
| Decoupled WD | 标准方式 | 与 Muon/AdamW 生态兼容 |
