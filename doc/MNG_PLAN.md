# Matrix Natural Gradient (Roo) 优化器实现计划

## Context

根据《面向强化学习的谱感知矩阵优化器》技术报告中的"方法二：矩阵自然梯度"，在 Megatron-LM 中实现该优化器，使其可在 slime RL 训练框架中通过 `--optimizer roo` 直接使用。

Roo 的核心谱变换是 f(σ) = σ/(σ²+ε²)，反转奇异值大小关系，放大 off-principal 方向更新——这正是 RLVR 训练动力学（Three-Gate Theory）所需要的。

## 核心算法

```
M_t = β * M_{t-1} + G_t                    # 动量累积
A = M_t^T @ M_t + ε² I                     # Gram 矩阵 + 正则化
A_inv ≈ NS_inverse(A, steps=5)             # Newton-Schulz 逆矩阵迭代（纯 GEMM）
Φ = M_t @ A_inv                            # 矩阵自然梯度方向
Φ = Φ / RMS(Φ) * scale_factor              # RMS 归一化
W_t = W_{t-1} - η * (Φ + λ * W_{t-1})     # 权重更新 (decoupled weight decay)
```

### 为什么用 Newton-Schulz 而非 torch.linalg.solve

`torch.linalg.solve` 底层走 Cholesky/LU 分解，包含大量标量操作和复杂内存访问模式，**无法利用 GPU Tensor Cores**。而 Newton-Schulz 逆矩阵迭代全程只有矩阵乘法（纯 GEMM），完美吃满 A100/H100 的 Tensor Cores，且与 Muon 的 NS 基础设施高度一致。

### Newton-Schulz 逆矩阵迭代

对 A = M^T M + ε²I 求逆：

```
# 缩放保证收敛
c = ||A||_F   (或 Tr(A)，计算极快)
Ã = A / c

# 迭代：X_k → A^{-1}/c
X_0 = I
X_{k+1} = X_k (2I - Ã X_k)    # 每步仅 2 次 GEMM
                                 # 重复 5~10 步

# 反归一化
A^{-1} ≈ X_K / c
```

收敛条件：||I - Ã X_0||₂ < 1。用 ||A||_F 归一化后 X_0=I 即满足。对优化器而言，5 步近似逆已完全足够——其带来的隐式正则化效应甚至比精确求逆更好。

## 需要修改/创建的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `megatron/core/optimizer/matrix_natural_gradient.py` | **新建** | Roo 优化器核心类 + 工厂函数 |
| `megatron/core/optimizer/optimizer_config.py` | 修改 | 增加 `roo_*` 配置字段 |
| `megatron/training/arguments.py` | 修改 | 增加 `'roo'` 到 optimizer choices + CLI 参数 |
| `megatron/training/training.py` | 修改 | 增加 Roo 分发路径 |
| `slime/backends/megatron_utils/model.py` | 修改 | 增加 Roo 分发路径 |

## 详细步骤

### Step 1: `optimizer_config.py` — 增加 Roo 配置字段

在 `OptimizerConfig` 的 Muon 配置段后（~line 157）添加：

```python
# Roo (Matrix Natural Gradient)
roo_momentum: float = 0.95
"""Momentum coefficient for Roo optimizer."""

roo_epsilon: float = 0.01
"""Regularization epsilon for Roo inverse-spectral transform."""

roo_scale_factor: float = 1.0
"""Scale factor applied after RMS normalization."""

roo_split_qkv: bool = True
"""Whether to split QKV parameters for Roo optimizer."""

roo_tp_mode: str = "allreduce_gram"
"""TP handling mode: 'allreduce_gram' or 'blockwise'."""

roo_num_ns_steps: int = 5
"""Number of Newton-Schulz iteration steps for gram matrix inversion."""

roo_fp32_matmul_prec: str = "medium"
"""FP32 matmul precision for NS iteration ('low', 'medium', 'high')."""
```

### Step 2: `arguments.py` — CLI 参数

- 把 `--optimizer` 的 choices 从 `['adam', 'sgd', 'muon', 'dist_muon']` 改为 `['adam', 'sgd', 'muon', 'dist_muon', 'roo']`
- 在 Muon 参数段后添加 Roo CLI 参数：
  - `--roo-momentum` (float, default 0.95)
  - `--roo-epsilon` (float, default 0.01)
  - `--roo-scale-factor` (float, default 1.0)
  - `--roo-no-split-qkv` (action, dest='roo_split_qkv')
  - `--roo-tp-mode` (str, choices=['allreduce_gram', 'blockwise'])
  - `--roo-num-ns-steps` (int, default 5)
  - `--roo-fp32-matmul-prec` (str, choices=['low', 'medium', 'high'])
- 在 validation 段（~line 1226）添加 Roo 检查（不支持 distributed optimizer / FSDP）
- 在 `get_megatron_optimizer_config()`（line 1185）条件里加上 `or args.optimizer == 'roo'`

### Step 3: `matrix_natural_gradient.py` — 核心实现（新建）

**A. Newton-Schulz 逆矩阵求解函数**

```python
def newton_schulz_inverse(A: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """用 Newton-Schulz 迭代求 A 的近似逆。A 必须对称正定。

    全程只有 GEMM 操作，完美利用 Tensor Cores。

    Args:
        A: 对称正定矩阵 [n, n]（即 M^T M + ε²I）
        steps: 迭代步数（默认 5）

    Returns:
        A 的近似逆矩阵 [n, n]
    """
    n = A.shape[0]

    # 归一化：确保 ||I - Ã||₂ < 1
    c = A.norm()  # Frobenius 范数
    A_tilde = A / c

    # X_0 = I
    X = torch.eye(n, dtype=A.dtype, device=A.device)

    # 迭代：X_{k+1} = X_k @ (2I - A_tilde @ X_k)
    I_2 = 2.0 * torch.eye(n, dtype=A.dtype, device=A.device)
    for _ in range(steps):
        X = X @ (I_2 - A_tilde @ X)

    # 反归一化
    return X / c
```

**可选高阶变体**（更快收敛，与 Muon 的五次多项式 NS 一致）：

```python
# 三阶 NS (每步 3 次 GEMM，但收敛更快):
# X_{k+1} = X_k (3I - A_tilde @ X_k (3I - A_tilde @ X_k))
# 或等价：X_{k+1} = X_k (a I + b A_tilde X_k + c (A_tilde X_k)^2)
# 系数可优化以最大化收敛盆地
```

**B. TP 感知的 Gram 矩阵计算**

```python
def compute_gram_with_tp(
    M: torch.Tensor,
    eps: float,
    tp_group: ProcessGroup | None,
    partition_dim: int | None,
    tp_mode: str,
) -> torch.Tensor:
    """计算 M^T M + ε²I，处理 TP 分片。"""
    gram = M.t() @ M

    # partition_dim == 0 (行切分): M^T M = Σ M_i^T M_i，需要 allreduce
    if (tp_group is not None
        and tp_mode == "allreduce_gram"
        and partition_dim == 0):
        torch.distributed.all_reduce(gram, group=tp_group)

    # partition_dim == 1 (列切分): 各 rank 独立，无需通信

    gram.diagonal().add_(eps * eps)
    return gram
```

**C. `MatrixNaturalGradient(torch.optim.Optimizer)` 类**

独立的 PyTorch 优化器（不依赖 `emerging_optimizers`），关键方法：

- `__init__`: 存储超参、TP 配置、QKV 分割配置、NS 步数
- `step()`: 遍历参数组，对 2D 参数调用 `_roo_transform`
- `_roo_transform_single(M, eps, sf, tp_group, partition_dim)`:
  1. 保存/设置 `torch.backends.cuda.matmul.allow_tf32` 精度
  2. 将 M upcast 到 float32（数值稳定）
  3. `gram = compute_gram_with_tp(M, eps, tp_group, partition_dim, tp_mode)`
  4. `A_inv = newton_schulz_inverse(gram, steps=num_ns_steps)`
  5. `Φ = M @ A_inv`
  6. 恢复原始 dtype
  7. RMS 归一化 + scale
- `_roo_transform_qkv`: 将 QKV 参数拆分为 Q/K/V，分别做 Roo 变换，再拼回

**TP 处理逻辑**：
- `partition_dim == 0`（行切分，如 RowParallelLinear）：gram 需 allreduce，因为 M^T M = Σ M_i^T M_i
- `partition_dim == 1`（列切分，如 ColumnParallelLinear）：每个 TP rank 独立求解，无需通信
- `partition_dim is None` 或 `blockwise`：无通信

**D. `get_megatron_roo_optimizer()` 工厂函数**

完全仿照 `get_megatron_muon_optimizer()`（muon.py:164-332）的模式：
1. 将参数分为 linear_params（2D 非 embedding）和 nonlinear_params
2. 冻结 nonlinear → 创建 Roo 的 param_groups → 创建 `MatrixNaturalGradient`
3. 包装 `Float16OptimizerWithFloat16Params`（bf16）
4. 冻结 linear → 创建 Adam（通过 `get_megatron_optimizer`）
5. 解冻全部 → 返回 `ChainedOptimizer([roo_wrapped, adam_wrapped])`

### Step 4: `training.py` — 分发路径

- 导入 `get_megatron_roo_optimizer`
- `get_megatron_optimizer_config()` line 1185: 条件改为 `if args.optimizer == 'adam' or 'muon' in args.optimizer or args.optimizer == 'roo':`
- `setup_model_and_optimizer()` line 1234: 改为三路分支
  ```python
  if 'muon' not in config.optimizer and config.optimizer != 'roo':
      optimizer = get_megatron_optimizer(...)
  elif config.optimizer == 'roo':
      optimizer = get_megatron_roo_optimizer(...)
  else:
      optimizer = get_megatron_muon_optimizer(...)
  ```

### Step 5: `slime/backends/megatron_utils/model.py` — Slime 集成

- 添加导入（try/except 模式）：
  ```python
  try:
      from megatron.core.optimizer.matrix_natural_gradient import get_megatron_roo_optimizer
      HAS_ROO_OPTIMIZER = True
  except ImportError:
      HAS_ROO_OPTIMIZER = False
  ```
- `setup_model_and_optimizer()` line 126: 改为三路分支
  ```python
  if HAS_MUON_OPTIMIZER and 'muon' in config.optimizer:
      ...  # 现有 Muon 路径
  elif HAS_ROO_OPTIMIZER and config.optimizer == 'roo':
      optimizer = get_megatron_roo_optimizer(config=config, model_chunks=model, ...)
  else:
      ...  # 现有 Adam/SGD 路径
  ```

## Weight Decay 设计

### 分析

Roo 的 ε（Tikhonov 正则化）和 weight decay λ 控制的是正交的两件事：
- **ε** 控制更新的**方向**分配（谱空间中哪些分量被强调）
- **λ** 控制参数的**幅度**衰减（权重空间中的全局缩小）

标准 decoupled WD 对 W 的所有奇异值等比例缩小：σᵢ → (1-ηλ)σᵢ。
这与 RL 的"保护 principal 方向"目标有微妙冲突（大奇异值的绝对衰减量最大）。
但 RL 训练步数短（几百步）、λ 值小，实际影响可忽略。

### 决定：采用标准 decoupled weight decay

```
W_t = (1 - η·λ) W_{t-1} - η·Φ
```

- 与 Muon/AdamW 行为一致，生态兼容
- 默认 λ=0.01（可通过 `--weight-decay` 调节，设 0 即关闭）
- ε 是 Roo 的核心超参，weight decay 是辅助角色

## 计算开销分析

设 Gram 矩阵尺寸为 n×n（n = 权重矩阵的较小维度）：

| 操作 | FLOPs | 说明 |
|------|-------|------|
| M^T @ M | 2mn² | 一次 GEMM |
| NS 每步：A_tilde @ X | 2n³ | GEMM |
| NS 每步：X @ (2I - ...) | 2n³ | GEMM |
| NS 共 K 步 | 4Kn³ | K=5 时 ≈20n³ |
| M @ A_inv | 2mn² | 一次 GEMM |
| **总计** | 4mn² + 20n³ | 全部 Tensor Core 可加速 |

对比 Muon（5 步 NS for msign，每步 3 次 GEMM on M 本身）：Muon 总量 ≈ 15·2mn·min(m,n)。Roo 和 Muon 量级相当，但 Roo 多了一个 n×n 的 NS 循环。对于 TP 后的局部维度，n 通常不大，开销可接受。

## 使用方式

Slime 启动脚本中：
```bash
--optimizer roo \
--roo-momentum 0.95 \
--roo-epsilon 0.01 \
--roo-num-ns-steps 5 \
--roo-scale-factor 1.0 \
--lr 3e-4
```

## 验证方案

1. **数学正确性**：小矩阵上验证 NS 逆矩阵迭代与 `torch.linalg.inv` 结果一致（误差 < 1e-4）
2. **谱变换正确性**：验证 Roo 输出的谱变换与 f(σ) = σ/(σ²+ε²) 的 SVD 显式计算一致
3. **TP 正确性**：TP=2 下 `allreduce_gram` 模式结果与 TP=1 对比
4. **训练跑通**：用小模型 `--optimizer roo` 跑几步确认 loss 下降
5. **Checkpoint**：保存/加载 checkpoint 确认 momentum_buffer 正确恢复
6. **Slime 端到端**：通过 slime 的 RL 训练脚本使用 `--optimizer roo` 验证完整流程
