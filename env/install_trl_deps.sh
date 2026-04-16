#!/bin/bash
# PeRL TRL 一键安装缺失依赖
# 在 pod 内运行: bash /mnt/llm-train-5p/shenzhennan/PeRL/env/install_trl_deps.sh

set -e

echo "=== 安装 PeRL TRL 路线缺失依赖 ==="
echo "已有: torch, transformers, datasets, accelerate, flash-attn, triton, wandb, ray"
echo ""

# 核心 (trl + peft + fire 是必须的)
echo "[1/3] 核心依赖: trl, peft, fire"
pip install trl peft fire

# 训练加速 (deepspeed 用于 ZeRO-2, vllm 用于 rollout 生成)
echo "[2/3] 训练加速: deepspeed, vllm"
pip install deepspeed
pip install vllm

# 工具 (math-verify 用于数学 reward, liger-kernel 可选但脚本里有开关)
echo "[3/3] 工具: math-verify, liger-kernel"
pip install math-verify liger-kernel

echo ""
echo "=== 安装完成，验证 ==="
python3 -c "
import trl; print(f'trl: {trl.__version__}')
import peft; print(f'peft: {peft.__version__}')
import deepspeed; print(f'deepspeed: {deepspeed.__version__}')
import vllm; print(f'vllm: {vllm.__version__}')
import fire; print(f'fire: {fire.__version__}')
print('All good!')
"
