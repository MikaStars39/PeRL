#!/bin/bash
# PeRL TRL 路线依赖检测脚本
# 进 pod 后运行: bash /mnt/llm-train-5p/shenzhennan/PeRL/env/check_deps.sh

echo "=== Python & CUDA ==="
python3 --version 2>/dev/null || echo "python3: NOT FOUND"
python3 -c "import torch; print(f'torch: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}')" 2>/dev/null || echo "torch: NOT FOUND"

echo ""
echo "=== 核心训练依赖 ==="
CORE_PKGS=(
    "transformers"
    "trl"
    "peft"
    "datasets"
    "accelerate"
    "deepspeed"
    "vllm"
)
for pkg in "${CORE_PKGS[@]}"; do
    ver=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$pkg'))" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  $pkg: $ver"
    else
        echo "  $pkg: MISSING"
    fi
done

echo ""
echo "=== GPU 加速 ==="
GPU_PKGS=(
    "flash_attn:flash-attn"
    "xformers:xformers"
    "triton:triton"
)
for entry in "${GPU_PKGS[@]}"; do
    import_name="${entry%%:*}"
    pip_name="${entry##*:}"
    ver=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$pip_name'))" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  $pip_name: $ver"
    else
        echo "  $pip_name: MISSING"
    fi
done

echo ""
echo "=== 工具 & 日志 ==="
UTIL_PKGS=(
    "wandb"
    "fire"
    "ray"
    "liger_kernel:liger-kernel"
    "math_verify:math-verify"
)
for entry in "${UTIL_PKGS[@]}"; do
    if [[ "$entry" == *:* ]]; then
        pip_name="${entry##*:}"
    else
        pip_name="$entry"
    fi
    ver=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('$pip_name'))" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  $pip_name: $ver"
    else
        echo "  $pip_name: MISSING"
    fi
done

echo ""
echo "=== 完整 pip list (供对照) ==="
pip list 2>/dev/null | wc -l
echo "(共以上数量的包)"
