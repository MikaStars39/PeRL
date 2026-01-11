# 设置环境变量
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

PROJECT_DIR=/root/slime/
source ${PROJECT_DIR}/scripts/models/qwen3-30B-A3B.sh

# 使用 torchrun 启动，nproc_per_node 设为你的 GPU 数量（这里是 8）
PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 \
    ${PROJECT_DIR}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s1-0103/ \
    --save /mnt/llm-train/users/explore-train/qingyu/.cache/qwen3-30b-s1-0103_torch_dist/