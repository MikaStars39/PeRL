# 设置环境变量
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_DIR=/jpfs/qingyu/PeRL/modules/slime
source $PROJECT_DIR/scripts/models/qwen2.5-1.5B.sh
# DeepSeek-R1-Distill-Qwen-1.5B has the same vocab_size=151936 as base Qwen2.5
# Use make-vocab-size-divisible-by 2 to avoid padding mismatch during HF→Megatron conversion
MODEL_ARGS+=(--make-vocab-size-divisible-by 2)

PARALLEL_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 4
   --expert-model-parallel-size 1
)

PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 \
    ${PROJECT_DIR}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    --hf-checkpoint /jpfs/models/DeepSeek-R1-Distill-Qwen-1.5B \
    --save /jpfs/qingyu/PeRL/ckpt/DeepSeek-R1-Distill-Qwen-1.5B_megatron