export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

: "${SOURCE_DIR:?Set SOURCE_DIR}"
source $SOURCE_DIR
# DeepSeek-R1-Distill-Qwen-1.5B has the same vocab_size=151936 as base Qwen2.5
# Use make-vocab-size-divisible-by 2 to avoid padding mismatch during HF→Megatron conversion
# MODEL_ARGS+=(--make-vocab-size-divisible-by 2)

PARALLEL_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
)

PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 \
  ${PROJECT_DIR}/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  ${PARALLEL_ARGS[@]} \
  --hf-checkpoint $1 \
  --save $2
