# 设置环境变量
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_DIR=/root/slime
source $PROJECT_DIR/scripts/models/moonlight.sh
# DeepSeek-R1-Distill-Qwen-1.5B has the same vocab_size=151936 as base Qwen2.5
# Use make-vocab-size-divisible-by 2 to avoid padding mismatch during HF→Megatron conversion
# MODEL_ARGS+=(--make-vocab-size-divisible-by 2)

PARALLEL_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 2
   --expert-tensor-parallel-size 1
)

PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=8 \
    ${PROJECT_DIR}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    --hf-checkpoint /jpfs-5p/chenyanxu.9/model/Moonlight-16B-A3B-cpt-yarn-32k/iter_0002711-hf \
    --save /jpfs-5p/chenyanxu.9/model/Moonlight-16B-A3B-cpt-yarn-32k/iter_0002711_torch_dist

