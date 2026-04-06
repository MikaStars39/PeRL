#!/bin/bash

# Moonlight-16B-A3B RL training with DAPO (GRPO + asymmetric clipping + dynamic sampling)
# Model: Moonlight-16B-A3B-dolci-think-yarn-32k (SFT checkpoint with YaRN 8k->32k)
# Dataset: Polaris-Dataset (math)
# Backend: Megatron (8 nodes, 64 GPUs)

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ---- paths (edit these) ----
PROJECT_DIR=${PROJECT_DIR:-"/jpfs/chenyanxu.9/PeRL/modules/slime"}
MEGATRON_PATH=${MEGATRON_PATH:-"/jpfs/chenyanxu.9/PeRL/modules/Megatron-LM"}
SCRIPT_DIR="${PROJECT_DIR}/scripts"
HF_CKPT="/jpfs-5p/chenyanxu.9/model/Moonlight-16B-A3B-dolci-think-yarn-32k/iter_0004351-hf"
MEGATRON_CKPT="/jpfs-5p/chenyanxu.9/model/Moonlight-16B-A3B-dolci-think-yarn-32k/iter_0004351_torch_dist"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SAVE_DIR:-/jpfs-5p/chenyanxu.9/model/Moonlight-16B-A3B-dapo-rl-${TIMESTAMP}}"
DATA_PATH="/jpfs/chenyanxu.9/data/Polaris-V2-RL-14K/train-00000-of-00001.parquet"
LOG_DIR=${SAVE_DIR}/output.log
mkdir -p ${SAVE_DIR}

# ---- model architecture (Moonlight-16B-A3B) ----
source "${SCRIPT_DIR}/models/moonlight.sh"

# ---- checkpoints ----
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --load ${MEGATRON_CKPT}
   --save ${SAVE_DIR}
   --save-interval 32
)

# ── YaRN context extension: 8192 -> 32768 ──
# The SFT model was trained with YaRN; RL must use the same RoPE config.
YARN_ARGS=(
   --original-max-position-embeddings 8192
   --rotary-scaling-factor 4.0
   --beta-fast 32.0
   --beta-slow 1.0
   --mscale 1.0
   --mscale-all-dim 1.0
)

# ---- rollout & data ----
ROLLOUT_ARGS=(
   --prompt-data ${DATA_PATH}
   --input-key prompt
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --num-rollout 2000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 31000
   --rollout-temperature 1.0
   --global-batch-size 256
)

# ---- reward ----
RM_ARGS=(
   --rm-type deepscaler
)

# ---- DAPO / GRPO algorithm ----
GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

# ---- optimizer (Adam) ----
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ---- sglang rollout engine ----
# Moonlight-16B needs TP=2 for sglang inference as well
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --rollout-num-gpus 32
   --sglang-mem-fraction-static 0.8
)

# ---- performance / parallelism ----
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-distributed-optimizer
   --use-dynamic-batch-size
   --max-tokens-per-gpu 31000
)

# ---- misc ----
MISC_ARGS=(
   --actor-num-nodes 4
   --actor-num-gpus-per-node 8
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

EVAL_ARGS=(
   --eval-interval 32
   --eval-prompt-data aime /jpfs-5p/qingyu/data/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 31000
   --eval-top-p 0.95
)

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export WANDB_API_KEY=local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6
export WANDB_ENTITY=automl
wandb login --relogin --host=http://11.71.1.218:8082 ${WANDB_API_KEY}

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-sft
   --wandb-group moonlight-16b-dolci-think-yarn-32k-rl-dapo
)

# ---- launch ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"http://11.71.1.218:8082\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train.py \
   ${EVAL_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${YARN_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} 2>&1 | tee ${LOG_DIR}
