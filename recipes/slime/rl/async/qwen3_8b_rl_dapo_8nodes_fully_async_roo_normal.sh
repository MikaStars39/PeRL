#!/bin/bash

# Qwen3-8B RL training with DAPO — FULLY ASYNC (off-policy) — RooNormal optimizer
# Model: Qwen3-8B-Base-sft-dolci-think
# Dataset: Polaris-Dataset-53K (math)
# Backend: Megatron (8 nodes, 64 GPUs)
# Optimizer: RooNormal (SVD-based spectral clipping with Muon-style EMA)

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
HF_CKPT="/jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think/iter_0005375-hf"
MEGATRON_CKPT="/jpfs-5p/chenyanxu.9/model/Qwen3-8B-Base-sft-dolci-think/iter_0005375_torch_dist"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SAVE_DIR:-/jpfs-5p/chenyanxu.9/model/Qwen3-8B-offpolicy-roo-normal-${TIMESTAMP}}"
DATA_PATH="/jpfs-5p/qingyu/data/profiling_20260402181029/filtered.jsonl"
LOG_DIR=${SAVE_DIR}/output.log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p ${SAVE_DIR}

# ---- model architecture (Qwen3-8B) ----
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

# ---- checkpoints ----
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --load ${MEGATRON_CKPT}
   --save ${SAVE_DIR}
   --save-interval 32
)

# ---- rollout & data ----
ROLLOUT_ARGS=(
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data ${DATA_PATH}
   --input-key prompt
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --num-rollout 2000
   --rollout-batch-size 64
   --over-sampling-batch-size 96
   --n-samples-per-prompt 8
   --rollout-max-response-len 30000
   --rollout-temperature 1.0
   --global-batch-size 128
   --partial-rollout
   --mask-offpolicy-in-partial-rollout
)

# ---- reward ----
RM_ARGS=(
   --rm-type deepscaler
)

# ---- CISPO algorithm ----
ALGO_ARGS=(
   --advantage-estimator grpo
   --loss-type custom_loss
   --custom-loss-function-path examples.cispo.cispo_loss.cispo_loss_function
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip-high 5.0
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --use-tis
   --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function
   --tis-clip 5.0
   --tis-clip-low 0.5
)

# ---- optimizer (RooNormal — SVD-based spectral clipping with Muon-style EMA) ----
# RooNormal uses explicit SVD: f(σ) = min(1/(σ+ε), clip_value)
# Momentum: Muon-style EMA with Nesterov acceleration
# Linear (2D) params use RooNormal; nonlinear params use Adam internally.
OPTIMIZER_ARGS=(
   --optimizer roo_normal
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --roo-normal-momentum 0.95
   --roo-normal-clip-value 20.0
   --roo-normal-epsilon 1e-7
   --roo-normal-scale-factor 1.0
   --roo-normal-svd-log-interval 10
   --roo-normal-svd-log-dir ${SAVE_DIR}/svd_logs
   # Nesterov is ON by default; add --roo-normal-no-nesterov to disable
   # Adam defaults for nonlinear params (embeddings, biases, norms)
   --adam-beta1 0.9
   --adam-beta2 0.98
   --adam-eps 1e-15
)

# ---- sglang rollout engine ----
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --rollout-num-gpus 48
   --sglang-mem-fraction-static 0.85
   --sglang-server-concurrency 256
)

# ---- performance / parallelism ----
# NOTE: RooNormal does NOT support --use-distributed-optimizer
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 30000
)

# ---- misc ----
MISC_ARGS=(
   --actor-num-nodes 2
   --actor-num-gpus-per-node 8
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export WANDB_API_KEY=local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6 # your_wandb_key
export WANDB_ENTITY=automl # your_wandb_entity
wandb login --relogin --host=http://11.71.1.218:8082 ${WANDB_API_KEY}

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-rl-optim
   --wandb-group roo-normal-cispo
)


# ---- launch ----
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}/:${PROJECT_DIR}/examples/fully_async\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"http://11.71.1.218:8082\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train_async.py \
   ${EVAL_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${ALGO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} 2>&1 | tee ${LOG_DIR}
