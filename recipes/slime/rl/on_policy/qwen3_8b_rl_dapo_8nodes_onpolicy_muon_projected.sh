#!/bin/bash

# Qwen3-8B RL training with DAPO + MuonProjected optimizer
# Model: Qwen3-8B-Base-sft-dolci-think
# Dataset: Polaris-Dataset-53K (math)
# Backend: Megatron (8 nodes, 64 GPUs)
# Optimizer: MuonProjected = (I - alpha * W0 W0^T / ||W0||_F^2) @ Muon(G)

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
SAVE_DIR="${SAVE_DIR:-/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-muon-projected-${TIMESTAMP}}"
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
   --prompt-data ${DATA_PATH}
   --input-key prompt
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --num-rollout 2000
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 30000
   --rollout-temperature 1.0
   --global-batch-size 512
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
   # DAPO asymmetric clipping
   --eps-clip 0.2
   --eps-clip-high 0.28
   # DAPO dynamic sampling: filter out prompts where all samples have same reward
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

# ---- optimizer (MuonProjected) ----
OPTIMIZER_ARGS=(
   --optimizer muon_projected
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   # MuonProjected: (I - alpha * W0 W0^T / ||W0||_F^2) @ Muon(G)
   --muon-projected-momentum 0.95
   --muon-projected-projection-alpha 0.5
   --muon-projected-num-ns-steps 5
   --muon-projected-scale-mode spectral
   --muon-projected-extra-scale-factor 0.2
   # Adam for non-linear params (embedding, output, 1D)
   --adam-beta1 0.9
   --adam-beta2 0.98
   --adam-eps 1e-15
)

# ---- sglang rollout engine ----
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --rollout-num-gpus 64
   --sglang-mem-fraction-static 0.8
)

# ---- performance / parallelism ----
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 4
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 30000
)

# ---- misc ----
MISC_ARGS=(
   --actor-num-nodes 8
   --actor-num-gpus-per-node 8
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
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
export WANDB_API_KEY=local-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 # your_wandb_key
export WANDB_ENTITY=duo # your_wandb_entity
wandb login --relogin --host=http://11.71.1.153:8080 ${WANDB_API_KEY}

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-rl-optim
   --wandb-group qwen3-8b-onpolicy-profiling-muon-projected-speed
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
    \"WANDB_BASE_URL\": \"http://11.71.1.153:8080\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train.py \
   ${EVAL_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} 2>&1 | tee ${LOG_DIR}
