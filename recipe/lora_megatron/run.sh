#!/bin/bash

set -ex

# export SGLANG_LORA_PROFILE=1
# export SGLANG_LORA_PROFILE_INTERVAL=10
# export SGLANG_LORA_ENABLE_FUSION=1

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /mnt/llm-train/users/explore-train/qingyu/miles/scripts/models/qwen3-4B.sh

PROJECT_DIR=/mnt/llm-train/users/explore-train/qingyu/miles
WANDB_HOST="http://11.71.1.218:8082"
FIXED_PROJECT_NAME="qy-lora"
SAVE_DIR=/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/Qwen3-4B-lora-ckpt
LOG_FILE=$SAVE_DIR/run.log
mkdir -p $SAVE_DIR

export WANDB_API_KEY=local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6 
export WANDB_ENTITY=automl 
export WANDB_PROJECT=${FIXED_PROJECT_NAME} 
export WANDB_NAME=${RUN_NAME}

wandb login --relogin --host=http://11.71.1.218:8082 ${WANDB_API_KEY}
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"RAY_ENABLE_OPENTELEMETRY\": \"0\",
    \"RAY_DISABLE_METRICS_COLLECTION\": \"1\",
    \"RAY_USAGE_STATS_DISABLED\": \"1\",
    \"GRPC_ENABLE_FORK_SUPPORT\": \"0\",
    \"WANDB_MODE\": \"online\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_HOST}\",
    \"WANDB_PROJECT\": \"${FIXED_PROJECT_NAME}\", 
    \"WANDB_NAME\": \"${RUN_NAME}\",
    \"WANDB_START_METHOD\": \"thread\",
    \"WANDB_INIT_TIMEOUT\": \"300\"
  }
}"


CKPT_ARGS=(
   --hf-checkpoint /mnt/llm-train/users/explore-train/qingyu/.cache/Qwen3-4B
   --save $SAVE_DIR
   --save-interval 50
)

LORA_ARGS=(
   --lora-rank 64
   --lora-alpha 32
   --lora-dropout 0.0  # +fsdp
   --target-modules all-linear
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/llm-train/users/explore-train/qingyu/.cache/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100 
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --global-batch-size 64
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime24 /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${FIXED_PROJECT_NAME}  # 这里改成了固定的 Project Name
   --wandb-group language-rl
   --wandb-key "local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6" 
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-decode-log-interval 1000
   # --sglang-enable-metrics # -fsdp
   --sglang-mem-fraction-static 0.4 # +fsdp, memory usage on H200 = 140*0.4=56GB per GPU
   # --sglang-attention-backend fa3  # +fsdp
   # --sglang-attention-backend flashinfer
   --sglang-chunked-prefill-size 4096
)

MEGATRON_ARGS=(
   # --no-offload-train
   # --no-offload-rollout
   --megatron-to-hf-mode bridge
   # --offload-rollout-level kv_cache weight  # -fsdp: not supported in megatron
   # --train-backend fsdp  # -fsdp: use megatron instead
   --train-backend megatron  # +fsdp
   --attention-dropout 0.0  # +fsdp: default dropout in megatron is 0.1
   --hidden-dropout 0.0  # +fsdp: default dropout in megatron is 0.1
   --accumulate-allreduce-grads-in-fp32  # +fsdp, megatron specific
   --attention-softmax-in-fp32  # +fsdp, megatron specific
   # --attention-backend flash  # +fsdp, megatron specific
   # --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' # +fsdp, otherwise OOM
)

PERF_ARGS=(
   # --gradient-checkpointing # +fsdp
   # --sequence-parallel # +fsdp
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --colocate
   # --rollout-num-gpus 8
   # --rollout-num-gpus-per-engine 1 # tp size for sglang
   --calculate-per-token-loss # +fsdp
   --use-miles-router # +fsdp
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 $PROJECT_DIR/train.py \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MEGATRON_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee $LOG_FILE