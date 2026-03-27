#!/bin/bash

# usage: bash examples/on_policy_distillation/run-qwen3-8B-opd.sh

set -ex

function nvlink() {

   export PYTHONBUFFERED=16
   
   NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)

   if [ "$NVLINK_COUNT" -gt 0 ]; then
      HAS_NVLINK=1
   else
      HAS_NVLINK=0
   fi

   echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
}

function prepare() {

   FIXED_PROJECT_NAME="slime-opd-40b-experiments"
   PROJECT_DIR=/mnt/llm-train/users/explore-train/qingyu/slime # /root/slime is mounted from the docker env
   SCRIPT_DIR=${PROJECT_DIR}/scripts # where is the scripts
   LOCAL_IP=$(hostname -I | awk '{print $1}') # get master node local ip for submitting
   TIMESTAMP=$(date +%Y%m%d_%H%M%S) # timestamp for naming
   SAVE_DIR=/mnt/llm-train/users/explore-train/qingyu/ckpt/${TIMESTAMP}_Qwen3-8B-LoRA
   LOG_DIR=$SAVE_DIR/output.log # where to save log
   mkdir -p ${SAVE_DIR} # create save dir
   DATA_DIR=/mnt/llm-train/users/explore-train/qingyu/.cache

   WANDB_HOST="http://11.71.1.218:8082"
   export WANDB_API_KEY=local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6 
   export WANDB_ENTITY=automl 
   export WANDB_PROJECT=${FIXED_PROJECT_NAME} 
   export WANDB_NAME="${TIMESTAMP}_Qwen3-8B-LoRA"
   wandb login --relogin --host=http://11.71.1.218:8082 ${WANDB_API_KEY}
}

function launch_ray_server() {
   LOCAL_IP=$(hostname -I | awk '{print $1}')

   export RAY_ENABLE_OPENTELEMETRY=0
   export RAY_DISABLE_METRICS_COLLECTION=1
   export GRPC_ENABLE_FORK_SUPPORT=0
   export RAY_USAGE_STATS_DISABLED=1
   export GRPC_POLL_STRATEGY=poll
   export OTEL_SDK_DISABLED=true
   export OTEL_TRACES_EXPORTER=none
   export OTEL_METRICS_EXPORTER=none
   export RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=0

   echo "🚀 Starting Ray Head on $LOCAL_IP..."

   ray start --head \
      --node-ip-address "$LOCAL_IP" \
      --num-gpus 8 \
      --dashboard-host 0.0.0.0 \
      --disable-usage-stats

   echo "✅ Master is up. Dashboard: http://$LOCAL_IP:8265"
}

nvlink
prepare
# launch_ray_server

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
   --hf-checkpoint /mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-7B
   --ref-load /mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-7B
   --load /mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-7B
   --save ${SAVE_DIR}
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/llm-train/users/explore-train/qingyu/.cache/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 2000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1
   --global-batch-size 64
)

GRPO_ARGS=(
   --use-kl-loss
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
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --rollout-num-gpus 24
   --sglang-mem-fraction-static 0.75
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

PERF_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216

   --lora-rank 128
   --lora-alpha 256
   --target-modules all-linear
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   # --colocate
   --use-fault-tolerance
   --dump-details /root/shared_data/qwen3-4B-fsdp-1116-noref/dump_details
   # --fsdp-cpu-offload
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${FIXED_PROJECT_NAME}
   --wandb-group language-rl
   --wandb-key "local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6" 
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} 2>&1 | tee ${LOG_DIR}