#!/bin/bash

# # for rerun the task
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python

set -exo pipefail

# check env variable
: "${PROMPT_DATA:?Set PROMPT_DATA}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY}"
: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${PYTHONPATH:?Set PYTHONPATH}"
: "${WANDB_BASE_URL:?Set WANDB_BASE_URL}"
: "${SCRIPT_DIR:?Set SCRIPT_DIR}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT}"
: "${MCORE_CHECKPOINT:?Set MCORE_CHECKPOINT}"
: "${SAVE_DIR:?Set SAVE_DIR}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_GROUP:?Set WANDB_GROUP}"

cd $PROJECT_DIR
LOG_DIR=${SAVE_DIR}/output.log
mkdir -p $(dirname $LOG_DIR)

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${SCRIPT_DIR}/scripts/models/moonlight.sh"

CKPT_ARGS=(
  --hf-checkpoint $HF_CHECKPOINT
  --load $MCORE_CHECKPOINT
  --save $SAVE_DIR
  --save-interval 128
)

YARN_ARGS=(
  --original-max-position-embeddings 8192
  --rotary-scaling-factor 4.0
  --beta-fast 32.0
  --beta-slow 1.0
  --mscale 1.0
  --mscale-all-dim 1.0
)

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data $PROMPT_DATA
  --input-key messages
  # data is already in conversation format, no need to apply chat template
  #--apply-chat-template
  --rollout-shuffle
  --num-epoch 5
  --rollout-batch-size 256
  --global-batch-size 256
  --rollout-max-context-len 32000
  --seq-length 32000

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 64000
)

OPTIMIZER_ARGS=(
  --optimizer muon
  --lr 5e-5
  --lr-decay-style cosine
  --min-lr 0
  --lr-warmup-fraction 0.05
  --weight-decay 0.01
  --muon-momentum 0.95
  --muon-num-ns-steps 5
  --muon-scale-mode spectral
  --muon-extra-scale-factor 0.2
)

WANDB_ARGS=(
  --use-wandb
  --wandb-project $WANDB_PROJECT
  --wandb-group $WANDB_GROUP
  --wandb-key $WANDB_API_KEY
  --wandb-host $WANDB_BASE_URL
)

MISC_ARGS=(
  # default dropout in megatron is 0.1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  # should be good for model performance
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash

  --moe-enable-deepep
  --moe-token-dispatcher-type flex
)

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
wandb login --relogin --host=${WANDB_BASE_URL} ${WANDB_API_KEY}

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"WANDB_MODE\": \"online\",
    \"WANDB_DIR\": \"/tmp/wandb\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 ${PROJECT_DIR}/modules/slime/train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${YARN_ARGS[@]} 2>&1 | tee $LOG_DIR
