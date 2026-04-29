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

set -ex

# check env variable
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY}"
: "${PROJECT_DIR:?Set PROJECT_DIR}"
: "${PYTHONPATH:?Set PYTHONPATH}"
: "${WANDB_BASE_URL:?Set WANDB_BASE_URL}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY}"

cd $PROJECT_DIR
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

: "${SCRIPT_DIR:?Set SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/models/moonlight.sh"

CKPT_ARGS=(
  --hf-checkpoint $HF_CHECKPOINT
  --load $MCORE_CHECKPOINT
  --save $SAVE_DIR
  --save-interval 128
)

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data
  --input-key messages
  # data is already in conversation format, no need to apply chat template
  #--apply-chat-template
  --rollout-shuffle
  --num-epoch 5
  --rollout-batch-size 256
  --global-batch-size 256
  --rollout-max-context-len 8000

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 32000
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
  --wandb-project slime-sft
  --wandb-group Moonlight-16B-A3B-sft-muon
)

MISC_ARGS=(
  # default dropout in megatron is 0.1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  # should be good for model performance
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  # MLA model, do not use flash attention backend
  # --attention-backend flash

  --moe-token-dispatcher-type alltoall
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
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 /jpfs/chenyanxu.9/slime_optim/train_async.py \
  --actor-num-nodes 8 \
  --actor-num-gpus-per-node 8 \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${SFT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${MISC_ARGS[@]}
