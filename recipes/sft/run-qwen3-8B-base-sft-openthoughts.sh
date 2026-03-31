#!/bin/bash

# for rerun the task
# pkill -9 sglang
# sleep 3
# ray stop --force
# pkill -9 ray
# pkill -9 python
# sleep 3
# pkill -9 ray
# pkill -9 python

set -ex
PROJECT_DIR=/mnt/public_02/chenyanxu/slime_optim
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="/mnt/public_02/chenyanxu/slime_optim/scripts"
source "${SCRIPT_DIR}/models/qwen3-8B.sh"

# ── Preprocess: convert OpenThoughts3 ShareGPT format to OpenAI format ──
# CONVERTED_DATA="/mnt/public_02/chenyanxu/dataset/OpenThoughts3-1.2M/openthoughts3_messages.jsonl"
# if [ ! -f "$CONVERTED_DATA" ]; then
#     python3 "${SCRIPT_DIR}/convert_openthoughts3.py"
# fi

CKPT_ARGS=(
   --hf-checkpoint /mnt/public_02/chenyanxu/model/zhennanshen/Qwen3-8B-Base/
   --ref-load /mnt/public_02/chenyanxu/model/zhennanshen/Qwen3-8B-Base_torch_dist
   --load /mnt/public_02/chenyanxu/model/zhennanshen/Qwen3-8B-Base_slime/
   --save /mnt/public_02/chenyanxu/model/zhennanshen/Qwen3-8B-Base_slime/
   --save-interval 128
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data /mnt/public_02/chenyanxu/dataset/OpenThoughts3-1.2M/openthoughts3_messages
   --input-key messages
   #--apply-chat-template
   --rollout-shuffle
   --num-epoch 5
   --rollout-batch-size 256
   --global-batch-size 256
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
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 2

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32000
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-sft
   --wandb-group qwen3-8b-base-sft-openthoughts
   --wandb-key 83966b717ea21767bfadff00107b8f73b6ba3982
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 $PROJECT_DIR/train_async.py \
   --actor-num-nodes 8 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
