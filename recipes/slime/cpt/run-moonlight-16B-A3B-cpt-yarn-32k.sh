#!/bin/bash

# Moonlight-16B-A3B CPT with YaRN context extension (8k -> 32k)
# Continued Pre-Training on plain text data with full causal LM loss.

set -ex

PROJECT_DIR=/jpfs/chenyanxu.9/slime_optim
cd $PROJECT_DIR

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="/jpfs/chenyanxu.9/slime_optim/scripts"
source "${SCRIPT_DIR}/models/moonlight.sh"

CKPT_ARGS=(
   --hf-checkpoint /jpfs/chenyanxu.9/model/Moonlight-16B-A3B
   --load /jpfs/chenyanxu.9/model/Moonlight-16B-A3B/Moonlight-16B-A3B_torch_dist
   --save /jpfs-5p/chenyanxu.9/model/Moonlight-16B-A3B-cpt-yarn-32k
   --save-interval 128
)

# ── YaRN context extension: 8192 -> 32768 ──
# Moonlight pretrained with max_position_embeddings=8192.
# scaling_factor = 32768 / 8192 = 4.0
YARN_ARGS=(
   --original-max-position-embeddings 8192
   --rotary-scaling-factor 4.0
   --beta-fast 32.0
   --beta-slow 1.0
   --mscale 1.0
   --mscale-all-dim 1.0
)

CPT_ARGS=(
   --rollout-function-path slime.rollout.cpt_rollout.generate_rollout
   --prompt-data /jpfs-5p/hanrongduo/data/20260404/jsonl2
   --input-key text
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 256
   --global-batch-size 256
   --seq-length 32000

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
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

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export WANDB_API_KEY=local-b0d90ad40bfaa2dd58fa4525f18c82ccb8aca2c6
export WANDB_ENTITY=automl
wandb login --relogin --host=http://11.71.1.218:8082 ${WANDB_API_KEY}

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-cpt
   --wandb-group Moonlight-16B-A3B-cpt-yarn-32k
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32

   --attention-backend flash

   --moe-enable-deepep
   --moe-token-dispatcher-type flex
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
# ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/jpfs/chenyanxu.9/PeRL/modules/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"http://11.71.1.218:8082\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /jpfs/chenyanxu.9/PeRL/modules/slime/train_async.py \
   --actor-num-nodes 4 \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${YARN_ARGS[@]} \
   ${CPT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
