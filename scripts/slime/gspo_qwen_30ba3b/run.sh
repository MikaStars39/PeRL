#!/bin/bash

# ---------------------------- system config ---------------------------
# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

PROJECT_DIR=/root/slime
SCRIPT_DIR=${PROJECT_DIR}/scripts
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh" # load model args

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

LOCAL_IP=$(hostname -I | awk '{print $1}')

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR=/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/${TIMESTAMP}_gspo_qwen30ba3b
mkdir -p ${SAVE_DIR}
LOG_DIR=$SAVE_DIR/output.log

# ---------------------------- running config ---------------------------

CKPT_ARGS=(
   --hf-checkpoint /mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s1-0103/
   --ref-load /mnt/llm-train/users/explore-train/qingyu/.cache/qwen3-30b-s1-0103_torch_dist/
   --load /mnt/llm-train/users/explore-train/qingyu/.cache/qwen3-30b-s1-0103_torch_dist/
   --save ${SAVE_DIR}
   --save-interval 32
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/llm-train/users/explore-train/qingyu/.cache/dapo-math-17k/dapo-math-17k.jsonl # training data path
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 32
   --eval-prompt-data aime /mnt/llm-train/users/explore-train/qingyu/.cache/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

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

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator gspo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 3e-4
   --eps-clip-high 4e-4
   # ref: verl gspo
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # offloading to cpu
#    --optimizer-cpu-offload
#    --overlap-cpu-optimizer-d2h-h2d
#    --use-precision-aware-optimizer
)

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
#    --sglang-enable-ep-moe
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

   # actor and rollout gpu allocations
   --actor-num-nodes 2
   --actor-num-gpus-per-node 8
   --rollout-num-gpus 16 # total rollout gpus i.e., nnodes * per_node_gpus
   --rollout-num-gpus-per-engine 8 # tp size for sglang
)

ray job submit --address="http://${LOCAL_IP}:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${PROJECT_DIR}/train.py \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} 2>&1 | tee ${LOG_DIR}