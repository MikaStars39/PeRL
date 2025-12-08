#! /bin/bash

set -exo pipefail
ulimit -n 65535

PROJECT_DIR="."
BASE_MODEL_PATH="/mnt/shared-storage-user/p1-shared/Qwen/DeepSeek-R1-Distill-Qwen-1.5B"

DATASET="aime2024@512,aime2025@512,amc2023@32,math500@8,minerva@8,hmmt2025@32"
# DATASET="aime2024@512,aime2025@512" # for test

export PYTHONPATH="${PROJECT_DIR}"
export HF_ENDPOINT="https://hf-mirror.com"
export LD_LIBRARY_PATH="/root/miniconda3/envs/perl/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
# export VLLM_LOGGING_LEVEL="DEBUG"

TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DP_SIZE=8
TP_SIZE=1
MAX_NUM_REQUEST=2000
GPU_MEMORY_UTILIZATION=0.95

function kill_vllm_processes() {
  pkill -9 python || true;
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
  sleep 1;
  pkill -9 python || true;
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
}

function eval_model_with_adapter() {
  kill_vllm_processes;
  
  RESULT_DIR="$1"
  MODEL_DIR="$2"
  ADAPTER_DIR="$3"

  mkdir -p "${RESULT_DIR}"
  
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${PROJECT_DIR}/perl/eval.py" \
    --result-dir "${RESULT_DIR}" \
    --model "${MODEL_DIR}" \
    --adapter "${ADAPTER_DIR}" \
    --dataset "${DATASET}" \
    --serve-port 8000 \
    --dp-size "${DP_SIZE}" \
    --tp-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "42" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-num-request "${MAX_NUM_REQUEST}" \
    --dtype "bfloat16" 2>&1 | tee "eval.log";
}

set +e

eval_model_with_adapter \
  "${PROJECT_DIR}/outputs/eval/deepseek-r1-1.5b" \
  "${BASE_MODEL_PATH}" \
  ""

eval_model_with_adapter \
   "${PROJECT_DIR}/outputs/eval/test-perl-20251208" \
   "${BASE_MODEL_PATH}" \
   "${PROJECT_DIR}/ckpts/perl"

eval_model_with_adapter \
  "${PROJECT_DIR}/outputs/eval/test-full-20251208" \
  "${PROJECT_DIR}/ckpts/grpo_full_qwen2_5_3b_20251121_111716/checkpoint-1024" \
  ""

eval_model_with_adapter \
  "${PROJECT_DIR}/outputs/eval/lora_bsz_32_1920-20251208" \
  "${BASE_MODEL_PATH}" \
  "${PROJECT_DIR}/ckpts/lora_bsz_32_1920"
