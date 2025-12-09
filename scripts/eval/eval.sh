#! /bin/bash

set -exo pipefail
ulimit -n 65535

PROJECT_DIR="."
BASE_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# DATASET="aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"

DATASET="aime2024@128,aime2025@128" # for test

export HF_ENDPOINT="https://hf-mirror.com"

TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"
CUDA_VISIBLE_DEVICES=0,3
DP_SIZE=2
TP_SIZE=1
MAX_NUM_REQUEST=1024
GPU_MEMORY_UTILIZATION=0.95

function kill_vllm_processes() {
  pkill -9 python;
  pkill -9 -f "vllm.entrypoints.openai.api_server";
  pkill -9 -f "VLLM::EngineCore";
  sleep 1;
  pkill -9 python;
  pkill -9 -f "vllm.entrypoints.openai.api_server";
  pkill -9 -f "VLLM::EngineCore";
}

function eval_model_with_adapter() {

  kill_vllm_processes;
  
  RESULT_DIR="${PROJECT_DIR}/outputs/eval/$1___$2"
  LOG_DIR="${RESULT_DIR}/eval.log"

  mkdir -p "${RESULT_DIR}"
  
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${PROJECT_DIR}/perl/eval.py" \
    --result-dir "${RESULT_DIR}" \
    --model "${BASE_MODEL_PATH}" \
    --adapter "$2" \
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
    --dtype "bfloat16" 2>&1 | tee "${LOG_DIR}";
}

set +e
eval_model_with_adapter "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "outputs/dapo_lora_qwen_1_5b_bsz_32_20251205_194748/checkpoint-512"