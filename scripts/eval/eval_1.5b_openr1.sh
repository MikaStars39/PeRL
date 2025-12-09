#! /bin/bash

set -exo pipefail
ulimit -n 65535

PROJECT_DIR="."
BASE_MODEL_PATH="/root/models/DeepSeek-R1-Distill-Qwen-1.5B"

# DATASET="aime2024@512,aime2025@512,amc2023@32,math500@8,minerva@8,hmmt2025@32"
DATASET="aime2024@32,aime2025@32,amc2023@32,math500@4,minerva@4,hmmt2025@32"
# DATASET="aime2024@512,aime2025@512" # for test

export PYTHONPATH="${PROJECT_DIR}"
export HF_ENDPOINT="https://hf-mirror.com"
export LD_LIBRARY_PATH="/root/miniconda3/envs/perl/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"
# export VLLM_LOGGING_LEVEL="DEBUG"

TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"
# MAX_NEW_TOKENS="65536"
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
    --prompt-format "open-r1" \
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
  "${PROJECT_DIR}/outputs/eval/openr1-deepseek-r1-1.5b" \
  "${BASE_MODEL_PATH}" \
  ""

eval_model_with_adapter \
   "${PROJECT_DIR}/outputs/eval/openr1-test-perl-20251208" \
   "${BASE_MODEL_PATH}" \
   "${PROJECT_DIR}/ckpts/perl"

eval_model_with_adapter \
  "${PROJECT_DIR}/outputs/eval/openr1-test-full-20251208" \
  "${PROJECT_DIR}/ckpts/grpo_full_qwen2_5_3b_20251121_111716/checkpoint-1024" \
  ""


function eval_model_with_adapter_from_hf() {
  mkdir -p "${PROJECT_DIR}/ckpts/$1/$2";

  hf download MikaStars39/PeRL \
      --repo-type model \
      --local-dir "${PROJECT_DIR}/ckpts/" \
      --include "$1/$2/*";

  ls -a "${PROJECT_DIR}/ckpts/$1/$2";

  eval_model_with_adapter \
      "${PROJECT_DIR}/outputs/eval/openr1-$1___$2" \
      "${BASE_MODEL_PATH}" \
      "${PROJECT_DIR}/ckpts/$1/$2";
}

eval_model_with_adapter_from_hf "dapo_dora_qwen2_5_1_5b_20251126_115730"      "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_ia3_qwen2_5_1_5b_20251128_120647"       "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_layernorm_qwen2_5_1_5b_20251127_195534" "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_lora_qwen_1_5b"                         "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_milora_qwen2_5_1_5b_20251126_224006"    "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_miss_qwen2_5_1_5b_20251124_220354"      "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_pissa_qwen2_5_1_5b_20251126_192154"     "checkpoint-1024"
eval_model_with_adapter_from_hf "dapo_vera_qwen2_5_1_5b_20251126_190555"      "checkpoint-1024"