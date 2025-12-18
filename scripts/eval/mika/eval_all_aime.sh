#! /bin/bash

set -exo pipefail
ulimit -n 65535

PROJECT_DIR="."
BASE_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

DATASET="aime2024@32"

export PYTHONPATH="${PROJECT_DIR}"
export HF_ENDPOINT="https://hf-mirror.com"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY  # 取消代理
export VLLM_TORCH_COMPILE="0"

TEMPERATURE="0.7"
TOP_P="0.9"
MAX_NEW_TOKENS="31744"
# MAX_NEW_TOKENS="65536"
CUDA_VISIBLE_DEVICES=0,1,2,3
DP_SIZE=4
TP_SIZE=1
MAX_NUM_REQUEST="$((200 * ${DP_SIZE}))"
GPU_MEMORY_UTILIZATION=0.95

function kill_vllm_processes() {
  pkill -9 -f "vllm.entrypoints.openai.api_server" || true;
  pkill -9 -f "VLLM::EngineCore" || true;
  sleep 1;
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
    --dtype "bfloat16" 2>&1 | tee "${RESULT_DIR}/eval.log";
}

function eval_model_with_adapter_from_hf() {
  local MODEL_NAME="$1"
  local CKPT_NUM="$2"
  local HF_REPO="${3:-MikaStars39/PeRL}"
  local CHECKPOINT_NAME="checkpoint-${CKPT_NUM}"
  
  echo "=========================================="
  echo "Evaluating model: ${MODEL_NAME}, checkpoint: ${CHECKPOINT_NAME}"
  echo "From repository: ${HF_REPO}"
  echo "=========================================="
  
  mkdir -p "${PROJECT_DIR}/ckpts/${MODEL_NAME}/${CHECKPOINT_NAME}";

  hf download "${HF_REPO}" \
      --repo-type model \
      --local-dir "${PROJECT_DIR}/ckpts/" \
      --include "${MODEL_NAME}/${CHECKPOINT_NAME}/*" || true;

  ls -a "${PROJECT_DIR}/ckpts/${MODEL_NAME}/${CHECKPOINT_NAME}";

  # 检查下载是否成功
  if [[ ! -f "${PROJECT_DIR}/ckpts/${MODEL_NAME}/${CHECKPOINT_NAME}/adapter_config.json" ]]; then
    echo "WARNING: Download failed or checkpoint not found, skipping ${MODEL_NAME}/${CHECKPOINT_NAME}"
    return 0
  fi

  eval_model_with_adapter \
      "${PROJECT_DIR}/outputs/eval/aime-${MODEL_NAME}___${CHECKPOINT_NAME}" \
      "${BASE_MODEL_PATH}" \
      "${PROJECT_DIR}/ckpts/${MODEL_NAME}/${CHECKPOINT_NAME}";
}

set +e

# 格式: "MODEL_NAME:HF_REPO" 或 "MODEL_NAME" (默认使用 MikaStars39/PeRL)
MODELS=(
  # MikaStars39/PeRL
  "dapo_miss_qwen2_5_1_5b_20251124_220354"
  "dapo_pissa_qwen2_5_1_5b_20251126_192154"
  "dapo_vera_qwen2_5_1_5b_20251126_190555"
  # 5456es/perl_results
  "dapo_lora_plus_20251204_160304:5456es/perl_results"
  "dapo_lora_fa_20251204_152725:5456es/perl_results"
)

# 定义要评估的 checkpoint 列表
CHECKPOINTS=(64 128 192 256 320 384 448 512 576 640 704 768 832 896 960 1024)

# 遍历所有模型和 checkpoint 组合
for MODEL_ENTRY in "${MODELS[@]}"; do
  # 解析模型名和仓库名
  if [[ "${MODEL_ENTRY}" == *":"* ]]; then
    MODEL_NAME="${MODEL_ENTRY%%:*}"
    HF_REPO="${MODEL_ENTRY#*:}"
  else
    MODEL_NAME="${MODEL_ENTRY}"
    HF_REPO="MikaStars39/PeRL"
  fi

  echo "=========================================="
  echo "Starting evaluation for model: ${MODEL_NAME}"
  echo "From repository: ${HF_REPO}"
  echo "=========================================="
  
  for CKPT_NUM in "${CHECKPOINTS[@]}"; do
    eval_model_with_adapter_from_hf "${MODEL_NAME}" "${CKPT_NUM}" "${HF_REPO}"
  done
  
  # 评估完该模型的所有checkpoint后，删除下载的模型文件（保留评估结果）
  echo "=========================================="
  echo "All checkpoints evaluated for ${MODEL_NAME}, cleaning up downloaded model files..."
  echo "=========================================="
  rm -rf "${PROJECT_DIR}/ckpts/${MODEL_NAME}"
  echo "Deleted ${PROJECT_DIR}/ckpts/${MODEL_NAME}"
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

