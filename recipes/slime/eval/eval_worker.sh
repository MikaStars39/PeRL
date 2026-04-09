#!/usr/bin/env bash
# eval_worker.sh — 在单个 pod 上执行：对分配到的 checkpoint 依次 convert → eval
#
# 用法:
#   bash eval_worker.sh <model_base:iter_num> ...
#   例: bash eval_worker.sh /path/to/model:1151 /path/to/model:2175
#
set -euo pipefail
pkill -9 python
pkill -9 sglang

# ========== 配置 ==========
ORIGIN_HF="/jpfs/chenyanxu.9/model/zhennanshen/Qwen3-8B-Base"
TASK_DIR="/jpfs/chenyanxu.9/data/nano-eval"
EVAL_SCRIPT="/jpfs/qingyu/nano-eval/run.py"

if [ $# -eq 0 ]; then
    echo "[ERROR] 用法: $0 <model_base:iter_num> ..."
    exit 1
fi

for TASK in "$@"; do
    # 解析 model_base:iter_num
    MODEL_BASE="${TASK%:*}"
    ITER_NUM="${TASK##*:}"

    # 补零到7位
    ITER_PADDED=$(printf "iter_%07d" "$ITER_NUM")
    INPUT_DIR="${MODEL_BASE}/${ITER_PADDED}"
    HF_DIR="${MODEL_BASE}/${ITER_PADDED}-hf"

    echo "============================================================"
    echo "[INFO] Processing ${MODEL_BASE##*/}/${ITER_PADDED} ..."
    echo "============================================================"

    # ---------- 跳过已有结果的 ----------
    if [ -f "${HF_DIR}/eval_results.json" ]; then
        echo "[SKIP] ${ITER_PADDED} 已有 eval_results.json，跳过"
        continue
    fi

    # ---------- 1. Convert (如果 -hf 目录不存在) ----------
    if [ ! -d "${HF_DIR}" ] || [ ! -f "${HF_DIR}/config.json" ]; then
        echo "[INFO] Converting ${ITER_PADDED} -> HF ..."
        cd /root/slime
        PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
            --input-dir "${INPUT_DIR}" \
            --output-dir "${HF_DIR}" \
            --origin-hf-dir "${ORIGIN_HF}" \
            --force
        echo "[OK] Convert done: ${HF_DIR}"
    else
        echo "[SKIP] ${ITER_PADDED}-hf 已存在，跳过转换"
    fi
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    # ---------- 3. Eval ----------
    WORKDIR="${HF_DIR}/eval_${TIMESTAMP}"
    mkdir -p "${WORKDIR}"

    echo "[INFO] Running eval for ${ITER_PADDED} ..."
    python "${EVAL_SCRIPT}" \
        --stage all \
        --task-dir "${TASK_DIR}" \
        --tasks "gpqa_diamond@4,hmmt2025@4,aime2024@32,aime2025@32,math500@4,minerva@4" \
        --output "${WORKDIR}/step01_prepared.jsonl" \
        --inference-output "${WORKDIR}/step02_inference.jsonl" \
        --score-output "${WORKDIR}/step03_score.jsonl" \
        --final-eval-output "${WORKDIR}/step03_final_eval.jsonl" \
        --backend offline \
        --model-path "${HF_DIR}" \
        --tp-size 1 \
        --dp-size 8 \
        --temperature 1.0 \
        --top-p 0.95 \
        --max-tokens 30000 \
        --concurrency 32 \
        2>&1 | tee "${WORKDIR}/run.log"

    # ---------- 4. 提取最终结果到 eval_results.json ----------
    if [ -f "${WORKDIR}/step03_final_eval.jsonl" ]; then
        cp "${WORKDIR}/step03_final_eval.jsonl" "${HF_DIR}/eval_results.json"
        echo "[OK] Eval done: ${HF_DIR}/eval_results.json"
    else
        echo "[WARN] ${ITER_PADDED} eval 未产出 final_eval 文件"
    fi

    echo ""
done

echo "[DONE] Worker finished all assigned checkpoints."
