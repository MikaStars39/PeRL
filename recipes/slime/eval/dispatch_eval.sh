#!/usr/bin/env bash
# dispatch_eval.sh — 在 CPU 机器上运行，把 checkpoint 分配给多个 k8s pod 并行执行
#
# 用法:
#   bash dispatch_eval.sh <job_keyword>
#   例: bash dispatch_eval.sh cyx-eval
#
# 逻辑:
#   1. 扫描多个模型目录的所有 checkpoint
#   2. 均匀分配到 pod，每个 pod 后台执行 eval_worker.sh
#   3. 等待全部完成
#
set -uo pipefail

# ========== 配置 ==========
MODEL_BASES=(
    "/jpfs-5p/chenyanxu.9/model/Qwen3-8B-onpolicy-profiling-20260403_091551"
)
WORKER_SCRIPT="/jpfs/chenyanxu.9/PeRL/recipes/slime/eval/eval_worker.sh"
NAMESPACE="${NAMESPACE:-explore-train}"
NUM_PODS=4

if command -v kubectl >/dev/null 2>&1; then
    KUBECTL=(kubectl)
elif command -v kt >/dev/null 2>&1; then
    KUBECTL=(kt)
else
    echo "[ERROR] 未找到 kubectl 或 kt"
    exit 1
fi

KEYWORD="${1:-}"
if [ -z "$KEYWORD" ]; then
    echo "Usage: $0 <job_keyword>"
    echo "  例: $0 cyx-eval"
    exit 1
fi

# ========== 1. 收集需要测的 checkpoint ==========
# 格式: "model_base:iter_num"
TODO=()

for MODEL_BASE in "${MODEL_BASES[@]}"; do
    echo "[INFO] Scanning checkpoints in ${MODEL_BASE} ..."

    ALL_ITERS=()
    while IFS= read -r d; do
        num=$(echo "$d" | sed 's/iter_0*//')
        ALL_ITERS+=("$num")
    done < <(ls "${MODEL_BASE}" | grep -E "^iter_[0-9]+$" | sort -V)

    if [ ${#ALL_ITERS[@]} -eq 0 ]; then
        echo "[WARN] No checkpoints found in ${MODEL_BASE}, skipping"
        continue
    fi

    echo "[INFO]   ${MODEL_BASE##*/}: found ${#ALL_ITERS[@]} ckpts"

    for num in "${ALL_ITERS[@]}"; do
        ITER_PADDED=$(printf "iter_%07d" "$num")
        HF_DIR="${MODEL_BASE}/${ITER_PADDED}-hf"
        if [ -f "${HF_DIR}/eval_results.json" ]; then
            echo "[SKIP] ${MODEL_BASE##*/}/${ITER_PADDED} already evaluated"
        else
            TODO+=("${MODEL_BASE}:${num}")
        fi
    done
done

if [ ${#TODO[@]} -eq 0 ]; then
    echo "[DONE] No checkpoints to evaluate!"
    exit 0
fi

echo ""
echo "[INFO] Total ${#TODO[@]} checkpoints to evaluate"

# ========== 2. 获取 Pod 列表 ==========
ALL_PODS_RAW=$("${KUBECTL[@]}" get pods -n "$NAMESPACE" --field-selector=status.phase=Running \
    --no-headers -o custom-columns=":metadata.name" 2>/dev/null) || true

mapfile -t PODS < <(echo "$ALL_PODS_RAW" | grep "^${KEYWORD}" | sort -V | head -n "$NUM_PODS")
ACTUAL_PODS=${#PODS[@]}

if [ "$ACTUAL_PODS" -eq 0 ]; then
    echo "[ERROR] 没有匹配 ^${KEYWORD} 的 Running Pod"
    exit 1
fi
echo "[INFO] Found ${ACTUAL_PODS} pods"

# ========== 3. 分配 checkpoint 到 pod ==========
# 均匀分配: pod i 拿 TODO[i], TODO[i+N], TODO[i+2N], ...
declare -a POD_TASKS
for ((i = 0; i < ACTUAL_PODS; i++)); do
    POD_TASKS[$i]=""
done

for ((j = 0; j < ${#TODO[@]}; j++)); do
    pod_idx=$((j % ACTUAL_PODS))
    POD_TASKS[$pod_idx]="${POD_TASKS[$pod_idx]} ${TODO[$j]}"
done

# ========== 4. 下发任务到每个 pod ==========
LOG_DIR="/jpfs/chenyanxu.9/logs/dispatch_$(date +%Y%m%d%H%M%S)"
mkdir -p "${LOG_DIR}"

PIDS=()
for ((i = 0; i < ACTUAL_PODS; i++)); do
    tasks="${POD_TASKS[$i]}"
    if [ -z "${tasks// }" ]; then
        echo "[INFO] Pod ${PODS[$i]}: no tasks assigned, skipping"
        continue
    fi

    POD="${PODS[$i]}"
    LOG_FILE="${LOG_DIR}/pod_${i}_${POD}.log"

    echo "[INFO] Pod ${POD} -> checkpoints:${tasks}"

    # 后台 kubectl exec，日志写到文件
    "${KUBECTL[@]}" exec -n "$NAMESPACE" "$POD" -- \
        bash -lc "bash ${WORKER_SCRIPT} ${tasks}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[INFO] All ${#PIDS[@]} pods dispatched. Logs: ${LOG_DIR}/"
echo "[INFO] Waiting for all pods to finish ..."

# ========== 5. 等待并汇报 ==========
FAILED=0
for ((i = 0; i < ${#PIDS[@]}; i++)); do
    if wait "${PIDS[$i]}"; then
        echo "[OK] Pod task $i finished"
    else
        echo "[FAIL] Pod task $i failed, check log: ${LOG_DIR}/"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "$FAILED" -gt 0 ]; then
    echo "[WARN] ${FAILED}/${#PIDS[@]} pod tasks failed"
else
    echo "[DONE] All evaluations completed successfully!"
fi
echo "Logs: ${LOG_DIR}/"
for mb in "${MODEL_BASES[@]}"; do
    echo "Results: ${mb}/iter_*-hf/eval_results.json"
done
echo "============================================================"
