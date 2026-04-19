#!/bin/bash
# GPU 空闲检测 + 自动启动脚本
# 每 5 分钟检查一次 GPU 利用率，连续 3 次 ≤5% 则启动指定命令

INTERVAL=300        # 检查间隔 (秒)
THRESHOLD=5          # 利用率阈值 (%)
MAX_IDLE_COUNT=3     # 连续空闲次数
idle_count=0

# ========== 在这里写你要启动的命令 ==========
CMD="echo 'TODO: 替换成你的命令'"
# ============================================

echo "[gpu_watchdog] started, interval=${INTERVAL}s, threshold=${THRESHOLD}%, trigger after ${MAX_IDLE_COUNT} consecutive idles"

while true; do
    # 取所有 GPU 的最大利用率
    max_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{if($1>max)max=$1} END{print max+0}')
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$max_util" -le "$THRESHOLD" ]; then
        idle_count=$((idle_count + 1))
        echo "[${ts}] GPU max util: ${max_util}% ≤ ${THRESHOLD}% (idle ${idle_count}/${MAX_IDLE_COUNT})"
    else
        idle_count=0
        echo "[${ts}] GPU max util: ${max_util}% — active, reset counter"
    fi

    if [ "$idle_count" -ge "$MAX_IDLE_COUNT" ]; then
        echo "[${ts}] GPU idle ${MAX_IDLE_COUNT} consecutive times, launching command..."
        eval "$CMD"
        echo "[${ts}] command finished, exiting watchdog"
        exit 0
    fi

    sleep $INTERVAL
done
