#!/bin/bash

export MASTER_REAL_IP="11.249.248.180"

# 检查环境变量
if [ -z "$MASTER_REAL_IP" ]; then
    echo "❌ Error: Please set MASTER_IP environment variable."
    echo "Usage: MASTER_IP=11.x.x.x ./worker_up.sh"
    exit 1
fi

LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "👷 Joining Master at $MASTER_IP..."

# 加入集群
ray start \
    --address "$MASTER_REAL_IP:6379" \
    --node-ip-address "$LOCAL_IP" \
    --num-gpus 8 \
    --disable-usage-stats

echo "✅ Worker has joined the cluster."