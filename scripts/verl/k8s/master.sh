#!/bin/bash
# 自动获取本机 IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo "🚀 Starting Ray Head on $LOCAL_IP..."

# 清理旧进程
ray stop --force
pkill -9 python sglang || true

# 启动 Head 节点 (监听所有网卡以接收 Service 转发或直连)
ray start --head \
    --node-ip-address "$LOCAL_IP" \
    --num-gpus 8 \
    --dashboard-host 0.0.0.0 \
    --disable-usage-stats

echo "✅ Master is up. Dashboard: http://$LOCAL_IP:8265"