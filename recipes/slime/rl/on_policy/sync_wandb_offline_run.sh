#!/usr/bin/env bash
# 将本地离线 W&B 目录（如 wandb/offline-run-* 或已生成的 run-*）同步到自建服务器。
# 用法：
#   export WANDB_API_KEY=...   # 与训练脚本一致
#   export WANDB_ENTITY=automl # 可选，默认 automl
#   ./sync_wandb_offline_run.sh /path/to/wandb/run-20260407_143055-5oxqq13g
#   ./sync_wandb_offline_run.sh   # 默认下面 OFFLINE_RUN 路径
#
# 说明：若目录是 offline-run-*，wandb sync 同样支持；路径请填「包含 run- 前缀的那一层文件夹」。

set -euo pipefail
export WANDB_API_KEY='local-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'   # 与训练脚本里一致
export WANDB_ENTITY=duo
WANDB_BASE_URL="${WANDB_BASE_URL:-http://11.71.1.153:8080}"
WANDB_ENTITY="${WANDB_ENTITY:-duo}"

# 默认：当前目录下的 run（可按需改成你机器上的绝对路径）
DEFAULT_RUN="./wandb/run-20260408_085258-uijyw7se"
OFFLINE_RUN="${1:-$DEFAULT_RUN}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[ERROR] 请先 export WANDB_API_KEY（与 qwen3_8b_rl_dapo_8nodes_onpolicy_muon.sh 中一致）"
  exit 1
fi

if [[ ! -d "$OFFLINE_RUN" ]]; then
  echo "[ERROR] 目录不存在: $OFFLINE_RUN"
  echo "       请把离线 run 拷到本机，或传入正确路径，例如："
  echo "       $0 /jpfs/xxx/wandb/run-20260409_042913-bjne0qpr"
  exit 1
fi

export WANDB_BASE_URL
export WANDB_ENTITY

echo "[INFO] WANDB_BASE_URL=$WANDB_BASE_URL"
echo "[INFO] WANDB_ENTITY=$WANDB_ENTITY"
echo "[INFO] SYNC: $OFFLINE_RUN"

wandb login --relogin --host="${WANDB_BASE_URL}" "${WANDB_API_KEY}"
wandb sync "${OFFLINE_RUN}"

echo "[OK] 同步命令已执行。若 UI 未更新，请刷新自建 W&B 页面或稍等索引。"
