export PYTHONPATH=/mnt/llm-train/users/explore-train/qingyu/PeRL/AReaL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_DEBUG=WARN
TIMESTAMP=$(date +%Y%m%d%H%M%S)
LOG_DIR=/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/Qwen3-4B-lora-ckpt/run.log

python3 recipe/areal/run_rl.py \
    --config recipe/areal/lora_qwen.yaml \
    scheduler.type=ray \
    experiment_name=lora_rl_qwen_distill \
    trial_name=${TIMESTAMP} \
    allocation_mode=sglang:d24p1t1+archon:d2p1t4 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=8 \
    +actor.archon.enable_compile=false 2>&1 | tee $LOG_DIR
