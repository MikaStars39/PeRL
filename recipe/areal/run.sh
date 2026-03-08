export PYTHONPATH=/mnt/llm-train/users/explore-train/qingyu/AReaL
TIMESTAMP=$(date +%Y%m%d%H%M%S)
python3 recipe/areal/run_rl.py \
    --config recipe/areal/lora_qwen.yaml \
    scheduler.type=ray \
    experiment_name=lora_rl_qwen_distill \
    trial_name=${TIMESTAMP} \
    allocation_mode=sglang:d24p1t1+fsdp:d4p1t2 \
    cluster.n_nodes=4 \
    cluster.n_gpus_per_node=8
