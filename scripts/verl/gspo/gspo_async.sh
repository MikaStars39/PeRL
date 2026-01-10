# run Qwen3-30B GSPO with new model engine
set -euox pipefail

# ---------------------------- system config ---------------------------
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_HOME="/usr/local/cuda"

# ---------------------------- wandb config ----------------------------

debug=true
backend=megatron # fsdp, fsdp2, megatron
project_name=gspo
experiment_name=qwen3-30B-base-grpo-$backend
timestamp=$(date +%Y%m%d_%H%M%S)
default_local_dir=/mnt/llm-train/users/explore-train/qingyu/PeRL/outputs/${timestamp}_${experiment_name}

if [ "$debug" = true ]; then
    logger=console
else
    logger=console,wandb
fi

# ---------------------------- Algorithm ----------------------------

adv_estimator=grpo
loss_mode=gspo
use_kl_in_reward=False 
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_low=3e-4
clip_ratio_high=4e-4
actor_lr=1e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0
clip_ratio_c=10.0
use_dynamic_bsz=True
use_remove_padding=True

# ---------------------------- Data/Model ----------------------------

train_files=/mnt/llm-train/users/explore-train/qingyu/PeRL/data/DAPO-Math-17k-verl/train.parquet
test_files=/mnt/llm-train/users/explore-train/qingyu/PeRL/data/aime_2024-verl/train.parquet

actor_model_path=/mnt/llm-train/users/explore-train/zhangyuqi60/Nomerge/ms-swift/hf_outputs/qwen3-30b-s1-0103

max_prompt_length=2048
max_response_length=32768
enable_overlong_buffer=True
overlong_buffer_len=8192
overlong_penalty_factor=1.0

# [修改点 1] 资源分配：Fully Async 需要将训练节点和推理节点物理隔离
# 假设你有 4 个节点 (ARNOLD_WORKER_NUM=4)，建议 2 个用于训练，2 个用于 Rollout
# 请根据实际物理节点数调整这两个变量
NNODES_TRAIN=2
NNODES_ROLLOUT=2
NGPUS_PER_NODE=8

ppo_mini_batch_size=128
n_resp_per_prompt=8
n_resp_per_prompt_val=1

# ---------------------------- Training ----------------------------

actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3))

# Megatron parallelism config
TP_SIZE=4
CP_SIZE=1
PP_SIZE=1
VPP_SIZE=null
EP_SIZE=8
ETP_SIZE=1

param_offload=True
grad_offload=True
optimizer_offload=True

moe_router_dtype=fp32
moe_permute_fusion=True
recompute_method=uniform
recompute_granularity=full
recompute_num_layers=1
apply_rope_fusion=True
gradient_accumulation_fusion=True
use_mbridge=True

# ---------------------------- Inference / Async  ----------------------------

infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.8
top_p=0.7
temperature=1.0

rollout_mode="async"
rollout_name="vllm" 

if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
    train_batch_size=0 # async mode doesn't need train batch size
    # [修改点 2] gen_batch_size 必须设为 1 以启用流式生成
    gen_batch_size=1 
fi

test_freq=32
save_freq=128

# [修改点 3] Async Stream with Partial Rollout 核心参数
staleness_threshold=0.5          # 允许一定的过时样本 (Mode 3/4 必须 > 0)
trigger_parameter_sync_step=4    # 推荐调整：根据文档 30B 实验，设为 4 比较合适 (相当于 512 batch size)
partial_rollout=True             # 开启 Partial Rollout (Mode 4 必须为 True)
require_batches=1                # 流式训练，集齐 1 个 mini-batch 就训练

calculate_log_probs=True
total_rollout_steps=$(((512*400)))

# ---------------------------- wrapper config ----------------------------
ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=$param_offload \
    actor_rollout_ref.actor.megatron.grad_offload=$grad_offload \
    actor_rollout_ref.actor.megatron.optimizer_offload=$optimizer_offload \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=$moe_router_dtype \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=$moe_permute_fusion \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=$recompute_method \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=$recompute_granularity \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=$recompute_num_layers \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=$apply_rope_fusion \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=$gradient_accumulation_fusion \
    actor_rollout_ref.actor.megatron.use_mbridge=$use_mbridge"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu"

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=$top_p \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temperature \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.calculate_log_probs=$calculate_log_probs \
    actor_rollout_ref.rollout.total_rollout_steps=$total_rollout_steps"

REWARD_CONFIG="
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}"

ALGORITHM_CONFIG="
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam"

DATA_CONFIG="
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.gen_batch_size=$gen_batch_size \
    data.trust_remote_code=True \
    data.train_batch_size=$train_batch_size \
    data.truncation='error'"

# [修改点 4] 更新 Trainer 和 Rollout 的资源配置
TRAINER_CONFIG="
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=$logger \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES_TRAIN \
    trainer.val_before_train=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq"

# [修改点 5] 添加 require_batches 到 Async 配置
ASYNC_CONFIG="
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.partial_rollout=$partial_rollout \
    async_training.require_batches=$require_batches"

CONFIG_NAME=ppo_megatron_trainer
ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_MEGATRON_CONFIG"

# ---------------------------- Run Training ----------------------------

mkdir -p $default_local_dir

python3 -m verl.trainer.main_ppo \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    actor_rollout_ref.rollout.nnodes=$NNODES_ROLLOUT \
    actor_rollout_ref.rollout.n_gpus_per_node=$NGPUS_PER_NODE \
    $TRAINER_CONFIG \
    $ACTOR_CONFIG \
    $ROLLOUT_CONFIG \
    $REWARD_CONFIG \
    $ALGORITHM_CONFIG \
    $DATA_CONFIG \
    $ASYNC_CONFIG 2>&1 | tee $default_local_dir/train.log