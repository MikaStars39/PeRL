#! /bin/bash

set -euo pipefail

PROJECT_DIR="."
LORA_RANK=32
BASE_MODEL="/root/models/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_PATH="open-r1/DAPO-Math-17k-Processed"
DATASET_NAME="${DATASET_PATH##*/}"
export LD_LIBRARY_PATH="/root/miniconda3/envs/perl/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

unset WANDB_DISABLED
OUTPUT_DIR="outputs/mem_dapo_lora_1_5b_r${LORA_RANK}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${OUTPUT_DIR}
LOG_FILE=${OUTPUT_DIR}/output.log

export HF_ENDPOINT="https://hf-mirror.com"

ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29500 \
    --config_file scripts/accelerate/ds_zero2_8gpu.yaml \
    run.py train \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "${BASE_MODEL}" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft true \
    --config.peft.type "lora" \
    --config.peft.task_type "CAUSAL_LM" \
    --config.peft.r "${LORA_RANK}" \
    --config.peft.lora_alpha 64 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.total_step 1000 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 1e-5 \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 4 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 16384 \
    --config.training.num_generations 8 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 4 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 64 \
    --config.training.max_steps 8192 \
    --config.training.use_vllm true \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "constant" \
    --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
    --config.training.vllm_mode "colocate" \
    --config.training.vllm_gpu_memory_utilization 0.1 \
    --config.training.use_liger_kernel false \
    --config.training.loss_type "dapo" \
    --config.training.report_to '["wandb"]' \
    --config.logging.trackio_space_id "Open-Tinker/Open-Tinker" \
    --config.logging.trackio_project "grpo-full-qwen3-4b" \
    --config.logging.wandb_project "grpo-full-qwen3-4b" \
    --config.dataset.dataset_name_or_path "${DATASET_PATH}" \
    --config.dataset.example_numbers 1000000000 \
    &> ${LOG_FILE}
