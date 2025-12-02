#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --port 8001 \
    --served-model-name "local-model"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --data-parallel-size 8 \
    --port 8002 \
    --served-model-name "local-model"

