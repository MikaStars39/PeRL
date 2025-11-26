# outputs/grpo_full_qwen2_5_3b_20251121_111716/checkpoint-${ckpt}
# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1,2,3 lm_eval \
#     --model vllm \
#     --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,tensor_parallel_size=4,dtype=bfloat16,max_model_len=32768,gpu_memory_utilization=0.9 \
#     --tasks aime24 \
#     --batch_size auto \
#     --gen_kwargs "temperature=0.6,top_p=0.95,n=32,max_gen_toks=16384" \
#     --log_samples \
#     --output_path "outputs/results_deepseek_aime24" \
#     --seed 42 \
#     --trust_remote_code

opencompass debug.py --debug --dump-eval-details