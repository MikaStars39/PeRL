
# !/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python perl/eval/pass@k_lin.py \
  --input_file bench_data/bench.generated.parquet \
  --output_file outputs/DeepSeek-R1-Distill-Qwen-1.5B_baseline_math_full.parquet \
  --model_name /home/shenzhennan/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --tensor_parallel_size 4 \
  --port 9998 \
  --n_samples 1 \
  --max_workers 64 \
  --max_tokens 30000
