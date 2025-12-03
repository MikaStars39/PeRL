
# !/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python perl/eval/pass@k_lin.py \
  --input_file bench_data/valid.math.dedup.by_source/math_aime25.parquet \
  --output_file outputs/valid.math.dedup.test.aime25.lin.parquet \
  --model_name /mnt/moonfs/zhennan-m2/PeRL/grpo_full_qwen2_5_3b_20251121_111716 \
  --tensor_parallel_size 4 \
  --port 9999 \
  --n_samples 1 \
  --max_workers 64 \
  --max_tokens 30000
