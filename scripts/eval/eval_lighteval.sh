
lighteval endpoint litellm \
    "scripts/eval/eval_test.yaml" \
    "math_500_avg_4@n=4,minerva_pass_4@k=4,amc23_pass_32@k=32,aime24_pass_32@k=32,aime25_pass_32@k=32,gpqa_diamond_pass_4@k=4,olympiadbench_pass_4@k=4" \
    --custom-tasks config/math.py &> output.log