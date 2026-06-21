# 在稳定到达率下，TTFT 是否还异常高？
# TPOT 是否随着 RPS 上升而恶化？
# P99 ITL 是否随着 RPS 上升而放大？

MODEL=/data/workspace/vllm-ascend-dly/tutorial/models/Qwen3-8B
DATA=/root/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model $MODEL \
  --backend vllm \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path $DATA \
  --num-prompts 100 \
  --request-rate 5 \
  --burstiness 5 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --plot-timeline \
  --timeline-itl-thresholds 25 50 \
  --result-dir bench_results/sharegpt \
  --result-filename qwen3_8b_sharegpt_rps5_np100.json