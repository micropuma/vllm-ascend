# 在并发上限为 16 时，系统最大 output tok/s 大概是多少？
# TTFT 是否因为排队明显升高？
# P99 TPOT / P99 ITL 是否恶化？

MODEL=/data/workspace/vllm-ascend-dly/tutorial/models/Qwen3-8B
DATA=/root/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model $MODEL \
  --backend vllm \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path $DATA \
  --num-prompts 200 \
  --max-concurrency 16 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --plot-timeline \
  --timeline-itl-thresholds 25 50 \
  --result-dir bench_results/sharegpt \
  --result-filename qwen3_8b_sharegpt_c16_np200.json