# 看大量请求burst下的性能抖动

MODEL=/data/workspace/vllm-ascend-dly/tutorial/models/Qwen3-8B
DATA=/root/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json

vllm bench serve \
  --model $MODEL \
  --backend vllm \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path $DATA \
  --num-prompts 100 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --plot-timeline \
  --timeline-itl-thresholds 25 50 \
  --plot-dataset-stats \
  --result-dir bench_results/sharegpt \
  --result-filename qwen3_8b_sharegpt_burst_np100.json