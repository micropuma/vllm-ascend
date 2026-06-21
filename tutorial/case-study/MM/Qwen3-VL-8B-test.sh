#!/usr/bin/env bash

# ========== basic config ==========
export VLLM_USE_MODELSCOPE=false

MODEL=${MODEL:-/root/models/Qwen3-VL-8B-Instruct}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

DATASET=${DATASET:-/root/datasets/vision-arena-bench-v0.1}
HF_NAME="lmarena-ai/vision-arena-bench-v0.1"   # 新增：对应的 HF 仓库 ID
HF_SPLIT=${HF_SPLIT:-train}

OUT_DIR=${OUT_DIR:-/data/workspace/vllm-ascend-dly/tutorial/case-study/MM}
mkdir -p "$OUT_DIR"

# ========== Hugging Face dataset cache ==========
mkdir -p /root/datasets/hf_cache

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=/root/datasets/hf_cache
export HF_DATASETS_CACHE=/root/datasets/hf_cache/datasets
export HUGGINGFACE_HUB_CACHE=/root/datasets/hf_cache/hub

# 如果你要走本地代理，而不是 hf-mirror，则注释上面的 HF_ENDPOINT，打开下面四行：
# export http_proxy=http://127.0.0.1:10808
# export https_proxy=http://127.0.0.1:10808
# export HTTP_PROXY=http://127.0.0.1:10808
# export HTTPS_PROXY=http://127.0.0.1:10808
# unset HF_ENDPOINT

echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "HF_NAME=$HF_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "SERVER=http://${HOST}:${PORT}"

# ========== 0. check server ==========
echo "[0/3] Checking server..."
curl -s "http://${HOST}:${PORT}/v1/models" | python -m json.tool || {
  echo "ERROR: vLLM server is not ready at http://${HOST}:${PORT}"
  exit 1
}

# ========== 1. smoke test ==========
echo "[1/3] Running smoke benchmark: num_prompts=10"

vllm bench serve \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path "$DATASET" \
  --hf-name "$HF_NAME" \
  --hf-split "$HF_SPLIT" \
  --num-prompts 10 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --result-dir "$OUT_DIR" \
  --result-filename qwen3vl_visionarena_smoke_np10.json

# ========== 2. throughput test ==========
echo "[2/3] Running throughput benchmark: num_prompts=500, max_concurrency=16"

vllm bench serve \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path "$DATASET" \
  --hf-name "$HF_NAME" \
  --hf-split "$HF_SPLIT" \
  --num-prompts 500 \
  --max-concurrency 16 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --plot-timeline \
  --timeline-itl-thresholds 25 50 \
  --plot-dataset-stats \
  --result-dir "$OUT_DIR" \
  --result-filename qwen3vl_visionarena_throughput_c16_np500.json

# ========== 3. latency test ==========
echo "[3/3] Running latency benchmark: num_prompts=500, request_rate=10, burstiness=5"

vllm bench serve \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path "$DATASET" \
  --hf-name "$HF_NAME" \
  --hf-split "$HF_SPLIT" \
  --num-prompts 500 \
  --request-rate 10 \
  --burstiness 5 \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --plot-timeline \
  --timeline-itl-thresholds 25 50 \
  --plot-dataset-stats \
  --result-dir "$OUT_DIR" \
  --result-filename qwen3vl_visionarena_latency_rps10_burst5_np500.json

echo "Done. Results saved to: $OUT_DIR"