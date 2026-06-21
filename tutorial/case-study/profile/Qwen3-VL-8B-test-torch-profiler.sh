#!/usr/bin/env bash
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}

MODEL=${MODEL:-/root/models/Qwen3-VL-8B-Instruct}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

DATASET=${DATASET:-/root/datasets/vision-arena-bench-v0.1}
HF_NAME=${HF_NAME:-lmarena-ai/vision-arena-bench-v0.1}
HF_SPLIT=${HF_SPLIT:-train}
NUM_PROMPTS=${NUM_PROMPTS:-10}

PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-ascend-dly/tutorial/case-study/profile}
OUT_DIR=${OUT_DIR:-${PROFILE_ROOT}/bench_results}
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/vllm_profile}
RESULT_FILENAME=${RESULT_FILENAME:-qwen3vl_visionarena_smoke_profile_np10.json}
ANALYZE_PROFILE=${ANALYZE_PROFILE:-1}

mkdir -p "$OUT_DIR"
mkdir -p "$TORCH_PROFILER_DIR"
mkdir -p /root/datasets/hf_cache

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/root/datasets/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/datasets/hf_cache/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/root/datasets/hf_cache/hub}

echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "HF_NAME=$HF_NAME"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "SERVER=http://${HOST}:${PORT}"
echo "OUT_DIR=$OUT_DIR"
echo "TORCH_PROFILER_DIR=$TORCH_PROFILER_DIR"

stop_profile() {
  curl -sf -X POST "http://${HOST}:${PORT}/stop_profile" >/dev/null || true
}

trap stop_profile EXIT

echo "[0/4] Checking server..."
curl -sf "http://${HOST}:${PORT}/v1/models" | python -m json.tool >/dev/null

echo "[1/4] Starting aten-pytorch profile..."
curl -sf -X POST "http://${HOST}:${PORT}/start_profile" >/dev/null

echo "[2/4] Running smoke benchmark..."
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
  --num-prompts "$NUM_PROMPTS" \
  --temperature 0 \
  --save-result \
  --save-detailed \
  --result-dir "$OUT_DIR" \
  --result-filename "$RESULT_FILENAME"

echo "[3/4] Stopping aten-pytorch profile..."
stop_profile
trap - EXIT

latest_profile_dir=$(find "$TORCH_PROFILER_DIR" -maxdepth 1 -mindepth 1 -type d -name '*_ascend_pt' | sort | tail -n 1)

if [[ -z "${latest_profile_dir}" ]]; then
  echo "WARNING: no *_ascend_pt profile directory found under $TORCH_PROFILER_DIR"
fi

echo "[4/4] Latest profile dir: $latest_profile_dir"

if [[ "$ANALYZE_PROFILE" == "1" ]]; then
  if python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("torch_npu") else 1)'; then
    echo "Analyzing profile output with torch_npu..."
    PROFILE_RUN_DIR="$latest_profile_dir" python -c 'import os; from torch_npu.profiler.profiler import analyse; analyse(os.environ["PROFILE_RUN_DIR"])'
  else
    echo "Skipping local analyse(): torch_npu is not available in the current Python environment."
  fi
fi

echo "Smoke profile finished."
echo "Benchmark result: $OUT_DIR/$RESULT_FILENAME"
echo "Profiler output: $latest_profile_dir"
