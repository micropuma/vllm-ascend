#!/usr/bin/env bash

# -------------------------------------------------------------------
# 环境变量（带默认值，可外部覆盖）
# -------------------------------------------------------------------

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
MSPERF_ROOT=${MSPERF_ROOT:-${PROFILE_ROOT}/msperf_profile}
RESULT_FILENAME=${RESULT_FILENAME:-qwen3vl_visionarena_smoke_msperf_np10.json}
PARSE_PROFILE=${PARSE_PROFILE:-1}
PARSE_OUTPUT_DIRNAME=${PARSE_OUTPUT_DIRNAME:-output}

# ---- 直接指定 MS Service Profiler 输出目录 ----
# 必须与服务端启动脚本中 prof_dir 配置保持一致
MSPERF_OUTPUT_DIR="/data/workspace/vllm-ascend-dly/tutorial/case-study/profile/msperf_output"

mkdir -p "$OUT_DIR"
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
echo "MSPERF_OUTPUT_DIR=$MSPERF_OUTPUT_DIR"

echo "[0/4] Checking server..."
curl -sf "http://${HOST}:${PORT}/v1/models" | python -m json.tool >/dev/null

echo "[1/4] Running smoke benchmark..."
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

# 直接在指定目录下查找最新子目录
latest_profile_dir=$(find "$MSPERF_OUTPUT_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1)

if [[ -z "${latest_profile_dir}" ]]; then
  echo "ERROR: No msperf profile data found in $MSPERF_OUTPUT_DIR"
  echo "Possible reasons:"
  echo "  - Server was not started with SERVICE_PROF_CONFIG_PATH pointing to a config that sets prof_dir to this path."
  echo "  - Server process lacks write permissions to this directory."
  echo "  - Profiler data has not been flushed yet (try waiting a few seconds after the benchmark)."
  echo "  - The server may be writing to the default path (~/.ms_server_profiler)."
fi

echo "[2/4] Latest profile dir: $latest_profile_dir"

if [[ "$PARSE_PROFILE" == "1" ]]; then
  # 解析需要 msserviceprofiler，若未安装则提示并退出
  if ! command -v msserviceprofiler >/dev/null 2>&1; then
    echo "ERROR: msserviceprofiler not found in PATH."
    echo "Install it (git clone https://gitcode.com/Ascend/msserviceprofiler.git && cd msserviceprofiler && bash scripts/build_and_upgrade.sh)"
    echo "or load the CANN toolkit environment before running this script."
  fi

  parse_output_dir="${latest_profile_dir}/${PARSE_OUTPUT_DIRNAME}"
  if [[ -e "${parse_output_dir}" ]]; then
    timestamp=$(date -u +%Y%m%d-%H%M%S)
    parse_output_dir="${latest_profile_dir}/${PARSE_OUTPUT_DIRNAME}_${timestamp}"
  fi

  echo "[3/4] Parsing msperf data to: $parse_output_dir"
  msserviceprofiler parse \
    --input-path="${latest_profile_dir}" \
    --output-path "${parse_output_dir}"

  echo "[4/4] Parsed files:"
  find "$parse_output_dir" -maxdepth 1 -type f | sort
else
  echo "[3/4] Skipping parse because PARSE_PROFILE=$PARSE_PROFILE"
  echo "[4/4] Raw profile dir: $latest_profile_dir"
fi

echo "Smoke msperf finished."
echo "Benchmark result: $OUT_DIR/$RESULT_FILENAME"
echo "Raw profiler output: $latest_profile_dir"