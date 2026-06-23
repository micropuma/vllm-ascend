#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DeepSeek-V2 Server — Ascend / vLLM-Ascend (DBO disabled)
#
# 启动：
#   bash deepseek-v2-server-no-dbo.sh
#
# 指定端口：
#   PORT=8001 bash deepseek-v2-server-no-dbo.sh
#
# 采集 profiler：
#   ENABLE_PROFILER=1 PORT=8001 bash deepseek-v2-server-no-dbo.sh
#
# profiler 小规模压测：
#   curl -X POST http://127.0.0.1:8001/start_profile
#
#   LABEL=no_dbo PORT=8001 \
#   INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=4 MAX_CONCURRENCY=2 REQUEST_RATE=inf \
#   bash deepseek-v2-dbo-test.sh
#
#   curl -X POST http://127.0.0.1:8001/stop_profile
#
# 正式 benchmark：
#   LABEL=no_dbo PORT=8001 \
#   INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96 REQUEST_RATE=inf \
#   bash deepseek-v2-dbo-test.sh
# ============================================================


# ------------------------------------------------------------
# Basic env
# ------------------------------------------------------------
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

# 两卡 TP=2；如果换 4 卡，这里和 TP 都要同步改
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/data/torch_cache}
export VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-/data/vllm_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/data/triton_cache}
export HF_HOME=${HF_HOME:-/data/huggingface_cache}
export TMPDIR=${TMPDIR:-/data/tmp}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/torch_inductor_cache}
export VLLM_COMPILE_CACHE_PATH=${VLLM_COMPILE_CACHE_PATH:-/data/vllm_compile_cache}

mkdir -p \
  "$TORCH_EXTENSIONS_DIR" \
  "$VLLM_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$HF_HOME" \
  "$TMPDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$VLLM_COMPILE_CACHE_PATH"


# ------------------------------------------------------------
# Ascend communication config
# ------------------------------------------------------------

# 推荐 AI_CPU：尽量避免 HCCL 通信占用 AI Core / AI Vector 计算资源
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AI_CPU}

# no-DBO baseline：FlashComm 默认关闭，保证和 DBO 路径解耦
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=${VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE:-0}
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=${VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED:-0}

# 显式关闭 Ascend DBO
export VLLM_ASCEND_ENABLE_DBO=${VLLM_ASCEND_ENABLE_DBO:-0}

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}


# ------------------------------------------------------------
# Model / server config
# ------------------------------------------------------------
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}

TP=${TP:-2}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}


# ------------------------------------------------------------
# Profiler config
# ------------------------------------------------------------
ENABLE_PROFILER=${ENABLE_PROFILER:-0}
PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-ascend/profile}

# no-DBO 单独目录，避免和 DBO profile 混在一起
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/no_dbo_profile}

# profiler 时建议小一点，避免 trace 爆炸
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-20}

mkdir -p "$TORCH_PROFILER_DIR"


# ------------------------------------------------------------
# Log stats
# ------------------------------------------------------------
# 调试时可以 LOG_STATS=1，正式 benchmark 建议保持 0
LOG_STATS=${LOG_STATS:-0}


# ------------------------------------------------------------
# Print config
# ------------------------------------------------------------
echo "============================================================"
echo " DeepSeek-V2 Server (DBO disabled)"
echo "============================================================"
echo "  MODEL                               = $MODEL"
echo "  SERVER                              = http://${HOST}:${PORT}"
echo "  ASCEND_RT_VISIBLE_DEVICES           = $ASCEND_RT_VISIBLE_DEVICES"
echo "  TP                                  = $TP"
echo "  MAX_MODEL_LEN                       = $MAX_MODEL_LEN"
echo "  MAX_NUM_BATCHED_TOKENS              = $MAX_NUM_BATCHED_TOKENS"
echo "  MAX_NUM_SEQS                        = $MAX_NUM_SEQS"
echo ""
echo "  --enable-dbo                        = OFF"
echo "  VLLM_ASCEND_ENABLE_DBO               = $VLLM_ASCEND_ENABLE_DBO"
echo ""
echo "  HCCL_OP_EXPANSION_MODE               = $HCCL_OP_EXPANSION_MODE"
echo "  VLLM_ASCEND_ENABLE_FLASHCOMM1        = $VLLM_ASCEND_ENABLE_FLASHCOMM1"
echo "  VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = $VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"
echo "  VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED= $VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"
echo "  VLLM_LOGGING_LEVEL                   = $VLLM_LOGGING_LEVEL"
echo ""
echo "  ENABLE_PROFILER                      = $ENABLE_PROFILER"
echo "  TORCH_PROFILER_DIR                   = $TORCH_PROFILER_DIR"
echo "  PROFILER_MAX_ITERATIONS              = $PROFILER_MAX_ITERATIONS"
echo "  LOG_STATS                            = $LOG_STATS"
echo "============================================================"
echo ""
echo "DBO 已禁用："
echo "  - 不传 --enable-dbo"
echo "  - VLLM_ASCEND_ENABLE_DBO=0"
echo "  - 不会触发 ubatch / DBO 双缓冲重叠"
echo ""


# ------------------------------------------------------------
# Build serve args
# ------------------------------------------------------------
serve_args=(
  serve "$MODEL"

  --host "$HOST"
  --port "$PORT"

  --dtype bfloat16

  --distributed-executor-backend mp
  --tensor-parallel-size "$TP"
  --enable-expert-parallel

  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ "$LOG_STATS" == "0" ]]; then
  serve_args+=(--disable-log-stats)
fi


# ------------------------------------------------------------
# Profiler args
# ------------------------------------------------------------
if [[ "$ENABLE_PROFILER" == "1" ]]; then
  # stop_profile flush 可能较慢，避免 RPC 超时
  export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-1800000}

  # 简洁版 profiler config：
  #   - torch_profiler_with_stack=true：采 Python call stack
  #   - torch_profiler_record_shapes=true：采 shape
  #   - torch_profiler_use_gzip=false：不压缩，方便 grep / 检查
  #   - max_iterations：限制采集轮数，防止 trace 过大
  PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_use_gzip\":true,\"torch_profiler_with_memory\":true,\"torch_profiler_with_flops\":false,\"max_iterations\":${PROFILER_MAX_ITERATIONS}}"

  echo "  PROFILER_CONFIG                     = ${PROFILER_CONFIG}"

  serve_args+=(
    --profiler-config "$PROFILER_CONFIG"
  )
fi


# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
vllm "${serve_args[@]}"