#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DeepSeek-V2 DBO Server — Ascend / vLLM-Ascend
#
# 启动：
#   bash deepseek-v2-dbo-server-dp.sh
#
# 指定端口：
#   PORT=8001 bash deepseek-v2-dbo-server-dp.sh
#
# 采集 profiler：
#   ENABLE_PROFILER=1 PORT=8001 bash deepseek-v2-dbo-server-dp.sh
#
# profiler 小规模压测：
#   curl -X POST http://127.0.0.1:8001/start_profile
#
#   LABEL=dbo PORT=8001 \
#   INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=4 MAX_CONCURRENCY=2 REQUEST_RATE=inf \
#   bash deepseek-v2-dbo-test.sh
#
#   curl -X POST http://127.0.0.1:8001/stop_profile
#
# 正式 benchmark：
#   LABEL=dbo PORT=8001 \
#   INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96 REQUEST_RATE=inf \
#   bash deepseek-v2-dbo-test.sh
# ============================================================


# ------------------------------------------------------------
# Basic env
# ------------------------------------------------------------
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

# 两卡 DP=2；每个 DP rank 使用一张卡，TP 固定为 1
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
# Ascend DBO / communication config
# ------------------------------------------------------------

# 推荐 AI_CPU：尽量避免 HCCL 通信占用 AI Core / AI Vector 计算资源
# TODO(leon)：目前这个配置和flashcomm2 产生不兼容bug
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AI_CPU}

# DP-only 基线：TP=1 时 FlashComm1 不受支持；保持关闭以验证 DP+EP+DBO+compile
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=${VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE:-0}
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=${VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED:-0}

# 显式打开 Ascend DBO 环境变量
export VLLM_ASCEND_ENABLE_DBO=${VLLM_ASCEND_ENABLE_DBO:-1}

# 调试 DBO 触发时可设 DEBUG；正式性能测试建议 INFO
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}


# ------------------------------------------------------------
# Model / server config
# ------------------------------------------------------------
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}

TP=${TP:-1}
DP=${DP:-2}
DP_LOCAL=${DP_LOCAL:-2}

# 为 4K prompt 预留 chat template / special token 空间
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# DBO 需要足够大的 batch token 才有意义
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}


# ------------------------------------------------------------
# DBO thresholds
# ------------------------------------------------------------

# 第一轮建议只验证 prefill DBO：
#   - prefill token >= 1024 才触发 DBO
#   - decode DBO 默认关闭，避免 decode 阶段小 batch ubatch 开销污染结果
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-1024}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-1000000000}


# ------------------------------------------------------------
# Profiler config
# ------------------------------------------------------------
ENABLE_PROFILER=${ENABLE_PROFILER:-0}
PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-ascend/profile}
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/dbo_profile}

# profiler 时建议小一点，避免 trace 爆炸
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-20}

mkdir -p "$TORCH_PROFILER_DIR"


# ------------------------------------------------------------
# Log stats
# ------------------------------------------------------------
# 调试时可以 LOG_STATS=1，正式 benchmark 建议保持 0
LOG_STATS=${LOG_STATS:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}


# ------------------------------------------------------------
# Print config
# ------------------------------------------------------------
echo "============================================================"
echo " DeepSeek-V2 DBO Server"
echo "============================================================"
echo "  MODEL                               = $MODEL"
echo "  SERVER                              = http://${HOST}:${PORT}"
echo "  ASCEND_RT_VISIBLE_DEVICES           = $ASCEND_RT_VISIBLE_DEVICES"
echo "  TP                                  = $TP"
echo "  DP                                  = $DP"
echo "  DP_LOCAL                            = $DP_LOCAL"
echo "  MAX_MODEL_LEN                       = $MAX_MODEL_LEN"
echo "  MAX_NUM_BATCHED_TOKENS              = $MAX_NUM_BATCHED_TOKENS"
echo "  MAX_NUM_SEQS                        = $MAX_NUM_SEQS"
echo ""
echo "  --enable-dbo                        = ON"
echo "  DBO_PREFILL_TOKEN_THRESHOLD          = $DBO_PREFILL_TOKEN_THRESHOLD"
echo "  DBO_DECODE_TOKEN_THRESHOLD           = $DBO_DECODE_TOKEN_THRESHOLD"
echo ""
echo "  HCCL_OP_EXPANSION_MODE               = $HCCL_OP_EXPANSION_MODE"
echo "  VLLM_ASCEND_ENABLE_FLASHCOMM1        = $VLLM_ASCEND_ENABLE_FLASHCOMM1"
echo "  VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = $VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"
echo "  VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED= $VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"
echo "  VLLM_ASCEND_ENABLE_DBO               = $VLLM_ASCEND_ENABLE_DBO"
echo "  VLLM_LOGGING_LEVEL                   = $VLLM_LOGGING_LEVEL"
echo ""
echo "  ENABLE_PROFILER                      = $ENABLE_PROFILER"
echo "  TORCH_PROFILER_DIR                   = $TORCH_PROFILER_DIR"
echo "  PROFILER_MAX_ITERATIONS              = $PROFILER_MAX_ITERATIONS"
echo "  LOG_STATS                            = $LOG_STATS"
echo "  ENFORCE_EAGER                        = $ENFORCE_EAGER"
echo "============================================================"
echo ""
echo "DBO 触发确认："
echo "  grep -n \"should_ubatch: True\" dbo_server.log"
echo "  grep -n \"AllgatherTemplate\\|DeepseekAllgather\\|select_dbo_templates\" dbo_server.log"
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
  --data-parallel-size "$DP"
  --data-parallel-size-local "$DP_LOCAL"
  --enable-expert-parallel

  --enable-dbo
  --all2all-backend deepep_low_latency
  --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
  --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"

  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  serve_args+=(--enforce-eager)
fi

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