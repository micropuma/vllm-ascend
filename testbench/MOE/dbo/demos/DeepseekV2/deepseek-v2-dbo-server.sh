#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DeepSeek-V2 DBO Server — Ascend / vLLM-Ascend
#
# 启动：
#   bash deepseek-v2-dbo-server.sh
#
# 指定端口：
#   PORT=8001 bash deepseek-v2-dbo-server.sh
#
# 采集 profiler：
#   ENABLE_PROFILER=1 PORT=8001 bash deepseek-v2-dbo-server.sh
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
# Ascend DBO / communication config
# ------------------------------------------------------------

# 推荐 AI_CPU：尽量避免 HCCL 通信占用 AI Core / AI Vector 计算资源
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AI_CPU}

# TODO(leon)：目前flashcomm1/flashcomm2在dbo下和torch compile不兼容，先默认关闭
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}
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

TP=${TP:-2}

# 为 4K prompt 预留 chat template / special token 空间
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# DBO 需要足够大的 batch token 才有意义
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ADDITIONAL_CONFIG=${ADDITIONAL_CONFIG:-}


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
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/${LABEL:-dbo}_profile}

# profiler 时建议小一点，避免 trace 爆炸
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-20}

# Keep kernel/communication analysis and Python stack analysis separate.  Stack
# collection on both TP ranks makes the operator trace unnecessarily large and
# has produced incomplete Ascend trace JSONs in long DBO runs.
PROFILER_MODE=${PROFILER_MODE:-operator} # operator | host_stack
case "$PROFILER_MODE" in
    operator)
        PROFILER_WITH_STACK=false
        PROFILER_RECORD_SHAPES=true
        ;;
    host_stack)
        PROFILER_WITH_STACK=true
        PROFILER_RECORD_SHAPES=false
        ;;
    *)
        echo "Unknown PROFILER_MODE=$PROFILER_MODE (expected operator or host_stack)" >&2
        exit 2
        ;;
esac

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
echo " DeepSeek-V2 DBO Server"
echo "============================================================"
echo "  MODEL                               = $MODEL"
echo "  SERVER                              = http://${HOST}:${PORT}"
echo "  ASCEND_RT_VISIBLE_DEVICES           = $ASCEND_RT_VISIBLE_DEVICES"
echo "  TP                                  = $TP"
echo "  MAX_MODEL_LEN                       = $MAX_MODEL_LEN"
echo "  MAX_NUM_BATCHED_TOKENS              = $MAX_NUM_BATCHED_TOKENS"
echo "  MAX_NUM_SEQS                        = $MAX_NUM_SEQS"
echo "  ADDITIONAL_CONFIG                   = ${ADDITIONAL_CONFIG:-<unset>}"
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
echo "  PROFILER_MODE                        = $PROFILER_MODE"
echo "  TORCH_PROFILER_DIR                   = $TORCH_PROFILER_DIR"
echo "  PROFILER_MAX_ITERATIONS              = $PROFILER_MAX_ITERATIONS"
echo "  LOG_STATS                            = $LOG_STATS"
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
  --generation-config vllm

  --distributed-executor-backend mp
  --tensor-parallel-size "$TP"
  --enable-expert-parallel

  --all2all-backend deepep_low_latency
  --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
  --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"

  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
)

if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  serve_args+=(--enforce-eager)
fi

if [[ -n "$ADDITIONAL_CONFIG" ]]; then
  jq empty <<<"$ADDITIONAL_CONFIG"
  serve_args+=(--additional-config "$ADDITIONAL_CONFIG")
fi

if [[ "$VLLM_ASCEND_ENABLE_DBO" == "1" ]]; then
  serve_args+=(--enable-dbo)
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

  # torch_npu must reach RECORD_AND_SAVE before it stops; stopping directly
  # from RECORD can drop the CANN PROF_* export. Keep vLLM's own worker limit
  # disabled and let the torch_npu schedule own the bounded capture window.
  # The frontend remains disabled so only worker forward steps are captured.
  PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":${PROFILER_WITH_STACK},\"torch_profiler_record_shapes\":${PROFILER_RECORD_SHAPES},\"torch_profiler_use_gzip\":true,\"torch_profiler_with_memory\":true,\"torch_profiler_with_flops\":false,\"ignore_frontend\":true,\"warmup_iterations\":1,\"active_iterations\":${PROFILER_MAX_ITERATIONS},\"max_iterations\":0}"

  echo "  PROFILER_CONFIG                     = ${PROFILER_CONFIG}"

  serve_args+=(
    --profiler-config "$PROFILER_CONFIG"
  )
fi


# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
vllm "${serve_args[@]}"
