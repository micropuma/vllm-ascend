#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Warmup Experiment Server — vLLM-Ascend DBO
#
# 用法：
#   EXPERIMENT=compile_range bash server.sh
#   EXPERIMENT=aclgraph bash server.sh
#
# 默认会创建全新 RUN_ID / cache 目录，避免复用旧缓存。
# 如需完全隔离，请不要手动复用 RUN_ID。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT=${EXPERIMENT:-compile_range}
DUMP_MODE=${DUMP_MODE:-none}

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
COLD_ROOT=${COLD_ROOT:-${SCRIPT_DIR}/cache/${RUN_ID}}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
mkdir -p "$COLD_ROOT" "$LOG_DIR"

SERVER_LOG="${LOG_DIR}/server_${RUN_ID}.log"

MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
TP=${TP:-2}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

# 这两个参数是 warmup-exp 重点关心的实验变量，允许外部显式覆盖。
COMPILE_RANGES=${COMPILE_RANGES:-}
CAPTURE_SIZES=${CAPTURE_SIZES:-}

case "$EXPERIMENT" in
    compile_range)
        # 减少 ACLGraph bucket 干扰，把 capture size 压到最小。
        CAPTURE_SIZES=${CAPTURE_SIZES:-'[16]'}
        # 默认给几个边界，实际可由调用侧覆盖。
        COMPILE_RANGES=${COMPILE_RANGES:-'[1024,2048,4096,8192,16384]'}
        ;;
    aclgraph)
        # 减少 piecewise range 干扰，把 compile range 收敛到单个大区间。
        CAPTURE_SIZES=${CAPTURE_SIZES:-'[1,2,4,8,16,32,64,128,256]'}
        COMPILE_RANGES=${COMPILE_RANGES:-'[16384]'}
        ;;
    *)
        echo "Unknown EXPERIMENT=$EXPERIMENT" >&2
        echo "Valid values: compile_range, aclgraph" >&2
        exit 2
        ;;
esac

DBO=${DBO:-1}
FC1=${FC1:-1}
FC2=${FC2:-0}
FC2_OSHARED=${FC2_OSHARED:-0}
HCCL_MODE=${HCCL_MODE:-AI_CPU}

export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-$HCCL_MODE}
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-$FC1}
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=${VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE:-$FC2}
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=${VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED:-$FC2_OSHARED}
export VLLM_ASCEND_ENABLE_DBO=${VLLM_ASCEND_ENABLE_DBO:-$DBO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

export VLLM_CACHE_ROOT="${COLD_ROOT}/vllm_cache"
export TORCH_EXTENSIONS_DIR="${COLD_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${COLD_ROOT}/triton_cache"
export HF_HOME=${HF_HOME:-/data/huggingface_cache}
export TMPDIR=${TMPDIR:-/data/tmp}
export TORCHINDUCTOR_CACHE_DIR="${COLD_ROOT}/torch_inductor_cache"

if [[ -z "${COMPILE_CACHE_DIR:-}" ]]; then
    export VLLM_COMPILE_CACHE_PATH="${COLD_ROOT}/vllm_compile_cache"
else
    export VLLM_COMPILE_CACHE_PATH="$COMPILE_CACHE_DIR"
fi

mkdir -p \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TRITON_CACHE_DIR" \
  "$HF_HOME" \
  "$TMPDIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$VLLM_COMPILE_CACHE_PATH"

export TORCH_TRACE_DIR="${COLD_ROOT}/torch_trace"
case "$DUMP_MODE" in
    dynamo)
        export TORCH_TRACE="${TORCH_TRACE_DIR}"
        export TORCH_LOGS="+dynamic,guards,recompiles"
        DUMP_DESC="Dynamo FX Graph"
        ;;
    compile)
        unset TORCH_TRACE
        export TORCH_LOGS="+dynamic"
        DUMP_DESC="compile cache only"
        ;;
    backend)
        unset TORCH_TRACE
        unset TORCH_LOGS
        export VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE=1
        DUMP_DESC="backend AOT only"
        ;;
    all)
        export TORCH_TRACE="${TORCH_TRACE_DIR}"
        export TORCH_LOGS="+dynamic,guards,recompiles,recompiles_verbose,graph_breaks"
        export VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE=1
        DUMP_DESC="ALL"
        ;;
    none|*)
        unset TORCH_TRACE
        unset TORCH_LOGS
        DUMP_DESC="none (cold boot only, no dump overhead)"
        ;;
esac

ENABLE_PROFILER=${ENABLE_PROFILER:-0}
PROFILE_ROOT=${PROFILE_ROOT:-${SCRIPT_DIR}/profile}
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/warmup_exp_profile}
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-32}
mkdir -p "$TORCH_PROFILER_DIR"

LOG_STATS=${LOG_STATS:-0}

{
    echo "============================================================"
    echo " Warmup Experiment Server"
    echo "============================================================"
    echo "  RUN_ID               = $RUN_ID"
    echo "  EXPERIMENT           = $EXPERIMENT"
    echo "  DUMP_MODE            = $DUMP_MODE"
    echo "  DUMP_DESC            = $DUMP_DESC"
    echo ""
    echo "  MODEL                = $MODEL"
    echo "  SERVER               = http://${HOST}:${PORT}"
    echo "  ASCEND_RT_VISIBLE_DEVICES = $ASCEND_RT_VISIBLE_DEVICES"
    echo "  TP                   = $TP"
    echo "  MAX_MODEL_LEN        = $MAX_MODEL_LEN"
    echo "  MAX_NUM_BATCHED_TOKENS = $MAX_NUM_BATCHED_TOKENS"
    echo "  MAX_NUM_SEQS         = $MAX_NUM_SEQS"
    echo ""
    echo "  DBO                  = $DBO"
    echo "  FC1                  = $FC1"
    echo "  FC2                  = $FC2"
    echo "  HCCL_MODE            = $HCCL_MODE"
    echo ""
    echo "  COMPILE_RANGES       = $COMPILE_RANGES"
    echo "  CAPTURE_SIZES        = $CAPTURE_SIZES"
    echo ""
    echo "  COLD_ROOT            = $COLD_ROOT"
    echo "  VLLM_CACHE_ROOT      = $VLLM_CACHE_ROOT"
    echo "  VLLM_COMPILE_CACHE_PATH = $VLLM_COMPILE_CACHE_PATH"
    echo "  TORCHINDUCTOR_CACHE_DIR = $TORCHINDUCTOR_CACHE_DIR"
    echo "  TRITON_CACHE_DIR     = $TRITON_CACHE_DIR"
    echo ""
    echo "  ENABLE_PROFILER      = $ENABLE_PROFILER"
    echo "  SERVER_LOG           = $SERVER_LOG"
    echo "============================================================"
    echo ""
} | tee "$SERVER_LOG"

if [[ -d "$VLLM_COMPILE_CACHE_PATH" ]] && ls -A "$VLLM_COMPILE_CACHE_PATH" 2>/dev/null | grep -q .; then
    echo "⚠ WARNING: Compile cache dir is NOT empty!" | tee -a "$SERVER_LOG"
    ls -la "$VLLM_COMPILE_CACHE_PATH" | tee -a "$SERVER_LOG"
else
    echo "✓ Compile cache dir is empty — true cold start confirmed." | tee -a "$SERVER_LOG"
fi
echo "" | tee -a "$SERVER_LOG"

serve_args=(
  serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype bfloat16
  --distributed-executor-backend mp
  --tensor-parallel-size "$TP"
  --enable-expert-parallel
  --enable-dbo
  --all2all-backend deepep_low_latency
  --dbo-prefill-token-threshold "${DBO_PREFILL_TOKEN_THRESHOLD:-1024}"
  --dbo-decode-token-threshold "${DBO_DECODE_TOKEN_THRESHOLD:-1000000000}"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --compilation-config "{\"cudagraph_capture_sizes\":${CAPTURE_SIZES},\"compile_ranges_endpoints\":${COMPILE_RANGES},\"cache_dir\":\"${VLLM_COMPILE_CACHE_PATH}\"}"
)

if [[ "$LOG_STATS" == "0" ]]; then
  serve_args+=(--disable-log-stats)
fi

if [[ "$ENABLE_PROFILER" == "1" ]]; then
  export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-1800000}
  PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_use_gzip\":true,\"torch_profiler_with_memory\":true,\"torch_profiler_with_flops\":false,\"max_iterations\":${PROFILER_MAX_ITERATIONS}}"
  echo "  PROFILER_CONFIG      = ${PROFILER_CONFIG}" | tee -a "$SERVER_LOG"
  serve_args+=(--profiler-config "$PROFILER_CONFIG")
fi

echo "Launching vllm serve..." | tee -a "$SERVER_LOG"
echo "" | tee -a "$SERVER_LOG"
vllm "${serve_args[@]}" 2>&1 | tee -a "$SERVER_LOG"
