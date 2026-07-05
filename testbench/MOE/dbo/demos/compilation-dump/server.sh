#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Compilation Dump Server — vLLM-Ascend DBO
#
# 冷启动 + 编译图 dump，支持渐进式 dump 级别。
#
# 用法：
#   DUMP_MODE=all bash server.sh
#   DUMP_MODE=dynamo bash server.sh
#   DUMP_MODE=compile bash server.sh
#   DUMP_MODE=backend bash server.sh
#   DUMP_MODE=none bash server.sh
#
# 指定端口：
#   PORT=8001 bash server.sh
#
# 快速验证（小 capture size）：
#   CAPTURE_SIZES='[16]' bash server.sh
#
# 不同 FlashComm/DBO 组合：
#   FC1=0 FC2=0 DBO=1 bash server.sh
#   FC1=1 FC2=1 DBO=1 HCCL_MODE=AIV bash server.sh
# ============================================================

# ── Resolve script dir ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Dump mode ─────────────────────────────────────────────────────────────────
#   none    — 不 dump，仅运行（但保留冷启动隔离）
#   dynamo  — dump Dynamo 捕获的 FX Graph (TORCH_TRACE + TORCH_LOGS)
#   compile — dump vllm-ascend compile cache (VLLM_COMPILE_CACHE_PATH + TORCHINDUCTOR_CACHE_DIR)
#   backend — dump triton/torchair backend 编译产物 (AOT compile cache)
#   all     — 全部开启
DUMP_MODE=${DUMP_MODE:-all}

# ── Cold start config ─────────────────────────────────────────────────────────
# 每次启动使用全新 cache 目录以保证冷启动
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
COLD_ROOT=${COLD_ROOT:-${SCRIPT_DIR}/cache/${RUN_ID}}
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
mkdir -p "$COLD_ROOT" "$LOG_DIR"

SERVER_LOG="${LOG_DIR}/server_${RUN_ID}.log"

# ── Compilation parameters ────────────────────────────────────────────────────
# 可以通过环境变量覆盖：
#   CAPTURE_SIZES='[1,2,4,8,16,32]'  → 自定义 capture sizes
#   COMPILE_CACHE_DIR                  → 自定义 compile cache 路径
CAPTURE_SIZES=${CAPTURE_SIZES:-'[16]'}

# ── Basic env ─────────────────────────────────────────────────────────────────
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

# 两卡 TP=2
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}

# ── Cache 目录（均指向冷启动隔离目录）─────────────────────────────────────────
export VLLM_CACHE_ROOT="${COLD_ROOT}/vllm_cache"
export TORCH_EXTENSIONS_DIR="${COLD_ROOT}/torch_extensions"
export TRITON_CACHE_DIR="${COLD_ROOT}/triton_cache"
export HF_HOME=${HF_HOME:-/data/huggingface_cache}
export TMPDIR=${TMPDIR:-/data/tmp}
export TORCHINDUCTOR_CACHE_DIR="${COLD_ROOT}/torch_inductor_cache"

# compile cache 可按需覆盖
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

# ── Ascend DBO / communication config ─────────────────────────────────────────
FC1=${FC1:-1}
FC2=${FC2:-0}
FC2_OSHARED=${FC2_OSHARED:-0}
DBO=${DBO:-1}
HCCL_MODE=${HCCL_MODE:-AI_CPU}

export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-$HCCL_MODE}

export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-$FC1}
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=${VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE:-$FC2}
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=${VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED:-$FC2_OSHARED}
export VLLM_ASCEND_ENABLE_DBO=${VLLM_ASCEND_ENABLE_DBO:-$DBO}

export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# ── Dump config (取决于 DUMP_MODE) ────────────────────────────────────────────

TORCH_TRACE_DIR="${COLD_ROOT}/torch_trace"

case "$DUMP_MODE" in
    dynamo)
        # Layer 1: Dynamo 捕获的 FX Graph
        export TORCH_TRACE="${TORCH_TRACE_DIR}"
        export TORCH_LOGS="+dynamic,guards,recompiles"
        DUMP_DESC="Dynamo FX Graph (TORCH_TRACE + TORCH_LOGS)"
        ;;

    compile)
        # Layer 2: vllm-ascend 编译切图
        # compile cache 和 inductor cache 已在上面设置
        # 不加 TORCH_TRACE 避免 overhead
        unset TORCH_TRACE
        export TORCH_LOGS="+dynamic"
        DUMP_DESC="vllm-ascend compile cache + torch_inductor_cache"
        ;;

    backend)
        # Layer 3: triton/torchair backend 编译
        # AOT compile 产物会自动落到 VLLM_CACHE_ROOT
        unset TORCH_TRACE
        export TORCH_LOGS=""
        # 显式保留 AOT 产物
        export VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE=1
        DUMP_DESC="backend AOT compile artifacts (VLLM_CACHE_ROOT + compile_cache)"
        ;;

    all)
        # 全部 dump
        export TORCH_TRACE="${TORCH_TRACE_DIR}"
        export TORCH_LOGS="+dynamic,guards,recompiles,recompiles_verbose,graph_breaks"
        export VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE=1
        DUMP_DESC="ALL (Dynamo + compile cache + backend AOT)"
        ;;

    none|*)
        unset TORCH_TRACE
        export TORCH_LOGS=""
        DUMP_DESC="none (cold boot only, no dump overhead)"
        ;;
esac

# ── Model / server config ────────────────────────────────────────────────────
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}

TP=${TP:-2}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

# ── DBO thresholds ────────────────────────────────────────────────────────────
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-1024}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-1000000000}

# ── Profiler config ───────────────────────────────────────────────────────────
ENABLE_PROFILER=${ENABLE_PROFILER:-0}
PROFILE_ROOT=${PROFILE_ROOT:-${SCRIPT_DIR}/profile}
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/dbo_profile}
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-20}
mkdir -p "$TORCH_PROFILER_DIR"

# ── Log stats ─────────────────────────────────────────────────────────────────
LOG_STATS=${LOG_STATS:-0}

# ── Print config ──────────────────────────────────────────────────────────────
{
    echo "============================================================"
    echo " Compilation Dump Server"
    echo "============================================================"
    echo "  RUN_ID                              = $RUN_ID"
    echo "  DUMP_MODE                           = $DUMP_MODE"
    echo "  DUMP_DESC                           = $DUMP_DESC"
    echo ""
    echo "  MODEL                               = $MODEL"
    echo "  SERVER                              = http://${HOST}:${PORT}"
    echo "  ASCEND_RT_VISIBLE_DEVICES           = $ASCEND_RT_VISIBLE_DEVICES"
    echo "  TP                                  = $TP"
    echo "  MAX_MODEL_LEN                       = $MAX_MODEL_LEN"
    echo "  MAX_NUM_BATCHED_TOKENS              = $MAX_NUM_BATCHED_TOKENS"
    echo "  MAX_NUM_SEQS                        = $MAX_NUM_SEQS"
    echo ""
    echo "  --enable-dbo                        = $DBO"
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
    echo "  CAPTURE_SIZES                        = $CAPTURE_SIZES"
    echo ""
    echo "  TORCH_TRACE                          = ${TORCH_TRACE:-<unset>}"
    echo "  TORCH_LOGS                           = ${TORCH_LOGS:-<unset>}"
    echo "  VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE   = ${VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE:-<unset>}"
    echo ""
    echo "  COLD_ROOT                            = $COLD_ROOT"
    echo "  VLLM_CACHE_ROOT                      = $VLLM_CACHE_ROOT"
    echo "  VLLM_COMPILE_CACHE_PATH              = $VLLM_COMPILE_CACHE_PATH"
    echo "  TORCHINDUCTOR_CACHE_DIR              = $TORCHINDUCTOR_CACHE_DIR"
    echo "  TRITON_CACHE_DIR                     = $TRITON_CACHE_DIR"
    echo ""
    echo "  ENABLE_PROFILER                      = $ENABLE_PROFILER"
    echo "  SERVER_LOG                           = $SERVER_LOG"
    echo "============================================================"
    echo ""
} | tee "$SERVER_LOG"

# ── Cold start verification ───────────────────────────────────────────────────
# 确认 compile cache 目录为空（真正冷启动）
if [[ -d "$VLLM_COMPILE_CACHE_PATH" ]] && ls -A "$VLLM_COMPILE_CACHE_PATH" 2>/dev/null | grep -q .; then
    echo "⚠ WARNING: Compile cache dir is NOT empty! Cold start may be compromised." | tee -a "$SERVER_LOG"
    echo "  $VLLM_COMPILE_CACHE_PATH" | tee -a "$SERVER_LOG"
    echo "  Existing contents:" | tee -a "$SERVER_LOG"
    ls -la "$VLLM_COMPILE_CACHE_PATH" | tee -a "$SERVER_LOG"
else
    echo "✓ Compile cache dir is empty — true cold start confirmed." | tee -a "$SERVER_LOG"
fi
echo "" | tee -a "$SERVER_LOG"

# ── Quick verification hints ──────────────────────────────────────────────────
cat <<'HINTS' | tee -a "$SERVER_LOG"
After server starts, verify:
  # Check Dynamo graph was captured:
  grep -n "torch.compile took\|dynamo\|fullgraph" <server_log>

  # Check compile cache artifacts:
  ls -la ${COLD_ROOT}/vllm_compile_cache/

  # Check TORCH_TRACE output:
  ls -la ${COLD_ROOT}/torch_trace/
  tlparse ${COLD_ROOT}/torch_trace/

  # Check AOT compile artifacts:
  find ${COLD_ROOT} -name "model" -path "*/torch_aot_compile/*"

  # Check DBO trigger:
  grep -n "should_ubatch: True" <server_log>

  # Check graph capture:
  grep -n "Graph capturing finished\|Capturing CUDA graphs" <server_log>

HINTS

# ── Build serve args ──────────────────────────────────────────────────────────
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
  --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
  --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"

  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"

  --compilation-config "{\"cudagraph_capture_sizes\":${CAPTURE_SIZES},\"cache_dir\":\"${VLLM_COMPILE_CACHE_PATH}\"}"
)

if [[ "$LOG_STATS" == "0" ]]; then
  serve_args+=(--disable-log-stats)
fi

# ── Profiler args ─────────────────────────────────────────────────────────────
if [[ "$ENABLE_PROFILER" == "1" ]]; then
  export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-1800000}

  PROFILER_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_use_gzip\":true,\"torch_profiler_with_memory\":true,\"torch_profiler_with_flops\":false,\"max_iterations\":${PROFILER_MAX_ITERATIONS}}"

  echo "  PROFILER_CONFIG                     = ${PROFILER_CONFIG}" | tee -a "$SERVER_LOG"

  serve_args+=(
    --profiler-config "$PROFILER_CONFIG"
  )
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
echo "Launching vllm serve..." | tee -a "$SERVER_LOG"
echo "" | tee -a "$SERVER_LOG"

vllm "${serve_args[@]}" 2>&1 | tee -a "$SERVER_LOG"
