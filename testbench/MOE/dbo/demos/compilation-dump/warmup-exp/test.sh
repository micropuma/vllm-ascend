#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Warmup Experiment Test — vLLM-Ascend DBO
#
# 支持两类实验：
#   1. compile_range：围绕 compile range 边界序列发压
#   2. aclgraph：围绕 ACLGraph bucket 边界序列发压
#
# 通过 MODE 控制是否先 warmup 再正式压测：
#   MODE=observe   只跑正式序列
#   MODE=warmup    先跑一轮预热，再跑正式序列
#
# 建议和 server.sh 配套使用，且默认保持冷启动 cache 隔离。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
TEST_LOG="${LOG_DIR}/test_${RUN_ID}.log"

MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}

EXPERIMENT=${EXPERIMENT:-compile_range}
MODE=${MODE:-observe}

LABEL=${LABEL:-warmup-exp}
REQUEST_RATE=${REQUEST_RATE:-inf}
OUTPUT_LEN=${OUTPUT_LEN:-16}
NUM_PROMPTS=${NUM_PROMPTS:-4}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
SAVE_DETAILED=${SAVE_DETAILED:-0}

# 每个 step 都是固定输入长度，便于观察 first-hit vs replay。
case "$EXPERIMENT" in
    compile_range)
        # 这个模型的最大上下文是 8192，所以默认只探测到 8192 边界；
        # 但 chat template 会额外吃掉 token，默认不把 8192 放进实验组，
        # 避免因为模板膨胀导致请求直接超限。
        SEQUENCE=${SEQUENCE:-"1023,1024,1025,2047,2048,2049,4095,4096,4097,8191"}
        ;;
    aclgraph)
        SEQUENCE=${SEQUENCE:-"15,16,17,31,32,33,63,64,65,127,128,129,255,256"}
        ;;
    *)
        echo "Unknown EXPERIMENT=$EXPERIMENT" >&2
        echo "Valid values: compile_range, aclgraph" >&2
        exit 2
        ;;
esac

OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/results}
PROFILE_ROOT=${PROFILE_ROOT:-${SCRIPT_DIR}/profile}
mkdir -p "$OUT_DIR"

wait_server() {
    local port="$1"
    local url="http://${HOST}:${port}/v1/models"
    echo "  Waiting for server ${url} ..." | tee -a "$TEST_LOG"
    for _ in $(seq 1 120); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "  ✓ Server is ready" | tee -a "$TEST_LOG"
            return 0
        fi
        sleep 5
    done
    echo "  ✗ Server wait timeout after 600s. Check server logs." | tee -a "$TEST_LOG"
    return 1
}

run_vllm_bench() {
    local port="$1"
    local input_len="$2"
    local result_filename="$3"
    local save_result="${4:-1}"
    local bench_label="${5:-bench}"

    local args=(
        vllm bench serve
        --host "$HOST" --port "$port"
        --model "$MODEL" --tokenizer "$MODEL"
        --backend openai-chat
        --endpoint /v1/chat/completions
        --dataset-name random
        --num-prompts "$NUM_PROMPTS"
        --max-concurrency "$MAX_CONCURRENCY"
        --request-rate "$REQUEST_RATE"
        --random-input-len "$input_len"
        --random-output-len "$OUTPUT_LEN"
        --random-range-ratio 0.0
        --temperature 0
        --ignore-eos
        --percentile-metrics ttft,tpot,itl,e2el
        --result-dir "$OUT_DIR"
        --result-filename "$result_filename"
    )

    if [[ "$save_result" == "1" ]]; then
        args+=(--save-result)
    fi
    if [[ "$SAVE_DETAILED" == "1" ]]; then
        args+=(--save-detailed)
    fi

    echo "  [$bench_label] input_len=${input_len}" | tee -a "$TEST_LOG"
    "${args[@]}" 2>&1 | tee -a "$TEST_LOG"
}

print_header() {
    {
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Warmup Experiment Test  [${LABEL}]"
        echo "  RUN_ID            = ${RUN_ID}"
        echo "  EXPERIMENT        = ${EXPERIMENT}"
        echo "  MODE              = ${MODE}"
        echo "  server            = http://${HOST}:${PORT}"
        echo "  SEQUENCE          = ${SEQUENCE}"
        echo "  INPUT_LEN         = varying"
        echo "  OUTPUT_LEN        = ${OUTPUT_LEN}"
        echo "  NUM_PROMPTS       = ${NUM_PROMPTS}"
        echo "  MAX_CONCURRENCY   = ${MAX_CONCURRENCY}"
        echo "  REQUEST_RATE      = ${REQUEST_RATE}"
        echo "  TEST_LOG          = ${TEST_LOG}"
        echo "════════════════════════════════════════════════════════════"
        echo ""
    } | tee -a "$TEST_LOG"
}

run_sequence() {
    local phase="$1"
    local seq="$2"
    local idx=0
    IFS=',' read -r -a lens <<< "$seq"
    for input_len in "${lens[@]}"; do
        idx=$((idx + 1))
        local result_file="${LABEL}_${EXPERIMENT}_${phase}_step${idx}_in${input_len}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}.json"
        run_vllm_bench "$PORT" "$input_len" "$result_file" 1 "${phase}-step${idx}" || true
    done
}

print_header
wait_server "$PORT"

if [[ "$MODE" == "warmup" ]]; then
    echo "  [1/2] Warmup sequence..." | tee -a "$TEST_LOG"
    run_sequence "warmup" "$SEQUENCE"
fi

echo "  [2/2] Formal sequence..." | tee -a "$TEST_LOG"
run_sequence "formal" "$SEQUENCE"

echo ""
echo "  ✓ All results saved under ${OUT_DIR}" | tee -a "$TEST_LOG"
