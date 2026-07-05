#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Compilation Dump Test — benchmark & verification
#
# 配合 server.sh 使用，发压并验证 DBO 触发。
#
# 用法：
#   # 单次 benchmark
#   bash test.sh
#
#   # 指定端口和参数
#   PORT=8001 bash test.sh
#
#   # 快速验证（单请求）
#   PRESET=quick-verify bash test.sh
#
#   # 对比两个 server 结果
#   bash test.sh --compare
#
#   # Profile 采集
#   bash test.sh --profile
#
# Presets:
#   PRESET=quick-verify  INPUT_LEN=1024 OUTPUT_LEN=1  NUM_PROMPTS=1  MAX_CONCURRENCY=1
#   PRESET=small         INPUT_LEN=4096 OUTPUT_LEN=1  NUM_PROMPTS=32 MAX_CONCURRENCY=16
#   PRESET=prefill4k     INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96
#   PRESET=prefill8k     INPUT_LEN=8192 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96
#   PRESET=custom        Use explicitly supplied INPUT_LEN / OUTPUT_LEN / etc.
# ============================================================

# ── Resolve script dir ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/logs}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
TEST_LOG="${LOG_DIR}/test_${RUN_ID}.log"

# ── Basic parameters ──────────────────────────────────────────────────────────
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
LABEL=${LABEL:-compilation-dump}

PRESET=${PRESET:-prefill4k}

case "$PRESET" in
    quick-verify)
        DEFAULT_INPUT_LEN=1024
        DEFAULT_OUTPUT_LEN=1
        DEFAULT_NUM_PROMPTS=1
        DEFAULT_MAX_CONCURRENCY=1
        ;;
    small)
        DEFAULT_INPUT_LEN=4096
        DEFAULT_OUTPUT_LEN=1
        DEFAULT_NUM_PROMPTS=32
        DEFAULT_MAX_CONCURRENCY=16
        ;;
    prefill4k)
        DEFAULT_INPUT_LEN=4096
        DEFAULT_OUTPUT_LEN=16
        DEFAULT_NUM_PROMPTS=500
        DEFAULT_MAX_CONCURRENCY=96
        ;;
    prefill8k)
        DEFAULT_INPUT_LEN=8192
        DEFAULT_OUTPUT_LEN=16
        DEFAULT_NUM_PROMPTS=500
        DEFAULT_MAX_CONCURRENCY=96
        ;;
    custom)
        DEFAULT_INPUT_LEN=1024
        DEFAULT_OUTPUT_LEN=128
        DEFAULT_NUM_PROMPTS=200
        DEFAULT_MAX_CONCURRENCY=64
        ;;
    *)
        echo "Unknown PRESET=$PRESET" >&2
        echo "Valid: quick-verify, small, prefill4k, prefill8k, custom" >&2
        exit 2
        ;;
esac

INPUT_LEN=${INPUT_LEN:-$DEFAULT_INPUT_LEN}
OUTPUT_LEN=${OUTPUT_LEN:-$DEFAULT_OUTPUT_LEN}
NUM_PROMPTS=${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-$DEFAULT_MAX_CONCURRENCY}
REQUEST_RATE=${REQUEST_RATE:-inf}

WARMUP_PROMPTS=${WARMUP_PROMPTS:-16}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-16}

SAVE_DETAILED=${SAVE_DETAILED:-0}

OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/results}
PROFILE_ROOT=${PROFILE_ROOT:-${SCRIPT_DIR}/profile}
mkdir -p "$OUT_DIR"

RESULT_SUFFIX=${RESULT_SUFFIX:-}
RESULT_FILE="${OUT_DIR}/${LABEL}_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"

# ── Dump context headers ──────────────────────────────────────────────────────
print_header() {
    {
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Compilation Dump Test  [${LABEL}]"
        echo "  RUN_ID            = ${RUN_ID}"
        echo "  PRESET            = ${PRESET}"
        echo "  server            = http://${HOST}:${PORT}"
        echo "  INPUT_LEN         = ${INPUT_LEN}"
        echo "  OUTPUT_LEN        = ${OUTPUT_LEN}"
        echo "  NUM_PROMPTS       = ${NUM_PROMPTS}"
        echo "  MAX_CONCURRENCY   = ${MAX_CONCURRENCY}"
        echo "  REQUEST_RATE      = ${REQUEST_RATE}"
        echo "  RESULT_FILE       = ${RESULT_FILE}"
        echo "  TEST_LOG          = ${TEST_LOG}"
        echo "════════════════════════════════════════════════════════════"
        echo ""
    } | tee -a "$TEST_LOG"
}

# ── Wait for server ───────────────────────────────────────────────────────────
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

# ── Run benchmark ─────────────────────────────────────────────────────────────
run_vllm_bench() {
    local port="$1"
    local num_prompts="$2"
    local max_concurrency="$3"
    local result_filename="$4"
    local save_result="$5"
    local bench_label="$6"

    local args=(
        vllm bench serve
        --host "$HOST" --port "$port"
        --model "$MODEL" --tokenizer "$MODEL"
        --backend openai-chat
        --endpoint /v1/chat/completions
        --dataset-name random
        --num-prompts "$num_prompts"
        --max-concurrency "$max_concurrency"
        --request-rate "$REQUEST_RATE"
        --random-input-len "$INPUT_LEN"
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

    echo "  [$bench_label] Running benchmark..." | tee -a "$TEST_LOG"
    "${args[@]}" 2>&1 | tee -a "$TEST_LOG"
    echo "  [$bench_label] Done." | tee -a "$TEST_LOG"
}

run_bench() {
    local port="$1"
    local label="$2"
    local out_file="$3"
    local with_profile="${4:-0}"

    print_header

    # Warmup
    echo "  [1/3] Warmup..." | tee -a "$TEST_LOG"
    run_vllm_bench "$port" "$WARMUP_PROMPTS" "$WARMUP_CONCURRENCY" "warmup_${label}.json" 0 "warmup" \
        2>/dev/null || true

    if [[ "$with_profile" == "1" ]]; then
        echo "  [2/3] Starting torch profiler..." | tee -a "$TEST_LOG"
        curl -sf -X POST "http://${HOST}:${port}/start_profile" >/dev/null
        trap 'curl -sf -X POST "http://${HOST}:${port}/stop_profile" >/dev/null 2>&1 || true' EXIT
    fi

    echo "  [2/3] Formal benchmark..." | tee -a "$TEST_LOG"
    run_vllm_bench "$port" "$NUM_PROMPTS" "$MAX_CONCURRENCY" "$(basename "$out_file")" 1 "bench"

    if [[ "$with_profile" == "1" ]]; then
        echo "  [3/3] Stopping profiler and waiting for flush..." | tee -a "$TEST_LOG"
        curl -sf -X POST "http://${HOST}:${port}/stop_profile" >/dev/null
        trap - EXIT

        local profiler_dir="${PROFILE_ROOT}/dbo_profile"
        local latest
        latest=$(find "$profiler_dir" -maxdepth 1 -mindepth 1 -type d -name '*_ascend_pt' \
                 2>/dev/null | sort | tail -n 1 || true)
        if [[ -n "$latest" ]]; then
            echo "  ✓ Profile directory: $latest" | tee -a "$TEST_LOG"
            if python3 -c 'import torch_npu' 2>/dev/null; then
                echo "  Running torch_npu analyse..." | tee -a "$TEST_LOG"
                PROFILE_RUN_DIR="$latest" python3 - <<'PYEOF'
import os
from torch_npu.profiler.profiler import analyse
analyse(os.environ["PROFILE_RUN_DIR"])
print("analyse done. Profile ready for TensorBoard / MindStudio:", os.environ["PROFILE_RUN_DIR"])
PYEOF
            else
                echo "  tensorboard --logdir $latest" | tee -a "$TEST_LOG"
            fi
        else
            echo "  ⚠ Profile directory not found." | tee -a "$TEST_LOG"
        fi
    fi

    echo "  ✓ Result saved to $out_file" | tee -a "$TEST_LOG"
}

# ── Extract metrics ───────────────────────────────────────────────────────────
extract_metric() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "    (result file does not exist)"
        return
    fi
    python3 - "$file" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
def g(key):
    return d.get(key, None)
def fmt(key, unit=""):
    v = g(key)
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}{unit}"
    return f"{v}{unit}"
print(f"    output_throughput      : {fmt('output_throughput', ' tok/s')}")
print(f"    request_throughput     : {fmt('request_throughput', ' req/s')}")
print(f"    mean_ttft_ms           : {fmt('mean_ttft_ms', ' ms')}")
print(f"    p99_ttft_ms            : {fmt('p99_ttft_ms', ' ms')}")
print(f"    mean_tpot_ms           : {fmt('mean_tpot_ms', ' ms/tok')}")
print(f"    p99_tpot_ms            : {fmt('p99_tpot_ms', ' ms/tok')}")
print(f"    mean_itl_ms            : {fmt('mean_itl_ms', ' ms')}")
print(f"    p99_itl_ms             : {fmt('p99_itl_ms', ' ms')}")
print(f"    mean_e2el_ms           : {fmt('mean_e2el_ms', ' ms')}")
print(f"    p99_e2el_ms            : {fmt('p99_e2el_ms', ' ms')}")
print(f"    completed              : {g('completed')}")
print(f"    failed                 : {g('failed')}")
PYEOF
}

# ── Compare ────────────────────────────────────────────────────────────────────
compare_results() {
    local file_a="${1:-}"
    local file_b="${2:-}"

    if [[ -z "$file_a" ]]; then
        # find the two most recent result files
        local results
        results=$(find "$OUT_DIR" -maxdepth 1 -name "*.json" -not -name "warmup_*" \
                  -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -2 | awk '{print $2}')
        file_a=$(echo "$results" | head -1)
        file_b=$(echo "$results" | tail -1)
    fi

    if [[ -z "$file_a" || -z "$file_b" ]]; then
        echo "Need at least two result files to compare." >&2
        echo "Run two benchmarks with different LABEL first." >&2
        exit 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Compare Results"
    echo "════════════════════════════════════════════════════════════"
    echo "  A: $(basename "$file_a")"
    echo "  B: $(basename "$file_b")"
    echo ""

    echo "  [A]"
    extract_metric "$file_a"

    echo ""
    echo "  [B]"
    extract_metric "$file_b"

    echo ""
    echo "  Compare (B vs A):"
    python3 - "$file_a" "$file_b" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    a = json.load(f)
with open(sys.argv[2]) as f:
    b = json.load(f)

def fmt_ratio(key, higher_better):
    av, bv = a.get(key), b.get(key)
    if not av or not bv:
        return "N/A"
    if higher_better:
        ratio = bv / av
    else:
        ratio = av / bv
    arrow = "↑" if ratio > 1.005 else ("↓" if ratio < 0.995 else "→")
    pct = (ratio - 1) * 100
    return f"{ratio:.3f}x ({pct:+.1f}%) {arrow}"

print(f"    output_throughput      : {fmt_ratio('output_throughput', True)}")
print(f"    request_throughput     : {fmt_ratio('request_throughput', True)}")
print(f"    mean_ttft_ms           : {fmt_ratio('mean_ttft_ms', False)}")
print(f"    p99_ttft_ms            : {fmt_ratio('p99_ttft_ms', False)}")
print(f"    mean_tpot_ms           : {fmt_ratio('mean_tpot_ms', False)}")
print(f"    mean_e2el_ms           : {fmt_ratio('mean_e2el_ms', False)}")
PYEOF

    echo ""
    echo "  Result files:"
    echo "    A: $file_a"
    echo "    B: $file_b"
}

# ── Verify DBO trigger ────────────────────────────────────────────────────────
verify_dbo() {
    local server_log="${1:-}"
    if [[ -z "$server_log" ]]; then
        # try to find the most recent server log
        server_log=$(find "$LOG_DIR" -maxdepth 1 -name "server_*.log" -printf "%T@ %p\n" \
                     2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  DBO Trigger Verification"
    echo "════════════════════════════════════════════════════════════"

    if [[ -z "$server_log" || ! -f "$server_log" ]]; then
        echo "  No server log found. Specify with: SERVER_LOG=<path> bash test.sh --verify"
        return
    fi

    echo "  Log: $server_log"
    echo ""

    echo "  [DBO trigger]"
    local dbo_lines
    dbo_lines=$(grep -c "should_ubatch: True" "$server_log" 2>/dev/null || echo "0")
    echo "    should_ubatch: True  →  $dbo_lines occurrences"

    echo ""
    echo "  [Template selection]"
    grep -n "AllgatherTemplate\|DeepseekAllgather\|select_dbo_templates\|AlltoallTemplate" \
         "$server_log" 2>/dev/null | head -5 || echo "    (none)"

    echo ""
    echo "  [Compilation]"
    grep -n "torch.compile took\|compilation.*sec" "$server_log" 2>/dev/null | head -5 || echo "    (none)"

    echo ""
    echo "  [Graph capture]"
    grep -n "Capturing CUDA graphs\|Graph capturing finished\|NPUGraph" \
         "$server_log" 2>/dev/null | head -5 || echo "    (none)"

    echo ""
    echo "  [Startup]"
    grep -n "Application startup complete" "$server_log" 2>/dev/null | head -1 || echo "    (not found)"

    echo ""
    echo "  [Errors — should be empty or only benign warnings]"
    grep -n "ERROR\|EZ9999\|AddRmsNormBias\|name 'Min' is not defined\|shape.*must match" \
         "$server_log" 2>/dev/null | head -5 || echo "    (none found)"
}

# ── Main ──────────────────────────────────────────────────────────────────────
MODE="${1:-single}"

case "$MODE" in
    --compare)
        shift
        compare_results "${1:-}" "${2:-}"
        ;;

    --verify)
        shift
        verify_dbo "${1:-}"
        ;;

    --profile)
        wait_server "$PORT"
        run_bench "$PORT" "$LABEL" "$RESULT_FILE" 1
        ;;

    --auto)
        wait_server "$PORT"
        run_bench "$PORT" "$LABEL" "$RESULT_FILE" 0

        echo ""
        echo "Run compare with:"
        echo "  bash test.sh --compare"
        echo "Verify DBO trigger with:"
        echo "  bash test.sh --verify"
        ;;

    single|*)
        wait_server "$PORT"
        run_bench "$PORT" "$LABEL" "$RESULT_FILE" 0

        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Benchmark complete."
        echo "════════════════════════════════════════════════════════════"
        echo ""
        echo "  Extract metrics from result:"
        echo "    python3 -c \"import json; d=json.load(open('${RESULT_FILE}')); print(f'completed={d[\\\"completed\\\"]} failed={d[\\\"failed\\\"]}')\""
        echo ""
        echo "  Verify DBO trigger:"
        echo "    bash ${SCRIPT_DIR}/test.sh --verify"
        echo ""
        echo "  Compare two runs:"
        echo "    bash ${SCRIPT_DIR}/test.sh --compare"
        echo ""
        echo "  Profile (need ENABLE_PROFILER=1 on server):"
        echo "    bash ${SCRIPT_DIR}/test.sh --profile"
        ;;
esac
