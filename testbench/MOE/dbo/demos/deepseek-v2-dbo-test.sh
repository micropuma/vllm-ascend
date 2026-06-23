#!/usr/bin/env bash
set -euo pipefail

# DBO vs Baseline benchmark script for vLLM-Ascend / DeepSeek-V2.
#
# Main goal:
#   Use a prefill-heavy workload that is more suitable for observing DBO benefit.
#
# Recommended flow:
#   # terminal 1
#   PORT=8000 bash deepseek-v2-server.sh 2>&1 | tee baseline_server.log
#   # terminal 2
#   LABEL=baseline PORT=8000 bash deepseek-v2-dbo-test.sh
#
#   # terminal 1: stop baseline, then start DBO
#   PORT=8001 bash deepseek-v2-dbo-server.sh 2>&1 | tee dbo_server.log
#   # terminal 2
#   LABEL=dbo PORT=8001 bash deepseek-v2-dbo-test.sh
#
#   # compare
#   bash deepseek-v2-dbo-test.sh --compare
#
# Presets:
#   BENCH_PRESET=prefill4k   INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96
#   BENCH_PRESET=ttft4k      INPUT_LEN=4096 OUTPUT_LEN=1  NUM_PROMPTS=500 MAX_CONCURRENCY=96
#   BENCH_PRESET=prefill8k   INPUT_LEN=8192 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96
#   BENCH_PRESET=quick       INPUT_LEN=1024 OUTPUT_LEN=128 NUM_PROMPTS=200 MAX_CONCURRENCY=64
#   BENCH_PRESET=custom      Use explicitly supplied INPUT_LEN / OUTPUT_LEN / NUM_PROMPTS / MAX_CONCURRENCY.

# ── Basic parameters ────────────────────────────────────────────────────────
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
LABEL=${LABEL:-baseline}

# Default benchmark preset: prefill-heavy, close to the useful DBO test regime.
BENCH_PRESET=${BENCH_PRESET:-prefill4k}

case "$BENCH_PRESET" in
    prefill4k)
        DEFAULT_INPUT_LEN=4096
        DEFAULT_OUTPUT_LEN=16
        DEFAULT_NUM_PROMPTS=500
        DEFAULT_MAX_CONCURRENCY=96
        ;;
    ttft4k)
        DEFAULT_INPUT_LEN=4096
        DEFAULT_OUTPUT_LEN=1
        DEFAULT_NUM_PROMPTS=500
        DEFAULT_MAX_CONCURRENCY=96
        ;;
    prefill8k)
        DEFAULT_INPUT_LEN=8192
        DEFAULT_OUTPUT_LEN=16
        DEFAULT_NUM_PROMPTS=500
        DEFAULT_MAX_CONCURRENCY=96
        ;;
    quick)
        DEFAULT_INPUT_LEN=1024
        DEFAULT_OUTPUT_LEN=128
        DEFAULT_NUM_PROMPTS=200
        DEFAULT_MAX_CONCURRENCY=64
        ;;
    custom)
        DEFAULT_INPUT_LEN=1024
        DEFAULT_OUTPUT_LEN=128
        DEFAULT_NUM_PROMPTS=200
        DEFAULT_MAX_CONCURRENCY=64
        ;;
    *)
        echo "Unknown BENCH_PRESET=$BENCH_PRESET" >&2
        echo "Valid values: prefill4k, ttft4k, prefill8k, quick, custom" >&2
        exit 2
        ;;
esac

INPUT_LEN=${INPUT_LEN:-$DEFAULT_INPUT_LEN}
OUTPUT_LEN=${OUTPUT_LEN:-$DEFAULT_OUTPUT_LEN}
NUM_PROMPTS=${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-$DEFAULT_MAX_CONCURRENCY}
REQUEST_RATE=${REQUEST_RATE:-inf}

# Warmup remains small to avoid overlong cold-start stage.
WARMUP_PROMPTS=${WARMUP_PROMPTS:-16}
WARMUP_CONCURRENCY=${WARMUP_CONCURRENCY:-16}

# Optional detailed request records. Useful when locating TTFT / ITL tails.
SAVE_DETAILED=${SAVE_DETAILED:-0}

OUT_DIR=${OUT_DIR:-/data/workspace/vllm-ascend/testbench/MOE/dbo/results}
PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-ascend/profile}
mkdir -p "$OUT_DIR"

# Optional suffix for repeated runs, e.g. RESULT_SUFFIX=_run1.
RESULT_SUFFIX=${RESULT_SUFFIX:-}
RESULT_FILE="${OUT_DIR}/${LABEL}_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"

# ── Helper functions ────────────────────────────────────────────────────────

wait_server() {
    local port="$1"
    local url="http://${HOST}:${port}/v1/models"
    echo "  Waiting for server ${url} ..."
    for _ in $(seq 1 120); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "  ✓ Server is ready"
            return 0
        fi
        sleep 5
    done
    echo "  ✗ Server wait timeout after 600s. Check server logs." >&2
    return 1
}

print_bench_config() {
    local port="$1"
    local label="$2"
    local with_profile="$3"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Benchmark [$label]  server=http://${HOST}:${port}"
    echo "  BENCH_PRESET=${BENCH_PRESET}"
    echo "  INPUT_LEN=${INPUT_LEN}  OUTPUT_LEN=${OUTPUT_LEN}"
    echo "  NUM_PROMPTS=${NUM_PROMPTS}  MAX_CONCURRENCY=${MAX_CONCURRENCY}"
    echo "  REQUEST_RATE=${REQUEST_RATE}  SAVE_DETAILED=${SAVE_DETAILED}"
    echo "  PROFILE=${with_profile}"
    echo "════════════════════════════════════════════════════════════"
    if [[ "$INPUT_LEN" -lt 4096 && "$BENCH_PRESET" != "quick" ]]; then
        echo "  ⚠ INPUT_LEN < 4096. This may be too small to expose DBO prefill benefit."
    fi
    if [[ "$OUTPUT_LEN" -gt 32 && "$BENCH_PRESET" != "quick" ]]; then
        echo "  ⚠ OUTPUT_LEN > 32. Long decode may dilute prefill DBO benefit."
    fi
}

run_vllm_bench() {
    local port="$1"
    local num_prompts="$2"
    local max_concurrency="$3"
    local result_filename="$4"
    local save_result="$5"

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

    "${args[@]}"
}

run_bench() {
    local port="$1"
    local label="$2"
    local out_file="$3"
    local with_profile="${4:-0}"

    print_bench_config "$port" "$label" "$with_profile"

    # Warmup: not counted, no profiler. Failure does not abort the formal benchmark.
    echo "  [1/3] Warmup..."
    run_vllm_bench "$port" "$WARMUP_PROMPTS" "$WARMUP_CONCURRENCY" "warmup_${label}.json" 0 \
        2>/dev/null || true

    if [[ "$with_profile" == "1" ]]; then
        echo "  [2/3] Starting torch profiler..."
        curl -sf -X POST "http://${HOST}:${port}/start_profile" >/dev/null
        trap 'curl -sf -X POST "http://${HOST}:${port}/stop_profile" >/dev/null 2>&1 || true' EXIT
    fi

    echo "  [2/3] Formal benchmark..."
    run_vllm_bench "$port" "$NUM_PROMPTS" "$MAX_CONCURRENCY" "$(basename "$out_file")" 1

    if [[ "$with_profile" == "1" ]]; then
        echo "  [3/3] Stopping profiler and waiting for flush..."
        curl -sf -X POST "http://${HOST}:${port}/stop_profile" >/dev/null
        trap - EXIT

        local profiler_dir="${PROFILE_ROOT}/${label}_profile"
        local latest
        latest=$(find "$profiler_dir" -maxdepth 1 -mindepth 1 -type d -name '*_ascend_pt' \
                 2>/dev/null | sort | tail -n 1 || true)
        if [[ -n "$latest" ]]; then
            echo "  ✓ Profile directory: $latest"
            if python3 -c 'import torch_npu' 2>/dev/null; then
                echo "  Running torch_npu analyse..."
                PROFILE_RUN_DIR="$latest" python3 - <<'PYEOF'
import os
from torch_npu.profiler.profiler import analyse
analyse(os.environ["PROFILE_RUN_DIR"])
print("analyse done. Open with TensorBoard / MindStudio:", os.environ["PROFILE_RUN_DIR"])
PYEOF
            else
                echo "  torch_npu is unavailable. Skip analyse."
                echo "  tensorboard --logdir $latest"
            fi
        else
            echo "  ⚠ Profile directory not found. Confirm server was started with ENABLE_PROFILER=1."
            echo "    Expected: ${profiler_dir}/*_ascend_pt"
        fi
    fi

    echo "  ✓ Result saved to $out_file"
}

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

print(f"    Output throughput      : {fmt('output_throughput', ' tok/s')}")
print(f"    Request throughput     : {fmt('request_throughput', ' req/s')}")
print(f"    Mean TTFT              : {fmt('mean_ttft_ms', ' ms')}")
print(f"    P99 TTFT               : {fmt('p99_ttft_ms', ' ms')}")
print(f"    Mean TPOT              : {fmt('mean_tpot_ms', ' ms/tok')}")
print(f"    P99 TPOT               : {fmt('p99_tpot_ms', ' ms/tok')}")
print(f"    Mean ITL               : {fmt('mean_itl_ms', ' ms')}")
print(f"    P99 ITL                : {fmt('p99_itl_ms', ' ms')}")
print(f"    Mean E2E latency       : {fmt('mean_e2el_ms', ' ms')}")
print(f"    P99 E2E latency        : {fmt('p99_e2el_ms', ' ms')}")
PYEOF
}

compare_results() {
    local base_file="${OUT_DIR}/baseline_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"
    local dbo_file="${OUT_DIR}/dbo_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Compare Results"
    echo "════════════════════════════════════════════════════════════"
    echo "  BENCH_PRESET=${BENCH_PRESET}"
    echo "  INPUT_LEN=${INPUT_LEN} OUTPUT_LEN=${OUTPUT_LEN} NUM_PROMPTS=${NUM_PROMPTS} MAX_CONCURRENCY=${MAX_CONCURRENCY}"
    echo ""

    if [[ ! -f "$base_file" ]]; then
        echo "  ✗ baseline result missing: $base_file"
    else
        echo "  [Baseline]"
        extract_metric "$base_file"
    fi

    echo ""

    if [[ ! -f "$dbo_file" ]]; then
        echo "  ✗ DBO result missing: $dbo_file"
    else
        echo "  [DBO]"
        extract_metric "$dbo_file"
    fi

    if [[ -f "$base_file" && -f "$dbo_file" ]]; then
        echo ""
        python3 - "$base_file" "$dbo_file" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    b = json.load(f)
with open(sys.argv[2]) as f:
    d = json.load(f)

def val(obj, key):
    x = obj.get(key, None)
    return x if isinstance(x, (int, float)) and x != 0 else None

def higher(key):
    bv, dv = val(b, key), val(d, key)
    if bv is None or dv is None:
        return "N/A", None
    pct = (dv / bv - 1.0) * 100.0
    return f"{dv / bv:.3f}x ({pct:+.2f}%)", pct

def lower(key):
    bv, dv = val(b, key), val(d, key)
    if bv is None or dv is None:
        return "N/A", None
    pct = (1.0 - dv / bv) * 100.0
    return f"{bv / dv:.3f}x ({pct:+.2f}%)", pct

out_s, out_pct = higher("output_throughput")
req_s, req_pct = higher("request_throughput")
ttft_s, ttft_pct = lower("mean_ttft_ms")
p99_ttft_s, p99_ttft_pct = lower("p99_ttft_ms")
tpot_s, tpot_pct = lower("mean_tpot_ms")
e2e_s, e2e_pct = lower("mean_e2el_ms")

print("  ┌──────────────────────────────┬────────────────────────────┐")
print("  │ Metric                       │ DBO vs Baseline             │")
print("  ├──────────────────────────────┼────────────────────────────┤")
print(f"  │ Output throughput            │ {out_s:>26} │")
print(f"  │ Request throughput           │ {req_s:>26} │")
print(f"  │ Mean TTFT lower-is-better    │ {ttft_s:>26} │")
print(f"  │ P99 TTFT lower-is-better     │ {p99_ttft_s:>26} │")
print(f"  │ Mean TPOT lower-is-better    │ {tpot_s:>26} │")
print(f"  │ Mean E2E lower-is-better     │ {e2e_s:>26} │")
print("  └──────────────────────────────┴────────────────────────────┘")

positive = sum(1 for x in [out_pct, req_pct, ttft_pct, p99_ttft_pct, tpot_pct, e2e_pct] if x is not None and x > 0)
negative = sum(1 for x in [out_pct, req_pct, ttft_pct, p99_ttft_pct, tpot_pct, e2e_pct] if x is not None and x < 0)

print("")
if positive >= 4 and (req_pct or 0) > 2 and (ttft_pct or 0) > 2:
    print("  ✓ DBO shows positive benefit in this workload.")
elif positive > negative:
    print("  △ DBO shows mixed / small positive benefit. Repeat runs and inspect profiler.")
else:
    print("  △ DBO benefit is not observed in this workload.")
    print("    Suggested DBO-friendly workload:")
    print("      BENCH_PRESET=prefill4k  # INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96")
    print("      BENCH_PRESET=ttft4k     # INPUT_LEN=4096 OUTPUT_LEN=1  NUM_PROMPTS=500 MAX_CONCURRENCY=96")
PYEOF
    fi

    echo ""
    echo "  Result files:"
    echo "    baseline : $base_file"
    echo "    dbo      : $dbo_file"
    echo ""
    echo "  Verify DBO trigger in DBO server log:"
    echo "    grep -n 'should_ubatch: True' dbo_server.log"
    echo "    grep -n 'AllgatherTemplate\|DeepseekAllgather\|select_dbo_templates' dbo_server.log"
}

# ── Main ────────────────────────────────────────────────────────────────────

MODE="${1:-single}"

BASE_RESULT="${OUT_DIR}/baseline_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"
DBO_RESULT="${OUT_DIR}/dbo_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"

case "$MODE" in
    --compare)
        compare_results
        ;;

    --profile)
        wait_server "$PORT"
        run_bench "$PORT" "$LABEL" "$RESULT_FILE" 1
        ;;

    --auto)
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Auto Compare Mode"
        echo "════════════════════════════════════════════════════════════"
        echo ""
        echo "Step 1: start baseline server in another terminal:"
        echo "  PORT=8000 bash deepseek-v2-server.sh 2>&1 | tee baseline_server.log"
        echo "Press Enter to continue..."
        read -r

        wait_server 8000
        run_bench 8000 "baseline" "$BASE_RESULT" 0

        echo ""
        echo "Step 2: stop baseline server, then start DBO server:"
        echo "  PORT=8001 bash deepseek-v2-dbo-server.sh 2>&1 | tee dbo_server.log"
        echo "Press Enter to continue..."
        read -r

        wait_server 8001
        run_bench 8001 "dbo" "$DBO_RESULT" 0

        compare_results
        ;;

    single|*)
        wait_server "$PORT"
        run_bench "$PORT" "$LABEL" "$RESULT_FILE" 0

        echo ""
        echo "Single benchmark done. Compare with:"
        echo "  BENCH_PRESET=${BENCH_PRESET} INPUT_LEN=${INPUT_LEN} OUTPUT_LEN=${OUTPUT_LEN} NUM_PROMPTS=${NUM_PROMPTS} MAX_CONCURRENCY=${MAX_CONCURRENCY} bash $(basename "$0") --compare"
        echo "Profile mode, server must be started with ENABLE_PROFILER=1:"
        echo "  LABEL=${LABEL} PORT=${PORT} bash $(basename "$0") --profile"
        ;;
esac