#!/usr/bin/env bash
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    EXPERIMENT CONFIGURATION                                 ║
# ║  Edit the two case blocks below to change what is being compared.            ║
# ║  DP mode: server script is always deepseek-v2-dbo-server-dp.sh             ║
# ║           (pass DBO_ENABLED=0 or =1 via ENV)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Case 1 ─────────────────────────────────────────────────────────────────────
CASE1_LABEL="${CASE1_LABEL:-baseline}"
declare -a CASE1_ENV=(
  DBO_ENABLED="${CASE1_DBO:-0}"
  VLLM_ASCEND_ENABLE_FLASHCOMM1="${CASE1_FC1:-0}"
  VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE="${CASE1_FC2_SIZE:-0}"
  VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED="${CASE1_FC2_OSHARED:-0}"
)

# ── Case 2 ─────────────────────────────────────────────────────────────────────
CASE2_LABEL="${CASE2_LABEL:-dbo}"
CASE2_COLD_CACHE="${CASE2_COLD_CACHE:-1}"   # 0 = reuse warm cache, 1 = fresh cold cache
declare -a CASE2_ENV=(
  DBO_ENABLED="${CASE2_DBO:-1}"
  VLLM_ASCEND_ENABLE_FLASHCOMM1="${CASE2_FC1:-0}"
  VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE="${CASE2_FC2_SIZE:-0}"
  VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED="${CASE2_FC2_OSHARED:-0}"
)
# ═══════════════════════════════════════════════════════════════════════════════


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
VLLM_ROOT=${VLLM_ROOT:-/data/workspace/vllm-dbo-v0221/vllm}
VENV=${VENV:-/data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate}
ENV_SCRIPT=${ENV_SCRIPT:-/data/workspace/vllm-dbo-v0221/env.sh}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_ROOT=${RUN_ROOT:-"/data/tmp/deepseek-v2-dp-dbo-${RUN_ID}"}
PORT=${PORT:-8001}

# ── NPU memory management (auto_benchmark.sh pattern) ──────────────────────────
DEVICES="${DEVICES:-0,1}"
DEVICE_MEMORY_WAIT_TIMEOUT=${DEVICE_MEMORY_WAIT_TIMEOUT:-120}
DEVICE_MEMORY_THRESHOLD_PCT=${DEVICE_MEMORY_THRESHOLD_PCT:-10}
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
BENCH_PRESET=${BENCH_PRESET:-prefill4k}
RESULT_SUFFIX=${RESULT_SUFFIX:-"_${RUN_ID}"}
OUT_DIR=${OUT_DIR:-"${SCRIPT_DIR}/results/dp"}

source "$VENV"
set +u
source "$ENV_SCRIPT"
set -u
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export MODEL BENCH_PRESET RESULT_SUFFIX OUT_DIR
mkdir -p "$RUN_ROOT" "$OUT_DIR"

# ── Resolve benchmark parameters (mirrors deepseek-v2-dbo-test.sh presets) ────
case "$BENCH_PRESET" in
    prefill4k)  INPUT_LEN=4096;  OUTPUT_LEN=16;  NUM_PROMPTS=500;  MAX_CONCURRENCY=96 ;;
    ttft4k)     INPUT_LEN=4096;  OUTPUT_LEN=1;   NUM_PROMPTS=500;  MAX_CONCURRENCY=96 ;;
    prefill8k)  INPUT_LEN=8192;  OUTPUT_LEN=16;  NUM_PROMPTS=500;  MAX_CONCURRENCY=96 ;;
    quick)      INPUT_LEN=1024;  OUTPUT_LEN=128; NUM_PROMPTS=200;  MAX_CONCURRENCY=64 ;;
    custom)     INPUT_LEN=${INPUT_LEN:-1024}; OUTPUT_LEN=${OUTPUT_LEN:-128}; NUM_PROMPTS=${NUM_PROMPTS:-200}; MAX_CONCURRENCY=${MAX_CONCURRENCY:-64} ;;
    *)          echo "Unknown BENCH_PRESET=$BENCH_PRESET" >&2; exit 2 ;;
esac

RESULT1="${OUT_DIR}/${CASE1_LABEL}_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"
RESULT2="${OUT_DIR}/${CASE2_LABEL}_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}${RESULT_SUFFIX}.json"

# ── Cleanup: kill server, wait for port + NPU memory release ──────────────────

# ── NPU memory helpers (from auto_benchmark.sh pattern) ────────────────────────

# Map logical device IDs to physical NPU IDs via npu-smi
get_physical_npu_ids() {
    python3 - "$DEVICES" <<'PYEOF'
import subprocess, re, sys
logical_ids = [int(x.strip()) for x in sys.argv[1].split(',') if x.strip().isdigit()]
try:
    result = subprocess.run(['npu-smi', 'info', '-m'], capture_output=True, text=True, timeout=10)
    for line in result.stdout.split('\n'):
        m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+', line)
        if m:
            npu_id, logic_id = int(m.group(1)), int(m.group(3))
            if logic_id in logical_ids:
                print(f"{logic_id}:{npu_id}")
except Exception:
    for lid in logical_ids:
        print(f"{lid}:{lid}")
PYEOF
}

# Query HBM usage percentage for a physical NPU
get_npu_hbm_usage_pct() {
    local npu_id="$1"
    local pct
    pct=$(npu-smi info -t usages -i "$npu_id" 2>/dev/null | grep -oP 'HBM Usage Rate\(%\)\s+:\s+\K\d+' | head -1)
    echo "${pct:-100}"
}

# Wait until NPU HBM usage drops below threshold on all devices
wait_device_memory_free() {
    local max_wait=${1:-${DEVICE_MEMORY_WAIT_TIMEOUT}}
    local threshold_pct=${2:-${DEVICE_MEMORY_THRESHOLD_PCT}}

    echo "  Waiting for NPU memory release (threshold < ${threshold_pct}%, timeout ${max_wait}s) ..."

    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        local all_free=true
        local status_line=""
        while IFS= read -r mapping; do
            local logic_id="${mapping%%:*}"
            local npu_id="${mapping##*:}"
            local pct
            pct=$(get_npu_hbm_usage_pct "$npu_id")
            status_line="${status_line} dev${logic_id}=${pct}%"
            if [[ "$pct" -ge "$threshold_pct" ]]; then
                all_free=false
            fi
        done < <(get_physical_npu_ids)

        if $all_free; then
            echo "  ✓ NPU memory released (${status_line}, waited ${waited}s)"
            return 0
        fi

        if [[ $((waited % 15)) -eq 0 ]]; then
            echo "  ... NPU usage:${status_line} (waited ${waited}s)"
        fi
        sleep 3
        waited=$((waited + 3))
    done

    # Timeout: print final status
    local final_status=""
    while IFS= read -r mapping; do
        local logic_id="${mapping%%:*}"
        local npu_id="${mapping##*:}"
        local pct
        pct=$(get_npu_hbm_usage_pct "$npu_id")
        final_status="${final_status} dev${logic_id}=${pct}%"
    done < <(get_physical_npu_ids)
    echo "  ⚠ NPU memory not fully released (${final_status}, timeout ${max_wait}s) — proceeding anyway"
    return 1
}

# Deep clean: kill any residual process occupying the target port
cleanup_device_processes() {
    echo "  Deep cleaning residual processes on port $PORT ..."
    local existing
    existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        echo "    Killing port $PORT occupant (PID: $existing)"
        kill -TERM $existing 2>/dev/null || true
        sleep 3
        existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
        [[ -z "$existing" ]] || kill -KILL $existing 2>/dev/null || true
    fi
    local stale_workers
    local stale_servers
    stale_servers=$(ps -eo pid=,args= | awk -v port="$PORT" '$0 ~ /[v]llm serve/ && $0 ~ ("--port " port) {print $1}' || true)
    [[ -z "$stale_servers" ]] || kill -TERM $stale_servers 2>/dev/null || true
    stale_workers=$(ps -eo pid=,args= | awk '/VLLM::(Worker|EngineCore)/ {print $1}' || true)
    [[ -z "$stale_workers" ]] || kill -TERM $stale_workers 2>/dev/null || true
    sleep 2
    stale_workers=$(ps -eo pid=,args= | awk '/VLLM::(Worker|EngineCore)/ {print $1}' || true)
    [[ -z "$stale_workers" ]] || kill -KILL $stale_workers 2>/dev/null || true
    echo "  ✓ Deep cleanup done"
}
server_pid=""
cleanup_running=0
cleanup_needed=1
cleanup() {
  local rc=$?
  [[ "$cleanup_running" -eq 0 && "$cleanup_needed" -eq 1 ]] || return 0
  cleanup_running=1
  local server_pgid=""
  if [[ -n "$server_pid" ]]; then
    server_pgid=$(ps -o pgid= -p "$server_pid" 2>/dev/null | tr -d ' ' || true)
  fi
  echo ""
  echo "=== Stopping server and reclaiming NPU resources ==="

  if [[ -n "$server_pgid" ]] && kill -0 -"$server_pgid" 2>/dev/null; then
    # 1) SIGTERM to entire process group (setsid makes PGID == server_pid)
    kill -TERM -"$server_pgid" 2>/dev/null || true

    # 2) Wait up to 30s for process group to exit
    local waited=0
    while kill -0 -"$server_pgid" 2>/dev/null && [[ $waited -lt 30 ]]; do
      sleep 2
      waited=$((waited + 2))
    done

    # 3) SIGKILL if any process in the group survives
    if kill -0 -"$server_pgid" 2>/dev/null; then
      echo "  Process group still alive, sending SIGKILL"
      kill -KILL -"$server_pgid" 2>/dev/null || true
      sleep 3
    fi

    # 4) Port-level fallback: kill anything still listening on PORT
  fi
  cleanup_device_processes
  wait_device_memory_free || true
  server_pid=""
  cleanup_needed=0
  cleanup_running=0
  echo "=== Cleanup done ==="
  echo ""
  return "$rc"
}

trap cleanup EXIT INT TERM HUP

# ── Wait for server, fail fast if the process dies ────────────────────────────
wait_server() {
  local timeout=${1:-1800}
  local elapsed=0
  echo "  Waiting for server on http://127.0.0.1:${PORT} (timeout ${timeout}s)..."
  while (( elapsed < timeout )); do
    if [[ -n "$server_pid" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
      echo "  ✗ Server process (PID $server_pid) died unexpectedly!" >&2
      return 1
    fi
    if curl --noproxy '*' -sf --connect-timeout 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "  ✓ Server ready on http://127.0.0.1:${PORT}"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "  ✗ Server wait timeout after ${timeout}s." >&2
  return 1
}

# ── Start server with console + file output (DP: server script is hardcoded) ──
run_server() {
  local log_file=$1
  shift

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Starting: deepseek-v2-dbo-server-dp.sh"
  echo "  Log:      $log_file"
  echo "════════════════════════════════════════════════════════════"

  cleanup_needed=1
  cleanup_device_processes
  wait_device_memory_free || return 1
  env "$@" setsid bash -c "bash "$SCRIPT_DIR/deepseek-v2-dbo-server-dp.sh" 2>&1 | tee "$log_file"" &
  server_pid=$!
  wait_server || {
    echo "════════════════════════════════════════════════════════════"
    echo "  ✗ SERVER STARTUP FAILED!"
    echo "  Last 80 lines of $log_file:"
    echo "────────────────────────────────────────────────────────────"
    tail -80 "$log_file" >&2
    echo "────────────────────────────────────────────────────────────"
    return 1
  }
}

# ── Summary table ─────────────────────────────────────────────────────────────
print_summary() {
  local file_a="$1" file_b="$2" label_a="$3" label_b="$4"

  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════╗"
  echo "║                        E2E Results Summary                          ║"
  echo "╠══════════════════════════════════════════════════════════════════════╣"
  echo "║  BENCH_PRESET: ${BENCH_PRESET}  (in=${INPUT_LEN} out=${OUTPUT_LEN} np=${NUM_PROMPTS} c=${MAX_CONCURRENCY})"
  echo "║  RUN_ID:       ${RUN_ID}"
  echo "╚══════════════════════════════════════════════════════════════════════╝"

  python3 - "$file_a" "$file_b" "$label_a" "$label_b" <<'PYEOF'
import json, sys

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  ✗ Cannot read {path}: {e}")
        sys.exit(1)

base = load(sys.argv[1])
dbo  = load(sys.argv[2])
name_a = sys.argv[3]
name_b = sys.argv[4]

for label, data in [(name_a, base), (name_b, dbo)]:
    c = data.get("completed", 0)
    f = data.get("failed", 0)
    if c == 0 and f > 0:
        print(f"  ✗ {label}: all {f} requests FAILED — benchmark did not run correctly!")
    else:
        print(f"  ✓ {label}: {c} completed, {f} failed")

print()
print(f"  {'Metric':<30} {name_a:>10} {name_b:>10} {'%s/%s'%(name_b,name_a):>14}" % (name_b, name_a))
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*14}")

rows = [
    ("Output Throughput (tok/s)",    "output_throughput",   False),
    ("Total Token Throughput (t/s)", "total_token_throughput", False),
    ("Request Throughput (req/s)",   "request_throughput",  False),
    ("Mean TTFT (ms)",               "mean_ttft_ms",        True),
    ("P99 TTFT (ms)",                "p99_ttft_ms",         True),
    ("Mean TPOT (ms/tok)",           "mean_tpot_ms",        True),
    ("Mean ITL (ms)",                "mean_itl_ms",         True),
    ("Mean E2E Latency (ms)",        "mean_e2el_ms",        True),
    ("Duration (s)",                 "duration",            True),
]

any_change = False
for name, key, lower_better in rows:
    bv = base.get(key)
    dv = dbo.get(key)
    if bv is None or dv is None or bv == 0:
        print(f"  {name:<30} {'N/A':>10} {'N/A':>10} {'N/A':>14}")
        continue
    ratio = dv / bv
    pct = (ratio - 1) * 100
    if lower_better:
        verdict = "✓ better" if pct < -1 else ("✗ worse" if pct > 1 else "≈ same")
    else:
        verdict = "✓ better" if pct > 1 else ("✗ worse" if pct < -1 else "≈ same")
    if abs(pct) > 1:
        any_change = True
    print(f"  {name:<30} {bv:>10.2f} {dv:>10.2f} {ratio:>7.3f}x ({pct:>+5.1f}%) {verdict}")

print()
if any_change:
    pos = 0; neg = 0
    for name, key, lb in rows:
        bv, dv = base.get(key), dbo.get(key)
        if bv and dv and bv != 0:
            pct = (dv/bv - 1) * 100
            if (lb and pct < 0) or (not lb and pct > 0): pos += 1
            elif (lb and pct > 0) or (not lb and pct < 0): neg += 1
    if pos > neg:
        print(f"  → {name_b} shows net positive benefit over {name_a}.")
    elif neg > pos:
        print(f"  → {name_b} shows net regression vs {name_a} — check server logs and trigger evidence.")
    else:
        print(f"  → {name_b} shows mixed / negligible effect. Review test conditions.")
else:
    print(f"  → No significant difference between {name_a} and {name_b}.")
    print(f"  → Verify DBO trigger evidence: grep -c 'should_ubatch: True' <log>")

print()
print(f"  {name_a} result: {sys.argv[1]}")
print(f"  {name_b} result: {sys.argv[2]}")
PYEOF
  echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════════
echo "vLLM commit:        $(git -C "$VLLM_ROOT" rev-parse HEAD)"
echo "vLLM-Ascend commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
python -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Case 1
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 1: ${CASE1_LABEL}  (DP mode, DBO_ENABLED=${CASE1_ENV[0]##*=})"
echo "╚══════════════════════════════════════════════════════════════════════╝"

rm -f "$RESULT1"

run_server "$RUN_ROOT/${CASE1_LABEL}.log" PORT="$PORT" "${CASE1_ENV[@]}"

echo ""
echo "═══ Running ${CASE1_LABEL} benchmark... ═══"
LABEL="$CASE1_LABEL" PORT="$PORT" bash "$SCRIPT_DIR/deepseek-v2-dbo-test.sh"
echo "═══ ${CASE1_LABEL} benchmark complete ═══"

cleanup
server_pid=""

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Case 2
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  PHASE 2: ${CASE2_LABEL}  (DP mode, DBO_ENABLED=${CASE2_ENV[0]##*=})"
echo "╚══════════════════════════════════════════════════════════════════════╝"

rm -f "$RESULT2"

if [[ "$CASE2_COLD_CACHE" == "1" ]]; then
  CASE2_CACHE_ROOT="$RUN_ROOT/${CASE2_LABEL}-cache"
  mkdir -p "$CASE2_CACHE_ROOT/xdg" "$CASE2_CACHE_ROOT/torchinductor" "$CASE2_CACHE_ROOT/vllm-compile"
  CASE2_CACHE_ENV=(
    XDG_CACHE_HOME="$CASE2_CACHE_ROOT/xdg"
    TORCHINDUCTOR_CACHE_DIR="$CASE2_CACHE_ROOT/torchinductor"
    VLLM_COMPILE_CACHE_PATH="$CASE2_CACHE_ROOT/vllm-compile"
  )
else
  CASE2_CACHE_ENV=()
fi

run_server "$RUN_ROOT/${CASE2_LABEL}.log" PORT="$PORT" \
  "${CASE2_ENV[@]}" "${CASE2_CACHE_ENV[@]}"

echo ""
echo "═══ Running ${CASE2_LABEL} benchmark... ═══"
LABEL="$CASE2_LABEL" PORT="$PORT" bash "$SCRIPT_DIR/deepseek-v2-dbo-test.sh"
echo "═══ ${CASE2_LABEL} benchmark complete ═══"

# ── Post-run validation ───────────────────────────────────────────────────────
echo ""
echo "═══ Post-run validation ═══"

dbo_log="$RUN_ROOT/${CASE2_LABEL}.log"
dbo_hits=$(rg -c 'should_ubatch: True' "$dbo_log" || true)
if [[ "$dbo_hits" -lt 2 ]]; then
  echo "✗ DBO trigger evidence INSUFFICIENT: expected ≥2 'should_ubatch: True' entries, found ${dbo_hits:-0}" >&2
  echo "  The DBO server may not have triggered ubatching. Check: $dbo_log" >&2
else
  echo "✓ DBO trigger evidence: $dbo_hits 'should_ubatch: True' entries"
fi

if [[ "$CASE2_COLD_CACHE" == "1" ]]; then
  if ! find "$CASE2_CACHE_ROOT" -type f -print -quit | grep -q .; then
    echo "✗ Cold DBO cache root has no artifacts: $CASE2_CACHE_ROOT" >&2
  else
    echo "✓ Cold DBO cache artifacts present"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
if [[ -f "$RESULT1" && -f "$RESULT2" ]]; then
  print_summary "$RESULT1" "$RESULT2" "$CASE1_LABEL" "$CASE2_LABEL"
else
  echo ""
  echo "═══ Result file status ═══"
  [[ -f "$RESULT1" ]] && echo "  ✓ $RESULT1" || echo "  ✗ MISSING: $RESULT1"
  [[ -f "$RESULT2" ]] && echo "  ✓ $RESULT2" || echo "  ✗ MISSING: $RESULT2"
fi

echo "Artifacts: results=$OUT_DIR logs=$RUN_ROOT"
