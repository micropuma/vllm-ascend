#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Warmup Experiment Matrix Runner
#
# 默认跑 4 组核心矩阵：
#   1. compile_range + DBO=0 + observe
#   2. compile_range + DBO=0 + warmup
#   3. aclgraph + DBO=0 + observe
#   4. aclgraph + DBO=0 + warmup
#
# FULL=1 时再增加 4 组 DBO=1 对照。
# 每个 case 独立 RUN_ID / cache / log。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FULL=${FULL:-0}
PAUSE_BETWEEN=${PAUSE_BETWEEN:-10}
SERVER_WAIT=${SERVER_WAIT:-20}

MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}

run_case() {
  local experiment="$1"
  local dbo="$2"
  local mode="$3"
  local tag="$4"
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)_${tag}"

  echo ""
  echo "============================================================"
  echo " CASE: ${tag}"
  echo "  EXPERIMENT = ${experiment}"
  echo "  DBO        = ${dbo}"
  echo "  MODE       = ${mode}"
  echo "  RUN_ID     = ${run_id}"
  echo "============================================================"
  echo ""

  env \
    MODEL="$MODEL" \
    HOST="$HOST" \
    PORT="$PORT" \
    EXPERIMENT="$experiment" \
    DBO="$dbo" \
    MODE="$mode" \
    RUN_ID="$run_id" \
    bash "$SCRIPT_DIR/server.sh" &

  local server_pid=$!
  trap 'kill "${server_pid}" >/dev/null 2>&1 || true' EXIT

  sleep "$SERVER_WAIT"

  env \
    MODEL="$MODEL" \
    HOST="$HOST" \
    PORT="$PORT" \
    EXPERIMENT="$experiment" \
    MODE="$mode" \
    RUN_ID="$run_id" \
    LABEL="$tag" \
    bash "$SCRIPT_DIR/test.sh" || true

  kill "$server_pid" >/dev/null 2>&1 || true
  wait "$server_pid" >/dev/null 2>&1 || true
  trap - EXIT

  sleep "$PAUSE_BETWEEN"
}

main() {
  run_case "compile_range" 0 "observe" "compile_range_dbo0_observe"
  run_case "compile_range" 0 "warmup"  "compile_range_dbo0_warmup"
  run_case "aclgraph"      0 "observe" "aclgraph_dbo0_observe"
  run_case "aclgraph"      0 "warmup"  "aclgraph_dbo0_warmup"

  if [[ "$FULL" == "1" ]]; then
    run_case "compile_range" 1 "observe" "compile_range_dbo1_observe"
    run_case "compile_range" 1 "warmup"  "compile_range_dbo1_warmup"
    run_case "aclgraph"      1 "observe" "aclgraph_dbo1_observe"
    run_case "aclgraph"      1 "warmup"  "aclgraph_dbo1_warmup"
  fi

  echo ""
  echo "All matrix cases completed."
}

main
