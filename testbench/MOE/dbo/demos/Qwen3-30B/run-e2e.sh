#!/usr/bin/env bash
set -euo pipefail

# Runs matched baseline and cold-DBO prefill4k measurements and retains all artifacts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
VLLM_ROOT=${VLLM_ROOT:-/data/workspace/vllm-dbo-v0221/vllm}
VENV=${VENV:-/data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate}
ENV_SCRIPT=${ENV_SCRIPT:-/data/workspace/vllm-dbo-v0221/env.sh}
RUN_ID=${RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_ROOT=${RUN_ROOT:-"/data/tmp/qwen3-30b-dbo-${RUN_ID}"}
PORT=${PORT:-8001}
BENCH_PRESET=${BENCH_PRESET:-prefill4k}
RESULT_SUFFIX=${RESULT_SUFFIX:-"_${RUN_ID}"}
MODEL=${MODEL:-/data/models/Qwen3-30B/Qwen3-30B}

source "$VENV"
set +u
source "$ENV_SCRIPT"
set -u
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export MODEL BENCH_PRESET RESULT_SUFFIX
export OUT_DIR=${OUT_DIR:-"${SCRIPT_DIR}/results"}
mkdir -p "$RUN_ROOT" "$OUT_DIR"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_server() {
  for _ in $(seq 1 180); do
    curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && return 0
    sleep 5
  done
  return 1
}

run_server() {
  local mode=$1 log_file=$2
  "$SCRIPT_DIR/$mode" >"$log_file" 2>&1 &
  server_pid=$!
  wait_server || { tail -100 "$log_file" >&2; return 1; }
}

echo "vLLM commit: $(git -C "$VLLM_ROOT" rev-parse HEAD)"
echo "vLLM-Ascend commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
python -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'

baseline_log="$RUN_ROOT/baseline.log"
run_server Qwen3-30B-server.sh "$baseline_log"
LABEL=baseline PORT="$PORT" bash "$SCRIPT_DIR/Qwen3-30B-dbo-test.sh"
cleanup
server_pid=""

mkdir -p "$RUN_ROOT/xdg" "$RUN_ROOT/torchinductor" "$RUN_ROOT/vllm-compile"
dbo_log="$RUN_ROOT/dbo.log"
XDG_CACHE_HOME="$RUN_ROOT/xdg" TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/torchinductor" \
VLLM_COMPILE_CACHE_PATH="$RUN_ROOT/vllm-compile" run_server Qwen3-30B-dbo-server.sh "$dbo_log"
LABEL=dbo PORT="$PORT" bash "$SCRIPT_DIR/Qwen3-30B-dbo-test.sh"

dbo_hits=$(rg -c 'should_ubatch: True' "$dbo_log" || true)
if [[ "$dbo_hits" -lt 2 ]]; then
  echo "DBO trigger evidence insufficient: expected at least two worker log entries, found $dbo_hits" >&2
  exit 1
fi
if ! find "$RUN_ROOT" -type f -print -quit | grep -q .; then
  echo "Cold DBO cache root has no artifacts: $RUN_ROOT" >&2
  exit 1
fi

BENCH_PRESET="$BENCH_PRESET" RESULT_SUFFIX="$RESULT_SUFFIX" \
  bash "$SCRIPT_DIR/Qwen3-30B-dbo-test.sh" --compare
echo "Artifacts: results=$OUT_DIR logs=$RUN_ROOT"
