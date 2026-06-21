#!/usr/bin/env bash

set -euo pipefail

# Examples:
# 1. Use the default HF model id:
#    bash tutorial/kv-memory-bytes/run_three_cases.sh
#
# 2. Use a local model placed under this tutorial directory:
#    bash tutorial/kv-memory-bytes/run_three_cases.sh \
#      --model "$(pwd)/tutorial/kv-memory-bytes/models/Qwen3-8B"
#
# 3. Use any local absolute path:
#    bash tutorial/kv-memory-bytes/run_three_cases.sh \
#      --model /data/models/Qwen3-8B \
#      --python-bin /data/workspace/.venv/bin/python
#
# You can also pass --dtype or --runner if needed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=python3
MODEL=Qwen/Qwen3-8B
DTYPE=bfloat16
RUNNER="${SCRIPT_DIR}/run_simple.py"

usage() {
    cat <<EOF
Usage: bash tutorial/kv-memory-bytes/run_three_cases.sh [options]

Options:
  --model PATH_OR_ID      Model path or HF model id. Default: ${MODEL}
  --dtype DTYPE           Model dtype. Default: ${DTYPE}
  --python-bin PATH       Python executable. Default: ${PYTHON_BIN}
  --runner PATH           Runner script. Default: ${RUNNER}
  -h, --help              Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL=$2
            shift 2
            ;;
        --dtype)
            DTYPE=$2
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN=$2
            shift 2
            ;;
        --runner)
            RUNNER=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

run_simple() {
    "${PYTHON_BIN}" "${RUNNER}" \
        --model "${MODEL}" \
        --dtype "${DTYPE}" \
        "$@"
}

case1() {
    echo "[case1] same kv bytes, different gpu_memory_utilization"
    run_simple --gpu-memory-utilization 0.35 --kv-cache-memory-bytes 8G --max-model-len 8192
    run_simple --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 8G --max-model-len 8192
}

case2() {
    echo "[case2] larger kv bytes"
    run_simple --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 6G --max-model-len 8192
    run_simple --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 10G --max-model-len 8192
}

case3() {
    echo "[case3] too-small kv bytes should fail"
    if run_simple --gpu-memory-utilization 0.7 --kv-cache-memory-bytes 512M --max-model-len 32768; then
        echo "[case3] expected failure, but command succeeded" >&2
        return 1
    fi
    echo "[case3] observed expected failure"
}

case1
case2
case3
