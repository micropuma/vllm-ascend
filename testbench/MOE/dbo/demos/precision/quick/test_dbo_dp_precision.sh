#!/usr/bin/env bash
set -euo pipefail

# Deterministic TP=1, DP=2, EP=2 DBO correctness gate. FlashComm stays off;
# the DP launcher is used for both sides so DBO_ENABLED is the sole A/B change.
QUICK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${QUICK_DIR}/../../../../../.." && pwd)"

set +u
source "${REPO_ROOT}/../.venv-dbo/bin/activate"
source "${REPO_ROOT}/../env.sh"
set -u
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

exec python3 "${QUICK_DIR}/run_quick_precision.py" \
  --flashcomm1 0 --tp-size 1 --dp-size 2 --dp-local 2 \
  --server-script "${REPO_ROOT}/testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server-dp.sh" "$@"
