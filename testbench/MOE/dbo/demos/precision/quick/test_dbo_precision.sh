#!/usr/bin/env bash
set -euo pipefail

# Quick DBO correctness gate. It compares deterministic generated token
# sequences and sampled-token logprobs after proving the DBO path was used.
QUICK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${QUICK_DIR}/../../../../../.." && pwd)"

set +u
source "${REPO_ROOT}/../.venv-dbo/bin/activate"
source "${REPO_ROOT}/../env.sh"
set -u
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

exec python3 "${QUICK_DIR}/run_quick_precision.py" --flashcomm1 0 "$@"
