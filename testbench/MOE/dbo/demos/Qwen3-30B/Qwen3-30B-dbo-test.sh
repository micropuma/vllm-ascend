#!/usr/bin/env bash
set -euo pipefail

# Reuse the maintained benchmark implementation with Qwen-specific defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL=${MODEL:-/data/models/Qwen3-30B/Qwen3-30B}
export OUT_DIR=${OUT_DIR:-"${SCRIPT_DIR}/results"}
exec bash "${SCRIPT_DIR}/../DeepseekV2/deepseek-v2-dbo-test.sh" "$@"
