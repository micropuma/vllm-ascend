#!/usr/bin/env bash
# Quick smoke test: DBO baseline vs DBO comparison
set -euo pipefail

# ── Environment Setup ──
source /data/workspace/.venv-dbo/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set +u  # atb/set_env.sh uses ZSH_VERSION which may be unset
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export SOC_VERSION=ascend910b1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AI_CPU
export VLLM_LOGGING_LEVEL=DEBUG

ARCH=$(uname -m)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/${ARCH}-linux/devlib:/usr/local/lib

# ── Test Config ──
export MODEL=/data/models/DeepSeek-V2-Lite-Chat
export MODE=compare
export DBO_THRESHOLD=64
export NUM_PROMPTS=8
export PROMPT_TOKENS=64
export OUTPUT_TOKENS=8

echo "================================================"
echo "  DBO Quick Smoke Test (offline)"
echo "  MODEL:         $MODEL"
echo "  MODE:          $MODE"
echo "  DBO_THRESHOLD: $DBO_THRESHOLD"
echo "  NUM_PROMPTS:   $NUM_PROMPTS"
echo "  PROMPT_TOKENS: $PROMPT_TOKENS"
echo "  OUTPUT_TOKENS: $OUTPUT_TOKENS"
echo "================================================"

cd /data/workspace/vllm-ascend
python3 testbench/MOE/dbo/demos/deepseek-v2-offline-dbo.py 2>&1