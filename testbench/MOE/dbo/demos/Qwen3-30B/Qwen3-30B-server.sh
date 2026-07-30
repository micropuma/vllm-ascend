#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B baseline server. Override MODEL, PORT, TP, or batch limits as needed.
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AI_CPU}
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=${VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE:-0}
export VLLM_ASCEND_FLASHCOMM2_OSHARED=${VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED:-0}
export VLLM_ASCEND_ENABLE_DBO=0
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

MODEL=${MODEL:-/data/models/Qwen3-30B/Qwen3-30B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

echo "Starting Qwen3-30B baseline: model=$MODEL port=$PORT tp=$TP"
exec vllm serve "$MODEL" \
  --host "$HOST" --port "$PORT" --dtype bfloat16 --generation-config vllm \
  --distributed-executor-backend mp --tensor-parallel-size "$TP" --enable-expert-parallel \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" --disable-log-stats
