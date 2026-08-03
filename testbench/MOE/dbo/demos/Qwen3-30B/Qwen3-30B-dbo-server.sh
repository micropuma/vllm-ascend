#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B DBO server. Keep all shared options equal to Qwen3-30B-server.sh.
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AI_CPU}
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=${VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE:-0}
export VLLM_ASCEND_FLASHCOMM2_OSHARED=${VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED:-0}
export VLLM_ASCEND_ENABLE_DBO=1
# export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-DEBUG}
# Cold ACL-graph capture for this model can exceed vLLM's 600-second default.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}

MODEL=${MODEL:-/data/models/Qwen3-30B/Qwen3-30B}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
TP=${TP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-1024}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-1000000000}
ENABLE_PROFILER=${ENABLE_PROFILER:-0}
PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos/profile}
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/qwen3_dbo_profile}
PROFILER_MAX_ITERATIONS=${PROFILER_MAX_ITERATIONS:-20}
PROFILER_WITH_STACK=${PROFILER_WITH_STACK:-false}
PROFILER_RECORD_SHAPES=${PROFILER_RECORD_SHAPES:-true}

echo "Starting Qwen3-30B DBO: model=$MODEL port=$PORT tp=$TP prefill_threshold=$DBO_PREFILL_TOKEN_THRESHOLD"
serve_args=(
  serve "$MODEL" --host "$HOST" --port "$PORT" --dtype bfloat16 --generation-config vllm
  --distributed-executor-backend mp --tensor-parallel-size "$TP" --enable-expert-parallel
  --enable-dbo --all2all-backend deepep_low_latency
  --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
  --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS" --disable-log-stats
)
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  serve_args+=(--enforce-eager)
fi
if [[ "$ENABLE_PROFILER" == "1" ]]; then
  export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-1800000}
  mkdir -p "$TORCH_PROFILER_DIR"
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":${PROFILER_WITH_STACK},\"torch_profiler_record_shapes\":${PROFILER_RECORD_SHAPES},\"torch_profiler_use_gzip\":true,\"torch_profiler_with_memory\":true,\"torch_profiler_with_flops\":false,\"ignore_frontend\":true,\"warmup_iterations\":1,\"active_iterations\":${PROFILER_MAX_ITERATIONS},\"max_iterations\":0}"
  echo "Profiler enabled: dir=$TORCH_PROFILER_DIR max_iterations=$PROFILER_MAX_ITERATIONS"
  serve_args+=(--profiler-config "$profiler_config")
fi
exec vllm "${serve_args[@]}"
