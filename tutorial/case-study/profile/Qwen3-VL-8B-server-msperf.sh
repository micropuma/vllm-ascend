#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# 环境变量设置
# -------------------------------------------------------------------
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-true}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

# -------------------------------------------------------------------
# 模型与服务配置
# -------------------------------------------------------------------
MODEL=${MODEL:-/root/models/Qwen3-VL-8B-Instruct}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

# -------------------------------------------------------------------
# MS Service Profiler 配置（直接指定可写目录）
# -------------------------------------------------------------------
PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-ascend-dly/tutorial/case-study/profile}
MSPERF_ROOT=${MSPERF_ROOT:-${PROFILE_ROOT}/msperf_profile}

# 强制使用 /tmp 下的目录（容器内任何用户都可写）
MSPERF_OUTPUT_DIR="/data/workspace/vllm-ascend-dly/tutorial/case-study/profile/msperf_output"
SERVICE_PROF_CONFIG_PATH="${MSPERF_ROOT}/msserviceprofiler_config.json"
PROFILING_SYMBOLS_PATH="${MSPERF_ROOT}/service_profiling_symbols.yaml"

# 采集参数
MSPERF_ENABLE=${MSPERF_ENABLE:-1}
MSPERF_LEVEL=${MSPERF_LEVEL:-INFO}
MSPERF_ACL_TASK_TIME=${MSPERF_ACL_TASK_TIME:-0}
MSPERF_ACL_TASK_TIME_LEVEL=${MSPERF_ACL_TASK_TIME_LEVEL:-}
MSPERF_TIMELIMIT=${MSPERF_TIMELIMIT:-0}
MSPERF_DOMAIN=${MSPERF_DOMAIN:-}

# 确保所有需要的目录存在
mkdir -p "$MSPERF_ROOT" "$MSPERF_OUTPUT_DIR"

# 生成配置文件，强制使用 /tmp 下的输出路径
cat > "$SERVICE_PROF_CONFIG_PATH" <<EOF
{
  "enable": ${MSPERF_ENABLE},
  "prof_dir": "${MSPERF_OUTPUT_DIR}",
  "profiler_level": "${MSPERF_LEVEL}",
  "acl_task_time": ${MSPERF_ACL_TASK_TIME},
  "acl_prof_task_time_level": "${MSPERF_ACL_TASK_TIME_LEVEL}",
  "timelimit": ${MSPERF_TIMELIMIT},
  "domain": "${MSPERF_DOMAIN}"
}
EOF

export SERVICE_PROF_CONFIG_PATH
export PROFILING_SYMBOLS_PATH

echo "MODEL=$MODEL"
echo "SERVER=http://${HOST}:${PORT}"
echo "SERVICE_PROF_CONFIG_PATH=$SERVICE_PROF_CONFIG_PATH"
echo "PROFILING_SYMBOLS_PATH=$PROFILING_SYMBOLS_PATH"
echo "MSPERF_OUTPUT_DIR=$MSPERF_OUTPUT_DIR"

# -------------------------------------------------------------------
# 启动 vLLM 服务
# -------------------------------------------------------------------
vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384