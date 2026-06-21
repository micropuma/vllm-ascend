#!/usr/bin/env bash

# -------------------------------------------------------------------
# 环境变量设置（可通过外部覆盖）
# -------------------------------------------------------------------

# 是否使用 ModelScope 下载模型（默认 true）
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-true}
# vLLM 多进程启动方式（默认 spawn）
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
# 指定可见的 Ascend 设备（默认卡 0）
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}

# -------------------------------------------------------------------
# 模型与服务配置
# -------------------------------------------------------------------

# 模型路径（默认指向 Qwen3-VL-8B-Instruct）
MODEL=${MODEL:-/root/models/Qwen3-VL-8B-Instruct}
# 服务监听地址和端口
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

# -------------------------------------------------------------------
# 性能采集配置（Ascend PyTorch Profiler）
# -------------------------------------------------------------------

# 性能数据根目录
PROFILE_ROOT=${PROFILE_ROOT:-/data/workspace/vllm-ascend-dly/tutorial/case-study/profile}
# 性能数据输出子目录（存放 *_ascend_pt）
TORCH_PROFILER_DIR=${TORCH_PROFILER_DIR:-${PROFILE_ROOT}/vllm_profile}
# 是否包含 Python 调用栈信息（默认关闭以减少数据量）
TORCH_PROFILER_WITH_STACK=${TORCH_PROFILER_WITH_STACK:-false}

# 确保输出目录存在
mkdir -p "$TORCH_PROFILER_DIR"

# 打印关键配置，方便调试
echo "MODEL=$MODEL"
echo "SERVER=http://${HOST}:${PORT}"
echo "TORCH_PROFILER_DIR=$TORCH_PROFILER_DIR"
echo "TORCH_PROFILER_WITH_STACK=$TORCH_PROFILER_WITH_STACK"

# -------------------------------------------------------------------
# 启动 vLLM 在线推理服务
# -------------------------------------------------------------------

serve_args=(
  serve
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --dtype bfloat16

  # 限制每个请求最多携带 1 张图片，避免多模态导致显存爆炸
  --limit-mm-per-prompt '{"image": 1}'

  # 最大模型上下文长度 16384 tokens
  --max-model-len 16384

  # 单批次最多处理 16384 tokens
  --max-num-batched-tokens 16384

  # 开启 torch profiler 性能采集，指定输出目录和是否包含调用栈
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":${TORCH_PROFILER_WITH_STACK}}"
)

vllm "${serve_args[@]}"
