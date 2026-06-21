#!/usr/bin/env bash
set -e

MODEL_ID="Qwen/Qwen3-30B-A3B"
CACHE_DIR="/mnt/.cache/modelscope"

export VLLM_USE_MODELSCOPE=True
export MODELSCOPE_CACHE="${CACHE_DIR}"

mkdir -p "${CACHE_DIR}"

echo "[1/3] install modelscope"
python -m pip install -U modelscope

echo "[2/3] download ${MODEL_ID}"
MODEL_PATH=$(python - <<PY
from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen3-30B-A3B",
    cache_dir="/mnt/.cache/modelscope",
)
print(model_dir)
PY
)

echo "[3/3] write env"
cat > dbo_model.env <<EOF
export DBO_MODEL_ID="${MODEL_ID}"
export DBO_MODEL_PATH="${MODEL_PATH}"
export VLLM_USE_MODELSCOPE=True
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF

echo
echo "[OK] Download finished"
echo "MODEL_PATH=${MODEL_PATH}"
echo
echo "Use:"
echo "  source ./dbo_model.env"
echo "  echo \$DBO_MODEL_PATH"