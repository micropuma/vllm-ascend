#!/usr/bin/env bash

PYTHON_BIN="${PYTHON_BIN:-python}"

INDEX_URL="https://mirrors.ustc.edu.cn/pypi/simple"
TRUSTED_HOST="mirrors.ustc.edu.cn"
CONSTRAINT_FILE="/data/workspace/vllm-ascend-dbo/ascend_protect_constraints.txt"

LOG_DIR="${LOG_DIR:-/data/workspace/pip-dependency-logs}"
mkdir -p "$LOG_DIR"

# 禁止环境中的额外索引干扰解析。
export PIP_EXTRA_INDEX_URL=""
export PYTHONPATH=""

PACKAGES=(
"protobuf>=5,<7"
"pydantic>=2,<3"
"msgspec"
"mistral-common==1.11.0"
"aiohttp>=3.13.3"
"anthropic>=0.71.0"
"blake3"
"cachetools"
"cbor2"
"compressed-tensors==0.15.0.1"
"depyf==0.20.0"
"diskcache==5.6.3"
"einops"
"fastapi[standard]>=0.115.0,<0.124.0"
"gguf>=0.17.0"
"ijson"
"lark==1.2.2"
"llguidance>=1.3.0,<1.4.0"
"lm-format-enforcer==0.11.3"
"mcp"
"model-hosting-container-standards>=0.1.13,<1.0.0"
"openai>=2.0.0"
"openai-harmony>=0.0.3"
"opentelemetry-api>=1.27.0"
"opentelemetry-exporter-otlp>=1.27.0"
"opentelemetry-sdk>=1.27.0"
"opentelemetry-semantic-conventions-ai>=0.4.1"
"outlines_core==0.2.14"
"partial-json-parser"
"prometheus_client>=0.18.0"
"prometheus-fastapi-instrumentator>=7.0.0"
"py-cpuinfo"
"pybase64"
"python-json-logger"
"pyzmq>=25.0.0"
"sentencepiece"
"setproctitle"
"tokenizers>=0.21.1"
"transformers==5.5.3"
"watchfiles"
"xgrammar>=0.1.32,<1.0.0"
"msgpack"
"numba"
"pandas-stubs"
"quart"
"wheel"
)

declare -a SUCCEEDED=()
declare -a FAILED=()

if [[ ! -f "$CONSTRAINT_FILE" ]]; then
echo "[FATAL] Constraint file not found:"
echo "        $CONSTRAINT_FILE"
exit 1
fi

echo "============================================================"
echo " Python        : $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo " Pip           : $($PYTHON_BIN -m pip --version)"
echo " Index         : $INDEX_URL"
echo " Constraint    : $CONSTRAINT_FILE"
echo " Log directory : $LOG_DIR"
echo " Package count : ${#PACKAGES[@]}"
echo "============================================================"

for i in "${!PACKAGES[@]}"; do
pkg="${PACKAGES[$i]}"

# 将包规格转换成适合作为文件名的字符串。
safe_name="$(printf '%s' "$pkg" | sed 's/[^A-Za-z0-9._-]/_/g')"
log_file="$LOG_DIR/$((i + 1))_${safe_name}.log"

echo
echo "------------------------------------------------------------"
echo "[$((i + 1))/${#PACKAGES[@]}] Installing: $pkg"
echo "Log: $log_file"
echo "------------------------------------------------------------"

if "$PYTHON_BIN" -m pip install \
   --no-cache-dir \
   --index-url "$INDEX_URL" \
   --trusted-host "$TRUSTED_HOST" \
   --timeout 600 \
   --retries 20 \
   --constraint "$CONSTRAINT_FILE" \
   "$pkg" 2>&1 | tee "$log_file"; then

 SUCCEEDED+=("$pkg")
 echo "[OK] $pkg"
else
 exit_code=${PIPESTATUS[0]}
 FAILED+=("$pkg")
 echo "[FAILED] $pkg, pip exit code: $exit_code"
 echo "[CONTINUE] Proceeding to the next package."
fi
done

echo
echo "============================================================"
echo " Installation summary"
echo "============================================================"

echo
echo "Succeeded: ${#SUCCEEDED[@]}"
for pkg in "${SUCCEEDED[@]}"; do
echo "  [OK] $pkg"
done

echo
echo "Failed: ${#FAILED[@]}"
for pkg in "${FAILED[@]}"; do
echo "  [FAILED] $pkg"
done

echo
echo "Running pip check..."
"$PYTHON_BIN" -m pip check 2>&1 | tee "$LOG_DIR/pip-check.log" || true

echo
echo "Logs are stored in:"
echo "  $LOG_DIR"

if [[ ${#FAILED[@]} -gt 0 ]]; then
printf '%s\n' "${FAILED[@]}" > "$LOG_DIR/failed-packages.txt"
echo
echo "Failed package list:"
echo "  $LOG_DIR/failed-packages.txt"
exit 2
fi

echo
echo "[OK] All requested dependencies were installed successfully."
