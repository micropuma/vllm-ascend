#!/usr/bin/env bash
# 一键对标：顺序跑 baseline 和 DBO server，输出 speedup 对比
#
# 用法：
#   bash bench.sh                           # 交互引导模式（推荐首次使用）
#   SKIP_PROMPT=1 bash bench.sh             # 跳过提示，假设两个 server 已就绪
#
# 可覆盖的参数：
#   MODEL=/data/models/DeepSeek-V2-Lite-Chat
#   INPUT_LEN=1024   OUTPUT_LEN=128
#   NUM_PROMPTS=200  MAX_CONCURRENCY=64
#   BASE_PORT=8000   DBO_PORT=8001

set -euo pipefail

DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 参数 ─────────────────────────────────────────────────────────────────────
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
BASE_PORT=${BASE_PORT:-8000}
DBO_PORT=${DBO_PORT:-8001}

INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_PROMPTS=${NUM_PROMPTS:-200}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REQUEST_RATE=${REQUEST_RATE:-inf}

OUT_DIR=${OUT_DIR:-"${DEMOS_DIR}/../results"}
mkdir -p "$OUT_DIR"

BASE_RESULT="${OUT_DIR}/baseline_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}.json"
DBO_RESULT="${OUT_DIR}/dbo_in${INPUT_LEN}_out${OUTPUT_LEN}_np${NUM_PROMPTS}_c${MAX_CONCURRENCY}.json"

SKIP_PROMPT=${SKIP_PROMPT:-0}

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

wait_server() {
    local port="$1"
    local url="http://${HOST}:${port}/v1/models"
    printf "  等待 http://${HOST}:${port} 就绪 "
    for _ in $(seq 1 120); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo " ✓"
            return 0
        fi
        printf "."
        sleep 5
    done
    echo ""
    echo "  ✗ 超时（600s），请确认 server 是否已正常启动" >&2
    return 1
}

run_one() {
    local port="$1"
    local label="$2"
    local out_file="$3"

    echo ""
    echo "  ── 发压 [${label}] port=${port} ──────────────────────────────"

    # warmup
    echo "  [1/2] warmup..."
    vllm bench serve \
        --host "$HOST" --port "$port" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --dataset-name random \
        --num-prompts 16 --max-concurrency 16 --request-rate inf \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 0.0 --temperature 0 --ignore-eos \
        --result-dir "$OUT_DIR" \
        --result-filename "warmup_${label}.json" \
        >/dev/null 2>&1 || true

    # 正式压测
    echo "  [2/2] 正式压测 (num_prompts=${NUM_PROMPTS}, concurrency=${MAX_CONCURRENCY})..."
    vllm bench serve \
        --host "$HOST" --port "$port" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --request-rate "$REQUEST_RATE" \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 0.0 --temperature 0 --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result \
        --result-dir "$OUT_DIR" \
        --result-filename "$(basename "$out_file")"

    echo "  ✓ 结果: $out_file"
}

print_compare() {
    python3 - "$BASE_RESULT" "$DBO_RESULT" <<'PYEOF'
import json, sys, os

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fmt(v, unit=""):
    if v is None: return "  N/A"
    if isinstance(v, float): return f"{v:>8.2f}{unit}"
    return str(v)

b = load(sys.argv[1])
d = load(sys.argv[2])

print()
print("=" * 58)
print("  DBO vs Baseline 对比")
print("=" * 58)
print(f"  {'指标':<26}  {'Baseline':>10}  {'DBO':>10}  {'变化':>8}")
print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*8}")

rows = [
    ("output_throughput",   "吞吐量 (tok/s)",     " tok/s", True),
    ("request_throughput",  "吞吐量 (req/s)",     " req/s", True),
    ("mean_ttft_ms",        "TTFT mean (ms)",     " ms",    False),
    ("p99_ttft_ms",         "TTFT p99  (ms)",     " ms",    False),
    ("mean_tpot_ms",        "TPOT mean (ms/tok)", " ms",    False),
    ("mean_e2el_ms",        "E2E  mean (ms)",     " ms",    False),
]

for key, label, unit, higher_better in rows:
    bv = b.get(key) if b else None
    dv = d.get(key) if d else None
    if bv and dv:
        ratio = dv / bv if higher_better else bv / dv
        arrow = "↑" if ratio > 1.005 else ("↓" if ratio < 0.995 else "→")
        change = f"{ratio:.2f}x {arrow}"
    else:
        change = "  N/A"
    print(f"  {label:<26}  {fmt(bv, unit):>10}  {fmt(dv, unit):>10}  {change:>8}")

print("=" * 58)

if b and d:
    tput_b = b.get("output_throughput", 0)
    tput_d = d.get("output_throughput", 0)
    if tput_b and tput_d:
        if tput_d > tput_b * 1.02:
            print(f"  ✓ DBO 有效：吞吐量提升 {(tput_d/tput_b-1)*100:.1f}%")
        elif tput_d < tput_b * 0.98:
            print(f"  △ DBO 未观测到加速（当前负载可能不足）")
            print(f"    建议增大 INPUT_LEN(>= 1024) 或 MAX_CONCURRENCY(>= 64)")
        else:
            print(f"  → 性能持平（误差范围内）")
print()
PYEOF
}

# ── 主流程 ────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         DeepSeek-V2 DBO 一键对标                     ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  INPUT_LEN      = ${INPUT_LEN}"
echo "║  OUTPUT_LEN     = ${OUTPUT_LEN}"
echo "║  NUM_PROMPTS    = ${NUM_PROMPTS}"
echo "║  MAX_CONCURRENCY= ${MAX_CONCURRENCY}"
echo "║  BASE_PORT      = ${BASE_PORT}  (baseline)"
echo "║  DBO_PORT       = ${DBO_PORT}  (dbo)"
echo "╚══════════════════════════════════════════════════════╝"

# ── Step 1: baseline ──────────────────────────────────────────────────────────
echo ""
echo "【Step 1/2】Baseline server (DBO 关闭，端口 ${BASE_PORT})"
echo ""

if [[ "$SKIP_PROMPT" != "1" ]]; then
    echo "  请在另一个终端启动 baseline server："
    echo ""
    echo "    PORT=${BASE_PORT} bash ${DEMOS_DIR}/deepseek-v2-server.sh"
    echo ""
    echo "  启动后按 Enter 继续..."
    read -r
fi

wait_server "$BASE_PORT"
run_one "$BASE_PORT" "baseline" "$BASE_RESULT"

# ── Step 2: DBO ───────────────────────────────────────────────────────────────
echo ""
echo "【Step 2/2】DBO server (DBO 开启，端口 ${DBO_PORT})"
echo ""

if [[ "$SKIP_PROMPT" != "1" ]]; then
    echo "  请停掉 baseline server，然后在另一个终端启动 DBO server："
    echo ""
    echo "    PORT=${DBO_PORT} bash ${DEMOS_DIR}/deepseek-v2-dbo-server.sh"
    echo ""
    echo "  启动后按 Enter 继续..."
    read -r
fi

wait_server "$DBO_PORT"
run_one "$DBO_PORT" "dbo" "$DBO_RESULT"

# ── 对比输出 ──────────────────────────────────────────────────────────────────
print_compare

echo "  结果文件："
echo "    baseline : $BASE_RESULT"
echo "    dbo      : $DBO_RESULT"
echo ""
echo "  验证 DBO 触发（在 DBO server 日志里搜索）："
echo "    grep 'should_ubatch: True' <dbo_server_log>"
