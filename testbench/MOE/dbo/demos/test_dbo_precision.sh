#!/usr/bin/env bash
set -euo pipefail

# DBO 精度验证：DBO=OFF vs DBO=ON, 直接调 API 对比输出
# 用法: bash test_dbo_precision.sh

DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BASELINE="${DEMOS_DIR}/deepseek-v2-server.sh"
SERVER_DBO="${DEMOS_DIR}/deepseek-v2-dbo-server.sh"

MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}
TP=${TP:-2}
DEVICES=${DEVICES:-0,1}
NUM_PROMPTS=${NUM_PROMPTS:-50}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-128}
SEED=${SEED:-42}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-/data/workspace/vllm-ascend/testbench/MOE/dbo/results}
LOG_FILE="${DEMOS_DIR}/test_precision_${TIMESTAMP}.log"
mkdir -p "$OUT_DIR"

export VLLM_USE_MODELSCOPE=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export VLLM_LOGGING_LEVEL=WARNING
export MAX_MODEL_LEN=8192
export MAX_NUM_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=256
export MODEL HOST TP PORT
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0

SERVER_PID=""
SERVER_STOP_TIMEOUT=60

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

cleanup_port() {
    local existing; existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        kill -TERM $existing 2>/dev/null || true; sleep 3
        existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
        [[ -z "$existing" ]] || kill -KILL $existing 2>/dev/null || true
    fi
}

stop_server() {
    [[ -z "${SERVER_PID:-}" ]] && return 0
    log "  停止 server (PGID=$SERVER_PID)"
    kill -TERM -"$SERVER_PID" 2>/dev/null || true
    local w=0
    while kill -0 -"$SERVER_PID" 2>/dev/null && [[ $w -lt $SERVER_STOP_TIMEOUT ]]; do sleep 2; w=$((w+2)); done
    kill -0 -"$SERVER_PID" 2>/dev/null && { kill -KILL -"$SERVER_PID" 2>/dev/null || true; sleep 3; }
    cleanup_port; log "  ✓ Server 已停止"; SERVER_PID=""
}

cleanup() { trap - EXIT INT TERM; [[ -n "${SERVER_PID:-}" ]] && stop_server; }
trap cleanup EXIT INT TERM

wait_server() {
    local url="http://${HOST}:${PORT}/v1/models"
    log "  等待 server ..."
    local w=0
    while [[ $w -lt 600 ]]; do
        curl -sf "$url" >/dev/null 2>&1 && { log "  ✓ 就绪 (${w}s)"; return 0; }
        sleep 5; w=$((w+5))
    done
    log "  ✗ 超时"; return 1
}

# ── 发请求脚本 (写入临时文件, 通过 argparse 传参) ──
PRECISION_PY="${OUT_DIR}/_precision_test.py"
cat > "$PRECISION_PY" << 'HEREDOC_END'
import json, random, requests, sys, argparse

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--num-prompts", type=int, required=True)
p.add_argument("--input-len", type=int, required=True)
p.add_argument("--output-len", type=int, required=True)
p.add_argument("--host", required=True)
p.add_argument("--port", type=int, required=True)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

random.seed(args.seed)
url = f"http://{args.host}:{args.port}/v1/chat/completions"

results = []
for i in range(args.num_prompts):
    tokens = [random.randint(1000, 50000) for _ in range(args.input_len)]
    prompt_text = "test " * (args.input_len // 2)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": args.output_len,
        "temperature": 0.0,
        "seed": args.seed + i,
    }

    resp = requests.post(url, json=payload, timeout=300)
    if resp.status_code != 200:
        results.append({"error": f"HTTP {resp.status_code}"})
        print(f"  ERROR #{i}: HTTP {resp.status_code}", file=sys.stderr)
        continue

    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    results.append({"idx": i, "text": text})
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{args.num_prompts} done", file=sys.stderr)

with open(args.out, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"OK {len(results)} results", file=sys.stderr)
HEREDOC_END

run_config() {
    local label="$1" dbo="$2"
    log ""; log "══════ ${label} (DBO=${dbo}) ══════"

    export VLLM_ASCEND_ENABLE_DBO="$dbo"
    export HCCL_OP_EXPANSION_MODE="AI_CPU"
    cleanup_port

    local srv; [[ "$dbo" == "1" ]] && srv="$SERVER_DBO" || srv="$SERVER_BASELINE"
    setsid bash -c "bash \"${srv}\" 2>&1 | tee -a \"${LOG_FILE}\"" &
    SERVER_PID=$!
    wait_server || { stop_server; return 1; }

    local out="${OUT_DIR}/${label}_precision_${TIMESTAMP}.jsonl"
    python3 "$PRECISION_PY" \
        --seed "$SEED" --num-prompts "$NUM_PROMPTS" \
        --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
        --host "$HOST" --port "$PORT" --model "$MODEL" --out "$out" \
        2>&1 | tee -a "$LOG_FILE"

    stop_server
}

# ── Main ──
log "DBO 精度验证: ${NUM_PROMPTS} prompts × in=${INPUT_LEN} out=${OUTPUT_LEN}, seed=$SEED"
log "Log: $LOG_FILE"

run_config "nodbo" 0
run_config "dbo"   1

NODBO_FILE="${OUT_DIR}/nodbo_precision_${TIMESTAMP}.jsonl"
DBO_FILE="${OUT_DIR}/dbo_precision_${TIMESTAMP}.jsonl"

log ""; log "══════ 精度对比 ══════"

python3 - "$NODBO_FILE" "$DBO_FILE" << 'HEREDOC_END'
import json, sys

def load(path):
    rr = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line: rr.append(json.loads(line))
    return rr

a = load(sys.argv[1]); b = load(sys.argv[2])
if len(a) != len(b): print(f"✗ 数量不一致: {len(a)} vs {len(b)}"); sys.exit(1)

match = 0; mismatch = 0
for i, (ra, rb) in enumerate(zip(a, b)):
    ta = ra.get("text", ra.get("error", ""))
    tb = rb.get("text", rb.get("error", ""))
    if ta == tb:
        match += 1
    else:
        mismatch += 1
        if mismatch <= 3:
            print(f"  MISMATCH #{i}:")
            print(f"    nodbo: {ta[:80]}")
            print(f"    dbo:   {tb[:80]}")

print(f"\n  一致: {match}/{len(a)}  不一致: {mismatch}/{len(a)}")
if mismatch == 0:      print("  ✓ DBO 精度无损")
elif mismatch/len(a) < 0.02: print(f"  ⚠ {mismatch} 个不一致 ({mismatch/len(a)*100:.1f}%)")
else:                   print(f"  ✗ {mismatch} 个不一致 ({mismatch/len(a)*100:.1f}%)")
HEREDOC_END

log "nodbo: $NODBO_FILE"
log "dbo:   $DBO_FILE"
log "Log:   $LOG_FILE"
