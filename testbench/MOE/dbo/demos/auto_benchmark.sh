#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Auto Benchmark — DBO × FlashComm × HCCL Mode
#
# 一键自动遍历所有配置组合。Server 和 Test 直接复用已有脚本
# （deepseek-v2-server.sh / deepseek-v2-dbo-server.sh /
#  deepseek-v2-dbo-test.sh），确保结果与手动跑完全对齐。
#
# 用法：
#   bash auto_benchmark.sh                        # 跑全部 10 组配置
#   CONFIGS="baseline dbo fc2 dbo_fc2" bash ...   # 只跑指定配置
#   BENCH_PRESETS="prefill4k,prefill8k" bash ...  # 多 workload
#   DRY_RUN=1 bash auto_benchmark.sh              # 只打印，不执行
#
# Config 矩阵 (10 组):
#   baseline        DBO=0  FC1=0  FC2=0  HCCL=AI_CPU
#   baseline_aiv    DBO=0  FC1=0  FC2=0  HCCL=AIV
#   fc1             DBO=0  FC1=1  FC2=0  HCCL=AI_CPU
#   fc1_aiv         DBO=0  FC1=1  FC2=0  HCCL=AIV
#   fc2             DBO=0  FC1=1  FC2=1  HCCL=AIV   (FC2→AIV)
#   dbo             DBO=1  FC1=0  FC2=0  HCCL=AI_CPU
#   dbo_aiv         DBO=1  FC1=0  FC2=0  HCCL=AIV
#   dbo_fc1         DBO=1  FC1=1  FC2=0  HCCL=AI_CPU
#   dbo_fc1_aiv     DBO=1  FC1=1  FC2=0  HCCL=AIV
#   dbo_fc2         DBO=1  FC1=1  FC2=1  HCCL=AIV   (FC2→AIV)
# ============================================================


# ── 路径 ───────────────────────────────────────────────────────────────────
DEMOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BASELINE="${DEMOS_DIR}/deepseek-v2-server.sh"
SERVER_DBO="${DEMOS_DIR}/deepseek-v2-dbo-server.sh"
TEST_SCRIPT="${DEMOS_DIR}/deepseek-v2-dbo-test.sh"

# ── 基础参数（与已有脚本默认值严格对齐）────────────────────────────────────
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}                        # 与 server 脚本默认 8001 对齐
TP=${TP:-2}
DEVICES=${DEVICES:-0,1}
DRY_RUN=${DRY_RUN:-0}

# ── 超时 / 冷却 ─────────────────────────────────────────────────────────────
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}
SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT:-60}
INTER_CONFIG_SLEEP=${INTER_CONFIG_SLEEP:-5}

# ── 设备显存管理 ─────────────────────────────────────────────────────────────
# 每个 config 结束后等待设备显存释放的超时 & 阈值
DEVICE_MEMORY_WAIT_TIMEOUT=${DEVICE_MEMORY_WAIT_TIMEOUT:-120}
# HBM 使用率低于此百分比视为显存已释放（默认 10%，即 HBM 占用 <6553 MB for 64GB）
DEVICE_MEMORY_THRESHOLD_PCT=${DEVICE_MEMORY_THRESHOLD_PCT:-10}
# 深度清理: 是否在 config 之间强制 kill 残留进程
DEEP_CLEANUP=${DEEP_CLEANUP:-1}

# ── DBO 阈值（与 deepseek-v2-dbo-server.sh 对齐）───────────────────────────
DBO_PREFILL_TOKEN_THRESHOLD=${DBO_PREFILL_TOKEN_THRESHOLD:-1024}
DBO_DECODE_TOKEN_THRESHOLD=${DBO_DECODE_TOKEN_THRESHOLD:-1000000000}

# ── FlashComm2 默认参数 ─────────────────────────────────────────────────────
# FlashComm2 OTP group size. TP=2 时必须为 1（实现要求 OTP < TP 且 TP 可被 OTP 整除）。
FC2_PARALLEL_SIZE=${FC2_PARALLEL_SIZE:-1}
FC2_OSHARED=${FC2_OSHARED:-1}

# ── 结果 & 日志（OUT_DIR 与 deepseek-v2-dbo-test.sh 对齐）──────────────────
OUT_DIR=${OUT_DIR:-/data/workspace/vllm-ascend/testbench/MOE/dbo/results}
LOG_DIR=${LOG_DIR:-${DEMOS_DIR}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/auto_benchmark_${TIMESTAMP}.log"
RESULT_SUFFIX=${RESULT_SUFFIX:-_${TIMESTAMP}}

mkdir -p "$OUT_DIR" "$LOG_DIR"


# ── Config matrix ──────────────────────────────────────────────────────────
# 格式: "NAME DBO FC1 FC2_PARALLEL FC2_OSHARED HCCL_MODE"
# CONFIGS 环境变量可指定子集，如 CONFIGS="baseline dbo fc2"
DEFAULT_CONFIGS=(
  "baseline       0 0 0 0 AI_CPU"
  "baseline_aiv   0 0 0 0 AIV"
  "fc1            0 1 0 0 AI_CPU"
  "fc1_aiv        0 1 0 0 AIV"
  "fc2            0 1 ${FC2_PARALLEL_SIZE} ${FC2_OSHARED} AIV"
  "dbo            1 0 0 0 AI_CPU"
  "dbo_aiv        1 0 0 0 AIV"
  "dbo_fc1        1 1 0 0 AI_CPU"
  "dbo_fc1_aiv    1 1 0 0 AIV"
  "dbo_fc2        1 1 ${FC2_PARALLEL_SIZE} ${FC2_OSHARED} AIV"
)

# ── Benchmark presets（与 deepseek-v2-dbo-test.sh 对齐）─────────────────────
# 仅存 preset 名称 → 参数的映射，方便计算结果文件名
declare -A PRESET_INPUT_LEN
declare -A PRESET_OUTPUT_LEN
declare -A PRESET_NUM_PROMPTS
declare -A PRESET_MAX_CONCURRENCY

PRESET_INPUT_LEN[prefill4k]=4096
PRESET_OUTPUT_LEN[prefill4k]=16
PRESET_NUM_PROMPTS[prefill4k]=500
PRESET_MAX_CONCURRENCY[prefill4k]=96

PRESET_INPUT_LEN[ttft4k]=4096
PRESET_OUTPUT_LEN[ttft4k]=1
PRESET_NUM_PROMPTS[ttft4k]=500
PRESET_MAX_CONCURRENCY[ttft4k]=96

PRESET_INPUT_LEN[prefill8k]=8192
PRESET_OUTPUT_LEN[prefill8k]=16
PRESET_NUM_PROMPTS[prefill8k]=500
PRESET_MAX_CONCURRENCY[prefill8k]=96

PRESET_INPUT_LEN[quick]=1024
PRESET_OUTPUT_LEN[quick]=128
PRESET_NUM_PROMPTS[quick]=200
PRESET_MAX_CONCURRENCY[quick]=64

# 默认只用 prefill4k；可通过逗号分隔指定多个，如 BENCH_PRESETS="prefill4k,prefill8k"
BENCH_PRESETS=${BENCH_PRESETS:-"prefill4k"}


# ── Device memory helpers ───────────────────────────────────────────────────

# 将逻辑 device ID (0, 1, ...) 映射为物理 NPU ID (6, 7, ...)
# 通过解析 npu-smi info -m 的输出实现
get_physical_npu_ids() {
    local logical_ids_str="$1"
    python3 - "$logical_ids_str" <<'PYEOF'
import subprocess, re, sys

logical_ids = [int(x.strip()) for x in sys.argv[1].split(',') if x.strip().isdigit()]

# Parse npu-smi info -m to build logical->physical mapping
try:
    result = subprocess.run(['npu-smi', 'info', '-m'],
                          capture_output=True, text=True, timeout=10)
    for line in result.stdout.split('\n'):
        # Format: NPU_ID  Chip_ID  Chip_Logic_ID  Chip_Name
        m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+', line)
        if m:
            npu_id = int(m.group(1))
            logic_id = int(m.group(3))
            for lid in logical_ids:
                if lid == logic_id:
                    print(f"{lid}:{npu_id}")
                    break
except Exception:
    # Fallback: assume direct mapping (unlikely but safe)
    for lid in logical_ids:
        print(f"{lid}:{lid}")
PYEOF
}

# 查询指定物理 NPU 的 HBM 使用率 (%)
get_npu_hbm_usage_pct() {
    local npu_id="$1"
    local pct
    pct=$(npu-smi info -t usages -i "$npu_id" 2>/dev/null | grep -oP 'HBM Usage Rate\(%\)\s+:\s+\K\d+' | head -1)
    if [[ -z "$pct" ]]; then
        # 无法查询时返回 100 避免误判为空闲
        echo "100"
    else
        echo "$pct"
    fi
}

# 等待设备显存释放至阈值以下
wait_device_memory_free() {
    local max_wait=${1:-${DEVICE_MEMORY_WAIT_TIMEOUT}}
    local threshold_pct=${2:-${DEVICE_MEMORY_THRESHOLD_PCT}}

    log_msg "  等待设备显存释放 (阈值 < ${threshold_pct}%, 超时 ${max_wait}s) ..."

    local waited=0
    while [[ $waited -lt $max_wait ]]; do
        local all_free=true
        local status_line=""
        while IFS= read -r mapping; do
            local logic_id="${mapping%%:*}"
            local npu_id="${mapping##*:}"
            local pct
            pct=$(get_npu_hbm_usage_pct "$npu_id")
            status_line="${status_line} dev${logic_id}=${pct}%"
            if [[ "$pct" -ge "$threshold_pct" ]]; then
                all_free=false
            fi
        done < <(get_physical_npu_ids "$DEVICES")

        if $all_free; then
            log_msg "  ✓ 设备显存已释放 (${status_line}, 等待 ${waited}s)"
            return 0
        fi

        if [[ $((waited % 15)) -eq 0 ]]; then
            log_msg "  ... 显存占用:${status_line} (已等 ${waited}s)"
        fi
        sleep 3
        waited=$((waited + 3))
    done

    # 超时：打印最终状态
    local final_status=""
    while IFS= read -r mapping; do
        local logic_id="${mapping%%:*}"
        local npu_id="${mapping##*:}"
        local pct
        pct=$(get_npu_hbm_usage_pct "$npu_id")
        final_status="${final_status} dev${logic_id}=${pct}%"
    done < <(get_physical_npu_ids "$DEVICES")
    log_msg "  ⚠ 设备显存未完全释放 (${final_status}, 超时 ${max_wait}s)"
    log_msg "  ⚠ 继续下一配置，可能因显存不足失败"
    return 1
}

# 深度清理：只清理由本脚本启动且占用目标端口的进程
cleanup_device_processes() {
    log_msg "  深度清理残留进程 ..."

    # 清理端口占用
    local existing
    existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        log_msg "    清理端口 $PORT 占用 (PID: $existing)"
        kill -TERM $existing 2>/dev/null || true
        sleep 3
        existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
        [[ -z "$existing" ]] || kill -KILL $existing 2>/dev/null || true
    fi

    log_msg "  ✓ 深度清理完成"
}

# ── Helper functions ───────────────────────────────────────────────────────

log_msg() {
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${ts}] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo "$*" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
}

validate_config() {
    local name="$1" fc1="$2" fc2_par="$3" hccl="$4"
    if [[ "$fc2_par" -gt 0 && "$hccl" != "AIV" ]]; then
        log_msg "  ✗ 配置 [$name] 无效: FlashComm2 要求 HCCL=AIV，当前为 $hccl"
        return 1
    fi
    if [[ "$fc2_par" -gt 0 && "$fc2_par" -ge "$TP" ]]; then
        log_msg "  ✗ 配置 [$name] 无效: FlashComm2 OTP size=$fc2_par 必须小于 TP=$TP"
        return 1
    fi
    if [[ "$fc2_par" -gt 0 && $((TP % fc2_par)) -ne 0 ]]; then
        log_msg "  ✗ 配置 [$name] 无效: TP=$TP 必须能被 FlashComm2 OTP size=$fc2_par 整除"
        return 1
    fi
    if [[ "$fc2_par" -gt 0 && "$fc1" -eq 0 ]]; then
        log_msg "  ⚠ 配置 [$name]: FlashComm2 开启但 FC1=0，结果可能异常"
    fi
    return 0
}

wait_server() {
    local url="http://${HOST}:${PORT}/v1/models"
    log_msg "  等待 server 就绪 (${url}) ..."
    local waited=0
    while [[ $waited -lt $SERVER_START_TIMEOUT ]]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            log_msg "  ✓ Server 就绪 (耗时 ${waited}s)"
            return 0
        fi
        if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 -"$SERVER_PID" 2>/dev/null; then
            log_msg "  ✗ Server 进程组已提前退出 (PGID=$SERVER_PID)"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
        if [[ $((waited % 30)) -eq 0 ]]; then
            log_msg "  ... 已等待 ${waited}s"
        fi
    done
    log_msg "  ✗ Server 启动超时 (${SERVER_START_TIMEOUT}s)"
    return 1
}

start_server() {
    local name="$1" dbo="$2" fc1="$3" fc2_par="$4" fc2_osh="$5" hccl="$6"

    log_msg ""
    log_msg "  ── 启动 Server ─────────────────────────────────────"
    log_msg "  Config : $name"
    log_msg "  DBO    : $dbo   FC1: $fc1   FC2: par=${fc2_par} oshared=${fc2_osh}"
    log_msg "  HCCL   : $hccl"
    log_msg "  Port   : $PORT   TP: $TP   Devices: $DEVICES"

    # ── 导出环境变量，覆盖已有脚本的 ${VAR:-default} ──
    export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
    export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

    export HCCL_OP_EXPANSION_MODE="$hccl"
    export VLLM_ASCEND_ENABLE_FLASHCOMM1="$fc1"
    export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE="$fc2_par"
    export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED="$fc2_osh"
    export VLLM_ASCEND_ENABLE_DBO="$dbo"
    export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

    export DBO_PREFILL_TOKEN_THRESHOLD
    export DBO_DECODE_TOKEN_THRESHOLD
    export PORT MODEL HOST TP
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

    # ── 选择 server 脚本 ──
    local server_script
    if [[ "$dbo" == "1" ]]; then
        server_script="$SERVER_DBO"
    else
        server_script="$SERVER_BASELINE"
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        log_msg "  [DRY-RUN] bash ${server_script}"
        SERVER_PID=""
        return 0
    fi

    # 确保端口未被占用
    local existing
    existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        log_msg "  ⚠ 端口 $PORT 被占用 (PID: $existing)，尝试清理..."
        kill $existing 2>/dev/null || true
        sleep 3
    fi

    log_msg "  启动: setsid bash ${server_script}"
    # 使用 setsid 创建独立的进程组，确保后续能一次性清理整个进程树
    # (包括 vllm serve 及其 multiprocessing spawn 出的所有 worker)
    setsid bash -c "bash \"${server_script}\" 2>&1 | tee -a \"${LOG_FILE}\"" &
    SERVER_PID=$!
    log_msg "  Server PID = $SERVER_PID"

    if ! wait_server; then
        log_msg "  ✗ Server 启动失败，跳过此配置"
        stop_server
        return 1
    fi
}

stop_server() {
    if [[ "$DRY_RUN" == "1" ]]; then
        log_msg "  [DRY-RUN] 停止 server"
        SERVER_PID=""
        return 0
    fi

    if [[ -z "${SERVER_PID:-}" ]]; then
        return 0
    fi

    log_msg "  停止 server 进程组 (PGID=$SERVER_PID) ..."

    # 1) 向整个进程组发 SIGTERM（因为 start_server 用了 setsid，
    #    vllm serve 及其所有 mp worker 都在同一进程组内）
    kill -TERM -"$SERVER_PID" 2>/dev/null || true

    # 2) 等待进程组退出
    local waited=0
    while kill -0 -"$SERVER_PID" 2>/dev/null && [[ $waited -lt $SERVER_STOP_TIMEOUT ]]; do
        sleep 2
        waited=$((waited + 2))
    done

    # 3) 超时则 SIGKILL 整个进程组
    if kill -0 -"$SERVER_PID" 2>/dev/null; then
        log_msg "  ⚠ 进程组未退出，强制 kill -KILL"
        kill -KILL -"$SERVER_PID" 2>/dev/null || true
        sleep 3
    fi

    # 4) 兜底：清理端口占用
    local existing
    existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        log_msg "  ⚠ 端口 $PORT 仍被占用，强制清理 PID $existing"
        kill -KILL $existing 2>/dev/null || true
        sleep 2
    fi

    # 5) 深度清理残留的 vllm / mp 进程并等待显存释放
    if [[ "${DEEP_CLEANUP:-1}" == "1" ]]; then
        cleanup_device_processes
        wait_device_memory_free
    fi

    log_msg "  ✓ Server 已停止"
    SERVER_PID=""
}

run_benchmark() {
    # 调用已有的 deepseek-v2-dbo-test.sh，完全对齐手动压测
    local config_name="$1"
    local preset_name="$2"

    local il=${PRESET_INPUT_LEN[$preset_name]}
    local ol=${PRESET_OUTPUT_LEN[$preset_name]}
    local np=${PRESET_NUM_PROMPTS[$preset_name]}
    local mc=${PRESET_MAX_CONCURRENCY[$preset_name]}

    log_msg ""
    log_msg "    ── 压测: ${config_name} / ${preset_name} ───────────────"
    log_msg "    input=${il}  output=${ol}  prompts=${np}  conc=${mc}"

    # 计算期望的结果文件路径（与 deepseek-v2-dbo-test.sh 命名规则一致）
    local expected_result="${OUT_DIR}/${config_name}_in${il}_out${ol}_np${np}_c${mc}${RESULT_SUFFIX}.json"

    if [[ "$DRY_RUN" == "1" ]]; then
        log_msg "    [DRY-RUN] LABEL=${config_name} BENCH_PRESET=${preset_name} PORT=${PORT} bash ${TEST_SCRIPT}"
        log_msg "    [DRY-RUN] → expected result: $expected_result"
        return 0
    fi

    # 调已有的 test 脚本（single 模式）
    if LABEL="$config_name" \
       BENCH_PRESET="$preset_name" \
       PORT="$PORT" \
       MODEL="$MODEL" \
       HOST="$HOST" \
       OUT_DIR="$OUT_DIR" \
       RESULT_SUFFIX="$RESULT_SUFFIX" \
       bash "$TEST_SCRIPT" 2>&1 | tee -a "$LOG_FILE"; then

        if [[ -f "$expected_result" ]]; then
            GENERATED_RESULTS+=("$expected_result")
            log_msg "    ✓ 结果: $expected_result"
            extract_metrics "$expected_result" "    "
        else
            log_msg "    ⚠ 未找到结果文件: $expected_result"
        fi
    else
        log_msg "    ✗ 压测失败，跳过此 preset"
        return 1
    fi
}

extract_metrics() {
    local file="$1"
    local indent="${2:-}"
    if [[ ! -f "$file" ]]; then
        log_msg "${indent}(结果文件不存在)"
        return
    fi
    python3 - "$file" "$indent" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
indent = sys.argv[2]

def g(key):
    return d.get(key, None)

def fmt(key, unit=""):
    v = g(key)
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.2f}{unit}"
    return f"{v}{unit}"

print(f"{indent}Output throughput      : {fmt('output_throughput', ' tok/s')}")
print(f"{indent}Request throughput     : {fmt('request_throughput', ' req/s')}")
print(f"{indent}Mean TTFT              : {fmt('mean_ttft_ms', ' ms')}")
print(f"{indent}P99 TTFT               : {fmt('p99_ttft_ms', ' ms')}")
print(f"{indent}Mean TPOT              : {fmt('mean_tpot_ms', ' ms/tok')}")
print(f"{indent}P99 TPOT               : {fmt('p99_tpot_ms', ' ms/tok')}")
print(f"{indent}Mean ITL               : {fmt('mean_itl_ms', ' ms')}")
print(f"{indent}P99 ITL                : {fmt('p99_itl_ms', ' ms')}")
print(f"{indent}Mean E2E latency       : {fmt('mean_e2el_ms', ' ms')}")
print(f"{indent}P99 E2E latency        : {fmt('p99_e2el_ms', ' ms')}")
PYEOF
}

check_dbo_trigger() {
    if [[ "$DRY_RUN" == "1" ]]; then
        return
    fi
    local count
    count=$(grep -c "should_ubatch: True" "$LOG_FILE" 2>/dev/null || true)
    log_msg "  DBO ubatch 触发次数 (累计): ${count:-0}"
}

print_summary_table() {
    log_section "RESULTS COMPARISON"

    if [[ ${#GENERATED_RESULTS[@]} -eq 0 ]]; then
        log_msg "  本次运行没有生成结果文件"
        return
    fi

    # 去重
    local unique_files=()
    local seen=()
    for f in "${GENERATED_RESULTS[@]}"; do
        local bn
        bn=$(basename "$f")
        if [[ ! " ${seen[*]} " =~ " ${bn} " ]]; then
            seen+=("$bn")
            unique_files+=("$f")
        fi
    done

    python3 - "${unique_files[@]}" <<'PYEOF'
import json, sys, os
from collections import defaultdict

files = sys.argv[1:]
if not files:
    print("No result files found.")
    sys.exit(0)

groups = defaultdict(list)
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
    except Exception:
        continue
    basename = os.path.basename(f)
    parts = basename.split('_in')
    if len(parts) < 2:
        continue
    config = parts[0]
    rest = parts[1]
    out_parts = rest.split('_out')
    if len(out_parts) < 2:
        continue
    il = out_parts[0]
    ol_rest = out_parts[1]
    ol = ol_rest.split('_')[0]
    groups[(il, ol)].append((config, d))

METRICS = [
    ("output_throughput",  "Output (tok/s)",  True),
    ("request_throughput", "Request (req/s)", True),
    ("mean_ttft_ms",       "TTFT mean (ms)",  False),
    ("p99_ttft_ms",        "TTFT p99 (ms)",   False),
    ("mean_tpot_ms",       "TPOT mean (ms/t)",False),
    ("mean_e2el_ms",       "E2E mean (ms)",   False),
]

for (il, ol), entries in sorted(groups.items()):
    print()
    print(f"  ┌─ Workload: input={il}, output={ol} ─" + "─" * 40)
    header = f"  │ {'Metric':<22}"
    for config, _ in entries:
        header += f" | {config:>12}"
    print(header)
    sep = f"  │ {'-'*22}"
    for _ in entries:
        sep += f"-+-{'-'*12}"
    print(sep)

    for key, label, higher_better in METRICS:
        row = f"  │ {label:<22}"
        for config, d in entries:
            v = d.get(key)
            if isinstance(v, (int, float)):
                row += f" | {v:>12.2f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)
    print(f"  └" + "─" * 60)

# 整体最优
print()
print("  ┌─ Best per metric (all workloads) ─" + "─" * 30)
all_entries = []
for entries in groups.values():
    all_entries.extend(entries)

for key, label, higher_better in METRICS:
    best_val = None
    best_config = ""
    for config, d in all_entries:
        v = d.get(key)
        if not isinstance(v, (int, float)) or v == 0:
            continue
        if higher_better:
            if best_val is None or v > best_val:
                best_val = v
                best_config = config
        else:
            if best_val is None or v < best_val:
                best_val = v
                best_config = config
    if best_config:
        arrow = "↑" if higher_better else "↓"
        print(f"  │ {label:<22} → {best_config:<16} = {best_val:.2f} {arrow}")
print(f"  └" + "─" * 60)
PYEOF
}

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "${SERVER_PID:-}" ]]; then
        log_msg "清理: 检测到未停止的 server，执行完整停止流程"
        stop_server || true
    fi
    exit "$rc"
}


# ── Main ───────────────────────────────────────────────────────────────────

SERVER_PID=""
GENERATED_RESULTS=()
FAILED_BENCHMARKS=()
trap cleanup EXIT INT TERM

# 初始化日志
{
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     Auto Benchmark — DBO × FlashComm × HCCL Mode             ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "║  Log file  : $LOG_FILE"
    echo "║  Model     : $MODEL"
    echo "║  Devices   : $DEVICES"
    echo "║  TP        : $TP"
    echo "║  Port      : $PORT"
    echo "║  Git       : $(git -C "$DEMOS_DIR" rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
    echo "║  Dry run   : $DRY_RUN"
    echo "║  Presets   : $BENCH_PRESETS"
    echo "║  Server    : $SERVER_BASELINE  (DBO=0)"
    echo "║             : $SERVER_DBO      (DBO=1)"
    echo "║  Test      : $TEST_SCRIPT"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
} | tee "$LOG_FILE"

# 解析 presets
read -ra PRESET_NAMES <<< "${BENCH_PRESETS//,/ }"
for pn in "${PRESET_NAMES[@]}"; do
    pn=$(echo "$pn" | xargs)
    if [[ -z "${PRESET_INPUT_LEN[$pn]:-}" ]]; then
        log_msg "✗ 未知 preset: $pn"
        log_msg "  可用: prefill4k ttft4k prefill8k quick"
        exit 1
    fi
done

log_msg "Benchmark presets (${#PRESET_NAMES[@]}):"
for pn in "${PRESET_NAMES[@]}"; do
    pn=$(echo "$pn" | xargs)
    log_msg "  - ${pn}: in=${PRESET_INPUT_LEN[$pn]} out=${PRESET_OUTPUT_LEN[$pn]} prompts=${PRESET_NUM_PROMPTS[$pn]} conc=${PRESET_MAX_CONCURRENCY[$pn]}"
done

# 解析 configs
CONFIGS_STR="${CONFIGS:-}"
if [[ -n "$CONFIGS_STR" ]]; then
    IFS=' ' read -ra CONFIG_NAMES <<< "$CONFIGS_STR"
    CONFIGS_TO_RUN=()
    for cname in "${CONFIG_NAMES[@]}"; do
        found=0
        for def in "${DEFAULT_CONFIGS[@]}"; do
            read -r dname rest <<< "$def"
            if [[ "$dname" == "$cname" ]]; then
                CONFIGS_TO_RUN+=("$def")
                found=1
                break
            fi
        done
        if [[ "$found" -eq 0 ]]; then
            log_msg "✗ 未知 config: $cname"
            log_msg "  可用: baseline baseline_aiv fc1 fc1_aiv fc2 dbo dbo_aiv dbo_fc1 dbo_fc1_aiv dbo_fc2"
            exit 2
        fi
    done
else
    CONFIGS_TO_RUN=("${DEFAULT_CONFIGS[@]}")
fi

log_msg ""
log_msg "Configurations (${#CONFIGS_TO_RUN[@]}):"
for cfg in "${CONFIGS_TO_RUN[@]}"; do
    read -r cname dbo fc1 fc2p fc2o hccl <<< "$cfg"
    fc2_str="off"
    [[ "$fc2p" -gt 0 ]] && fc2_str="on(p=${fc2p},oshared=${fc2o})"
    log_msg "  - ${cname}: DBO=${dbo} FC1=${fc1} FC2=${fc2_str} HCCL=${hccl}"
done

# ── 主循环 ─────────────────────────────────────────────────────────────────
TOTAL=${#CONFIGS_TO_RUN[@]}
CURRENT=0
FAILED_CONFIGS=()

for cfg in "${CONFIGS_TO_RUN[@]}"; do
    CURRENT=$((CURRENT + 1))
    read -r cname dbo fc1 fc2p fc2o hccl <<< "$cfg"

    log_section "[${CURRENT}/${TOTAL}] Config: ${cname}"

    if ! validate_config "$cname" "$fc1" "$fc2p" "$hccl"; then
        FAILED_CONFIGS+=("$cname (验证失败)")
        continue
    fi

    # 启动 server（复用已有脚本）
    if ! start_server "$cname" "$dbo" "$fc1" "$fc2p" "$fc2o" "$hccl"; then
        FAILED_CONFIGS+=("$cname (server 启动失败)")
        continue
    fi

    # 对所有 preset 发压（复用已有脚本）
    preset_idx=0
    for pn in "${PRESET_NAMES[@]}"; do
        pn=$(echo "$pn" | xargs)
        preset_idx=$((preset_idx + 1))
        log_msg ""
        log_msg "  [Preset ${preset_idx}/${#PRESET_NAMES[@]}: ${pn}]"

        if ! run_benchmark "$cname" "$pn"; then
            log_msg "  ⚠ Preset 失败，继续下一个"
            FAILED_BENCHMARKS+=("${cname}/${pn}")
        fi
    done

    check_dbo_trigger
    stop_server

    log_msg "  ✓ 配置 [$cname] 完成"
    log_msg ""

    if [[ "$DRY_RUN" != "1" && "$CURRENT" -lt "$TOTAL" ]]; then
        log_msg "  缓冲 ${INTER_CONFIG_SLEEP}s (显存已在 stop_server 中回收) ..."
        sleep "$INTER_CONFIG_SLEEP"
    fi
done

# ── 报告失败的配置 ─────────────────────────────────────────────────────────
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    log_section "FAILED CONFIGURATIONS"
    for f in "${FAILED_CONFIGS[@]}"; do
        log_msg "  ✗ $f"
    done
fi

if [[ ${#FAILED_BENCHMARKS[@]} -gt 0 ]]; then
    log_section "FAILED BENCHMARKS"
    for f in "${FAILED_BENCHMARKS[@]}"; do
        log_msg "  ✗ $f"
    done
fi

# ── Summary ─────────────────────────────────────────────────────────────────
log_section "BENCHMARK SUMMARY"
log_msg ""
log_msg "  Model    : $MODEL"
log_msg "  Devices  : $DEVICES"
log_msg "  TP       : $TP"
log_msg "  Port     : $PORT"
log_msg "  Git      : $(git -C "$DEMOS_DIR" rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
log_msg "  Log file : $LOG_FILE"

print_summary_table

log_section "DONE"
log_msg "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_msg "  成功    : $((TOTAL - ${#FAILED_CONFIGS[@]}))/${TOTAL}"
log_msg "  失败    : ${#FAILED_CONFIGS[@]}"
log_msg "  压测失败: ${#FAILED_BENCHMARKS[@]}"
log_msg "  Log     : $LOG_FILE"
log_msg "  Results : $OUT_DIR"
log_msg ""
log_msg "  对比不同配置："
log_msg "    grep -A12 'Workload:' $LOG_FILE"
log_msg "  检查 DBO 触发："
log_msg "    grep 'should_ubatch: True' $LOG_FILE"
log_msg "  查看完整 server 日志："
log_msg "    grep -n 'Config:' $LOG_FILE"

if [[ ${#FAILED_CONFIGS[@]} -gt 0 || ${#FAILED_BENCHMARKS[@]} -gt 0 ]]; then
    exit 1
fi
