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
#   bash auto_benchmark.sh                              # 默认跑 DeepSeek TP P0
#   TEST_GROUPS=all bash auto_benchmark.sh              # 跑全部具名测试组
#   TEST_GROUPS="p0_deepseek_tp,p2_deepseek_decode" bash ...
#   CONFIGS="baseline dbo_fc1" bash ...                 # 在每个选中组中跑指定配置
#   BENCH_PRESETS="prefill4k,prefill4k16" bash ...     # 覆盖 workload
#   DEBUG_VALIDATION=0 bash auto_benchmark.sh            # 只跑性能轮
#   DRY_RUN=1 bash auto_benchmark.sh                     # 只打印，不执行
#
# Test groups:
#   p0_deepseek_tp, p0_deepseek_dp, p0_deepseek_dp_shared_expert
#   p1_deepseek_small_batch, p1_qwen_tp, p2_deepseek_decode
#   all
#
# DeepSeek TP config matrix:
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
DEEPSEEK_DIR="${DEMOS_DIR}/DeepseekV2"
SERVER_BASELINE="${DEEPSEEK_DIR}/deepseek-v2-server.sh"
SERVER_DBO="${DEEPSEEK_DIR}/deepseek-v2-dbo-server.sh"
SERVER_DP="${DEEPSEEK_DIR}/deepseek-v2-dbo-server-dp.sh"
TEST_SCRIPT="${DEEPSEEK_DIR}/deepseek-v2-dbo-test.sh"
QWEN_DIR="${DEMOS_DIR}/Qwen3-30B"
QWEN_SERVER_BASELINE="${QWEN_DIR}/Qwen3-30B-server.sh"
QWEN_SERVER_DBO="${QWEN_DIR}/Qwen3-30B-dbo-server.sh"
QWEN_TEST_SCRIPT="${QWEN_DIR}/Qwen3-30B-dbo-test.sh"
QWEN_MODEL=${QWEN_MODEL:-/data/models/Qwen3-30B/Qwen3-30B}

# ── 基础参数（与已有脚本默认值严格对齐）────────────────────────────────────
MODEL=${MODEL:-/data/models/DeepSeek-V2-Lite-Chat}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8001}                        # 与 server 脚本默认 8001 对齐
TP=${TP:-2}
DEVICES=${DEVICES:-0,1}
BASE_MODEL="$MODEL"
BASE_TP="$TP"
DRY_RUN=${DRY_RUN:-0}
# TEST_GROUPS accepts a comma- or space-separated group list, or "all".
# TEST_PLAN is retained as a backwards-compatible alias for older invocations.
TEST_GROUPS=${TEST_GROUPS:-${TEST_PLAN:-p0_deepseek_tp}}
TEST_PLAN=${TEST_PLAN:-$TEST_GROUPS}
# used for dbo validation
DEBUG_VALIDATION=${DEBUG_VALIDATION:-0}
PERFORMANCE_RUN=${PERFORMANCE_RUN:-1}
BENCH_PRESETS_EXPLICIT=${BENCH_PRESETS+x}

# By default all cases reuse the normal shared caches. Set USE_CASE_CACHE=1 to
# isolate validation/performance artifacts under RUN_ROOT (cold-cache behavior).
RUN_ROOT=${RUN_ROOT:-${OUT_DIR:-/data/workspace/vllm-ascend/testbench/MOE/dbo/testbench/results}/runs}
USE_CASE_CACHE=${USE_CASE_CACHE:-0}
SOURCE_ENV=${SOURCE_ENV:-1}
VENV_ROOT=${VENV_ROOT:-/data/workspace/vllm-dbo-v0221/.venv-dbo}
ENV_SCRIPT=${ENV_SCRIPT:-/data/workspace/vllm-dbo-v0221/env.sh}

# ── 超时 / 冷却 ─────────────────────────────────────────────────────────────
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-1800}
SERVER_STOP_TIMEOUT=${SERVER_STOP_TIMEOUT:-60}
INTER_CONFIG_SLEEP=${INTER_CONFIG_SLEEP:-5}
VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-1800}
export VLLM_ENGINE_READY_TIMEOUT_S

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
DEFAULT_DBO_PREFILL_TOKEN_THRESHOLD="$DBO_PREFILL_TOKEN_THRESHOLD"
DEFAULT_DBO_DECODE_TOKEN_THRESHOLD="$DBO_DECODE_TOKEN_THRESHOLD"

# ── FlashComm2 默认参数 ─────────────────────────────────────────────────────
# FlashComm2 OTP group size. TP=2 时必须为 1（实现要求 OTP < TP 且 TP 可被 OTP 整除）。
FC2_PARALLEL_SIZE=${FC2_PARALLEL_SIZE:-1}
FC2_OSHARED=${FC2_OSHARED:-1}

# ── 结果 & 日志（OUT_DIR 与 deepseek-v2-dbo-test.sh 对齐）──────────────────
OUT_DIR=${OUT_DIR:-/data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/testbench/results}
LOG_DIR=${LOG_DIR:-${OUT_DIR}}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/auto_benchmark_${TIMESTAMP}.log"
RESULT_SUFFIX=${RESULT_SUFFIX:-_${TIMESTAMP}}
RUN_ROOT="${RUN_ROOT}/${TIMESTAMP}"
ACTIVE_LOG_FILE="$LOG_FILE"

mkdir -p "$OUT_DIR" "$LOG_DIR"
mkdir -p "$RUN_ROOT"

if [[ "$SOURCE_ENV" == "1" ]]; then
    # Ascend's set_env.sh probes optional shell variables such as ZSH_VERSION.
    # Load external environment scripts without nounset, then restore it.
    set +u
    if [[ -f "$VENV_ROOT/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_ROOT/bin/activate"
    else
        echo "WARNING: virtualenv not found: $VENV_ROOT/bin/activate" >&2
    fi
    if [[ -f "$ENV_SCRIPT" ]]; then
        # shellcheck disable=SC1090
        source "$ENV_SCRIPT"
    else
        echo "WARNING: environment script not found: $ENV_SCRIPT" >&2
    fi
    set -u
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
    export no_proxy="$NO_PROXY"
fi


# ── Config matrix ──────────────────────────────────────────────────────────
# 格式: "NAME FAMILY DBO FC1 FC2_PARALLEL FC2_OSHARED HCCL_MODE ADDITIONAL_CONFIG"
# CONFIGS can select any subset of the named configurations for every selected
# group. Without it, each test group supplies its own matrix.
DEFAULT_CONFIGS=(
  "baseline                    deepseek_tp 0 0 0 0 AI_CPU -"
  "baseline_aiv                deepseek_tp 0 0 0 0 AIV -"
  "fc1                         deepseek_tp 0 1 0 0 AI_CPU -"
  "fc1_aiv                     deepseek_tp 0 1 0 0 AIV -"
  "fc2                         deepseek_tp 0 1 ${FC2_PARALLEL_SIZE} ${FC2_OSHARED} AIV -"
  "dbo                         deepseek_tp 1 0 0 0 AI_CPU -"
  "dbo_aiv                     deepseek_tp 1 0 0 0 AIV -"
  "dbo_fc1                     deepseek_tp 1 1 0 0 AI_CPU -"
  "dbo_fc1_aiv                 deepseek_tp 1 1 0 0 AIV -"
  "dbo_fc2                     deepseek_tp 1 1 ${FC2_PARALLEL_SIZE} ${FC2_OSHARED} AIV -"
  "dp_shared_baseline          deepseek_dp 0 0 0 0 AI_CPU {\"multistream_overlap_shared_expert\":true}"
  "dp_shared_dbo               deepseek_dp 1 0 0 0 AI_CPU {\"multistream_overlap_shared_expert\":true}"
  "dp_baseline                 deepseek_dp 0 0 0 0 AI_CPU -"
  "dp_dbo                      deepseek_dp 1 0 0 0 AI_CPU -"
  "qwen_baseline               qwen_tp     0 0 0 0 AI_CPU -"
  "qwen_baseline_aiv           qwen_tp     0 0 0 0 AIV -"
  "qwen_fc1                    qwen_tp     0 1 0 0 AI_CPU -"
  "qwen_fc1_aiv                qwen_tp     0 1 0 0 AIV -"
  "qwen_fc2                    qwen_tp     0 1 ${FC2_PARALLEL_SIZE} ${FC2_OSHARED} AIV -"
  "qwen_dbo                    qwen_tp     1 0 0 0 AI_CPU -"
  "qwen_dbo_aiv                qwen_tp     1 0 0 0 AIV -"
  "qwen_dbo_fc1                qwen_tp     1 1 0 0 AI_CPU -"
  "qwen_dbo_fc1_aiv            qwen_tp     1 1 0 0 AIV -"
  "qwen_dbo_fc2                qwen_tp     1 1 ${FC2_PARALLEL_SIZE} ${FC2_OSHARED} AIV -"
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

PRESET_INPUT_LEN[prefill4k16]=4096
PRESET_OUTPUT_LEN[prefill4k16]=16
PRESET_NUM_PROMPTS[prefill4k16]=16
PRESET_MAX_CONCURRENCY[prefill4k16]=16

# Decode-focused workload. The low decode threshold is applied only for P2.
PRESET_INPUT_LEN[decode]=128
PRESET_OUTPUT_LEN[decode]=128
PRESET_NUM_PROMPTS[decode]=128
PRESET_MAX_CONCURRENCY[decode]=64

# Each test group's default workload is set with its matrix below.
BENCH_PRESETS=${BENCH_PRESETS:-"prefill4k"}


# ── Device memory helpers ───────────────────────────────────────────────────

# 将逻辑 device ID (0, 1, ...) 映射为物理 NPU ID (6, 7, ...)
# 通过解析 npu-smi info -m 的输出实现
get_physical_npu_ids() {
    local logical_ids_str="$1"
    python3 - "$logical_ids_str" <<'PYEOF'
import subprocess, re, sys

logical_ids = [int(x.strip()) for x in sys.argv[1].split(',') if x.strip().isdigit()]
mapped = set()

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
                    mapped.add(lid)
                    break
except Exception:
    pass

# Keep memory polling conservative when npu-smi output is unavailable or has
# a different format. An empty mapping would otherwise look like "all free".
for lid in logical_ids:
    if lid not in mapped:
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
    if [[ "$ACTIVE_LOG_FILE" == "$LOG_FILE" ]]; then
        echo "[${ts}] $*" >> "$LOG_FILE"
    else
        echo "[${ts}] $*" | tee -a "$ACTIVE_LOG_FILE" "$LOG_FILE" >/dev/null
    fi
    echo "[${ts}] $*"
}

log_section() {
    log_msg ""
    log_msg "===================================================================="
    log_msg "$*"
    log_msg "===================================================================="
}

validate_config() {
    local name="$1" family="$2" fc1="$3" fc2_par="$4" hccl="$5"
    local case_tp="$BASE_TP"
    case "$family" in
        deepseek_tp|qwen_tp) ;;
        deepseek_dp)
            case_tp=1
            if [[ "$fc1" != "0" || "$fc2_par" != "0" ]]; then
                log_msg "  ✗ 配置 [$name] 无效: DP=2/TP=1 不支持 FlashComm"
                return 1
            fi
            ;;
        *)
            log_msg "  ✗ 配置 [$name] 无效: 未知 family=$family"
            return 1
            ;;
    esac
    if [[ "$fc2_par" -gt 0 && "$hccl" != "AIV" ]]; then
        log_msg "  ✗ 配置 [$name] 无效: FlashComm2 要求 HCCL=AIV，当前为 $hccl"
        return 1
    fi
    if [[ "$fc2_par" -gt 0 && "$fc2_par" -ge "$case_tp" ]]; then
        log_msg "  ✗ 配置 [$name] 无效: FlashComm2 OTP size=$fc2_par 必须小于 TP=$case_tp"
        return 1
    fi
    if [[ "$fc2_par" -gt 0 && $((case_tp % fc2_par)) -ne 0 ]]; then
        log_msg "  ✗ 配置 [$name] 无效: TP=$case_tp 必须能被 FlashComm2 OTP size=$fc2_par 整除"
        return 1
    fi
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
    local name="$1" family="$2" dbo="$3" fc1="$4" fc2_par="$5" fc2_osh="$6" hccl="$7" additional_config="$8" run_mode="${9:-perf}"
    local server_script case_model="$BASE_MODEL" case_tp="$BASE_TP" case_dp=1 case_dp_local=1
    local cache_root="shared"
    if [[ "$USE_CASE_CACHE" == "1" ]]; then
        cache_root="${RUN_ROOT}/${run_mode}/${name}"
        mkdir -p "$cache_root/xdg" "$cache_root/torchinductor" "$cache_root/vllm-compile"
    fi

    case "$family" in
        deepseek_tp)
            server_script=$([[ "$dbo" == 1 ]] && printf "%s" "$SERVER_DBO" || printf "%s" "$SERVER_BASELINE")
            ;;
        deepseek_dp)
            server_script="$SERVER_DP"
            case_tp=1
            case_dp=2
            case_dp_local=2
            ;;
        qwen_tp)
            case_model="$QWEN_MODEL"
            server_script=$([[ "$dbo" == 1 ]] && printf "%s" "$QWEN_SERVER_DBO" || printf "%s" "$QWEN_SERVER_BASELINE")
            ;;
        *)
            log_msg "  ✗ 不支持的 family: $family"
            return 2
            ;;
    esac

    log_msg ""
    log_msg "  ── 启动 Server ─────────────────────────────────────"
    log_msg "  Config : $name (family=$family)"
    log_msg "  DBO    : $dbo   FC1: $fc1   FC2: par=${fc2_par} oshared=${fc2_osh}"
    log_msg "  HCCL   : $hccl   TP: $case_tp   DP: $case_dp"
    log_msg "  Model  : $case_model"
    log_msg "  Mode   : $run_mode"
    log_msg "  Cache  : $cache_root"
    [[ "$additional_config" == "-" ]] || log_msg "  ADDITIONAL_CONFIG: $additional_config"

    export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-false}
    export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
    export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
    export HCCL_OP_EXPANSION_MODE="$hccl"
    export VLLM_ASCEND_ENABLE_FLASHCOMM1="$fc1"
    export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE="$fc2_par"
    export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED="$fc2_osh"
    export VLLM_ASCEND_ENABLE_DBO="$dbo"
    export VLLM_ENGINE_READY_TIMEOUT_S
    if [[ "$run_mode" == "debug" ]]; then
        export VLLM_LOGGING_LEVEL="${DEBUG_LOG_LEVEL:-DEBUG}"
    else
        export VLLM_LOGGING_LEVEL="${PERF_LOG_LEVEL:-INFO}"
    fi
    export DBO_PREFILL_TOKEN_THRESHOLD DBO_DECODE_TOKEN_THRESHOLD
    if [[ "$USE_CASE_CACHE" == "1" ]]; then
        export XDG_CACHE_HOME="$cache_root/xdg"
        export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
        export VLLM_COMPILE_CACHE_PATH="$cache_root/vllm-compile"
    fi
    export MODEL="$case_model" PORT HOST TP="$case_tp"
    export DP="$case_dp" DP_LOCAL="$case_dp_local"
    export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
    export MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
    if [[ "$family" == "deepseek_dp" ]]; then
        export DBO_ENABLED="$dbo"
    else
        unset DBO_ENABLED
    fi
    if [[ "$additional_config" == "-" ]]; then
        unset ADDITIONAL_CONFIG
    else
        export ADDITIONAL_CONFIG="$additional_config"
    fi

    if [[ "$DRY_RUN" == 1 ]]; then
        log_msg "  [DRY-RUN] bash $server_script"
        SERVER_PID=""
        return 0
    fi

    local existing
    existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$existing" ]]; then
        log_msg "  ⚠ 端口 $PORT 被占用 (PID: $existing)，尝试清理..."
        kill $existing 2>/dev/null || true
        local port_waited=0
        while [[ $port_waited -lt 10 ]] && lsof -ti :"$PORT" >/dev/null 2>&1; do
            sleep 1
            port_waited=$((port_waited + 1))
        done
        existing=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [[ -n "$existing" ]]; then
            log_msg "  ⚠ 端口 $PORT 未释放，强制清理 PID $existing"
            kill -KILL $existing 2>/dev/null || true
            sleep 2
        fi
        if lsof -ti :"$PORT" >/dev/null 2>&1; then
            log_msg "  ✗ 端口 $PORT 仍被占用，无法启动 ${name}"
            return 1
        fi
    fi

    log_msg "  启动: setsid bash $server_script"
    setsid bash -c "bash \"${server_script}\" 2>&1 | tee -a \"${ACTIVE_LOG_FILE}\"" &
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
        if ! wait_device_memory_free; then
            # HBM can remain cached by the runtime after all worker processes
            # have exited. Do not turn this advisory wait into a matrix abort.
            log_msg "  ⚠ 显存回收检查超时，继续下一配置"
        fi
    fi

    log_msg "  ✓ Server 已停止"
    SERVER_PID=""
}

run_benchmark() {
    local config_name="$1" family="$2" preset_name="$3" result_label="${4:-$config_name}" run_mode="${5:-perf}"
    local case_model="$BASE_MODEL" case_test_script="$TEST_SCRIPT"
    if [[ "$family" == "qwen_tp" ]]; then
        case_model="$QWEN_MODEL"
        case_test_script="$QWEN_TEST_SCRIPT"
    fi

    local il=${PRESET_INPUT_LEN[$preset_name]}
    local ol=${PRESET_OUTPUT_LEN[$preset_name]}
    local np=${PRESET_NUM_PROMPTS[$preset_name]}
    local mc=${PRESET_MAX_CONCURRENCY[$preset_name]}
    local test_preset="$preset_name"
    case "$preset_name" in
        prefill4k16|decode) test_preset=custom ;;
    esac

    log_msg ""
    log_msg "    ── 压测: ${config_name} / ${preset_name} ───────────────"
    log_msg "    input=${il}  output=${ol}  prompts=${np}  conc=${mc}"

    # 计算期望的结果文件路径（与 deepseek-v2-dbo-test.sh 命名规则一致）
    local result_dir="${OUT_DIR}/${run_mode}"
    local mode_suffix="${RESULT_SUFFIX}_${run_mode}"
    local expected_result="${result_dir}/${result_label}_in${il}_out${ol}_np${np}_c${mc}${mode_suffix}.json"
    mkdir -p "$result_dir"

    if [[ "$DRY_RUN" == "1" ]]; then
        log_msg "    [DRY-RUN] LABEL=${result_label} MODE=${run_mode} BENCH_PRESET=${test_preset} PORT=${PORT} bash ${case_test_script}"
        log_msg "    [DRY-RUN] → expected result: $expected_result"
        return 0
    fi

    # 调已有的 test 脚本（single 模式）
    if LABEL="$result_label" \
       BENCH_PRESET="$test_preset" \
       INPUT_LEN="$il" \
       OUTPUT_LEN="$ol" \
       NUM_PROMPTS="$np" \
       MAX_CONCURRENCY="$mc" \
       PORT="$PORT" \
       MODEL="$case_model" \
       HOST="$HOST" \
       OUT_DIR="$result_dir" \
       RESULT_SUFFIX="${mode_suffix}" \
       bash "$case_test_script" 2>&1 | tee -a "$ACTIVE_LOG_FILE" "$LOG_FILE"; then

        if [[ -f "$expected_result" ]]; then
            if [[ "$run_mode" == "perf" ]]; then
                GENERATED_RESULTS+=("$expected_result")
            fi
            log_msg "    ✓ 结果: $expected_result"
            extract_metrics "$expected_result" "    "
        else
            log_msg "    ⚠ 未找到结果文件: $expected_result"
            return 1
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
    local config_name="$1" family="$2" log_file="${3:-$ACTIVE_LOG_FILE}"
    if [[ "$DRY_RUN" == "1" ]]; then
        log_msg "  [DRY-RUN] DBO validation: ${config_name} (${family})"
        return 0
    fi
    if [[ ! -f "$log_file" ]]; then
        log_msg "  ✗ DBO validation failed: log not found: $log_file"
        return 1
    fi

    local expected_workers=2
    local evidence
    local rc=0
    if ! evidence=$(python3 - "$log_file" "$expected_workers" <<'PYEOF'
import re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
workers = sorted(set(re.findall(r"(Worker_(?:TP|DP)\\d+(?:_EP\\d+)?)", text)))
hits = sorted(set(re.findall(r"(Worker_(?:TP|DP)\\d+(?:_EP\\d+)?).*should_ubatch: True", text)))
print(f"workers={','.join(workers) or '-'}")
print(f"hits={','.join(hits) or '-'}")
print(f"hit_count={len(hits)}")
sys.exit(0 if len(hits) >= int(sys.argv[2]) else 1)
PYEOF
    ); then
        rc=1
    fi
    while IFS= read -r line; do
        log_msg "  DBO validation ${config_name}: ${line}"
    done <<< "$evidence"
    if [[ "$rc" -ne 0 ]]; then
        log_msg "  ✗ DBO validation failed for ${config_name}; performance run remains scheduled"
    else
        log_msg "  ✓ DBO validation passed for ${config_name}"
    fi
    return "$rc"
}

run_server_benchmark() {
    local name="$1" family="$2" dbo="$3" fc1="$4" fc2p="$5" fc2o="$6" hccl="$7" additional="$8"
    local preset="$9" mode="${10}" label="${11}"

    if ! start_server "$name" "$family" "$dbo" "$fc1" "$fc2p" "$fc2o" "$hccl" "$additional" "$mode"; then
        return 1
    fi
    local rc=0
    if ! run_benchmark "$name" "$family" "$preset" "$label" "$mode"; then
        rc=1
    fi
    stop_server || rc=1
    return "$rc"
}

run_config_case() {
    local cname="$1" family="$2" dbo="$3" fc1="$4" fc2p="$5" fc2o="$6" hccl="$7" additional="$8" preset="$9"
    local config_rc=0

    # Validation is intentionally separate from performance. DEBUG logs and
    # evidence parsing are never used as performance measurements.
    if [[ "$dbo" == "1" && "$DEBUG_VALIDATION" == "1" ]]; then
        ACTIVE_LOG_FILE="${RUN_ROOT}/debug/${cname}_${preset}.server.log"
        mkdir -p "$(dirname "$ACTIVE_LOG_FILE")"
        if ! run_server_benchmark "$cname" "$family" "$dbo" "$fc1" "$fc2p" "$fc2o" "$hccl" "$additional" "$preset" debug "${cname}_debug"; then
            VALIDATION_FAILURES+=("${cname}/${preset} (debug benchmark)")
            config_rc=1
        fi
        if ! check_dbo_trigger "$cname" "$family" "$ACTIVE_LOG_FILE"; then
            VALIDATION_FAILURES+=("${cname}/${preset} (missing worker evidence)")
            config_rc=1
        fi
    fi

    if [[ "$PERFORMANCE_RUN" != "1" ]]; then
        return "$config_rc"
    fi

    # DBO cases get a matched baseline with the same communication settings.
    if [[ "$dbo" == "1" ]]; then
        ACTIVE_LOG_FILE="${RUN_ROOT}/perf/${cname}_baseline_${preset}.server.log"
        mkdir -p "$(dirname "$ACTIVE_LOG_FILE")"
        if ! run_server_benchmark "${cname}_baseline" "$family" 0 "$fc1" "$fc2p" "$fc2o" "$hccl" "$additional" "$preset" perf "${cname}_baseline"; then
            PERFORMANCE_FAILURES+=("${cname}/${preset}/baseline")
            config_rc=1
        fi
    fi

    ACTIVE_LOG_FILE="${RUN_ROOT}/perf/${cname}_${preset}.server.log"
    mkdir -p "$(dirname "$ACTIVE_LOG_FILE")"
    if ! run_server_benchmark "$cname" "$family" "$dbo" "$fc1" "$fc2p" "$fc2o" "$hccl" "$additional" "$preset" perf "$cname"; then
        PERFORMANCE_FAILURES+=("${cname}/${preset}/${dbo:+dbo}")
        config_rc=1
    fi
    return "$config_rc"
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

generate_report() {
    local report_file="${OUT_DIR}/benchmark_report_${TIMESTAMP}.md"
    python3 - "$report_file" "$BASE_MODEL" "$TEST_PLAN" "$RUN_ROOT" "${GENERATED_RESULTS[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

report, model, plan, run_root, *files = sys.argv[1:]
entries = {}
for filename in files:
    try:
        data = json.loads(Path(filename).read_text())
    except (OSError, json.JSONDecodeError):
        continue
    label = Path(filename).name.split("_in", 1)[0]
    entries[label] = data

metrics = (
    ("output_throughput", "Output throughput (tok/s)", True),
    ("request_throughput", "Request throughput (req/s)", True),
    ("mean_ttft_ms", "Mean TTFT (ms)", False),
    ("p99_ttft_ms", "P99 TTFT (ms)", False),
    ("mean_tpot_ms", "Mean TPOT (ms/tok)", False),
    ("p99_tpot_ms", "P99 TPOT (ms/tok)", False),
    ("mean_itl_ms", "Mean ITL (ms)", False),
    ("p99_itl_ms", "P99 ITL (ms)", False),
    ("mean_e2el_ms", "Mean E2E (ms)", False),
    ("p99_e2el_ms", "P99 E2E (ms)", False),
)

lines = [
    "# DBO Benchmark Report",
    "",
    f"- Model: `{model}`",
    f"- Test plan: `{plan}`",
    f"- Run artifacts: `{run_root}`",
    "- Performance data is collected with INFO logging; DEBUG validation is separate.",
    "",
    "## Performance Comparison",
    "",
]

dbo_labels = sorted(label for label in entries if not label.endswith("_baseline"))
if not dbo_labels:
    lines.append("No performance result JSON was produced.")
for label in dbo_labels:
    current = entries[label]
    baseline = entries.get(f"{label}_baseline")
    lines.extend([f"### {label}", "", "| Metric | Baseline | Current | Delta |", "|---|---:|---:|---:|"])
    for key, display, higher_is_better in metrics:
        value = current.get(key)
        base = baseline.get(key) if baseline else None
        if isinstance(value, (int, float)) and isinstance(base, (int, float)) and base:
            delta = (value - base) / base * 100
            if not higher_is_better:
                delta = -delta
            base_text, value_text, delta_text = f"{base:.2f}", f"{value:.2f}", f"{delta:+.2f}%"
        else:
            base_text = f"{base:.2f}" if isinstance(base, (int, float)) else "N/A"
            value_text = f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"
            delta_text = "N/A"
        lines.append(f"| {display} | {base_text} | {value_text} | {delta_text} |")
    lines.append("")

Path(report).write_text("\n".join(lines) + "\n")
PYEOF

    {
        echo "## Validation And Failures"
        echo ""
        echo "Debug server logs are under: \`$RUN_ROOT/debug\`."
        echo ""
        echo "### DBO Validation Failures"
        if [[ ${#VALIDATION_FAILURES[@]} -eq 0 ]]; then
            echo "None."
        else
            for item in "${VALIDATION_FAILURES[@]}"; do echo "- $item"; done
        fi
        echo ""
        echo "### Performance Failures"
        if [[ ${#PERFORMANCE_FAILURES[@]} -eq 0 ]]; then
            echo "None."
        else
            for item in "${PERFORMANCE_FAILURES[@]}"; do echo "- $item"; done
        fi
        echo ""
        echo "### Configuration Failures"
        if [[ ${#FAILED_CONFIGS[@]} -eq 0 ]]; then
            echo "None."
        else
            for item in "${FAILED_CONFIGS[@]}"; do echo "- $item"; done
        fi
    } >> "$report_file"
    log_msg "  Report   : $report_file"
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
trap cleanup EXIT INT TERM HUP

# 初始化日志
{
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     Auto Benchmark — DBO × FlashComm × HCCL Mode             ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "║  Log file  : $LOG_FILE"
    echo "║  Model     : $BASE_MODEL"
    echo "║  Devices   : $DEVICES"
    echo "║  TP        : $BASE_TP"
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

# Named test groups map directly to testbench/task.md. A DBO case runs its
# matched baseline automatically, so no duplicate baseline entry is needed.
declare -A GROUP_CONFIGS GROUP_PRESETS GROUP_DECODE_THRESHOLDS
GROUP_ORDER=(
    p0_deepseek_tp
    p0_deepseek_dp
    p0_deepseek_dp_shared_expert
    p1_deepseek_small_batch
    p1_qwen_tp
    p2_deepseek_decode
)
GROUP_CONFIGS[p0_deepseek_tp]="baseline dbo fc1 fc1_aiv dbo_fc1 dbo_fc1_aiv dbo_fc2"
GROUP_PRESETS[p0_deepseek_tp]="prefill4k"
GROUP_CONFIGS[p0_deepseek_dp]="dp_baseline dp_dbo"
GROUP_PRESETS[p0_deepseek_dp]="prefill4k"
GROUP_CONFIGS[p0_deepseek_dp_shared_expert]="dp_shared_baseline dp_shared_dbo"
GROUP_PRESETS[p0_deepseek_dp_shared_expert]="prefill4k"
GROUP_CONFIGS[p1_deepseek_small_batch]="baseline dbo dp_baseline dp_dbo"
GROUP_PRESETS[p1_deepseek_small_batch]="prefill4k16"
GROUP_CONFIGS[p1_qwen_tp]="qwen_baseline qwen_dbo qwen_fc1 qwen_fc1_aiv qwen_dbo_fc1 qwen_dbo_fc1_aiv qwen_dbo_fc2"
GROUP_PRESETS[p1_qwen_tp]="prefill4k"
# Decode comparison uses the DP server so the decode path is evaluated with
# the same data-parallel communication topology as the DP performance group.
GROUP_CONFIGS[p2_deepseek_decode]="dp_dbo"
GROUP_PRESETS[p2_deepseek_decode]="decode"
GROUP_DECODE_THRESHOLDS[p2_deepseek_decode]="${P2_DECODE_TOKEN_THRESHOLD:-32}"

select_groups() {
    local requested="$1" group
    if [[ "$requested" == "all" ]]; then
        SELECTED_GROUPS=("${GROUP_ORDER[@]}")
        return
    fi

    SELECTED_GROUPS=()
    read -ra requested_groups <<< "${requested//,/ }"
    for group in "${requested_groups[@]}"; do
        case "$group" in
            # Compatibility aliases for earlier versions of this script.
            p0) SELECTED_GROUPS+=(p0_deepseek_tp p0_deepseek_dp p0_deepseek_dp_shared_expert) ;;
            p1) SELECTED_GROUPS+=(p1_deepseek_small_batch p1_qwen_tp) ;;
            p2) SELECTED_GROUPS+=(p2_deepseek_decode) ;;
            qwen) SELECTED_GROUPS+=(p1_qwen_tp) ;;
            *)
                if [[ -z "${GROUP_CONFIGS[$group]:-}" ]]; then
                    echo "Unknown TEST_GROUPS entry: $group" >&2
                    echo "Available: ${GROUP_ORDER[*]} all" >&2
                    exit 2
                fi
                SELECTED_GROUPS+=("$group")
                ;;
        esac
    done
    if [[ ${#SELECTED_GROUPS[@]} -eq 0 ]]; then
        echo "TEST_GROUPS must name at least one test group" >&2
        exit 2
    fi
}

find_config() {
    local wanted="$1" def dname
    for def in "${DEFAULT_CONFIGS[@]}"; do
        read -r dname _ <<< "$def"
        if [[ "$dname" == "$wanted" ]]; then
            printf '%s\n' "$def"
            return 0
        fi
    done
    return 1
}

list_config_names() {
    local def name
    for def in "${DEFAULT_CONFIGS[@]}"; do
        read -r name _ <<< "$def"
        printf '%s ' "$name"
    done
}

FAILED_CONFIGS=()
VALIDATION_FAILURES=()
PERFORMANCE_FAILURES=()
TOTAL=0
CURRENT=0

select_groups "$TEST_GROUPS"

# Validate and count every requested case before any server starts. This makes
# invalid input fail fast and keeps progress counts correct for multi-group runs.
for group_name in "${SELECTED_GROUPS[@]}"; do
    config_names="${CONFIGS:-${GROUP_CONFIGS[$group_name]}}"
    if [[ -n "$BENCH_PRESETS_EXPLICIT" ]]; then
        preset_names="$BENCH_PRESETS"
    else
        preset_names="${GROUP_PRESETS[$group_name]}"
    fi
    read -ra CONFIG_NAMES <<< "${config_names//,/ }"
    read -ra PRESET_NAMES <<< "${preset_names//,/ }"
    for cname in "${CONFIG_NAMES[@]}"; do
        if ! find_config "$cname" >/dev/null; then
            log_msg "✗ Unknown config: $cname"
            log_msg "  Available: $(list_config_names)"
            exit 2
        fi
    done
    for pn in "${PRESET_NAMES[@]}"; do
        if [[ -z "${PRESET_INPUT_LEN[$pn]:-}" ]]; then
            log_msg "✗ Unknown preset: $pn"
            log_msg "  Available: prefill4k prefill4k16 decode ttft4k prefill8k quick"
            exit 2
        fi
    done
    TOTAL=$((TOTAL + ${#CONFIG_NAMES[@]} * ${#PRESET_NAMES[@]}))
done

for group_name in "${SELECTED_GROUPS[@]}"; do
    DBO_PREFILL_TOKEN_THRESHOLD=$DEFAULT_DBO_PREFILL_TOKEN_THRESHOLD
    DBO_DECODE_TOKEN_THRESHOLD=$DEFAULT_DBO_DECODE_TOKEN_THRESHOLD
    if [[ -n "${GROUP_DECODE_THRESHOLDS[$group_name]:-}" ]]; then
        DBO_DECODE_TOKEN_THRESHOLD=${GROUP_DECODE_THRESHOLDS[$group_name]}
    fi

    config_names="${CONFIGS:-${GROUP_CONFIGS[$group_name]}}"
    if [[ -n "$BENCH_PRESETS_EXPLICIT" ]]; then
        preset_names="$BENCH_PRESETS"
    else
        preset_names="${GROUP_PRESETS[$group_name]}"
    fi
    read -ra CONFIG_NAMES <<< "${config_names//,/ }"
    read -ra PRESET_NAMES <<< "${preset_names//,/ }"
    CONFIGS_TO_RUN=()
    for cname in "${CONFIG_NAMES[@]}"; do
        CONFIGS_TO_RUN+=("$(find_config "$cname")")
    done

    log_section "TEST GROUP: ${group_name}"
    log_msg "Configurations (${#CONFIGS_TO_RUN[@]}): ${config_names}"
    log_msg "Workloads (${#PRESET_NAMES[@]}): ${preset_names}"
    [[ -n "${GROUP_DECODE_THRESHOLDS[$group_name]:-}" ]] && log_msg "DBO decode threshold: ${DBO_DECODE_TOKEN_THRESHOLD}"

    for cfg in "${CONFIGS_TO_RUN[@]}"; do
        read -r cname family dbo fc1 fc2p fc2o hccl additional_config <<< "$cfg"
        if ! validate_config "$cname" "$family" "$fc1" "$fc2p" "$hccl"; then
            FAILED_CONFIGS+=("${group_name}/${cname} (invalid configuration)")
            continue
        fi
        for pn in "${PRESET_NAMES[@]}"; do
            CURRENT=$((CURRENT + 1))
            log_section "[${CURRENT}/${TOTAL}] ${group_name}: ${cname}/${pn}"
            if ! run_config_case "$cname" "$family" "$dbo" "$fc1" "$fc2p" "$fc2o" "$hccl" "$additional_config" "$pn"; then
                FAILED_CONFIGS+=("${group_name}/${cname}/${pn}")
            fi
            if [[ "$DRY_RUN" != "1" && "$CURRENT" -lt "$TOTAL" ]]; then
                sleep "$INTER_CONFIG_SLEEP"
            fi
        done
    done
done

# ── 报告失败的配置 ─────────────────────────────────────────────────────────
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    log_section "FAILED CONFIGURATIONS"
    for f in "${FAILED_CONFIGS[@]}"; do
        log_msg "  ✗ $f"
    done
fi

if [[ ${#VALIDATION_FAILURES[@]} -gt 0 ]]; then
    log_section "FAILED DBO VALIDATIONS"
    for f in "${VALIDATION_FAILURES[@]}"; do
        log_msg "  ✗ $f"
    done
fi

if [[ ${#PERFORMANCE_FAILURES[@]} -gt 0 ]]; then
    log_section "FAILED PERFORMANCE RUNS"
    for f in "${PERFORMANCE_FAILURES[@]}"; do
        log_msg "  ✗ $f"
    done
fi

# ── Summary ─────────────────────────────────────────────────────────────────
log_section "BENCHMARK SUMMARY"
log_msg ""
log_msg "  Model    : $BASE_MODEL"
log_msg "  Devices  : $DEVICES"
log_msg "  TP       : $BASE_TP"
log_msg "  Port     : $PORT"
log_msg "  Git      : $(git -C "$DEMOS_DIR" rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
log_msg "  Log file : $LOG_FILE"

print_summary_table
generate_report

log_section "DONE"
log_msg "  完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
log_msg "  成功    : $((TOTAL - ${#FAILED_CONFIGS[@]}))/${TOTAL}"
log_msg "  失败    : ${#FAILED_CONFIGS[@]}"
log_msg "  DBO 验证失败: ${#VALIDATION_FAILURES[@]}"
log_msg "  性能测试失败: ${#PERFORMANCE_FAILURES[@]}"
log_msg "  Log     : $LOG_FILE"
log_msg "  Results : $OUT_DIR"
log_msg ""
log_msg "  对比不同配置："
log_msg "    grep -A12 'Workload:' $LOG_FILE"
log_msg "  检查 DBO 触发："
log_msg "    grep 'should_ubatch: True' $LOG_FILE"
log_msg "  查看完整 server 日志："
log_msg "    grep -n 'Config:' $LOG_FILE"

if [[ ${#FAILED_CONFIGS[@]} -gt 0 || ${#VALIDATION_FAILURES[@]} -gt 0 || ${#PERFORMANCE_FAILURES[@]} -gt 0 ]]; then
    exit 1
fi
