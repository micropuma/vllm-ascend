#!/usr/bin/env python3
"""
DeepSeek-V2 DBO (Dual Batch Overlap) 效果验证脚本

运行模式（通过环境变量控制）：
  MODE=dbo   python deepseek-v2-offline-dbo.py   # 启用 DBO（默认）
  MODE=nodbo python deepseek-v2-offline-dbo.py   # 禁用 DBO，作为对照基线

DBO 触发条件（需同时满足）：
  1. enable_dbo=True
  2. batch token 数 >= DBO_THRESHOLD（本脚本设为 64，方便触发）
  3. MoE 通信模式不为 MC2（小规模 TP=2/EP=2 下自动满足）
  4. padding 后第二个 microbatch 非空

参考文档：../dbo.md
"""

import gc
import logging
import os
import sys
import time

# ── 环境变量（必须在 import torch 之前设置）──────────────────────────────────

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"

os.environ["TORCH_EXTENSIONS_DIR"] = "/data/torch_cache"
os.environ["VLLM_CACHE_DIR"] = "/data/vllm_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/triton_cache"
os.environ["HF_HOME"] = "/data/huggingface_cache"
os.environ["TMPDIR"] = "/data/tmp"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/data/torch_inductor_cache"
os.environ["VLLM_COMPILE_CACHE_PATH"] = "/data/vllm_compile_cache"

# DBO 硬件并发必要条件：让 HCCL 通信跑在 AI_CPU 核上，避免与计算抢 AI Core
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AI_CPU")

# 开启 DEBUG 日志，运行时可搜索 "should_ubatch" 确认 DBO 是否触发
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

for d in ["/data/torch_cache", "/data/vllm_cache", "/data/triton_cache",
          "/data/huggingface_cache", "/data/tmp"]:
    os.makedirs(d, exist_ok=True)

# ── 运行时 import ─────────────────────────────────────────────────────────────

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────────────────────────────

MODEL = os.environ.get("MODEL", "/data/models/DeepSeek-V2-Lite-Chat")
MODE  = os.environ.get("MODE", "dbo").lower()   # "dbo" | "nodbo"

# DBO 触发阈值（调低到 64，方便 offline 小 batch 验证；生产默认值是 512）。
#
# 触发条件：单个 forward step 的 batch token 总数 >= dbo_prefill_token_threshold。
# ascend model_runner 中 uniform_decode 固定传 False，prefill/decode 统一走这个门槛。
#
# prefill step 的 batch token 总数 = scheduler 打包进同一 step 的所有 request
# 的 prefill token 之和（上限由 max_num_batched_tokens 控制，默认 2048）。
# decode step 的 batch token 总数 ≈ 当前并发 request 数（每条贡献 1 token）。
# 两个阶段只要任意一个 step 的 token 总数 >= DBO_THRESHOLD 就会触发。
DBO_THRESHOLD = int(os.environ.get("DBO_THRESHOLD", "64"))

# 128 条 × 64 token/条 ≈ 8192 token，远超阈值 64，prefill step 必然触发 DBO。
# decode 阶段 128 条并发 × 1 token = 128 token，同样超过阈值 64，也会触发。
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "64"))
NUM_PROMPTS   = int(os.environ.get("NUM_PROMPTS", "128"))
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "32"))

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def make_prompts(num: int, tokens_each: int) -> list[str]:
    """生成足够长的 prompt，保证 batch 总 token 数超过 DBO 触发阈值。"""
    base = "The quick brown fox jumps over the lazy dog. " * (tokens_each // 9 + 1)
    return [base[:tokens_each * 4]] * num   # 粗略按字符数截断，tokenizer 会精确计算


def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()


def run_inference(enable_dbo: bool) -> tuple[float, list]:
    """
    创建 LLM 实例，执行推理，返回 (耗时秒数, outputs)。
    enable_dbo=True  → DBO 模式
    enable_dbo=False → 基线模式（无 DBO）
    """
    tag = "DBO" if enable_dbo else "baseline"
    logger.info("=" * 60)
    logger.info("Running %s mode (enable_dbo=%s)", tag, enable_dbo)
    logger.info("  Model            : %s", MODEL)
    logger.info("  NUM_PROMPTS      : %d", NUM_PROMPTS)
    logger.info("  PROMPT_TOKENS    : ~%d per prompt", PROMPT_TOKENS)
    logger.info("  OUTPUT_TOKENS    : %d", OUTPUT_TOKENS)
    logger.info("  DBO_THRESHOLD    : %d", DBO_THRESHOLD)
    logger.info("  Total input ~tok : ~%d  (threshold=%d, DBO will %s)",
                NUM_PROMPTS * PROMPT_TOKENS, DBO_THRESHOLD,
                "TRIGGER ✓" if enable_dbo and NUM_PROMPTS * PROMPT_TOKENS >= DBO_THRESHOLD else "NOT trigger")
    logger.info("=" * 60)

    prompts = make_prompts(NUM_PROMPTS, PROMPT_TOKENS)
    sampling_params = SamplingParams(
        temperature=0.0,        # greedy，保证两次输出一致，方便 diff
        max_tokens=OUTPUT_TOKENS,
    )

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=4096,
        enable_expert_parallel=True,
        enable_dbo=enable_dbo,
        # 将 prefill 触发阈值调低到 DBO_THRESHOLD，方便在小 batch 下验证
        # 生产环境不需要设置这个，保持默认 512 即可
        dbo_prefill_token_threshold=DBO_THRESHOLD if enable_dbo else 512,
        dbo_decode_token_threshold=max(1, DBO_THRESHOLD // 16) if enable_dbo else 32,
    )

    # warmup（不计时）
    logger.info("[%s] Warmup...", tag)
    _ = llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=4))

    # 正式计时
    logger.info("[%s] Timed run...", tag)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    logger.info("[%s] Done. Elapsed: %.3f s", tag, elapsed)

    del llm
    clean_up()

    return elapsed, outputs


def print_results(elapsed_dbo: float | None, elapsed_base: float | None,
                  outputs_dbo: list | None, outputs_base: list | None):
    print("\n" + "=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)

    if elapsed_dbo is not None and elapsed_base is not None:
        speedup = elapsed_base / elapsed_dbo
        print(f"  Baseline  : {elapsed_base:.3f} s")
        print(f"  DBO       : {elapsed_dbo:.3f} s")
        print(f"  Speedup   : {speedup:.2f}x  ({'faster' if speedup > 1 else 'slower'})")

        # 验证输出一致性（DBO 不应改变生成结果）
        if outputs_dbo and outputs_base:
            mismatches = sum(
                1 for a, b in zip(outputs_base, outputs_dbo)
                if a.outputs[0].text.strip() != b.outputs[0].text.strip()
            )
            print(f"  Output match: {'✓ all match' if mismatches == 0 else f'✗ {mismatches} mismatch(es)'}")
    elif elapsed_dbo is not None:
        print(f"  DBO elapsed: {elapsed_dbo:.3f} s")
    elif elapsed_base is not None:
        print(f"  Baseline elapsed: {elapsed_base:.3f} s")

    if outputs_dbo:
        print("\n  DBO outputs:")
        for i, out in enumerate(outputs_dbo):
            print(f"    [{i}] {out.outputs[0].text[:80]!r}")
    elif outputs_base:
        print("\n  Baseline outputs:")
        for i, out in enumerate(outputs_base):
            print(f"    [{i}] {out.outputs[0].text[:80]!r}")

    print("=" * 60)
    print()
    print("  To confirm DBO triggered, search the log above for:")
    print("    should_ubatch: True")
    print("  To run comparison: MODE=compare python deepseek-v2-offline-dbo.py")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    elapsed_dbo, elapsed_base = None, None
    outputs_dbo, outputs_base = None, None

    if MODE == "compare":
        # 对照实验：先跑 baseline 再跑 DBO，打印 speedup
        elapsed_base, outputs_base = run_inference(enable_dbo=False)
        elapsed_dbo,  outputs_dbo  = run_inference(enable_dbo=True)
    elif MODE == "dbo":
        elapsed_dbo, outputs_dbo = run_inference(enable_dbo=True)
    elif MODE == "nodbo":
        elapsed_base, outputs_base = run_inference(enable_dbo=False)
    else:
        logger.error("Unknown MODE=%r. Use 'dbo', 'nodbo', or 'compare'.", MODE)
        return 1

    print_results(elapsed_dbo, elapsed_base, outputs_dbo, outputs_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
