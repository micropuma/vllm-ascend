#!/usr/bin/env python3
"""
最小复现：DBO + FlashComm2 AICPU AllToAll 故障

对比矩阵：
  RUN=1: DBO=on  FC2=off  (应通过)
  RUN=2: DBO=on  FC2=on   (预期 AICPU 错误)
  RUN=3: DBO=off FC2=on   (应通过)
  RUN=4: DBO=off FC2=off  (应通过)

用法:
  RUN=2 PROMPT_TOKENS=128 NUM_PROMPTS=16 OUTPUT_TOKENS=8 python reproduce_dbo_fc2.py
"""

import gc
import logging
import os
import sys
import time

# ── 基础环境变量 ──────────────────────────────────────────────────────
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HCCL_INTRA_ROCE_ENABLE", "1")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AI_CPU")
os.environ.setdefault("SOC_VERSION", "ascend910b1")
os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# 缓存路径
for d in ["/data/torch_cache", "/data/vllm_cache", "/data/triton_cache",
          "/data/huggingface_cache", "/data/tmp", "/data/torch_inductor_cache",
          "/data/vllm_compile_cache"]:
    os.makedirs(d, exist_ok=True)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/torch_cache")
os.environ.setdefault("VLLM_CACHE_DIR", "/data/vllm_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/triton_cache")
os.environ.setdefault("HF_HOME", "/data/huggingface_cache")
os.environ.setdefault("TMPDIR", "/data/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/torch_inductor_cache")
os.environ.setdefault("VLLM_COMPILE_CACHE_PATH", "/data/vllm_compile_cache")

# ── DBO 和 FlashComm 控制 ──────────────────────────────────────────────

RUN = int(os.environ.get("RUN", "2"))

if RUN == 1:
    # DBO=on, FC2=off → 应通过
    os.environ["VLLM_ASCEND_ENABLE_DBO"] = "1"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
    os.environ["VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"] = "0"
    TAG = "DBO_on_FC2_off"
elif RUN == 2:
    # DBO=on, FC2=on → 预期复现 AICPU 错误
    os.environ["VLLM_ASCEND_ENABLE_DBO"] = "1"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
    os.environ["VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"] = "1"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"] = "0"
    TAG = "DBO_on_FC2_on"
elif RUN == 3:
    # DBO=off, FC2=on → 应通过
    os.environ["VLLM_ASCEND_ENABLE_DBO"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
    os.environ["VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"] = "1"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"] = "0"
    TAG = "DBO_off_FC2_on"
elif RUN == 4:
    # DBO=off, FC2=off → 应通过
    os.environ["VLLM_ASCEND_ENABLE_DBO"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
    os.environ["VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"] = "0"
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"] = "0"
    TAG = "DBO_off_FC2_off"
else:
    print(f"Unknown RUN={RUN}, must be 1-4", file=sys.stderr)
    sys.exit(1)

# ── 日志级别：DEBUG 可看到 should_ubatch, plog 等关键信息 ─────────────────
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

# ── 测试参数 ──────────────────────────────────────────────────────────
MODEL = os.environ.get("MODEL", "/data/models/DeepSeek-V2-Lite-Chat")
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "128"))
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "16"))
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "8"))
DBO_THRESHOLD = int(os.environ.get("DBO_THRESHOLD", "64"))

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def make_prompts(num: int, tokens_each: int) -> list[str]:
    base = "The quick brown fox jumps over the lazy dog. " * (tokens_each // 9 + 1)
    return [base[:tokens_each * 4]] * num


def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()


def main():
    logger.info("=" * 70)
    logger.info("  Reproduce: DBO + FlashComm2 AICPU AllToAll")
    logger.info("  RUN              : %d (%s)", RUN, TAG)
    logger.info("  Model            : %s", MODEL)
    logger.info("  PROMPT_TOKENS    : %d per prompt", PROMPT_TOKENS)
    logger.info("  NUM_PROMPTS      : %d", NUM_PROMPTS)
    logger.info("  OUTPUT_TOKENS    : %d", OUTPUT_TOKENS)
    logger.info("  DBO_THRESHOLD    : %d", DBO_THRESHOLD)
    logger.info("  Total ~input tok : %d (threshold=%d)",
                NUM_PROMPTS * PROMPT_TOKENS, DBO_THRESHOLD)
    logger.info("  Env:")
    logger.info("    VLLM_ASCEND_ENABLE_DBO              = %s",
                os.environ.get("VLLM_ASCEND_ENABLE_DBO"))
    logger.info("    VLLM_ASCEND_ENABLE_FLASHCOMM1       = %s",
                os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1"))
    logger.info("    VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE = %s",
                os.environ.get("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE"))
    logger.info("    VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED = %s",
                os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED"))
    logger.info("    HCCL_OP_EXPANSION_MODE              = %s",
                os.environ.get("HCCL_OP_EXPANSION_MODE"))
    logger.info("=" * 70)

    prompts = make_prompts(NUM_PROMPTS, PROMPT_TOKENS)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_TOKENS,
    )

    enable_dbo = os.environ.get("VLLM_ASCEND_ENABLE_DBO") == "1"

    try:
        llm = LLM(
            model=MODEL,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            max_model_len=2048,
            enable_expert_parallel=True,
            enable_dbo=enable_dbo,
            dbo_prefill_token_threshold=DBO_THRESHOLD if enable_dbo else 512,
            dbo_decode_token_threshold=max(1, DBO_THRESHOLD // 16) if enable_dbo else 32,
            all2all_backend="deepep_low_latency" if enable_dbo else "allgather_reducescatter",
        )

        # warmup
        logger.info("[%s] Warmup...", TAG)
        try:
            _ = llm.generate(prompts[:2], SamplingParams(temperature=0.0, max_tokens=4))
            logger.info("[%s] Warmup OK", TAG)
        except Exception as e:
            logger.error("[%s] Warmup FAILED: %s: %s", TAG, type(e).__name__, e)
            logger.error("[%s] This is the AICPU error reproduction!", TAG)

        # 正式运行
        logger.info("[%s] Timed run...", TAG)
        t0 = time.perf_counter()
        try:
            outputs = llm.generate(prompts, sampling_params)
            elapsed = time.perf_counter() - t0
            logger.info("[%s] Timed run OK. Elapsed: %.3f s", TAG, elapsed)
            for i, out in enumerate(outputs):
                logger.info("[%s]   [%d] %r", TAG, i, out.outputs[0].text[:60])
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error("[%s] Timed run FAILED after %.3f s: %s: %s",
                         TAG, elapsed, type(e).__name__, e)
            import traceback
            traceback.print_exc()

        del llm
        clean_up()

    except Exception as e:
        logger.error("[%s] LLM init FAILED: %s: %s", TAG, type(e).__name__, e)
        import traceback
        traceback.print_exc()
        return 1

    logger.info("[%s] Done.", TAG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
