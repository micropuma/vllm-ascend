#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import regex as re

from vllm import LLM, SamplingParams

DEFAULT_PROMPT = "Write one short sentence about KV cache."
BYTE_SCALES = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}


def parse_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([KMGTP]?)(?:i?B?)?\s*", value)
    if not match:
        raise ValueError(f"Unsupported byte value: {value!r}")
    return int(match.group(1)) * BYTE_SCALES[match.group(2).upper()]


def cleanup_llm(llm) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(llm_engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del llm
    gc.collect()
    try:
        import torch

        torch.npu.empty_cache()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one offline vLLM generation.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--kv-cache-memory-bytes", default="8G")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=parse_bytes(args.kv_cache_memory_bytes),
        max_model_len=args.max_model_len,
    )
    try:
        outputs = llm.generate(
            [args.prompt],
            SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            ),
            use_tqdm=False,
        )
        generated_text = outputs[0].outputs[0].text
    finally:
        cleanup_llm(llm)

    print("offline generation finished")
    print(f"model={args.model}")
    print(f"dtype={args.dtype}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"kv_cache_memory_bytes={args.kv_cache_memory_bytes}")
    print(f"max_model_len={args.max_model_len}")
    print(f"generated_text={generated_text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
