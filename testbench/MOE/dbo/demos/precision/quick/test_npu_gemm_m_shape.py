#!/usr/bin/env python3
"""Diagnose M-shape dependence of the Ascend BF16 linear/GEMM path.

This is deliberately a diagnostic, rather than a CI assertion: a non-zero
full-vs-split difference reproduces the current issue, while zero means the
underlying kernel may have become batch invariant.  It uses the same logical
dimensions as DeepSeek-V2-Lite q_proj by default.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu  # noqa: F401  # Registers the NPU backend with PyTorch.


def _stats(full: torch.Tensor, split: torch.Tensor) -> dict[str, float | int]:
    diff = (full.float() - split.float()).abs()
    return {
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "nonzero": int((diff != 0).sum().cpu()),
        "numel": diff.numel(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2050)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=3072)
    parser.add_argument("--split", type=int, default=1025)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("NPU is required for this diagnostic")
    if not 0 < args.split < args.m:
        raise ValueError("split must be strictly between 0 and m")

    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    x = torch.randn((args.m, args.k), device="npu", dtype=torch.bfloat16)
    weight = torch.randn((args.n, args.k), device="npu", dtype=torch.bfloat16)

    full = torch.nn.functional.linear(x, weight)
    split = torch.cat(
        [
            torch.nn.functional.linear(x[: args.split], weight),
            torch.nn.functional.linear(x[args.split :], weight),
        ],
        dim=0,
    )
    torch.npu.synchronize()

    result = {
        "device": str(x.device),
        "dtype": str(x.dtype),
        "shape": {"full": [args.m, args.k, args.n], "split": [args.split, args.m - args.split]},
        "stats": _stats(full, split),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
