# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep Ascend's ``deepep_low_latency`` backend out of CUDA DeepEP paths."""

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig


@property
def _use_deepep_ll_kernels_on_ascend(self: FusedMoEParallelConfig) -> bool:
    """CUDA DeepEP-LL kernels are unavailable; Ascend provides this backend."""
    return False


FusedMoEParallelConfig.use_deepep_ll_kernels = _use_deepep_ll_kernels_on_ascend
