# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""DBO dispatch via ``torch.library.custom_op`` — fullgraph-compatible.

Every DBO injection point is a registered custom op.  Dynamo treats each
``call_function[torch.ops.vllm_ascend.dbo_*]`` node as an opaque ATen-style
op in the FX graph — it does **not** trigger a graph break and does **not**
trace inside.  The fake impl provides compile-time shape inference; the real
impl runs only at runtime and is free to call ``get_forward_context()``.

Non-DBO path cost: one extra custom-op invocation whose real impl checks
``dbo_enabled == False`` and immediately delegates to the standard
``torch.ops.vllm.*`` call — negligible overhead.
"""

from typing import Optional

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)

from vllm_ascend.dbo.snapshot import _FLASH_COMM_V1_SNAPSHOT
import sys; _dbo_log = sys.stderr


def _safe_sp_enabled() -> bool:
    """Check if sequence parallelism is enabled, safe for fake-impl context."""
    if _FLASH_COMM_V1_SNAPSHOT:
        return True
    try:
        from vllm_ascend.utils import enable_sp_by_pass
        return enable_sp_by_pass()
    except Exception:
        return False


def _safe_tp_size() -> int:
    """Get TP world size, safe for fake-impl context."""
    try:
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


# Re-export snapshot so callers can set it without importing register_custom_ops
__all__ = [
    "dbo_column_allgather_mlp",
    "dbo_column_allgather_sp",
    "dbo_row_allreduce",
    "dbo_moe_prepare_allgather",
    "dbo_moe_finalize_allgather",
    "dbo_mla_preprocess",
]


# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════


def _get_fc_attr(name: str, default=None):
    """Read *name* from the current forward context without crashing."""
    try:
        from vllm.forward_context import get_forward_context

        return getattr(get_forward_context(), name, default)
    except AssertionError:
        return default


def _noalias(result: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
    """Ensure *result* does not alias any of *inputs*."""
    for inp in inputs:
        if result is inp:
            return result.clone()
    return result


def _call_dbo_hook(hook_name: str, is_record: bool):
    """Invoke a named DBO template hook (``dbo_linear_column_hook``, etc.)."""
    fc = _get_fc_attr("dbo_template")
    if fc is None:
        return
    hook = getattr(fc, hook_name, None)
    if hook is not None:
        hook(is_record=is_record)


# ═══════════════════════════════════════════════════════════════════════
# 1.  Column Parallel AllGather  (MLPColumnParallelOp)
# ═══════════════════════════════════════════════════════════════════════

@torch.library.custom_op("vllm_ascend::dbo_column_allgather_mlp", mutates_args=())
def dbo_column_allgather_mlp(
    input_: torch.Tensor,
) -> torch.Tensor:
    """Real impl — runs at runtime only, safe to call ``get_forward_context()``."""
    fc = _get_fc_attr("dbo_enabled", False)
    if fc:
        from vllm_ascend.distributed.parallel_state import get_mlp_tp_group

        _call_dbo_hook("dbo_linear_column_hook", True)
        result = get_mlp_tp_group().all_gather(input_, 0)
        _call_dbo_hook("dbo_linear_column_hook", False)
        return result
    else:
        from vllm_ascend.distributed.parallel_state import get_mlp_tp_group

        return get_mlp_tp_group().all_gather(input_, 0)


@dbo_column_allgather_mlp.register_fake
def _(input_: torch.Tensor) -> torch.Tensor:
    """Fake impl — AllGather expands dim 0 by TP size."""
    tp = _safe_tp_size()
    print("dbo_column_allgather_mlp FAKE: in=%s tp=%s out=%s", input_.shape, tp, (input_.shape[0]*tp, *input_.shape[1:]))
    return torch.empty(
        (input_.shape[0] * tp, *input_.shape[1:]),
        device=input_.device, dtype=input_.dtype,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2.  Column Parallel AllGather  (SequenceColumnParallelOp — FlashComm1)
# ═══════════════════════════════════════════════════════════════════════

@torch.library.custom_op("vllm_ascend::dbo_column_allgather_sp", mutates_args=())
def dbo_column_allgather_sp(
    input_: torch.Tensor,
    need_gather: bool,
) -> torch.Tensor:
    """Real impl — DBO-aware maybe_all_gather_and_maybe_unpad wrapper."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    if fc.dbo_enabled:
        _call_dbo_hook("dbo_linear_column_hook", True)
        if fc.flash_comm_v1_enabled and need_gather:
            input_ = tensor_model_parallel_all_gather(input_, 0)
        _call_dbo_hook("dbo_linear_column_hook", False)
        return torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, do_comm=False, label=need_gather,
        )
    else:
        return _noalias(torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, label=need_gather,
        ), input_)


@dbo_column_allgather_sp.register_fake
def _(input_: torch.Tensor, need_gather: bool) -> torch.Tensor:
    """Fake impl — if FlashComm1 is on AND need_gather, dim-0 expands."""
    print("dbo_column_allgather_sp FAKE: in=%s need_gather=%s sp=%s tp=%s", input_.shape, need_gather, _safe_sp_enabled(), _safe_tp_size())
    if _safe_sp_enabled() and need_gather:
        tp = _safe_tp_size()
        return torch.empty(
            (input_.shape[0] * tp, *input_.shape[1:]),
            device=input_.device,
            dtype=input_.dtype,
        )
    return torch.empty_like(input_)


# ═══════════════════════════════════════════════════════════════════════
# 3.  Row Parallel AllReduce  (SequenceRowParallelOp)
# ═══════════════════════════════════════════════════════════════════════

@torch.library.custom_op("vllm_ascend::dbo_row_allreduce", mutates_args=())
def dbo_row_allreduce(
    output_parallel: torch.Tensor,
) -> torch.Tensor:
    """Real impl — DBO-aware AllReduce."""
    fc = _get_fc_attr("dbo_enabled", False)
    if fc:
        _call_dbo_hook("dbo_linear_row_hook", True)
        result = tensor_model_parallel_all_reduce(output_parallel)
        _call_dbo_hook("dbo_linear_row_hook", False)
        return result
    else:
        return tensor_model_parallel_all_reduce(output_parallel)


@dbo_row_allreduce.register_fake
def _(output_parallel: torch.Tensor) -> torch.Tensor:
    """Fake impl — AllReduce does not change shape."""
    return torch.empty_like(output_parallel)


# ═══════════════════════════════════════════════════════════════════════
# 4.  MoE Prepare (AllGather path)
# ═══════════════════════════════════════════════════════════════════════

@torch.library.custom_op("vllm_ascend::dbo_moe_prepare_allgather", mutates_args=())
def dbo_moe_prepare_allgather(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    pertoken_scale: torch.Tensor,  # empty (shape [0]) tensor when not used
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real impl — DBO-aware MoE prepare (AllGather path)."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    ps = pertoken_scale if pertoken_scale.numel() > 0 else None

    if fc.dbo_enabled:
        _call_dbo_hook("dbo_moe_prepare_hook", True)
        if fc.flash_comm_v1_enabled:
            if fc.dp_metadata is None:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                router_logits = tensor_model_parallel_all_gather(router_logits, 0)
                if ps is not None:
                    ps = tensor_model_parallel_all_gather(ps, 0)
            else:
                from vllm.distributed.parallel_state import get_ep_group

                ep = get_ep_group()
                hidden_states = ep.all_gather(hidden_states, 0)
                router_logits = ep.all_gather(router_logits, 0)
                if ps is not None:
                    ps = ep.all_gather(ps, 0)
            _call_dbo_hook("dbo_moe_prepare_hook", False)

        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, True, True, do_comm=False,
        )
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            router_logits, True, True, do_comm=False,
        )
        if ps is not None:
            ps = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(ps, True, True, do_comm=False)
    else:
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, True, True,
        )
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            router_logits, True, True,
        )
        if ps is not None:
            ps = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(ps, True, True)

    return hidden_states, router_logits, ps if ps is not None else pertoken_scale


@dbo_moe_prepare_allgather.register_fake
def _(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    pertoken_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake impl — shape unchanged by AllGather (pad/unpad within do_comm)."""
    print("dbo_moe_prepare FAKE: hs=%s rl=%s", hidden_states.shape, router_logits.shape)
    return (
        torch.empty_like(hidden_states),
        torch.empty_like(router_logits),
        torch.empty_like(pertoken_scale),
    )


# ═══════════════════════════════════════════════════════════════════════
# 5.  MoE Finalize (AllGather path)
# ═══════════════════════════════════════════════════════════════════════

@torch.library.custom_op("vllm_ascend::dbo_moe_finalize_allgather", mutates_args=())
def dbo_moe_finalize_allgather(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Real impl — DBO-aware MoE finalize (AllGather path)."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    if fc.dbo_enabled:
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
            hidden_states, True, do_comm=False,
        )
        _call_dbo_hook("dbo_moe_finalize_hook", True)
        if fc.flash_comm_v1_enabled:
            if fc.dp_metadata is None:
                hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            else:
                from vllm.distributed.parallel_state import get_ep_group

                hidden_states = get_ep_group().reduce_scatter(hidden_states, 0)
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        _call_dbo_hook("dbo_moe_finalize_hook", False)
        return hidden_states
    else:
        return _noalias(torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True), hidden_states)


@dbo_moe_finalize_allgather.register_fake
def _(hidden_states: torch.Tensor) -> torch.Tensor:
    """Fake impl — FlashComm1 reduce_scatter shrinks dim 0."""
    print("dbo_moe_finalize_allgather FAKE: in=%s sp=%s tp=%s", hidden_states.shape, _safe_sp_enabled(), _safe_tp_size())
    if _safe_sp_enabled():
        tp = _safe_tp_size()
        return torch.empty(
            (hidden_states.shape[0] // tp, *hidden_states.shape[1:]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    return torch.empty_like(hidden_states)


# ═══════════════════════════════════════════════════════════════════════
# 6.  MLA Preprocess
# ═══════════════════════════════════════════════════════════════════════

@torch.library.custom_op("vllm_ascend::dbo_mla_preprocess", mutates_args=())
def dbo_mla_preprocess(
    q_c: torch.Tensor,
    kv_no_split: torch.Tensor,
    need_gather_q_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real impl — DBO-aware MLA preprocess AllGather."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    if fc.dbo_enabled:
        _call_dbo_hook("dbo_mla_preprocess_hook", True)
        if fc.flash_comm_v1_enabled:
            q_c = tensor_model_parallel_all_gather(q_c.contiguous(), 0)
            kv_no_split = tensor_model_parallel_all_gather(kv_no_split.contiguous(), 0)
        _call_dbo_hook("dbo_mla_preprocess_hook", False)
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c, need_gather_q_kv, do_comm=False,
        )
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split, need_gather_q_kv, do_comm=False,
        )
    else:
        q_c = _noalias(torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c.contiguous(), need_gather_q_kv,
        ), q_c)
        kv_no_split = _noalias(torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split.contiguous(), need_gather_q_kv,
        ), kv_no_split)
    return q_c, kv_no_split


@dbo_mla_preprocess.register_fake
def _(
    q_c: torch.Tensor,
    kv_no_split: torch.Tensor,
    need_gather_q_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake impl — FlashComm1 AllGather expands dim 0."""
    print("dbo_mla_preprocess FAKE: qc=%s kv=%s ng=%s sp=%s tp=%s", q_c.shape, kv_no_split.shape, need_gather_q_kv, _safe_sp_enabled(), _safe_tp_size())
    if _safe_sp_enabled() and need_gather_q_kv:
        tp = _safe_tp_size()
        return (
            torch.empty((q_c.shape[0] * tp, *q_c.shape[1:]), device=q_c.device, dtype=q_c.dtype),
            torch.empty(
                (kv_no_split.shape[0] * tp, *kv_no_split.shape[1:]),
                device=kv_no_split.device,
                dtype=kv_no_split.dtype,
            ),
        )
    return torch.empty_like(q_c), torch.empty_like(kv_no_split)


# ═══════════════════════════════════════════════════════════════════════
# 7.  Micro hook custom ops  (for OProj / A2A paths)
# ═══════════════════════════════════════════════════════════════════════
#
# These are the smallest possible custom ops: they read forward_context,
# optionally call a DBO hook, and return immediately.  They exist solely to
# keep get_forward_context() / hook calls out of the compiled FX graph while
# staying fullgraph-compatible.

@torch.library.custom_op("vllm_ascend::dbo_linear_row_record", mutates_args=())
def dbo_linear_row_record() -> None:
    """Record linear-row DBO hook (real impl)."""
    _call_dbo_hook("dbo_linear_row_hook", True)
    return None


@dbo_linear_row_record.register_fake
def _() -> None:
    return None


@torch.library.custom_op("vllm_ascend::dbo_linear_row_wait", mutates_args=())
def dbo_linear_row_wait() -> None:
    """Wait linear-row DBO hook (real impl)."""
    _call_dbo_hook("dbo_linear_row_hook", False)
    return None


@dbo_linear_row_wait.register_fake
def _() -> None:
    return None


@torch.library.custom_op("vllm_ascend::dbo_moe_prepare_record", mutates_args=())
def dbo_moe_prepare_record() -> None:
    """Record MoE-prepare DBO hook (real impl)."""
    _call_dbo_hook("dbo_moe_prepare_hook", True)
    return None


@dbo_moe_prepare_record.register_fake
def _() -> None:
    return None


@torch.library.custom_op("vllm_ascend::dbo_moe_prepare_wait", mutates_args=())
def dbo_moe_prepare_wait() -> None:
    """Wait MoE-prepare DBO hook (real impl)."""
    _call_dbo_hook("dbo_moe_prepare_hook", False)
    return None


@dbo_moe_prepare_wait.register_fake
def _() -> None:
    return None


@torch.library.custom_op("vllm_ascend::dbo_moe_finalize_record", mutates_args=())
def dbo_moe_finalize_record() -> None:
    """Record MoE-finalize DBO hook (real impl)."""
    _call_dbo_hook("dbo_moe_finalize_hook", True)
    return None


@dbo_moe_finalize_record.register_fake
def _() -> None:
    return None


@torch.library.custom_op("vllm_ascend::dbo_moe_finalize_wait", mutates_args=())
def dbo_moe_finalize_wait() -> None:
    """Wait MoE-finalize DBO hook (real impl)."""
    _call_dbo_hook("dbo_moe_finalize_hook", False)
    return None


@dbo_moe_finalize_wait.register_fake
def _() -> None:
    return None
