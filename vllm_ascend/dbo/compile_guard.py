"""Compile-safe wrappers for DBO instrumentation hooks.

All DBO hook calls must be wrapped with torch.compiler.disable() so that Dynamo
does not attempt to trace through the dynamically-assigned template hooks.
Without this guard, Dynamo fails with "failed to find name in frame builtins"
when it encounters ``forward_context.dbo_template.dbo_*_hook(...)`` because the
template object is assigned at runtime outside the compiled forward graph.
"""

import torch


@torch.compiler.disable()
def _dbo_call_linear_column_hook(forward_context, is_record: bool) -> None:
    forward_context.dbo_template.dbo_linear_column_hook(is_record=is_record)


@torch.compiler.disable()
def _dbo_call_linear_row_hook(forward_context, is_record: bool) -> None:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=is_record)


@torch.compiler.disable()
def _dbo_call_moe_prepare_hook(forward_context, is_record: bool) -> None:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=is_record)


@torch.compiler.disable()
def _dbo_call_moe_finalize_hook(forward_context, is_record: bool) -> None:
    forward_context.dbo_template.dbo_moe_finalize_hook(is_record=is_record)


@torch.compiler.disable()
def _dbo_call_mla_preprocess_hook(forward_context, is_record: bool) -> None:
    forward_context.dbo_template.dbo_mla_preprocess_hook(is_record=is_record)
