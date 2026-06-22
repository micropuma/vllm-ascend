#!/usr/bin/env python3
"""Quick check that DBO infrastructure imports correctly after migration."""
import sys

# Check 1: Runtime primitives
from vllm_ascend.worker.ubatching import UBatchEventKey, AscendUBatchContext, dbo_record_current_stream
print(f"ubatching OK: UBatchEventKey={list(UBatchEventKey)}")

# Check 2: Utility functions
from vllm_ascend.worker.ubatch_utils import check_enable_ubatch, maybe_create_ubatch_slices
print("ubatch_utils OK")

# Check 3: Template base
from vllm_ascend.dbo.overlap_templates.base import UbatchOverlapBaseTemplate
print(f"base template OK: {UbatchOverlapBaseTemplate.__name__}")

# Check 4: Template selector
from vllm_ascend.dbo.utils import select_dbo_templates
print(f"dbo/utils OK: {select_dbo_templates.__name__}")

# Check 5: Forward context
from vllm_ascend.ascend_forward_context import create_ascend_forward_context, set_ascend_forward_context
print(f"forward_context OK: {create_ascend_forward_context.__name__}")

# Check 6: Stream utilities
from vllm_ascend.utils import dbo_current_stream, dbo_set_stream
print(f"utils OK: {dbo_current_stream.__name__}, {dbo_set_stream.__name__}")

# Check 7: Env vars
from vllm_ascend.envs import VLLM_ASCEND_DBO_COMM_AIC_NUM
print(f"envs OK")

# Check 8: AscendUBatchWrapper
from vllm_ascend.worker.npu_ubatch_wrapper import AscendUBatchWrapper
print(f"npu_ubatch_wrapper OK: {AscendUBatchWrapper.__name__}")

# Check 9: DeepSeek template
from vllm_ascend.dbo.overlap_templates.deepseek import DeepseekAllgatherTemplate, DeepseekAlltoallTemplate
print(f"deepseek templates OK")

print("\nAll DBO imports verified successfully!")
