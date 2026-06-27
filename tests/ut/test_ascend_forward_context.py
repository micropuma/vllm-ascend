from types import SimpleNamespace

import pytest

from vllm_ascend.ascend_forward_context import _get_ubatch_num_tokens
from vllm_ascend.ops.register_custom_ops import _get_reduce_scatter_num_tokens


@pytest.mark.parametrize(
    ("attn_metadata", "fallback", "expected"),
    [
        ({"layer": SimpleNamespace(num_actual_tokens=2049)}, 2051, 2049),
        (SimpleNamespace(num_actual_tokens=1025), 1026, 1025),
        (None, 2051, 2051),
        ({}, 2051, 2051),
    ],
)
def test_get_ubatch_num_tokens(attn_metadata, fallback, expected):
    assert _get_ubatch_num_tokens(attn_metadata, fallback) == expected


@pytest.mark.parametrize(
    ("num_tokens", "tp_size", "expected"),
    [(2049, 2, 1025), (2050, 2, 1025), (2051, 2, 1026), (3, 4, 1)],
)
def test_get_reduce_scatter_num_tokens(num_tokens, tp_size, expected):
    assert _get_reduce_scatter_num_tokens(num_tokens, tp_size) == expected
