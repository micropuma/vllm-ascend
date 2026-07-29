from unittest.mock import patch

import torch

from vllm_ascend.ops.mla import _resolve_mla_forward_inputs
from vllm_ascend.ops import register_custom_ops


@patch("vllm_ascend.ops.register_custom_ops.get_forward_context", return_value=object())
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_world_size", return_value=2)
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_rank", return_value=0)
def test_maybe_chunk_residual_pads_to_local_shard_rank0(*_mocks):
    x = torch.randn(5, 16)
    residual = torch.randn(8, 16)

    chunked = register_custom_ops._maybe_chunk_residual_impl(x, residual)

    assert chunked.shape == x.shape
    torch.testing.assert_close(chunked, residual[:5])


@patch("vllm_ascend.ops.register_custom_ops.get_forward_context", return_value=object())
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_world_size", return_value=2)
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_rank", return_value=1)
def test_maybe_chunk_residual_pads_to_local_shard_rank1(*_mocks):
    x = torch.randn(5, 16)
    residual = torch.randn(8, 16)

    chunked = register_custom_ops._maybe_chunk_residual_impl(x, residual)

    assert chunked.shape == x.shape
    torch.testing.assert_close(chunked[:3], residual[5:8])
    torch.testing.assert_close(chunked[3:], torch.zeros_like(chunked[3:]))


@patch("vllm_ascend.ops.register_custom_ops.get_forward_context", return_value=object())
def test_maybe_chunk_residual_keeps_matching_shape(*_mocks):
    x = torch.randn(8, 16)
    residual = torch.randn(8, 16)

    chunked = register_custom_ops._maybe_chunk_residual_impl(x, residual)

    assert chunked is residual


@patch("vllm_ascend.ops.register_custom_ops._FLASH_COMM_V1_SNAPSHOT", False)
@patch("vllm_ascend.ops.register_custom_ops.enable_sp_by_pass", return_value=True)
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_world_size", return_value=2)
def test_maybe_pad_and_reduce_fake_keeps_non_ep_shape(*_mocks):
    x = torch.empty(5, 16)

    output = register_custom_ops._maybe_pad_and_reduce_fake(x, is_ep_comm=False)

    assert output.shape == x.shape


@patch("vllm_ascend.ops.register_custom_ops._FLASH_COMM_V1_SNAPSHOT", False)
@patch("vllm_ascend.ops.register_custom_ops.enable_sp_by_pass", return_value=True)
@patch("vllm_ascend.ops.register_custom_ops.get_tensor_model_parallel_world_size", return_value=2)
def test_maybe_pad_and_reduce_fake_reduces_ep_shape(*_mocks):
    x = torch.empty(5, 16)

    output = register_custom_ops._maybe_pad_and_reduce_fake(x, is_ep_comm=True)

    assert output.shape == (3, 16)


def test_resolve_mla_forward_inputs_keeps_non_vl_output_global():
    hidden_states = torch.randn(4096, 16)

    resolved_hidden_states, output_tokens, need_gather_q_kv = _resolve_mla_forward_inputs(
        hidden_states, flash_comm_v1_enabled=True, tp_size=2, is_vl_first_layer=False
    )

    assert resolved_hidden_states.shape == (4096, 16)
    assert output_tokens == 4096
    assert need_gather_q_kv is True


def test_resolve_mla_forward_inputs_keeps_vl_first_layer_output_local():
    hidden_states = torch.randn(8, 16)

    resolved_hidden_states, output_tokens, need_gather_q_kv = _resolve_mla_forward_inputs(
        hidden_states, flash_comm_v1_enabled=True, tp_size=2, is_vl_first_layer=True
    )

    assert resolved_hidden_states.shape == (8, 16)
    assert output_tokens == 4
    assert need_gather_q_kv is False


def test_resolve_mla_forward_inputs_disables_gather_without_flashcomm():
    hidden_states = torch.randn(4096, 16)

    resolved_hidden_states, output_tokens, need_gather_q_kv = _resolve_mla_forward_inputs(
        hidden_states, flash_comm_v1_enabled=False, tp_size=2, is_vl_first_layer=False
    )

    assert resolved_hidden_states.shape == (4096, 16)
    assert output_tokens == 4096
    assert need_gather_q_kv is False
