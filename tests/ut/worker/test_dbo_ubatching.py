from types import SimpleNamespace

import pytest
import torch
from vllm.forward_context import DPMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice

from vllm_ascend.worker.npu_ubatch_wrapper import (
    make_ubatch_dp_metadata,
    validate_dbo_num_ubatches,
)
from vllm_ascend.worker.ubatching import DBO_NUM_UBATCHES


def test_dbo_accepts_two_microbatches():
    validate_dbo_num_ubatches(DBO_NUM_UBATCHES)


@pytest.mark.parametrize("num_ubatches", [1, 3])
def test_dbo_rejects_unsupported_microbatch_counts(num_ubatches):
    with pytest.raises(ValueError, match="supports exactly 2 microbatches"):
        validate_dbo_num_ubatches(num_ubatches)


def test_dbo_dp_metadata_preserves_peer_ubatch_sizes(monkeypatch):
    parallel_config = SimpleNamespace(data_parallel_rank=0)
    vllm_config = SimpleNamespace(parallel_config=parallel_config)
    parent_dp_metadata = SimpleNamespace(
        num_tokens_across_dp_cpu=torch.tensor([4100, 4400], dtype=torch.int32)
    )
    ubatch_slices = [
        UBatchSlice(slice(0, 1), slice(0, 2050)),
        UBatchSlice(slice(1, 2), slice(2050, 4100)),
    ]
    captured = []

    def make(_parallel_config, num_tokens, num_tokens_across_dp_cpu):
        captured.append((num_tokens, num_tokens_across_dp_cpu.tolist()))
        return object()

    monkeypatch.setattr(DPMetadata, "make", staticmethod(make))

    result = make_ubatch_dp_metadata(vllm_config, ubatch_slices, parent_dp_metadata)

    assert len(result) == 2
    assert captured == [(2050, [2050, 2200]), (2050, [2050, 2200])]
