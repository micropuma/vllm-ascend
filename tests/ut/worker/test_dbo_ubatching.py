import pytest

from vllm_ascend.worker.npu_ubatch_wrapper import validate_dbo_num_ubatches
from vllm_ascend.worker.ubatching import DBO_NUM_UBATCHES


def test_dbo_accepts_two_microbatches():
    validate_dbo_num_ubatches(DBO_NUM_UBATCHES)


@pytest.mark.parametrize("num_ubatches", [1, 3])
def test_dbo_rejects_unsupported_microbatch_counts(num_ubatches):
    with pytest.raises(ValueError, match="supports exactly 2 microbatches"):
        validate_dbo_num_ubatches(num_ubatches)
