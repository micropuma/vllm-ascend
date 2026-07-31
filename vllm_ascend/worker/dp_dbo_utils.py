"""Pure DP+DBO coordination rules used after the distributed allreduce."""

import torch

from vllm.config import CUDAGraphMode

from vllm_ascend.worker.ubatch_utils import is_last_ubatch_empty


def resolve_dbo_dp_metadata(
    logical_tokens: torch.Tensor,
    physical_tokens: torch.Tensor,
    dbo_candidates: torch.Tensor,
    cudagraph_modes: torch.Tensor,
    num_ubatches: int = 2,
) -> tuple[bool, torch.Tensor, torch.Tensor, CUDAGraphMode]:
    """Resolve globally consistent DBO and DP padding decisions.

    ``physical_tokens`` is the local pre-existing padded shape.  The returned
    physical vector is used by collectives; the logical vector is never padded
    and is used only to reconstruct valid outputs.
    """
    logical_tokens = logical_tokens.to(device="cpu", dtype=torch.int32)
    physical_tokens = physical_tokens.to(device="cpu", dtype=torch.int32)
    dbo_candidates = dbo_candidates.to(device="cpu", dtype=torch.int32)
    cudagraph_modes = cudagraph_modes.to(device="cpu", dtype=torch.int32)

    synced_mode = CUDAGraphMode(int(cudagraph_modes.min().item()))
    should_ubatch = bool(torch.all(dbo_candidates == 1).item())
    if should_ubatch:
        # A short rank must not receive an empty second ubatch. In that case
        # all ranks execute the ordinary single-batch path.
        should_ubatch = not is_last_ubatch_empty(
            int(logical_tokens.min().item()),
            int(physical_tokens.max().item()),
            num_ubatches,
        )

    if synced_mode != CUDAGraphMode.NONE or should_ubatch:
        # each rank must use the same physical shape for collectives, so pad to the max
        physical_tokens = torch.full_like(physical_tokens, int(physical_tokens.max().item()))
    return should_ubatch, physical_tokens, logical_tokens, synced_mode
