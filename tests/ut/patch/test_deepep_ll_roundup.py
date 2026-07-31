from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

import vllm_ascend.patch.worker.patch_deepep_ll_config  # noqa: F401


def test_ascend_does_not_select_cuda_deepep_ll_kernels() -> None:
    config = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=2, ep_size=2,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1,
        use_ep=True, all2all_backend="deepep_low_latency", enable_eplb=False,
    )
    assert not config.use_deepep_ll_kernels
