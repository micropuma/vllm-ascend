# DBO compiled full-graph and FlashComm2/AIV validation

Date: 2026-07-03

## Scope

This note records the follow-up validation after the compiled DBO shape fix.
The code change and the earlier `cudagraph_capture_sizes=[16]` results are
documented in `dbo-flashcomm1-compiled-fix-log.md`.

## Remaining Python forward-context shape reads

Static inspection confirms that Step 3 did not eliminate all compiled-graph
shape values read from Python forward context.

MLA:

- `vllm_ascend/attention/mla_v1.py:1676-1677` passes
  `forward_context.num_tokens` to post-attention unpadding.
- `vllm_ascend/attention/mla_v1.py:1755` uses `_EXTRA_CTX.num_tokens` when
  forming the o-projection input shape.
- `vllm_ascend/ops/mla.py:164` reads
  `_EXTRA_CTX.flash_comm_v1_enabled`. This is a configuration value rather
  than a dynamic shape, but it is still Python context consumed in the op.

MoE prepare/finalize:

- `vllm_ascend/ops/prepare_finalize.py:286` reads
  `_EXTRA_CTX.padded_num_tokens`.
- `vllm_ascend/ops/prepare_finalize.py:399` reads
  `forward_context.num_tokens`.
- `vllm_ascend/ops/prepare_finalize.py:580` forms a shape from
  `forward_context.num_tokens + forward_context.pad_size`.

Conclusion: the constant-capture risk remains real for MLA and MoE. The
current Step 3 fix only makes dense row-parallel reduce-scatter padding derive
from the runtime tensor shape. Each remaining read still needs a dynamic-shape
regression where different logical token counts reuse one compiled graph.

## Default complete graph capture: FlashComm1 + DBO

Configuration:

```text
TP=2
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
VLLM_ASCEND_ENABLE_DBO=1
HCCL_OP_EXPANSION_MODE=AI_CPU
```

No `cudagraph_capture_sizes` override was supplied. Default complete graph
capture finished, the API health check returned HTTP 200, and the full server
test script completed:

```text
random input length: 4096
random output length: 16
requests: 500
maximum concurrency: 96
successful / failed: 500 / 0
benchmark duration: 67.04 s
request throughput: 7.46 req/s
mean TTFT: 5093.26 ms
median TTFT: 4826.56 ms
P99 TTFT: 11733.46 ms
```

Artifacts:

- Server: `/data/workspace/logs/deepseek-v2-dbo-fc1-default-full-server.log`
- Test: `/data/workspace/logs/deepseek-v2-dbo-fc1-default-full-test.log`
- JSON:
  `/data/workspace/vllm-ascend/testbench/MOE/dbo/results/fc1_dbo_default_full_in4096_out16_np500_c96.json`

## Default complete graph capture: FlashComm1 + FlashComm2 + AIV + DBO

For TP=2, FlashComm2 parallel size `1` is valid; size `2` is rejected because
the FlashComm2 o-projection TP size must be smaller than global TP size.

Configuration:

```text
TP=2
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1
VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
VLLM_ASCEND_ENABLE_DBO=1
HCCL_OP_EXPANSION_MODE=AIV
```

No `cudagraph_capture_sizes` override was supplied. Default complete graph
capture finished, the API health check returned HTTP 200, and the same full
server test script completed:

```text
random input length: 4096
random output length: 16
requests: 500
maximum concurrency: 96
successful / failed: 500 / 0
benchmark duration: 77.15 s
request throughput: 6.48 req/s
mean TTFT: 5897.29 ms
median TTFT: 5578.79 ms
P99 TTFT: 13674.28 ms
```

No `ERROR`, traceback, assertion, or EngineCore failure was present in the
server log. The only matched warnings were CPU affinity binding warnings.

Artifacts:

- Server:
  `/data/workspace/logs/deepseek-v2-dbo-fc1-fc2-aiv-default-full-server.log`
- Test:
  `/data/workspace/logs/deepseek-v2-dbo-fc1-fc2-aiv-default-full-test.log`
- JSON:
  `/data/workspace/vllm-ascend/testbench/MOE/dbo/results/fc1_fc2_aiv_dbo_default_full_in4096_out16_np500_c96.json`

## Interpretation

The combined FlashComm1 + FlashComm2 + AIV + DBO path is functionally usable
for this DeepSeek-V2-Lite TP=2 workload under default complete graph capture.
It is not a performance win in this mixed server benchmark:

- request throughput decreased from 7.46 to 6.48 req/s;
- mean TTFT increased from 5093.26 to 5897.29 ms;
- P99 TTFT increased from 11733.46 to 13674.28 ms.

This comparison is one run per configuration and is a functional validation,
not a statistically controlled performance conclusion.

## Remaining validation

1. Replace or tensorize the dynamic MLA/MoE shape reads listed above.
2. Add a regression that reuses a compiled graph across varying odd/even
   logical token counts and validates MLA unpadding, MoE prepare/finalize, and
   row-parallel collectives independently.
3. Fix the DBO exception protocol so a failed ubatch cancels its peer and the
   main thread raises the original exception instead of waiting in `join()`.
4. Repeat performance comparisons with multiple runs and separate prefill-only
   and decode-only workloads before making an optimization claim.
