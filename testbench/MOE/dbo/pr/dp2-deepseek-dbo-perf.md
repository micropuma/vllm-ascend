# TP1 DP2 DeepSeek DBO Validation

## Scope

DeepSeek-V2-Lite-Chat on two Ascend 910B3 cards with `TP=1`, `DP=2`, `EP=2`,
`HCCL_OP_EXPANSION_MODE=AI_CPU`, and FlashComm disabled. The DP launcher and
precision gate are:

```text
testbench/MOE/dbo/demos/deepseek-v2-dbo-server-dp.sh
testbench/MOE/dbo/demos/precision/quick/test_dbo_dp_precision.sh
```

Default compile/ACL-graph is the performance target. Eager isolation is
diagnostic only.

## Startup Fix

The first DBO startup failed because upstream `all2all_utils.py` selected the
CUDA-only `DeepEPLLPrepareAndFinalize` metadata path from the generic
`deepep_low_latency` name. This is not a missing import: that class must remain
unavailable on Ascend. The worker patch makes
`FusedMoEParallelConfig.use_deepep_ll_kernels` false, leaving Ascend's own
communication implementation active. Focused tests pass.

## DP DBO Metadata Bug

`AscendUBatchWrapper` previously constructed each ubatch's DP metadata as
`[local_tokens] * dp_size`. For an uneven wave such as
`num_tokens_across_dp=[4100, 4400]`, rank 0 advertised `[2050,2050]` and rank 1
`[2200,2200]` instead of both using the peer vector `[2050,2200]`. This
corrupts EP MoE communication shape/count metadata. The wrapper now derives
peer ubatch sizes from the synchronized parent vector and asserts the
equal-index split contract. A unit regression test covers this case.

## Correctness Evidence and Blocker

The fixed eager run reached DBO on both workers without HTTP 400:

```text
/data/tmp/dbo-dp-20260730/precision-eager-dp-metadata-v2
triggered_workers = [dp0_tp0, dp1_tp0]
```

It still reported output/logprob differences. The A/A control is also
non-stable: two independent `DBO_ENABLED=0` eager launches on the same
32-request workload produced up to `0.05158` sampled-logprob delta and text
differences. The current cross-cold-server exact-output gate therefore cannot
distinguish DP scheduler packing/numerical nondeterminism from a DBO error.
The earlier compile run also had 7/16 HTTP 400 responses caused by non-finite
response values:

```text
/data/tmp/dbo-dp-20260730/precision-compile-fixed-v2
```

That compile result is not a valid performance run.

## Performance Status

No DP plain A/B or DP 20-step profile is claimed yet. The quality gate must be
made scheduler-controlled first: record identical per-request DP assignment
and batch packing, or compare against an A/A distributional tolerance. Then
run three alternating plain `prefill4k` repetitions and separate operator and
host-stack 20-forward-step traces for both DP ranks.

## Next Optimization Work

1. Add a deterministic DP correctness harness with fixed rank assignment and
   an A/A control; retain strict token equality only when batch schedules are
   identical.
2. Trace the first divergent layer under that controlled schedule. Do not add
   DP overlap hooks based only on the current cross-server comparison.
3. Quantify DBO overhead with the metadata fix first, then design a DP-aware
   collective ordering/template change.

## DP Plain A/B (2026-07-31)

The first scheduler-controlled plain comparison was completed on two Ascend
910B3 cards with `TP=1, DP=2, EP=2`, `HCCL_OP_EXPANSION_MODE=AI_CPU`,
FlashComm disabled, default compile/ACL-graph mode, and the only A/B variable
being DBO (`DBO_ENABLED=1` versus `0`). The fixed workload was random chat,
4096 input tokens, 16 output tokens, 200 requests, concurrency 96. Each
configuration ran three repetitions after a 16-request warmup.

Raw artifacts are under:

```text
/data/tmp/dp-dbo-ab-20260731/
```

| Metric | DBO mean +/- std | Baseline mean +/- std | DBO change |
|---|---:|---:|---:|
| Output throughput (tok/s) | 213.3 +/- 85.9 | 204.7 +/- 95.9 | +4.2% |
| Total token throughput (tok/s) | 54,921 +/- 22,119 | 52,703 +/- 24,689 | +4.2% |
| Request throughput (req/s) | 13.33 +/- 5.37 | 12.79 +/- 5.99 | +4.2% |
| Mean TTFT (ms) | 3471 +/- 1562 | 3884 +/- 1527 | -10.6% |
| Mean TPOT (ms) | 297.5 +/- 136.9 | 316.1 +/- 185.9 | -5.9% |
| Mean E2E (ms) | 7933 +/- 3560 | 8626 +/- 4301 | -8.0% |

All six runs completed 200/200 requests. The direction is positive, but the
standard deviation is large and the DBO point estimate is within run-to-run
variation. This is not a stable DP DBO speedup claim yet. The large variation
comes from graph/capture and batch-packing state after server startup; a
follow-up measurement must reuse a warmed server or use repeated alternating
windows after all graph shapes are captured.

### Formal `prefill4k` 500-request A/B

The repository's formal `prefill4k` script was subsequently run with the full
500-request workload: `TP=1, DP=2, EP=2`, 4096 input tokens, 16 output tokens,
concurrency 96, and only DBO changed. Both runs completed 500/500 requests:

| Metric | DBO | Baseline | DBO change |
|---|---:|---:|---:|
| Output throughput | 127.65 tok/s | 117.74 tok/s | **+8.42%** |
| Total token throughput | 32,869.98 tok/s | 30,317.52 tok/s | **+8.42%** |
| Request throughput | 7.98 req/s | 7.36 req/s | **+8.42%** |
| Mean TTFT | 4,243.98 ms | 4,641.78 ms | **-8.57%** |
| Mean TPOT | 509.89 ms | 545.37 ms | **-6.51%** |
| Mean E2E | 11,892.35 ms | 12,822.31 ms | **-7.25%** |

Artifacts:

```text
/data/tmp/dp-dbo-500-20260731/dp_dbo_500_in4096_out16_np500_c96_run1.json
/data/tmp/dp-dbo-500-20260731/dp_base_500_in4096_out16_np500_c96_run1.json
```

This is the correct formal workload result and supports a positive DP+DBO
signal. It is still one 500-request pair, so the result remains provisional
until two more pairs are collected after graph warmup.

## DP DBO Operator Profile (2026-07-31)

An operator-mode profile was collected for both DP ranks with
`PROFILER_MAX_ITERATIONS=20`, `ENFORCE_EAGER=1`, 8 profiled requests, and
concurrency 4. The profile is diagnostic only. Both rank directories contain
valid `step_trace_time.csv` and `trace_view.json`:

```text
/data/tmp/dp-dbo-profile-20260731/operator_eager2/
```

The profile exercised an uneven wave (`[16384, 12348]`) on both workers and
logged `should_ubatch: True`. CANN aggregates were:

| Rank | Stage | Compute | Comm exposed | Comm overlap | Free |
|---|---:|---:|---:|---:|---:|
| DP0 | 27,525.7 ms | 532.6 ms | 2,389.4 ms | 0.0 ms (0%) | 24,603.7 ms (89.4%) |
| DP1 | 26,420.2 ms | 538.6 ms | 1,831.7 ms | 56.1 ms (3.0% of comm) | 24,049.9 ms (91.0%) |

This low-concurrency eager capture is dominated by free/host scheduling gaps,
so it must not be interpreted as steady-state DBO overlap efficiency. It does
show that DP1 has lower communication exposure for the same uneven vector,
while both ranks remain far from a compute/communication-balanced profile.

## Evidence-Based Optimization Priorities

1. **P0 measurement:** repeat the plain A/B after one server warmup phase has
   captured all relevant graph shapes, alternating baseline and DBO windows on
   the same process. Report median and confidence interval in addition to mean.
2. **P0 host/runtime:** collect a host-stack trace with at least 32 concurrent
   4K requests. The current profile's 89--91% free time is a low-load artifact;
   the higher-concurrency trace must determine whether
   `dbo_wait_current_stream_and_yield`, Python handoff, or synchronization is
   responsible for idle gaps.
3. **P1 overlap:** on a representative high-concurrency operator trace,
   correlate exposed AllGather blocks (currently about 1.0--1.5 ms average in
   the eager trace) with DBO hooks before changing record/wait placement.
   Only uncovered blocks above 1 ms should receive a new hook.
4. **P1 DP balance:** retain the peer-vector metadata fix and record per-wave
   DP token vectors. Uneven vectors such as `[16384,12348]` are expected, but
   rank skew above 10% should trigger scheduling/load-balance investigation.
5. **P2 GEMM precision/performance:** the first DP+DBO numerical boundary is
   still layer-0 `q_proj`, matching the known BF16 M-shape sensitivity. Do not
   add padding or change GEMM accumulation until a fixed-input precision gate
   is available.

## DP Host-Stack Profile (2026-07-31)

A separate `host_stack` capture was completed for both ranks using the same
eager, 8-request diagnostic wave. Both CANN outputs parsed successfully:

```text
/data/tmp/dp-dbo-profile-20260731/host_eager/
```

The aggregate rows were:

| Rank | Stage | Compute | Comm exposed | Comm overlap | Free |
|---|---:|---:|---:|---:|---:|
| DP0 | 27,450.4 ms | 906.3 ms | 937.6 ms | 0 ms | 25,606.5 ms (93.3%) |
| DP1 | 27,699.2 ms | 738.2 ms | 4,248.0 ms | 54.2 ms (1.3% of comm) | 22,712.9 ms (82.0%) |

The host-stack trace does not justify a specific Python function change: the
very small diagnostic wave is dominated by frontend/device idle time, and it
does not provide a representative high-concurrency host gap distribution.
It does establish a strong next experiment: repeat `host_stack` with at least
32 concurrent 4K requests, then inspect CPU events around
`dbo_wait_current_stream_and_yield` and the two ubatch handoff events. The
large DP1 communication exposure relative to DP0 also makes rank-dependent
collective timing/load balance a higher priority than operator fusion.

## DP+EP Layer Precision Trace (2026-07-31)

The DP precision gate was rerun with the layer-boundary tracer using the same
`TP=1, DP=2, EP=2` topology and a 16-request, 2048-token concurrent wave. The
baseline and DBO runs both completed HTTP requests; both DBO workers logged
`should_ubatch: True`. The authoritative trace is:

```text
/data/tmp/dbo-dp-layer-trace-20260731T031000Z/divergence.json
```

The first divergence is:

```text
module: DeepseekV2ForCausalLM.model.layers.0.self_attn.q_proj
rank:   0
baseline shape: [3274, 3072]
DBO stitched shape: [3274, 3072]
baseline aggregate mean: -0.024164944887161255
DBO stitched aggregate mean: -0.02190499217367978
baseline abs_max: 21.75
DBO stitched abs_max: 21.75
```

These are reductions over `3274 * 3072` BF16 elements, not elementwise error
statistics. The aggregate mean changes by `0.0022599527134814744` (about 9.35%
relative to the baseline aggregate mean), while both `abs_max` values are
`21.75`. They cannot be interpreted as `mean(abs(baseline - DBO))` or
`max(abs(baseline - DBO))`: this serving trace does not attach a request id to
each module record, and baseline and DBO schedulers can pack different requests
into the same recorded shape. The trace establishes the first *candidate
boundary* (`layer-0 q_proj`), not a strict per-element error magnitude for
DP+EP.

The trace records 1541 baseline and 3146 DBO module boundaries. DBO batches
included `3274`, `16384`, and `7455` tokens, all with equal DP vectors on both
ranks. This rules out the unequal-peer DP metadata bug as the cause of this
run: the metadata fix is still required for uneven DP+EP batches, but it is a
no-op for `[N, N]` vectors.

The result matches the existing TP=2 probe in
`pr-dbo-bf16-gemm-shape-precision.md`: layer-0 MLA `q_proj` is the first
observable mismatch, while the attention-update reference error is only about
`3.6e-7` max. DP+EP therefore does not introduce a new first divergence; it
exposes the same Ascend BF16 GEMM M-shape dependence through DBO splitting.
The TP probe remains the stronger quantitative evidence because it recomputes
the same input tail with the same weights and reports elementwise `maxabs`,
`meanabs`, and nonzero counts. A final DP+EP magnitude claim requires request
identity or a captured identical input tensor for the baseline/DBO pair.

### Correct modification direction

Do not attempt to fix this by changing DP metadata, event ordering, or
`npu_attention_update`. The short-term correctness choices are to bypass DBO
for requests that require baseline-identical logits, or to pad each microbatch
to a fixed GEMM M and remove padding rows before attention, routing, collectives,
and KV updates. Padding must be measured because it may erase the overlap gain.

The durable fix belongs in the Ascend linear/GEMM path: provide an M-shape
invariant BF16 accumulation/tiling path (or a documented higher-precision
accumulation mode) and validate it with the actual q projection dimensions
`[M,2048] x [2048,3072]`, comparing full M against the DBO split. A Python-side
`VLLM_BATCH_INVARIANT` switch is not sufficient until the Ascend NZ/linear path
has passed that A/B.

## Profile Deep Dive and Optimization Hypotheses

The operator trace was inspected at event and kernel level rather than using
the profiler's free-time percentage as a throughput proxy. It confirms that
this run selected **A2/AllGather**, not A3/AlltoAll: on Ascend A2, DeepSeek's
64 experts over EP=2 gives 32 experts per device, which does not satisfy the
current MC2 guard (`<=24` experts/device and EP size `>=16`). The dominant
device families were therefore:

| Rank | GroupedMatmul | MatMulV2 | AllGather (aggregate) | Mean AllGather |
|---|---:|---:|---:|---:|
| DP0 | 226.3 ms / 1,352 calls | 73.5 ms / 3,723 calls | 2,494.9 ms / 1,802 calls | 1.38 ms |
| DP1 | 181.6 ms / 1,300 calls | 79.0 ms / 3,672 calls | 1,761.7 ms / 1,733 calls | 1.02 ms |

The largest AllGather payload is approximately 16.8 MB (`allgather-top1`),
with single-call durations around 0.65--0.86 ms and measured HCCS aggregate
bandwidth around 19--22 GB/s. This makes communication placement worth
optimizing, but it does not justify switching to A3 or changing the transport
without a hardware-valid implementation and an A/B trace.

The A2 template currently records `ATTN_POST` after row linear, then waits on
that same event before column linear and again before MoE prepare. That is a
deliberate dependency chain, not evidence of a missing hook. The safe question
is whether the second wait is redundant for a given layer/stream schedule.
It must be tested by changing one boundary at a time and checking for both
deadlocks and reduced *exposed* AllGather time; event-count reduction alone is
not a performance result.

The CPU trace contains two synchronization points per forward: metadata
`gloo:all_reduce` in `_sync_metadata_across_dp`, followed by a second Gloo
all-reduce for the global `should_ubatch` decision. It also shows allocation of
the packed metadata tensor and scalar reads through `aten::item`. One DP0
all-reduce reached roughly 868 ms in this low-concurrency capture, while most
other calls were much smaller. The outlier is consistent with rank skew or
cold-start synchronization and must not be charged as a steady-state per-step
cost; it does, however, identify a synchronization boundary that can amplify
tail latency. Do not remove either collective until a correctness-preserving
merge or fast path is demonstrated.

The two collectives cannot be blindly concatenated without changing semantics.
The first collective determines the cross-DP token vector and the minimum
cudagraph mode; that result can change local padding and therefore the inputs
to `check_enable_ubatch`. The second collective must then apply global-min
semantics to the post-padding, post-MoE-method local decision. A viable
optimization therefore has two safe forms:

* **Conservative eager fast path:** when cudagraph mode is NONE, sequence
  parallelism is disabled, and no DP padding is requested, compute the local
  ubatch flag before the metadata collective and include it as another packed
  row. This path still needs an A/A correctness test and a mixed-rank test.
* **General packed metadata path:** include enough per-rank inputs in one
  collective (unpadded tokens, padded tokens, cudagraph mode, and local ubatch
  eligibility), then reproduce the existing post-processing locally. This
  reduces launch count but increases payload and code complexity; it should be
  evaluated only after the eager fast path is measured.

Switching `dp_allreduce_on_npu=True` is a separate experiment, not an assumed
fix. The payload is tiny, while the observed long Gloo call is likely a rank
wait/cold-start outlier; moving it to NPU may trade CPU latency for device
stream synchronization.

### Verification matrix for the next iteration

1. Repeat the formal `prefill4k` script as alternating windows on one warmed
   server process (at least two 500-request pairs per configuration). Keep
   `TP=1, DP=2, EP=2`, 4096/16 tokens, concurrency 96, and change only DBO.
   Report median, spread, and completion count.
2. Capture operator and host-stack traces at concurrency 32 or 96 with the
   same 4096/16 workload. Use the traces only to measure per-step exposed
   communication, overlap, event waits, Gloo duration, and DP token skew.
3. Prototype synchronization reduction in isolation: reuse CPU metadata
   buffers, then evaluate a single packed collective carrying both metadata and
   the ubatch decision. Preserve the global-min semantics and compare request
   correctness before/after throughput.
4. Prototype one A2 event-boundary change at a time. Accept it only if exposed
   AllGather decreases on both ranks, overlap increases, no HTTP failures occur,
   and the warmed 500-request A/B improves beyond run-to-run variation.
5. Treat DP token vectors as a first-class metric. For each forward record
   `num_tokens_across_dp`, max/min skew, and which rank finishes last. If skew
   exceeds 10% repeatedly, fix scheduling/load balance before changing kernels.

The existing eager server log gives a first quantitative skew baseline. Across
107 logged DP vectors, 27 had a max/min ratio above 1.1 and 17 exceeded 2x;
the largest vector was [8207, 1]. These extreme ratios are mostly tiny
decode/tail batches, not DBO waves. The large DBO waves in this capture were
[16384,12348] (one uneven wave on each rank) plus two balanced waves. Thus
the next analysis must bucket skew by total tokens and should_ubatch: a
[8207,1] decode transition should not be mixed with a 16K-token prefill
wave when estimating DBO's load-balance cost.

The present evidence supports a provisional **+8.42% throughput** result for
DP+DBO on the formal 500-request workload, with lower TTFT/TPOT/E2E. It does
not yet establish a stable speedup distribution or prove that any single code
path is the limiting factor. The highest-value next measurement is a warmed,
high-concurrency profile that correlates DP synchronization and rank skew with
the exposed AllGather blocks.
