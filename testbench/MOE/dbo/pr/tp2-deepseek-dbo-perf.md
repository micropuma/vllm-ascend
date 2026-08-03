# [Perf][DBO] FlashComm1 + DBO analysis: DeepSeek-V2-Lite, TP=2 (2026-07-30)

## Decision summary

This document analyzes the combined **FlashComm1 + DBO** path, not DBO in
isolation. The target is DeepSeek-V2-Lite (MLA + MoE), TP=2/EP=2, BF16, two
Ascend 910B3 NPUs, with `HCCL_OP_EXPANSION_MODE=AI_CPU`.

The 20-forward-step operator profile proves that the path is overlapping
communication and compute:

| CANN aggregate over the repaired captured 20-step window | Rank 0 | Rank 1 |
|---|---:|---:|
| Stage | 9,694.67 ms | 9,693.40 ms |
| Compute | 6,289.21 ms | 6,586.45 ms |
| Communication, total | 5,603.48 ms | 4,088.15 ms |
| Communication exposed | 2,562.53 ms (26.4% stage) | 1,536.82 ms (15.9% stage) |
| **Communication overlapped** | **3,040.94 ms (54.27% comm)** | **2,551.34 ms (62.41% comm)** |
| Free | 842.92 ms (8.7% stage) | 1,570.13 ms (16.2% stage) |

The central result is therefore not "DBO has no overlap." The repaired trace
shows 54.27% and 62.41% communication overlap on rank 0 and rank 1,
respectively (57.70% when the two communication totals are combined). The
remaining communication is still exposed. Rank 1 also retains a 16.2% free
interval, making host/runtime and dependency gaps a material next target.

The profile was capped at `max_iterations=20`; both TP workers logged
`should_ubatch: True` repeatedly for 16,384-token batches. It is diagnostic
data only. It must not be used as serving latency or throughput.

## Environment and reproducibility

| Item | Value |
|---|---|
| Model | `/data/models/DeepSeek-V2-Lite-Chat`, BF16, MLA + MoE |
| Parallelism | TP=2, EP enabled (EP=2), DP=1, PP=1 |
| NPU | 2 x Ascend 910B3, `ASCEND_RT_VISIBLE_DEVICES=0,1` (physical 3,4) |
| Runtime | CANN 9.0.0, torch_npu 2.10.0, vLLM 0.22.1 |
| Revisions | vLLM `0decac0`; vLLM-Ascend `7d9972c7` |
| Serving policy | max model len 8192; max batched tokens 16384; max seqs 256 |
| DBO policy | prefill threshold 1024; decode threshold 1,000,000,000 |
| FlashComm1 / DBO / HCCL | `1 / 1 / AI_CPU` |
| Profile workload | random chat, 4096 input, 16 output, 500 requests, concurrency 96 |
| Profile cap | 20 model-forward iterations |

Every launcher and benchmark command sourced
`/data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate` and
`/data/workspace/vllm-dbo-v0221/env.sh`, then set both `NO_PROXY` and
`no_proxy` to `127.0.0.1,localhost`.

The repaired capture used the checked-in pair in operator mode. This separates
kernel/communication collection from Python-stack collection and disables the
otherwise unbounded frontend trace:

```bash
ENABLE_PROFILER=1 PROFILER_MODE=operator PROFILER_MAX_ITERATIONS=20 \
HCCL_OP_EXPANSION_MODE=AI_CPU VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
VLLM_ASCEND_ENABLE_DBO=1 VLLM_LOGGING_LEVEL=DEBUG PORT=8001 \
LABEL=fc1_dbo_20_operator PROFILE_ROOT=/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/profile \
TORCH_PROFILER_DIR=/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/profile/fc1_dbo_20_operator \
bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh

LABEL=fc1_dbo_20_operator PORT=8001 BENCH_PRESET=prefill4k \
INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96 \
PROFILE_ROOT=/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/profile \
TORCH_PROFILER_DIR=/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/profile/fc1_dbo_20_operator \
OUT_DIR=/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/results RESULT_SUFFIX=_20step \
bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh --profile
```

## Artifacts and capture integrity

The prior malformed capture was removed from the experiment path before the
repaired capture. The authoritative artifacts are:

```text
/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/fc1_dbo_20_operator_server.log
/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/results/fc1_dbo_20_operator_in4096_out16_np500_c96_20step.json
/data/tmp/dbo-perf-fc1-dbo-20260730/fixed/profile/fc1_dbo_20_operator/
  dp0_pp0_tp0_dcp0_ep0_rank0_674795_20260730161919172_ascend_pt/
  dp0_pp0_tp1_dcp0_ep1_rank1_674796_20260730161919173_ascend_pt/
```

The benchmark completed 500/500 requests. `torch_npu.profiler.analyse` was
run separately for both ranks. Both now contain a non-empty CANN
`step_trace_time.csv` and a `trace_view.json` accepted by `jq empty`; the
previous rank-1 trailing-comma failure is gone. Each CSV contains one aggregate
row (empty `Step` field), so timings are capture-window aggregates rather than
per-step percentiles.

The harness was changed as part of this work: `PROFILER_MODE=operator` sets
`with_stack=false`, retains operator shapes, and `ignore_frontend=true` bounds
the capture to the worker iteration cap. The test script now analyses and
validates every `*_ascend_pt` rank directory rather than silently analysing
only the lexically last rank. A profile run fails if either its CANN step CSV
is missing or its trace JSON is invalid.

## Why FlashComm1 + DBO overlaps

FlashComm1 enables sequence parallelism for MoE models whenever it is enabled
and tokens are present. It pads only to TP alignment (at TP=2, at most one
token) in [ascend_forward_context.py](../../../../vllm_ascend/ascend_forward_context.py#L120).

For MLA, the combined path is explicit in
[mla_v1.py](../../../../vllm_ascend/attention/mla_v1.py#L1671): when both DBO
and FlashComm1 are enabled, DBO records an event, executes TP AllGather for
`q_c` and `kv_no_split`, then waits/yields and unpads. That is the
FlashComm1-specific collective which DBO schedules against work from the other
microbatch.

```text
ubatch 0 / compute stream:  MLA or MoE compute  ---------------------->
ubatch 1 / comm stream:                 record -> FC1 q_c/kv AG -> wait/yield
                                                    ^ overlaps the other stream
```

DeepSeek's A3 template implements the dependency policy in
[deepseek.py](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L39):
the MLA pre-process uses `ATTN_PRE`; the row hook permits `wait=False`; MoE
dispatch uses `MOE_DISPATCH`; finalize records the next `ATTN_PRE`. The server
uses `--all2all-backend deepep_low_latency`, so the profile helper's heuristic
label "MoE AllGather/A2" must not be read as a statement that the actual EP
algorithm was A2. FlashComm1's TP AllGather is expected even on the A3 MoE
path.

The two Python ubatch workers exchange ownership through `threading.Event` in
[ubatching.py](../../../../vllm_ascend/worker/ubatching.py#L129); device events
make the compute/communication stream dependencies explicit.

## Operator profile findings

### Measured overlap and exposure

The CANN aggregates are the authoritative profile measurements for this run:
rank 0 has `3,040.94 / 5,603.48 = 54.27%` communication overlap and rank 1
has `2,551.34 / 4,088.15 = 62.41%`. The rank-to-rank gap is 8.14 percentage
points, below the 10-point comparison guard but worth tracking with EP token
balance. Therefore DBO is working, but it is not saturated.

The repaired raw timeline is valid on both ranks. This report intentionally
uses CANN's full communication classification for the overlap ratio rather
than a hand-selected kernel-union metric: the latter excludes protocol work
and can only be a sanity check, not the numerator for a scheduling decision.

### Kernel distribution (rank 0)

Category shares are sums of kernel duration and can overlap; they are not
stage-wall-time shares.

| Kernel / category | Total | Evidence-based implication |
|---|---:|---|
| `FusedInferAttentionScore` | 2400.70 ms, 2268 calls | Attention is the largest compute block available to hide communication. |
| `MatMulV3` | 1779.65 ms, 3024 calls | Dense/MLA projections remain material; do not optimize only collectives. |
| `GroupedMatmul` | 1550.00 ms, 1456 calls | MoE expert GEMM is the second largest block; scheduling must preserve its overlap opportunity. |
| `MatMulV2` | 481.07 ms, 4177 calls | Lower-order projections still occupy a visible compute budget. |
| `AddRmsNormBias` / `SwiGlu` | 461.79 / 450.22 ms | Fusion coverage is already present; lower priority than exposed communication. |
| `allgatherAicpuKernel` | 364.78 ms, 3427 calls | FlashComm1/TP communication is high-frequency but largely hidden. Reduce exposure rather than merely reducing call count. |
| Slice / copy operations | 172.43 ms slice; rank1 has 318.89 ms cast-copy | Layout/copy cleanup is worthwhile only after exposed communication. |

`communication.json` lists individual collective protocol events. Its sum of
event elapsed times is an aggregate across overlapping streams and protocol
sub-events, not 20-step wall time or per-layer latency. Use it to locate a
specific exposed collective only after correlating it with the valid raw trace;
use `step_trace_time.csv` for aggregate overlap/exposure decisions.

### Serving result retained as diagnostic only

The repaired profile-enabled formal run completed 500/500 requests:

| Metric | Value |
|---|---:|
| Total-token throughput | 30,160.81 tok/s |
| Output throughput | 117.13 tok/s |
| Mean TTFT | 5,230.51 ms |
| Mean TPOT | 509.93 ms |
| Mean E2E | 12,879.52 ms |

These values are not used for a serving claim because torch profiler was
enabled.

An earlier plain comparison found FC1+DBO at 30,884.89 total tok/s versus a
no-FC1/no-DBO baseline at 24,744.60 tok/s (+24.81%) and mean TTFT -20.03%.
That remains an end-to-end combined configuration result, not a DBO-only
causal attribution because both FC1 and DBO changed together.

## Optimization plan

| Priority | Candidate | Evidence | Implementation path | Expected result / gate |
|---|---|---|---|---|
| P0 | Keep the repaired two-rank profile gate | The operator-mode 20-step trace is valid on both ranks; overlap differs by 8.14 points. | Keep `PROFILER_MODE=operator`, `ignore_frontend=true`, and per-rank CANN/JSON validation for every overlap change. Use `host_stack` only as a separate short diagnostic capture. | Both rank summaries present and parseable; investigate when overlap skew exceeds 10 points. |
| P0 | Measure DBO's incremental value with FC1 fixed | Existing plain +24.81% changes FC1 and DBO together. | Alternate `FC1=1, DBO=0` and `FC1=1, DBO=1` three times, same hardware/workload/caches; retain raw JSON. | Mean +/- std for throughput, TTFT, TPOT, E2E; 500/500 and both-rank DBO trigger. |
| P0 | Reduce exposed communication | The repaired capture exposes 45.73% of rank0 and 37.59% of rank1 communication (42.30% when totals are combined). | Annotate all collectives in the raw trace by hook interval; prioritize any >1 ms block outside `ATTN_PRE`, `ATTN_POST`, or `MOE_DISPATCH`. | Profile: lower exposed ratio and no decrease in overlap; then plain A/B and deadlock test. |
| P1 | Add hooks only for uncovered material collectives | Shared-expert TP all-reduce has no apparent DBO hook in [fused_moe.py](../../../../vllm_ascend/ops/fused_moe/fused_moe.py#L625). `MLPRowParallelOp` reduce-scatter is unhooked in [linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L219). | First correlate each raw collective to a hook. Add one hook per PR only when exposed cost is >1 ms per layer/block. | CANN exposed comm falls; TP=2 correctness/logprob and deadlock gate pass. |
| P1 | Repartition/merge event boundaries | 42.30% of combined communication remains exposed. | Merge only consecutive collective blocks separated by <200 us compute; preserve separation above 500 us. Move record/wait only after per-block balance is measured. | Target block efficiency `min(comm, compute)/max(comm, compute) >80%`; no unsafe cross-layer dependency. |
| P1 | Host/ubatch handoff investigation | Free is 16.1%, above the skill's 10% threshold. `threading.Event` handoff is in the hot DBO path. | Re-run an explicitly host-stack-focused trace after fixing completion; inspect `dbo_wait_current_stream_and_yield`, sync calls, and CPU/NUMA affinity. | Free ratio and host gaps fall without changing numerical output. |
| P2 | Stream core split sweep | AI_CPU prevents HCCL from consuming AIC/AIV compute resources, but core-limit choices can still change compute balance. | Sweep existing AIC/AIV split controls only after profile completeness and fixed-FC1 A/B. | Three plain repetitions per setting; retain only a repeatable winner. |
| P2 | Operator/precision changes | Attention/GEMM dominate after scheduling. | Consider further attention/layout fusion or precision changes only after P0/P1; do not trade BF16 quality for overlap. | Deterministic generation, logprob/layer-output precision gates. |

## Operational finding: cold start

This profile server spent about 400 seconds capturing 34 PIECEWISE and 34 FULL
graphs. That is a material cold-start problem but is not part of steady-state
throughput or overlap. Treat it as a separate issue: compare graph counts and
capture policy with FC1 fixed before proposing an eager/capture change.

## Merge gates

1. No DBO-only performance claim without the FC1-fixed, three-repeat plain A/B.
2. Every scheduling change requires a complete two-rank profile, lower exposed
   communication or higher overlap, plus plain serving confirmation.
3. Every DBO result requires 500/500 success and `should_ubatch: True` evidence
   on both TP workers.
4. Deadlock and TP=2 precision validation block merge of hook/event changes.
5. Profiler latency remains diagnostic; only non-profiler runs enter a serving
   performance table.
