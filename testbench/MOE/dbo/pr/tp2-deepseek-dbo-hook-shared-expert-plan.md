# [Plan][DBO] TP=2 DeepSeek hook and shared-expert optimization (2026-07-31)

## Scope and decision rule

This is a profile-guided plan for DeepSeek-V2-Lite (MLA + MoE) on two Ascend
910B3 cards with TP=2 and EP=2. It examines two distinct opportunities:

1. whether a material collective is outside the existing DBO hook boundaries;
2. whether the existing shared-expert stream can hide additional MoE time.

Do not use profiler-enabled throughput or latency as a serving result. A
candidate advances only when its diagnostic trace is complete on both TP ranks,
its three-run plain A/B is repeatable, and its precision and deadlock gates
pass. Change exactly one variable in each experiment.

## Frozen scene

| Item | Value |
|---|---|
| Model | `/data/models/DeepSeek-V2-Lite-Chat`, BF16 |
| Hardware | 2 x Ascend 910B3, TP=2 / EP=2, eager mode |
| MoE communication | A3, `deepep_low_latency` |
| FlashComm1 | enabled; `HCCL_OP_EXPANSION_MODE=AI_CPU` |
| DBO | prefill threshold 1024; decode disabled |
| Workload | 4096 input, 16 output, 500 prompts, concurrency 96 |
| Results root | `/data/tmp/dbo-perf-tp2-20260731/` |

All candidates retain this topology and workload. A2/A3 or TP/EP comparisons
are separate future experiments, not variables in this plan.

## Current evidence

### Plain measurements: shared-expert stream

The following are non-profiler, eager-mode measurements with DBO and FC1
enabled. Each configuration has three repetitions.

| Configuration | Total throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---:|---:|---:|
| Normal DBO | 30,132.17 +/- 63.97 | 5,040.55 +/- 0.95 | 505.74 +/- 0.76 |
| `multistream_overlap_shared_expert=true` | 32,624.70 +/- 244.49 | 4,651.82 +/- 30.27 | 466.07 +/- 2.68 |
| Difference | +8.27% | -7.71% | -7.84% |

The shared-expert stream is active and its split-vs-integrated local validation
passed in the server log. This is evidence that it is a promising *compute*
overlap policy. It is not evidence that it is ready to enable by default:
the strict E2E precision gate fails for both ordinary DBO and shared-stream
DBO (16/16 requests; maximum logprob drift 0.06495 and 0.05453 respectively,
where the threshold is 1e-3). The shared stream does not create this failure.

The existing diagnosis in
[`pr-dbo-bf16-gemm-shape-precision.md`](pr-dbo-bf16-gemm-shape-precision.md)
locates the pre-existing DBO divergence at layer-0 `q_proj`: Ascend BF16 GEMM
is not invariant between the full M shape and the DBO-split M shapes. Event
placement must therefore not be treated as a remedy for this precision issue.

### Profile state

The NPU profiler wrapper now honors its schedule and calls `profiler.step()`;
the previous `Incorrect schedule` problem is fixed. However, CANN offline
analysis of the new captures still fails with `start_info path is not exist`.
It does not produce a usable `step_trace_time.csv`, and the trace view is
truncated. Consequently, no overlap ratio, exposed-communication ratio, or
critical-path duration may be claimed from these captures. Repairing a raw JSON
array is useful only for locating event names, never for reporting overlap.

### P0 execution record (2026-07-31)

The launcher and harness were changed to make profiling export-only in workers,
wait for every rank's `PROF_*/host/start_info.done`, and then run CANN analysis
once in the harness. Three fresh eager TP=2 captures were attempted under the
frozen scene:

| Capture | Distinguishing profile variable | Result |
|---|---|---|
| `p0-unscheduled-EKdEUG` | no torch schedule; bounded worker `max_iterations=8` | CANN analysis raced raw export; no usable CSV. |
| `p0-export-only-7yqMvK` | no torch schedule; export-only worker | Both rank `PROF_*` directories were created but no `start_info.done` appeared before timeout. |
| `p0-explicit-stop-2vYijT` | no torch schedule; explicit `/stop_profile` | The same two-rank missing-marker failure recurred. |

An isolated torch_npu program confirms that the no-schedule path stops from
`RECORD`, which torch_npu explicitly warns can yield incomplete data. It must
not be used for DBO analysis. A scheduled isolation capture that reaches its
record-and-save window also fails to write raw CANN artifacts, so the remaining
blocker is the local CANN/torch_npu raw exporter rather than DBO scheduling.
The server logs and artifact roots are retained for runtime escalation. Until
the exporter writes both markers and CANN produces valid two-rank CSV/JSON, do
not calculate a new overlap metric or add the MLP ReduceScatter hook.

## What the current A3 hooks mean

The policy in
[`deepseek.py`](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py)
defines dependency boundaries, rather than assigning one hook to every model
operator:

| DeepSeek block boundary | A3 event policy | Intent | Change status |
|---|---|---|---|
| MLA pre-process | wait `ATTN_PRE`; layer 0 records it | Start MLA/TP AG after prior layer finalization | keep |
| Attention OProj row collective | record `ATTN_POST`; its wait is `wait=False` | Permit independent ubatch progress while A3 work is scheduled | keep |
| MoE prepare / dispatch | record or wait `MOE_DISPATCH` | Establish dispatch-to-expert dependency | keep |
| MoE finalize | record `ATTN_PRE` | Carry completion to next-layer MLA | keep |

The placement follows data dependencies. A record belongs immediately after
the producer of a tensor; a wait belongs immediately before its first
consumer. Moving an event merely to make a timeline look busier risks using
incomplete tensors or creating a cross-ubatch cycle. The A3 `wait=False` on
the row boundary is already the allowed relaxation because the corresponding
QKV/MLA AllGather has no dependency on the other ubatch's row ReduceScatter.

There is one source-level hook candidate: `MLPRowParallelOp` executes a TP
ReduceScatter without a DBO hook in
[`linear_op.py`](../../../../vllm_ascend/ops/linear_op.py). It remains a
candidate, not an implementation task. First prove in a valid trace that this
exact collective is exposed and is material per layer/block. The priority
threshold is >1 ms (P0); 0.5-1 ms is P1; smaller calls do not justify event
and handoff overhead for TP=2.

## Shared expert: topology conclusion

At TP=2 with FlashComm1, sequence parallelism is enabled.
`shared_expert_dp_enabled()` consequently returns true, so the shared experts
are replicated and the TP all-reduce in the A3 shared-expert path is skipped.
There is no shared-expert AllReduce to hook in this topology.

The relevant existing optimization is instead
`additional_config.multistream_overlap_shared_expert=true`:

```text
routed-expert stream: routing -> dispatch A3 -> grouped expert GEMM -> combine A3
shared-expert stream:                 gate/up+activation ->            down
```

The implementation waits on routed-expert events before the shared gate/up and
before the shared down, then makes the default stream wait for the shared
stream at the join. Thus its available overlap windows are dispatch versus
shared gate/up and combine versus shared down. The next profile should measure
whether either shared segment remains exposed; it must not add a nonexistent
collective hook.

## Ordered experiment plan

| Priority | Question and one changed variable | Profile procedure and evidence needed | Plain / correctness gates | Outcome rule |
|---|---|---|---|---|
| P0 | Can the profiler produce complete two-rank data? Change only profile mode. | Run bounded operator capture; require valid trace JSON plus CANN `step_trace_time.csv` on both ranks. Follow with a short `host_stack` capture only if gaps are visible. | No performance claim. Run profiler-wrapper UT. | Block hook timing decisions while CANN metadata is incomplete. |
| P0 | Is the unhooked MLP ReduceScatter exposed? Change only trace annotation. | Map each TP AG/RS and A3 dispatch/combine to `ATTN_PRE`, `ATTN_POST`, `MOE_DISPATCH`, or no event. Measure duration and compute between adjacent collectives per rank. | No behavior change; baseline requests succeed. | Promote only an uncovered RS >1 ms. Otherwise retain template. |
| P1 | Does one MLP row hook reduce exposure? Change only that hook. | Compare the same short operator capture. Require lower exposed communication and no new rank skew. Merge only when intervening compute <200 us; keep separation above 500 us. Target `min(comm, compute)/max(comm, compute) >80%`. | Three alternating plain runs, 500/500 success, both-rank trigger, precision and 30-minute deadlock/stress gate. | Keep only repeatable gain with passing quality. |
| P0 | Does the existing shared stream hide compute? Change only `multistream_overlap_shared_expert`. | Verify dispatch/gate-up and combine/down windows, stream join, and rank starvation in a valid trace. Compare to normal DBO, not profiler throughput. | Existing plain result is +8.27%; rerun after runtime/DBO changes. Strict precision and deadlock gates remain required. | Keep opt-in until the known DBO BF16 M-shape issue has a mitigation or relaxed policy. |
| P1 | Is host handoff limiting the window? Change only profile mode to `host_stack`. | Attribute gaps to `dbo_wait_current_stream_and_yield`, syncs, CPU affinity, or GIL contention. Investigate when host/free time exceeds 10%. | Plain A/B and peak-memory check for handoff changes. | Prefer host fixes over event relocation when NPU is starved. |

## Reproducible commands

Use separate servers for plain and profile modes. The launcher supports DBO-off
baseline and `ADDITIONAL_CONFIG` forwarding.

```bash
# Plain shared-expert A/B; repeat each side three times and alternate order.
ENFORCE_EAGER=1 VLLM_ASCEND_ENABLE_DBO=1 \
  ADDITIONAL_CONFIG='{"multistream_overlap_shared_expert":true}' PORT=8001 \
  bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh

LABEL=tp2_fc1_dbo1_shared_stream PORT=8001 BENCH_PRESET=prefill4k \
  INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96 \
  OUT_DIR=/data/tmp/dbo-perf-tp2-20260731/results \
  bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh

# Diagnostic only: never use profiler latency as serving data.
ENABLE_PROFILER=1 PROFILER_MODE=operator PROFILER_MAX_ITERATIONS=8 \
  ENFORCE_EAGER=1 VLLM_ASCEND_ENABLE_DBO=1 PORT=8001 \
  TORCH_PROFILER_DIR=/data/tmp/dbo-perf-tp2-20260731/profile/candidate \
  bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh

# Exact quality gate; both ranks must log should_ubatch: True.
ENFORCE_EAGER=1 PORT=8018 \
  python3 testbench/MOE/dbo/demos/precision/quick/run_quick_precision.py \
  --flashcomm1 1 \
  --server-script "$PWD/testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh" \
  --out-dir /data/tmp/dbo-perf-tp2-20260731/precision-candidate
```

## Selection

The justified next implementation work is P0 profiler artifact repair and
trace attribution of the unhooked MLP ReduceScatter. No additional DeepSeek
hook should be added before that evidence exists. The shared-expert stream has
a measured plain-performance upside, but it remains opt-in until the known
DBO BF16 M-shape precision issue has an agreed mitigation or an explicit
relaxed quality policy.
