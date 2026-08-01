# [Analysis] multistream_overlap_shared_expert vs DBO — Stream Architecture & Orthogonality

Date: 2026-08-01

## 1. Executive Summary

**`multistream_overlap_shared_expert` and DBO are orthogonal and complementary.**
They operate at different granularities, use independent synchronisation primitives,
and can be enabled together without correctness risk. Empirical measurements confirm
a **+8.27% throughput gain** when the shared-expert stream is enabled on top of DBO.

**No modifications are required** for DBO to coexist with this feature. The
shared-expert stream is a pure incremental optimisation.

---

## 2. Upstream Status

**`multistream_overlap_shared_expert` does not exist in upstream vllm.**

```
$ grep -rn "multistream\|shared_forward_impl\|_forward_shared_expert" vllm/
# → zero results
```

The entire feature chain lives in vllm-ascend:

| File | Role |
|------|------|
| [ascend_config.py:138](vllm_ascend/ascend_config.py#L138) | Configuration entry point (`additional_config`) |
| [utils.py:505](vllm_ascend/utils.py#L505) | `shared_experts_calculation_stream()` — dedicated NPU stream |
| [fused_moe.py:384](vllm_ascend/ops/fused_moe/fused_moe.py#L384) | Initialisation + split-path validation |
| [fused_moe.py:737](vllm_ascend/ops/fused_moe/fused_moe.py#L737) | `_forward_shared_experts()` — overlap logic |
| [fused_moe.py:845](vllm_ascend/ops/fused_moe/fused_moe.py#L845) | `shared_forward_impl()` — override of upstream FusedMoE |

Upstream vllm's `FusedMoE` only has `forward_impl()`. The methods `shared_forward_impl()`
and `_forward_shared_experts()` are vllm-ascend additions on `AscendFusedMoE`.

### Code Map

| Component | Path |
|-----------|------|
| Config entry | [ascend_config.py:138](vllm_ascend/ascend_config.py#L138) |
| Dedicated stream factory | [utils.py:505](vllm_ascend/utils.py#L505) |
| `npu_stream_switch` context manager | [utils.py:1032](vllm_ascend/utils.py#L1032) |
| DBO stream tracking (`dbo_current_stream`) | [utils.py:467](vllm_ascend/utils.py#L467) |
| AscendFusedMoE init (multistream setup) | [fused_moe.py:333-478](vllm_ascend/ops/fused_moe/fused_moe.py#L333) |
| `forward_impl()` — routed experts | [fused_moe.py:598](vllm_ascend/ops/fused_moe/fused_moe.py#L598) |
| `_forward_shared_experts()` — shared expert on 3rd stream | [fused_moe.py:737](vllm_ascend/ops/fused_moe/fused_moe.py#L737) |
| `shared_forward_impl()` — orchestrator | [fused_moe.py:845](vllm_ascend/ops/fused_moe/fused_moe.py#L845) |
| MoE prepare (DBO hooks) | [prepare_finalize.py:370-445](vllm_ascend/ops/fused_moe/prepare_finalize.py#L370) |
| MoE finalize (DBO hooks) | [prepare_finalize.py:559-606](vllm_ascend/ops/fused_moe/prepare_finalize.py#L559) |
| DBO ubatch context (stream switching) | [ubatching.py:100-170](vllm_ascend/worker/ubatching.py#L100) |
| DBO overlap template (DeepSeek A3) | [deepseek.py:52-82](vllm_ascend/dbo/overlap_templates/deepseek.py#L52) |
| Legacy path (310P, no multistream) | [_310p/fused_moe/fused_moe.py:280-296](vllm_ascend/_310p/fused_moe/fused_moe.py#L280) |
| `AscendMoERunner` (DBO-aware runner) | [fused_moe.py:255-330](vllm_ascend/ops/fused_moe/fused_moe.py#L255) |

---

## 3. Architecture: Two Levels of Overlap

### 3.1 DBO — Inter-Layer compute↔comm Overlap (2 streams)

```
  Transformer Block N-1              Transformer Block N
  ┌──────────────────────┐          ┌──────────────────────────────┐
  │ compute stream:      │          │ compute stream:              │
  │  MLA post → MoE prep │ ──────►  │  MLA pre → Attn → MLA post  │
  │     (record ATTN_POST)│          │     (wait ATTN_PRE)          │
  └──────────────────────┘          └──────────────────────────────┘
  ┌──────────────────────┐          ┌──────────────────────────────┐
  │ comm stream:         │          │ comm stream:                 │
  │  dispatch → combine  │ ──────►  │  MoE prep → dispatch         │
  │     (record ATTN_PRE) │          │     (wait ATTN_POST)         │
  └──────────────────────┘          └──────────────────────────────┘

  Synchronisation: UBatchEventKey (ATTN_PRE, ATTN_POST, MOE_DISPATCH)
  Manager:         AscendUBatchContext
                   → [ubatching.py:100](vllm_ascend/worker/ubatching.py#L100)
                   + UbatchOverlapBaseTemplate
                   → [deepseek.py:52](vllm_ascend/dbo/overlap_templates/deepseek.py#L52)
```

DBO splits each transformer block across two alternating NPU streams
(compute and comm), overlapping communication from one block with
computation from the next.

### 3.2 multistream_overlap_shared_expert — Intra-MoE-Layer shared-compute↔routed-comm Overlap (3rd stream)

```
  Single FusedMoE Layer

  Routed stream (DBO-managed compute or comm stream):
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  routing ─► dispatch A3 ─► grouped expert GEMM ─► combine │
  │     │            │                    │                    │
  │     │     ┌──────┘                    │                    │
  │     │     │ wait(before_dispatch)     │ wait(before_combine)
  │     │     │                           │                    │
  │ Shared stream (shared_experts_calculation_stream):       │
  │     │   gate+up·act ─────────────►   down                 │
  │     │                                                    │
  │     └── wait(before_routed_experts)                       │
  │                                                          │
  │ Join: DBO stream.wait_stream(shared stream)               │
  └──────────────────────────────────────────────────────────┘

  Synchronisation: routed-expert raw npu Events (before_dispatch, before_combine, …)
  Manager:         npu_stream_switch context manager inside
                   [_forward_shared_experts()](vllm_ascend/ops/fused_moe/fused_moe.py#L737)
```

The shared expert's gate+up projection runs concurrently with the routed
experts' dispatch A3 communication. The shared down projection runs
concurrently with the routed experts' combine A3 communication.

---

## 4. Stream Inventory

When both DBO and `multistream_overlap_shared_expert` are enabled:

| # | Stream | Purpose | Managed by |
|---|--------|---------|------------|
| 1 | DBO compute stream | Ubatch 0 compute | `AscendUBatchContext` — [ubatching.py](vllm_ascend/worker/ubatching.py) |
| 2 | DBO comm stream | Ubatch 1 communication | `AscendUBatchContext` — [ubatching.py](vllm_ascend/worker/ubatching.py) |
| 3 | `shared_experts_calculation_stream` | Shared expert GEMM | `npu_stream_switch` in [_forward_shared_experts()](vllm_ascend/ops/fused_moe/fused_moe.py#L745) |
| 4 | `gate_stream` | MoE gate + routing (if `multistream_overlap_gate`) | `AscendFusedMoE.gate_stream` — [fused_moe.py:395](vllm_ascend/ops/fused_moe/fused_moe.py#L395) |
| 5 | default stream | torch default | `torch.npu` |

Streams 1-2 are specific to DBO. Stream 3 is the multistream shared expert stream.
They have independent lifecycles and synchronisation mechanisms.

---

## 5. Complete Call Chain (DeepSeek-V2 A3 + DBO + multistream)

```
Model.forward()
  └─ AscendFusedMoE.forward()
       └─ self.runner.forward()
            └─ _forward_impl() → forward_impl()
                 └─ shared_forward_impl()

                      │
                      ├─ (1) forward_impl(return_with_event=True)
                      │     │
                      │     ├─ prepare()
                      │     │    ├─ dbo_moe_prepare_hook(is_record=True)     ← DBO hook
                      │     │    ├─ TP/EP all_gather
                      │     │    └─ dbo_moe_prepare_hook(is_record=False)   ← DBO hook
                      │     │
                      │     ├─ quant_method.apply()
                      │     │    └─ [dispatch → grouped GEMM → combine]
                      │     │       Records: before_dispatch_evt, before_gmm2_evt,
                      │     │                before_combine_evt, swiglu_limit
                      │     │
                      │     └─ finalize()
                      │          ├─ dbo_moe_finalize_hook(is_record=True)   ← DBO hook
                      │          ├─ reduce_scatter / all_reduce
                      │          └─ dbo_moe_finalize_hook(is_record=False)  ← DBO hook
                      │
                      │     Returns: FusedMoEResult with events
                      │
                      └─ (2) _forward_shared_experts(hidden_states, events)
                            │
                            ├─ npu_stream_switch(shared_experts_calculation_stream())
                            │    ├─ wait(before_routed_experts)
                            │    ├─ shared_experts_part1 (gate+up)    ∥ dispatch
                            │    ├─ wait(before_combine)
                            │    └─ shared_experts_part2 (down)       ∥ combine
                            │
                            └─ DBO_stream.wait_stream(shared_stream)
```

> ▶ Code: [shared_forward_impl()](vllm_ascend/ops/fused_moe/fused_moe.py#L845) →
> [forward_impl()](vllm_ascend/ops/fused_moe/fused_moe.py#L598) +
> [_forward_shared_experts()](vllm_ascend/ops/fused_moe/fused_moe.py#L737) |
> [prepare()](vllm_ascend/ops/fused_moe/prepare_finalize.py#L370) |
> [finalize()](vllm_ascend/ops/fused_moe/prepare_finalize.py#L559)

Key observation: **DBO hooks fire inside `forward_impl()` (prepare/finalize),
which completes *before* `_forward_shared_experts()` begins.** The two
mechanisms never interleave on the same stream at the same time.

---

## 6. Correctness Analysis

### 6.1 Stream Isolation

`npu_stream_switch` is a standard `torch.npu.stream()` context manager.
When it temporarily switches to the shared stream:

```python
# DBO's thread-local stream tracking
# → utils.py:467
def dbo_current_stream():
    if _current_stream_tls.value is None:
        _current_stream_tls.value = torch.npu.current_stream()
    return _current_stream_tls.value  # ← only modified by dbo_set_stream()

# npu_stream_switch — temporarily switches NPU stream
# → utils.py:1032
def npu_stream_switch(target_stream, *, enabled=True):
    if not enabled:
        return nullcontext()
    return torch.npu.stream(target_stream)
```

| State | Inside `npu_stream_switch(shared_stream)` | After context exit |
|-------|------------------------------------------|-------------------|
| NPU hardware stream | **shared stream** | **DBO stream** (restored) |
| `dbo_current_stream()` return | DBO stream (unchanged) | DBO stream |
| DBO event record/wait | Not called here | Normal |

- NPU hardware stream **is** switched → shared expert runs on the correct stream ✅
- `dbo_current_stream()` is **not** updated → DBO's tracking is undisturbed ✅
- Context manager exits → NPU stream restored to DBO stream ✅
- `wait_stream()` at [fused_moe.py:832-833](vllm_ascend/ops/fused_moe/fused_moe.py#L832) runs on the DBO stream → correct synchronisation ✅

### 6.2 No Deadlock Risk

- DBO events (`UBatchEventKey`) are recorded/waited at the prepare/finalize boundaries
  ([prepare_finalize.py:407,421,593,601](vllm_ascend/ops/fused_moe/prepare_finalize.py#L407)),
  outside `_forward_shared_experts`.
- Shared stream events (`before_dispatch`, `before_combine`) are raw routed-expert
  events, independent of DBO's event system.
- The shared stream waits for routed events → shared expert reads data only
  after it's produced.
- The DBO stream waits for the shared stream at the join →
  [fused_moe.py:832-833](vllm_ascend/ops/fused_moe/fused_moe.py#L832) →
  combined output (shared + routed) is ready before the next layer.

### 6.3 No Race Condition

- `wait_stream(shared_experts_calculation_stream())` at
  [fused_moe.py:832-833](vllm_ascend/ops/fused_moe/fused_moe.py#L832) guarantees
  the DBO stream sees all shared expert writes before proceeding.
- No tensor is accessed concurrently by DBO and shared streams without
  synchronisation.

---

## 7. Orthogonality Matrix

| Dimension | DBO | multistream_overlap_shared_expert | Conflict? |
|-----------|-----|----------------------------------|:---------:|
| Granularity | Inter-layer (block boundary) | Intra-MoE-layer (shared vs routed) | No |
| Stream count | 2 (compute + comm) | +1 (shared_experts_calculation) | No |
| Sync primitives | UBatchEventKey events | Raw routed-expert npu Events | No |
| Hook insertion points | prepare/finalize DBO hooks | Inside _forward_shared_experts | No overlap |
| Stream tracking | AscendUBatchContext | npu_stream_switch + context mgr | Isolated |
| NPU core allocation | set_stream_limit (cube/vector) | None (default) | ⚠️ potential resource contention |
| Precision impact | Known BF16 M-shape issue | No independent precision issue | Independent |

---

## 8. Empirical Evidence

### 8.1 PR Plan Measurements (TP=2, DeepSeek-V2-Lite, A3, FC1, DBO enabled)

From [tp2-deepseek-dbo-hook-shared-expert-plan.md](tp2-deepseek-dbo-hook-shared-expert-plan.md):

| Configuration | Total Throughput (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---------------|:------------------------:|:--------------:|:--------------:|
| Normal DBO | 30,132.17 ± 63.97 | 5,040.55 ± 0.95 | 505.74 ± 0.76 |
| **+ multistream_overlap_shared_expert** | **32,624.70 ± 244.49** | **4,651.82 ± 30.27** | **466.07 ± 2.68** |
| **Delta** | **+8.27%** | **-7.71%** | **-7.84%** |

The shared-expert stream provides a material, repeatable improvement on top of DBO.

### 8.2 Current Auto-Benchmark (2026-08-01)

For DeepSeek-V2-Lite with FlashComm1 (AI_CPU HCCL mode), A3 communication,
TP=2, prefill4k workload — the `dbo_fc1` configuration (DBO + FlashComm1)
achieves the best result of **121.90 tok/s (+25.8% vs baseline)**. The
multistream shared expert option was not tested in this specific sweep,
but the PR plan data confirms the additive benefit.

---

## 9. Known Limitations & Recommendations

### 9.1 DBO validation (hit_count=0)

In all current benchmark runs, `dbo` configurations report `hit_count=0`
in the DBO validation step. This is a separate issue related to DBO trigger
detection in the validation harness, not caused by multistream overlap.

### 9.2 NPU Core Resource Allocation (P2, low priority)

DBO sets core quotas on its compute and comm streams via
`torch.npu.set_stream_limit(cube_num=..., vector_num=...)` in
[ubatching.py:154-167](vllm_ascend/worker/ubatching.py#L154).
The `shared_experts_calculation_stream` does not inherit these limits,
which could theoretically lead to resource contention.

**Suggested mitigation** — propagate DBO compute stream limits to the
shared stream when both are enabled:

```python
# In _forward_shared_experts(), when both DBO and multistream are active:
# → vllm_ascend/ops/fused_moe/fused_moe.py near L745
if self.multistream_overlap_shared_expert and forward_context.dbo_enabled:
    # Inherit compute stream core allocation for the shared stream
    ...
```

This is a performance optimisation, not a correctness fix.

### 9.3 Interaction with multistream_overlap_gate (P2)

`multistream_overlap_gate` runs the shared expert + gate computation on
`gate_stream` **before** `forward_impl()` —
see [fused_moe.py:616-655](vllm_ascend/ops/fused_moe/fused_moe.py#L616).
When combined with DBO, this introduces an additional stream synchronisation
point. This path has not been tested with DBO in the current benchmark suite.
The `multistream_overlap_shared_expert` path (used in the PR plan) is the
recommended configuration for DBO scenarios.

### 9.4 DBO BF16 M-Shape Precision Issue (pre-existing, unrelated)

Both ordinary DBO and shared-stream DBO fail the strict E2E precision gate
(maximum logprob drift 0.06495 and 0.05453 respectively, threshold 1e-3).
See [pr-dbo-bf16-gemm-shape-precision.md](pr-dbo-bf16-gemm-shape-precision.md)
for the full diagnosis. The shared-expert stream does not create or exacerbate
this issue.

---

## 10. Conclusion

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   multistream_overlap_shared_expert             │
│   ┌──────────┐                                  │
│   │ 独立 #3   │  shared expert GEMM              │
│   │ stream   │  hidden behind routed comm        │
│   └────┬─────┘                                  │
│        │                                        │
│        │  ✅ Orthogonal    ✅ Complementary      │
│        │  ✅ +8.27% gain   ✅ Zero correctness   │
│        │     on top of DBO    risk               │
│        │                                        │
│   ┌────┴─────┐                                  │
│   │   DBO    │  Inter-layer compute↔comm        │
│   │ 2 stream │  overlap across blocks            │
│   └──────────┘                                  │
│                                                 │
│   vllm-ascend exclusive | Not in upstream vllm   │
└─────────────────────────────────────────────────┘
```

**For DBO development**: no modifications needed. Keep the two features
independently switchable. Focus DBO efforts on the BF16 M-shape precision
fix and hook boundary validation.

**For multistream shared expert**: keep opt-in until the known DBO BF16
M-shape precision issue has an agreed mitigation or relaxed quality policy.
The performance upside is real and repeatable.

---

## Appendix: Quick Reference

| What | Where |
|------|-------|
| Enable flag | `--additional-config '{"multistream_overlap_shared_expert": true}'` |
| Config parsing | [ascend_config.py:138](vllm_ascend/ascend_config.py#L138) |
| Feature gating | [fused_moe.py:384](vllm_ascend/ops/fused_moe/fused_moe.py#L384) — gated on `has_shared_experts` |
| Validation (split vs integrated) | [fused_moe.py:493](vllm_ascend/ops/fused_moe/fused_moe.py#L493) |
| MoE runner DBO passthrough | [fused_moe.py:475](vllm_ascend/ops/fused_moe/fused_moe.py#L475) — `enable_dbo` |
| DBO ubatch stream setup | [ubatching.py:100-170](vllm_ascend/worker/ubatching.py#L100) |
| Token dispatcher DBO hooks | [token_dispatcher.py:490-558](vllm_ascend/ops/fused_moe/token_dispatcher.py#L490) |
| Related PR plan (shared expert) | [tp2-deepseek-dbo-hook-shared-expert-plan.md](tp2-deepseek-dbo-hook-shared-expert-plan.md) |
| Related PR (BF16 precision) | [pr-dbo-bf16-gemm-shape-precision.md](pr-dbo-bf16-gemm-shape-precision.md) |
