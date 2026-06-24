# DBO Regime Analysis

## Goal

This document analyzes the current `DBO + torch.compile/fullgraph + ACL graph`
regime behavior. The main questions are:

1. Why the recent changes make `fullgraph=True` easier to compile.
2. Whether the newly explicit runtime metadata introduces new graph-splitting
   dimensions or descriptor/shape mismatches at runtime.

This document focuses on compile-time contracts, runtime regimes, and graph
cache/dispatch consistency. It does not analyze the DBO overlap algorithm
itself.

## Background

The current uncommitted changes have one main theme: several DBO custom-op
fake implementations no longer read runtime context internally. Instead, the
caller now passes explicit shape metadata into the custom op.

Typical examples include:

- `vllm_ascend::dbo_column_allgather_sp`
- `vllm_ascend::dbo_moe_prepare_allgather`
- `vllm_ascend::dbo_moe_finalize_allgather`
- `vllm_ascend::dbo_mla_preprocess`

The newly explicit metadata is mainly:

- `tp_size`
- `flash_comm_enabled`
- `output_num_tokens`
- `need_gather_q_kv` / `need_all_gather`

These changes primarily fix the compile-time shape contract of the custom ops.
They do not directly redefine the runtime execution policy.

## Why Fullgraph Now Compiles More Easily

`torch.compile(fullgraph=True)` does not require the graph to be independent of
all runtime information. Instead, it requires:

- stable compile-time output metadata inference,
- fake implementations that do not depend on implicit Python runtime state
  outside the graph,
- custom-op fake implementations that behave like local metadata functions.

Before the current changes, fake implementations directly or indirectly depended
on:

- `get_forward_context()`
- `_EXTRA_CTX`
- `enable_sp_by_pass()`
- `get_tensor_model_parallel_world_size()`

The problem was not that these are runtime values. The problem was that fake
implementations queried them internally as external state during fake tensor
propagation.

After the changes, the caller computes:

- `flash_comm_enabled`
- `tp_size`
- `output_num_tokens`

and passes them explicitly into the custom op. As a result:

- compile-time shape inference is more stable,
- fake implementations no longer read `forward_context` directly,
- the custom ops are closer to a compile-safe contract,
- `fullgraph=True` has a better chance of succeeding.

## This Does Not Mean True Per-Run Dynamic Inputs

It is important to distinguish two different properties:

1. **Compileability**
   - Whether Dynamo can successfully build a graph for a concrete trace case.
2. **Runtime polymorphism**
   - Whether the compiled graph can naturally handle later batches with
     different runtime metadata.

Reading `_EXTRA_CTX.flash_comm_v1_enabled` or `_EXTRA_CTX.num_tokens` in Python
and then passing them into a custom op as `bool`/`int` does not automatically
mean that those values remain truly per-run dynamic within one compiled graph.
In the current design, they are more likely to be:

- specialized as Python constants during trace, or
- turned into graph guards that split regimes.

Therefore, the current changes mainly improve compileability. They do not
automatically create one polymorphic graph that handles all later batches.

## Similar Upstream vLLM Situations

Upstream vLLM has similar problems, but its strategy is not "one fully dynamic
graph". Instead, it reduces runtime variability into a limited set of regimes
and reuses multiple graphs through descriptors and caches.

### ForwardContext Allows Runtime Attachments

Upstream `ForwardContext` already allows platform-specific runtime metadata to
be attached through `additional_kwargs`.

Reference:

- <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/forward_context.py>

This means upstream does not assume that all useful execution metadata can be
made static.

### Graph Dispatch Uses a Minimal BatchDescriptor

Upstream uses `BatchDescriptor` as the graph dispatch key. Its fields are
deliberately minimal, such as:

- `num_tokens`
- `num_reqs`
- `uniform`
- `has_lora`
- `num_active_loras`

References:

- <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/forward_context.py>
- <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/compilation/cuda_graph.py>

This shows that upstream accepts multiple graphs, but tries to keep graph
splitting under control by minimizing descriptor dimensions.

### Upstream Also Works to Reduce Opaque Custom Ops

Upstream issue `#31985` explicitly points out that large custom-op wrappers,
especially around MoE paths, interfere with `torch.compile` and complicate
caching and DBO-related paths.

Reference:

- <https://github.com/vllm-project/vllm/issues/31985>

The current vllm-ascend direction is consistent with this overall upstream
strategy: make fake implementations compile-safe first, then control regime
splitting separately.

## Parameter-by-Parameter Analysis

Below is the current assessment of whether each explicit parameter is likely to
introduce a new graph regime dimension.

### `tp_size`

Source:

- `get_tensor_model_parallel_world_size()`
- or cached layer attributes such as `self.tp_size`

Assessment:

- Low risk.
- This is a service-level topology constant, not batch-sensitive metadata.
- Even if trace specializes it once, that is usually expected behavior.

Conclusion:

- It should not create a new graph-splitting dimension.

### `need_gather_q_kv` / `need_all_gather`

Source:

- local MLA / linear path logic

Assessment:

- Low risk.
- This is usually closer to a layer-path choice than a batch-sensitive runtime
  regime.
- If compile specializes it, that typically matches an existing code-path split
  rather than introducing unstable runtime behavior.

Conclusion:

- Not a primary graph-splitting source.

### `dbo_enabled`

Source:

- `create_ascend_forward_context()` sets it for ubatch contexts.

Assessment:

- The current handling is reasonable.
- `dbo_enabled` mainly affects real-implementation overlap hooks and runtime
  communication behavior.
- It does not directly determine fake output shape.
- The current changes do **not** pull it into fake-shape metadata, which should
  remain true.

Conclusion:

- Not a primary compile-time regime dimension.

### `flash_comm_enabled`

Source:

- `set_ascend_forward_context()`
- `create_ascend_forward_context()`

Typical policy:

- dense model: `enable_sp(...) and num_tokens > 1000`
- MoE model: `enable_sp(...) and num_tokens is not None`

Assessment:

- Real regime-splitting dimension.
- Batch-sensitive boolean.
- Can produce at least two graph regimes: `False` and `True`.
- If runtime token counts often oscillate around the threshold, later batches
  may switch between those regimes repeatedly.

Conclusion:

- Creates limited graph splitting, but usually in a bounded number of cases.

### `output_num_tokens`

Source:

- `_EXTRA_CTX.num_tokens`
- effective token count after all-gather + unpad or similar DBO/FlashComm paths

How fake impls use it:

- directly as output dimension-0,
- to model all-gather + unpad or prepare/preprocess output shapes.

Assessment:

- This is the most sensitive parameter in the current design.
- It is not a boolean flag. It is exact shape metadata.
- If specialized exactly, it can easily become a graph-variant dimension.

Conclusion:

- Highest-priority regime parameter to review further.

### `pad_size`

Source:

- derived from `num_tokens` and `tp_size`

Assessment:

- Not usually the root cause.
- It will drift with `num_tokens/output_num_tokens`.
- If `num_tokens` is not fully absorbed by the existing bucket/descriptor
  mechanism, `pad_size` will also behave like an indirect shape-splitting
  factor.

Conclusion:

- Medium risk. Usually secondary to `num_tokens`.

### `flash_comm_v1_snapshot` Getter

Change:

- old code imported `_FLASH_COMM_V1_SNAPSHOT` directly,
- new code calls `get_flash_comm_v1_snapshot()`.

Assessment:

- Correct fix, not a new risk.
- It fixes stale Python module-level `bool` binding behavior.

Conclusion:

- Should be kept.

## Most Important Current Risk: DBO Ubatch Descriptor Alignment

There is one issue that is more concrete than "maybe more recompiles".

In DBO mode:

- each ubatch context sets
  `new_forward_context.num_tokens = ubatch_slices[ubatch_num].num_tokens`,
- `pad_size`, `max_tokens_across_dp`, and `padded_num_tokens` are recomputed
  based on that ubatch,

but:

- `batch_descriptor` appears to be inherited from the outer batch instead of
  being rebuilt for each ubatch.

The current code path indicates:

- outer forward constructs `batch_desc` in `model_runner_v1.py`,
- `set_ascend_forward_context(...)` uses that descriptor,
- `npu_ubatch_wrapper.py` passes the same `batch_descriptor` into every ubatch
  context,
- `create_ascend_forward_context(...)` does not rebuild the descriptor per
  `ubatch_num`.

That suggests:

1. the real ubatch `num_tokens/output_num_tokens` already changed,
2. but the graph dispatch key may still describe the outer batch,
3. while fake-shape contracts may already depend on ubatch-local
   `output_num_tokens`.

If that reading is correct, this is currently the clearest potential
descriptor/shape regime mismatch.

## Risk Ranking

### Reasonable and Should Stay

- explicit `tp_size`
- explicit `need_gather_q_kv` / `need_all_gather`
- snapshot getter change

### Causes Limited Regime Splitting but Still Looks Manageable

- `flash_comm_enabled`

### Highest Priority to Review Further

- `output_num_tokens`
- `pad_size`
- DBO ubatch `num_tokens` versus `batch_descriptor` consistency

## Recommended Follow-Up Checks

### 1. Confirm the Real Graph Dispatch Key in DBO Mode

Need to confirm whether fullgraph / ACL graph replay is dispatched:

- by the outer `batch_descriptor`, or
- by some deeper ubatch-local descriptor/bucket that already absorbs the true
  ubatch shape.

If ubatch-local shape is not explicitly represented in the dispatch key, then
`output_num_tokens` may introduce a compile-time shape dimension that dispatch
does not model.

### 2. Re-evaluate Whether `output_num_tokens` Must Be Exact

If fake implementations require exact token counts, graph variants may grow with
runtime shape diversity.

Need to evaluate whether a more stable bucketed value is possible, for example:

- a padded or bucketed token count already absorbed by graph capture sizing,
- or avoiding exact unpadded token count as compile-time metadata when possible.

### 3. Keep Compile-Safe and Runtime-Polymorphic as Separate Goals

The current changes can be viewed as:

- **compile-safe fix**: mostly achieved,
- **runtime regime control**: still needs alignment with upstream descriptor and
  bucketing principles.

## Conclusion

The current direction is broadly reasonable. It repairs the compile-time shape
contract of DBO custom ops and makes DBO-related paths more compatible with
`torch.compile(fullgraph=True)`.

However, this should not be interpreted as automatic support for one truly
polymorphic graph.

The more accurate description is:

- the system moves from "hard to compile in fullgraph" toward "able to compile
  concrete regimes and cache multiple graphs",
- `tp_size` is safe as a service-level constant,
- `flash_comm_enabled` is a bounded boolean regime split,
- `output_num_tokens` remains a clear risk as exact shape metadata,
- and DBO ubatch descriptor alignment is currently the most concrete potential
  issue to investigate next.

If further risk reduction is required, the priority should be:

1. verify whether DBO ubatch descriptors must be rebuilt,
2. check whether `output_num_tokens` can be absorbed into existing buckets,
3. avoid adding new batch-sensitive exact values to compile-time graph keys.
