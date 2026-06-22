# vLLM-Ascend DBO + MoE + torch.compile 源码导览

本文只做一件事：给代码 dump / trace 一个清晰读图顺序。

主线分三条：

1. **FusedMoE 主线**：`AscendFusedMoE` 如何从 `prepare -> expert MLP -> finalize` 进入 AllGather / AllToAll 通信。
2. **DBO 主线**：batch 如何切成两个 ubatch，两个线程如何靠 hook / event / stream 交替。
3. **compile / CUDAGraph 主线**：`torch.compile`、piecewise FX graph、NPU graph capture/replay 与 DBO 的边界在哪里。

结论先放前面：

- DBO 不等于 torch.compile，也不等于 CUDAGraph。DBO 是 runtime scheduler；torch.compile 是 FX 编译；CUDAGraph/NPUGraph 是设备执行录制。
- vLLM v1 支持 mixed batch cudagraph，但默认 `FULL_AND_PIECEWISE` 语义通常是 decode 走 FULL、mixed/prefill 走 PIECEWISE。
- 当前 DBO 最常见路径是 **PIECEWISE compile + eager DBO**：`CUDAGraphDispatcher` 返回 `PIECEWISE` 时，`AscendUBatchWrapper` 不做 PIECEWISE NPUGraph capture，只启动两个 ubatch 线程跑 model。
- FULL CUDAGraph + DBO 只有在 dispatch 返回 `FULL` 时触发，首次走 `_capture_ubatches()`，之后按 token 数 replay。
- FlashComm + DBO 的 compile 风险点主要在 custom op fake impl 和 live forward context：fake impl 读取 `_EXTRA_CTX`，而 compile 阶段和 runtime 阶段的 `_EXTRA_CTX` 不一定一致。

## 0. 路径索引

当前仓库路径均相对 `/mnt/home/douliyang/mlsys/vllm-ascend`。

| 主题 | 入口 |
|------|------|
| model runner 主线 | `vllm_ascend/worker/model_runner_v1.py` |
| DBO 判断 / ubatch 切分 | `vllm_ascend/worker/ubatch_utils.py` |
| DBO wrapper | `vllm_ascend/worker/npu_ubatch_wrapper.py` |
| DBO 线程 / event / stream | `vllm_ascend/worker/ubatching.py` |
| forward context | `vllm_ascend/ascend_forward_context.py` |
| DBO template 选择 | `vllm_ascend/dbo/utils.py` |
| DBO template 基类 | `vllm_ascend/dbo/overlap_templates/base.py` |
| Qwen3 MoE template | `vllm_ascend/dbo/overlap_templates/qwen3_moe.py` |
| DeepSeek template | `vllm_ascend/dbo/overlap_templates/deepseek.py` |
| FusedMoE 主实现 | `vllm_ascend/ops/fused_moe/fused_moe.py` |
| MoE comm method | `vllm_ascend/ops/fused_moe/moe_comm_method.py` |
| A2 AllGather prepare/finalize | `vllm_ascend/ops/fused_moe/prepare_finalize.py` |
| A3 AllToAll dispatch/combine | `vllm_ascend/ops/fused_moe/token_dispatcher.py` |
| linear DBO hook | `vllm_ascend/ops/linear_op.py` |
| FlashComm custom op / fake impl | `vllm_ascend/ops/register_custom_ops.py` |
| Ascend compile config | `vllm_ascend/platform.py` |
| Ascend compiler | `vllm_ascend/compilation/compiler_interface.py` |
| NPU graph wrapper | `vllm_ascend/compilation/acl_graph.py` |

上游 vLLM 0.20.2 参考：

- `vllm/v1/cudagraph_dispatcher.py`
- `vllm/compilation/piecewise_backend.py`
- `vllm/config/compilation.py`

## 1. 三层边界

先分清三层，后面所有图都按这三层解释。

```text
Layer A: torch.compile / FX graph
  Dynamo 捕获 Python forward，生成 FX Graph。
  PiecewiseBackend 可按 splitting_ops 把 FX Graph 切成多个 piece。

Layer B: CUDAGraph / NPUGraph
  torch.npu.graph() 录制设备执行。
  Ascend 侧存储在 ACLGraphWrapper 或 AscendUBatchWrapper.cudagraphs。

Layer C: DBO runtime
  CPU 线程、forward_context、NPU stream、event、yield。
  入口是 AscendUBatchWrapper + overlap template hook。
```

不要把这三件事混成一个状态机：

| 名词 | 真实含义 |
|------|----------|
| `PIECEWISE` compile | FX Graph 按 splitting_ops 切 piece。不是按 token range 切。 |
| `PIECEWISE` CUDAGraph dispatch | 当前 batch 满足 piecewise graph key。只说明 dispatch 结果，不说明 graph object 已存在。 |
| DBO | 两个 ubatch 的 runtime overlap。是否使用 NPUGraph 取决于 dispatch 是否为 `FULL`。 |

当前 `AscendUBatchWrapper` 的关键分支：

```text
有 ubatch_slices:
  dispatch == FULL 且未捕获:
    _capture_ubatches()

  dispatch == FULL 且已捕获:
    cudagraph.replay()

  dispatch == PIECEWISE/NONE:
    _run_ubatches()  # eager DBO，无 ubatch NPUGraph capture/replay
```

源码：`vllm_ascend/worker/npu_ubatch_wrapper.py:346`。

## 2. 总时序

一次 step 的源码路径：

```text
execute_model()
  -> _prepare_inputs()
  -> _determine_batch_execution_and_padding()
       -> CUDAGraphDispatcher.dispatch()
       -> check_enable_ubatch()
  -> maybe_create_ubatch_slices()
  -> _build_attention_metadata()
  -> set_ascend_forward_context(...)
  -> _model_forward()
  -> self.model(...)
       -> AscendUBatchWrapper.__call__()        # enable_dbo 时
          -> _make_ubatch_metadata()
          -> _run_ubatches() / _capture_ubatches() / replay()
             -> child thread model(...)
                -> linear / attention / AscendFusedMoE
```

关键代码：

- `execute_model`: `vllm_ascend/worker/model_runner_v1.py:1699`
- batch mode / DBO 判断: `vllm_ascend/worker/model_runner_v1.py:2661`
- ubatch 切分: `vllm_ascend/worker/ubatch_utils.py:96`
- wrapper 分流: `vllm_ascend/worker/npu_ubatch_wrapper.py:346`
- DBO thread context: `vllm_ascend/worker/ubatching.py:67`
- MoE forward: `vllm_ascend/ops/fused_moe/fused_moe.py:584`

### 2.1 PIECEWISE compile + eager DBO 时序

这是当前最常见的 DBO 运行形态：dispatch 返回 `PIECEWISE`，但 DBO wrapper 不 capture PIECEWISE NPUGraph。

```text
ModelRunner.execute_model()
│
├─ _determine_batch_execution_and_padding()
│   ├─ CUDAGraphDispatcher.dispatch()
│   │   └─ returns CUDAGraphMode.PIECEWISE
│   └─ check_enable_ubatch()
│       └─ should_ubatch = True
│
├─ maybe_create_ubatch_slices()
│   └─ [ubatch0 token_slice, ubatch1 token_slice]
│
├─ set_ascend_forward_context()
│   ├─ cudagraph_runtime_mode = PIECEWISE
│   ├─ ubatch_slices = [...]
│   └─ dbo_enabled = False              # 外层 context
│
└─ self.model(...)
    └─ AscendUBatchWrapper.__call__()
        ├─ _make_ubatch_metadata()
        │   ├─ ubatch0 context: dbo_enabled = True
        │   └─ ubatch1 context: dbo_enabled = True
        │
        └─ _run_ubatches()              # PIECEWISE/NONE 分支
            ├─ thread0: model(ubatch0)  # eager 或 compiled pieces
            └─ thread1: model(ubatch1)  # eager 或 compiled pieces
```

读图要点：`PIECEWISE` 在这里主要描述 FX compile / dispatch eligibility；实际 DBO 线程路径是 `_run_ubatches()`。

### 2.2 warmup shape -> FULL dispatch -> DBO capture/replay 完整时序

DBO 不自己判断某个 warmup shape 是否能 FULL capture。FULL eligibility 在 attention backend 初始化后已经由 `CudagraphDispatcher.initialize_cudagraph_keys()` 预先生成：它根据 resolved `cudagraph_mode`、`cudagraph_capture_sizes`、uniform decode query length、LoRA cases 等创建一批 `BatchDescriptor` key。

因此 warmup 阶段的 `dispatch()` 不是查“已经捕获好的 graph object”，而是查“这个 warmup/runtime batch descriptor 是否属于预先允许 capture/replay 的 FULL key 集合”。只有 dispatcher 返回 `FULL`，并且模型已经被 `AscendUBatchWrapper(..., CUDAGraphMode.FULL, ...)` 包住，DBO wrapper 才会真正尝试 `_capture_ubatches()`。

完整链路如下：

```text
attention backend / dispatcher 初始化阶段
│
├─ initialize_attn_backend()
│   └─ _check_and_update_cudagraph_mode(...)
│       ├─ attention backend 上报 cudagraph support
│       ├─ compilation_config.resolve_cudagraph_mode_and_sizes(...)
│       └─ cudagraph_dispatcher.initialize_cudagraph_keys(resolved_mode, ...)
│           ├─ 根据 cudagraph_capture_sizes 生成 padded size 映射
│           ├─ mixed_mode 生成 PIECEWISE/FULL mixed keys
│           └─ decode_mode == FULL 时，为 uniform decode 生成 FULL keys
│

load_model() 阶段
│
├─ compilation_config.cudagraph_mode.has_full_cudagraphs()?
│
├─ DBO off:
│   └─ self.model = ACLGraphWrapper(..., runtime_mode=FULL)
│
└─ DBO on:
    └─ self.model = AscendUBatchWrapper(..., runtime_mode=FULL)


Graph warmup / capture 阶段
│
├─ capture_model()                         # Ascend 包一层后调用上游 GPUModelRunner.capture_model
│
├─ for each capture descriptor / warmup shape:
│   └─ _dummy_run(
│        num_tokens=warmup_num_tokens,
│        num_reqs=warmup_num_reqs,
│        cudagraph_runtime_mode=expected_mode,
│        allow_microbatching=...,
│        is_graph_capturing=True,
│      )
│
│      ├─ 构造 warmup batch
│      │   ├─ num_scheduled_tokens
│      │   ├─ num_tokens_unpadded
│      │   ├─ max_query_len
│      │   └─ uniform_decode / lora state
│      │
│      ├─ _determine_batch_execution_and_padding(...)
│      │   │
│      │   ├─ dispatch_cudagraph(num_tokens_padded, disable_full=...)
│      │   │   └─ CUDAGraphDispatcher.dispatch(...)
│      │   │       ├─ 先把 runtime batch 归一化成 BatchDescriptor
│      │   │       ├─ 查预先初始化的 FULL eligibility key
│      │   │       ├─ 查预先初始化的 PIECEWISE eligibility key
│      │   │       └─ returns (_cudagraph_mode, batch_desc)
│      │   │
│      │   ├─ check_enable_ubatch(..., moe_comm_type=select_moe_comm_method(...))
│      │   │   └─ returns should_ubatch
│      │   │
│      │   └─ returns (_cudagraph_mode, batch_desc, should_ubatch, ...)
│      │
│      ├─ if caller passed cudagraph_runtime_mode:
│      │   └─ assert cudagraph_runtime_mode == _cudagraph_mode
│      │      # 这里校验 warmup shape 实际 dispatch 结果是否符合预期
│      │
│      ├─ maybe_create_ubatch_slices(should_ubatch, ...)
│      │   └─ returns ubatch_slices / ubatch_slices_padded
│      │
│      ├─ _build_attention_metadata(...,
│      │     ubatch_slices = ubatch_slices_padded if FULL else ubatch_slices,
│      │     for_cudagraph_capture = True,
│      │   )
│      │
│      ├─ set_ascend_forward_context(...,
│      │     aclgraph_runtime_mode = cudagraph_runtime_mode,
│      │     batch_descriptor = batch_desc,
│      │     ubatch_slices = ubatch_slices_padded,
│      │   )
│      │
│      └─ _model_forward(...)
│          └─ self.model(...)              # DBO on 时是 AscendUBatchWrapper
│              │
│              ├─ if ubatch_slices is None:
│              │   └─ 非 DBO / fallback 路径
│              │
│              ├─ if cudagraph_runtime_mode == FULL
│              │      and num_tokens not in self.cudagraphs:
│              │   │
│              │   ├─ _make_ubatch_metadata(..., cudagraph_runtime_mode=NONE)
│              │   │   ├─ ubatch0 forward_context: dbo_enabled=True
│              │   │   └─ ubatch1 forward_context: dbo_enabled=True
│              │   │
│              │   └─ _capture_ubatches()
│              │       ├─ start thread0/thread1
│              │       │   └─ 线程进入 AscendUBatchContext 后等待 cpu event
│              │       │
│              │       ├─ with torch.npu.graph(aclgraph, stream=compute_stream):
│              │       │   ├─ wake ubatch0
│              │       │   ├─ thread0/thread1 通过 DBO hook/yield 交替执行
│              │       │   ├─ model forward enqueue NPU kernel/comm task
│              │       │   ├─ join thread0/thread1
│              │       │   └─ cat outputs
│              │       │
│              │       └─ self.cudagraphs[num_tokens] = aclgraph metadata
│              │
│              ├─ elif cudagraph_runtime_mode == FULL
│              │        and num_tokens in self.cudagraphs:
│              │   └─ self.cudagraphs[num_tokens].aclgraph.replay()
│              │
│              └─ else:  # PIECEWISE / NONE
│                  └─ _run_ubatches()      # eager DBO；不 capture ubatch NPUGraph
```

读图要点：

- `FULL` 的来源不是 capture 成功与否，也不是 torch.compile fullgraph；它来自 dispatcher 预先初始化的 FULL eligibility key。
- warmup 阶段命中的是 `BatchDescriptor` key，不是已经存在的 graph object。真正 graph object 在后面的 `_capture_ubatches()` 里创建并写入 `self.cudagraphs`。
- `_dummy_run()` 会 assert 调用方期望的 `cudagraph_runtime_mode` 和实际 dispatch 结果一致；不一致说明这个 warmup shape 不在预期的 FULL eligibility 集合里。
- `_capture_ubatches()` 是真正的 DBO NPUGraph capture 点。它 capture 的是两个 ubatch 线程在 capture 上下文内 enqueue 到 NPU stream 的设备任务。
- capture 流程里传给 `_make_ubatch_metadata()` 的 `cudagraph_runtime_mode=CUDAGraphMode.NONE` 是为了避免 ubatch 内部递归触发 graph capture；外层是否 capture 已由 wrapper 分支决定。

关键代码位置：

- 上游 key 初始化：`vllm/v1/cudagraph_dispatcher.py:166`
- 上游 dispatch 查 key：`vllm/v1/cudagraph_dispatcher.py:235`
- 上游 capture desc 遍历：`vllm/v1/worker/gpu_model_runner.py:6558`
- 上游 warmup/capture 调 `_dummy_run()`：`vllm/v1/worker/gpu_model_runner.py:6601`
- Ascend 模型包装：`vllm_ascend/worker/model_runner_v1.py:3517`
- warmup shape dispatch：`vllm_ascend/worker/model_runner_v1.py:3193`
- warmup mode assert：`vllm_ascend/worker/model_runner_v1.py:3226`
- ubatch slice 创建：`vllm_ascend/worker/model_runner_v1.py:3240`
- FULL attention metadata padding：`vllm_ascend/worker/model_runner_v1.py:3298`
- forward context 写入 mode / descriptor / ubatch：`vllm_ascend/worker/model_runner_v1.py:3379`
- DBO wrapper FULL capture 分支：`vllm_ascend/worker/npu_ubatch_wrapper.py:400`
- 真正 `torch.npu.graph(...)` capture 点：`vllm_ascend/worker/npu_ubatch_wrapper.py:185`
- replay 分支：`vllm_ascend/worker/npu_ubatch_wrapper.py:419`

### 2.3 DBO 线程切换时序

`_run_ubatches()` 和 `_capture_ubatches()` 都依赖同一套 ubatch context/yield 机制。差别只是外层是否包了 `torch.npu.graph()`。

```text
main thread
│
├─ start thread0
├─ start thread1
├─ ready_barrier.wait()
└─ ubatch0.cpu_wait_event.set()

thread0 / ubatch0                       thread1 / ubatch1
│                                       │
├─ ready_barrier.wait()                 ├─ ready_barrier.wait()
├─ wait cpu_wait_event                  ├─ wait cpu_wait_event
├─ restore forward_context(ubatch0)     │
├─ switch compute_stream0               │
├─ run model until hook wait/yield       │
├─ signal thread1 cpu event ───────────► ├─ restore forward_context(ubatch1)
│                                       ├─ switch compute_stream1
│                                       ├─ run model until hook wait/yield
├─ wait own cpu event ◄───────────────── ├─ signal thread0 cpu event
├─ continue model                       ├─ wait own cpu event
└─ final synchronize + yield            └─ final synchronize + yield
```

DBO hook 只负责在合适位置 `record event`、`wait event`、`yield CPU`。真正的通信/计算仍由 linear、attention、MoE 代码发起。

## 3. FusedMoE 主线

### 3.1 MoE forward

`AscendFusedMoE.forward_impl()` 是 MoE 的主入口：

```text
AscendFusedMoE.forward_impl()
  -> _EXTRA_CTX.moe_comm_method.prepare(...)
  -> quant_method.apply(...)
       -> select_experts
       -> token dispatch
       -> expert MLP
       -> token combine
  -> _EXTRA_CTX.moe_comm_method.finalize(...)
```

源码：

- `forward_impl`: `vllm_ascend/ops/fused_moe/fused_moe.py:584`
- `prepare`: `vllm_ascend/ops/fused_moe/fused_moe.py:643`
- expert MLP: `vllm_ascend/ops/fused_moe/fused_moe.py:661`
- `finalize`: `vllm_ascend/ops/fused_moe/fused_moe.py:705`

### 3.2 comm method 选择

`set_ascend_forward_context()` 会选择 `moe_comm_type` 和 `moe_comm_method`。实际规则在：

- `MoECommType`: `vllm_ascend/ascend_forward_context.py:27`
- `select_moe_comm_method`: `vllm_ascend/ascend_forward_context.py:366`
- `get_moe_comm_method`: `vllm_ascend/ops/fused_moe/moe_comm_method.py:51`

粗略读法：

| 场景 | 常见通信 |
|------|----------|
| A2 / 310P / 小规模 fallback | `ALLGATHER` |
| A3 大 token EP | `ALLTOALL` |
| 小 token 且满足条件 | `MC2` / `FUSED_MC2` |

注意：`check_enable_ubatch()` 明确排除 `MoECommType.MC2`，所以 DBO 主要看 `ALLGATHER` 和 `ALLTOALL`。

### 3.3 A2 AllGather / FlashComm 路径

入口：`PrepareAndFinalizeWithAllGather`。

prepare 的 DBO hook 落点：

```text
_prepare_with_ep_group()
  if dbo_enabled:
    dbo_moe_prepare_hook(True)
    tensor_model_parallel_all_gather / ep_group.all_gather
    dbo_moe_prepare_hook(False)
    maybe_all_gather_and_maybe_unpad(..., do_comm=False)
  else:
    maybe_all_gather_and_maybe_unpad(..., do_comm=True)
```

源码：`vllm_ascend/ops/fused_moe/prepare_finalize.py:369`。

finalize 的 DBO hook 落点：

```text
_finalize_with_ep_group()
  if dbo_enabled:
    maybe_pad_and_reduce(..., do_comm=False)
    dbo_moe_finalize_hook(True)
    tensor_model_parallel_reduce_scatter / ep_group.reduce_scatter
    dbo_moe_finalize_hook(False)
  else:
    maybe_pad_and_reduce(..., do_comm=True)
```

源码：`vllm_ascend/ops/fused_moe/prepare_finalize.py:515`。

### 3.4 A3 AllToAll 路径

入口：`MoEAlltoAllTokenDispatcher`。

```text
token_dispatch()
  _dispatch_preprocess()
  dbo_moe_prepare_hook(True)
  async_all_to_all(...).wait()
  dbo_moe_prepare_hook(False)
  _dispatch_postprocess()

token_combine()
  _combine_preprocess()
  dbo_moe_finalize_hook(True)
  async_all_to_all(...).wait()
  dbo_moe_finalize_hook(False)
```

源码：

- dispatch: `vllm_ascend/ops/fused_moe/token_dispatcher.py:466`
- combine: `vllm_ascend/ops/fused_moe/token_dispatcher.py:532`

## 4. DBO 主线

### 4.1 DBO 启用条件

入口：`check_enable_ubatch()`。

必须满足：

1. `parallel_config.enable_dbo == True`
2. 上游 `check_ubatch_thresholds()` 通过
3. `moe_comm_type != MoECommType.MC2`
4. padding 后最后一个 ubatch 不是空的

源码：`vllm_ascend/worker/ubatch_utils.py:68`。

### 4.2 ubatch 切分

入口：`maybe_create_ubatch_slices()`。

默认两个 ubatch，按 token 总数二分：

```text
split_point = num_tokens_padded // num_ubatches
token_split_points = [split_point]
create_ubatch_slices(...)
_pad_out_ubatch_slices(...)  # padding 落到最后一个 ubatch
```

切分结果同时有：

```text
UBatchSlice(
  request_slice = 覆盖的 request 范围,
  token_slice   = 覆盖的 token 范围,
)
```

源码：`vllm_ascend/worker/ubatch_utils.py:96`。

### 4.3 wrapper 分流

`AscendUBatchWrapper.__call__()` 的读法：

```text
无 ubatch_slices:
  FULL -> ACLGraphWrapper
  PIECEWISE/NONE -> runnable eager/compiled callable

有 ubatch_slices:
  FULL + 未捕获 -> _capture_ubatches()
  FULL + 已捕获 -> replay()
  PIECEWISE/NONE -> _run_ubatches()
```

源码：`vllm_ascend/worker/npu_ubatch_wrapper.py:346`。

需要注意一个现状：DBO graph key 使用：

```python
num_tokens = (ubatch_slices[0].token_slice.stop
              - ubatch_slices[0].token_slice.start) * 2
```

这隐含两个 ubatch 等长。若 request boundary、padding 或未来 `num_ubatches != 2` 让切分不等长，这个 key 可能不等于真实 padded token 总数。调试 FULL DBO replay 命中率时要先看这里。

### 4.4 两个线程如何启动

`_run_ubatches()`：

```text
main thread:
  override_forward_context(None)
  start thread0
  start thread1
  ready_barrier.wait()
  wake ubatch0
  join thread0/thread1
  concat outputs

ubatch thread:
  with AscendUBatchContext:
    wait ready_barrier
    wait cpu_wait_event
    restore ubatch forward_context
    switch to compute_stream
    model(...)
    current_stream.synchronize()
    dbo_yield()
```

源码：

- `_run_ubatches`: `vllm_ascend/worker/npu_ubatch_wrapper.py:194`
- `AscendUBatchContext.__enter__`: `vllm_ascend/worker/ubatching.py:67`
- `_cpu_yield`: `vllm_ascend/worker/ubatching.py:107`

### 4.5 stream 分配

`make_ubatch_contexts()` 给两个 ubatch 交错分配 stream：

```text
ubatch0:
  compute_stream = 外层 current stream
  comm_stream    = wrapper.comm_stream

ubatch1:
  compute_stream = wrapper.comm_stream
  comm_stream    = 外层 current stream
```

源码：`vllm_ascend/worker/ubatching.py:209`。

### 4.6 hook 的真实语义

template hook 最后都落到这两个函数：

```text
dbo_record_current_stream(event)
  -> record compute_done event on current context

dbo_wait_current_stream_and_yield(event, wait=True)
  -> current compute stream waits event
  -> CPU yield to another ubatch thread
```

源码：`vllm_ascend/worker/ubatching.py:130`、`vllm_ascend/worker/ubatching.py:139`。

`_register_ubatch_function()` 用 `threading.get_ident()` 找当前线程对应的 ubatch context。这里是 runtime scheduler 逻辑，不适合进入 FX compiled graph。

## 5. 校验后的 DBO 时序图

下面的图按“单层 hook 顺序”和“两 ubatch overlap 关系”分开看。前者用于对源码，后者用于对 trace。

### 5.1 Qwen3 MoE / A2 AllGather

模板：`QwenMoEAllgatherTemplate`。

单层 hook 顺序：

```text
linear column:
  dbo_linear_column_hook(True)    # QKV/MLP AllGather 前
  [AllGather]
  dbo_linear_column_hook(False)   # wait ATTN_PRE, yield

attention compute

linear row:
  dbo_linear_row_hook(True)       # o_proj comm 前 record ATTN_POST
  [o_proj ReduceScatter / comm]
  # A2 template 的 row_hook(False) 不做事

MoE prepare:
  dbo_moe_prepare_hook(True)      # A2 template 不做事
  [EP AllGather]
  dbo_moe_prepare_hook(False)     # wait ATTN_POST, yield

expert compute

MoE finalize:
  dbo_moe_finalize_hook(True)     # record ATTN_PRE
  [EP ReduceScatter / AllReduce]
  dbo_moe_finalize_hook(False)    # A2 template 不做事
```

对应 overlap：

```text
ubatch0 o_proj comm        overlaps ubatch1 attention compute
ubatch1 o_proj comm        overlaps ubatch0 MoE prepare / expert window
ubatch0 MoE finalize comm  overlaps ubatch1 next-layer QKV/attention pre-window
```


两 ubatch 的相对时序可以这样看：

```text
时间 ─────────────────────────────────────────────────────────►

ubatch0:
  QKV AG | Attention compute | o_proj RS | MoE AG | Expert | MoE RS | next QKV AG
          record ATTN_POST ──┐            wait ATTN_POST
                              │
                              └──────────── overlaps ───────────────┐

ubatch1:
          wait ATTN_PRE | QKV AG | Attention compute | o_proj RS | MoE AG | Expert
          ▲                                                       ▲
          │                                                       │
  first layer 或上一层 MoE finalize record ATTN_PRE               ubatch0 MoE finalize record ATTN_PRE
```

A2 template 没有为 every communication 都配成对的 hook；它把 o_proj post-comm、MoE prepare、MoE finalize 这几个窗口串起来，重点覆盖 attention 后通信和 MoE 通信。

源码校验：

- template: `vllm_ascend/dbo/overlap_templates/qwen3_moe.py:7`
- linear column hook: `vllm_ascend/ops/linear_op.py:439`
- MoE prepare hook: `vllm_ascend/ops/fused_moe/prepare_finalize.py:390`
- MoE finalize hook: `vllm_ascend/ops/fused_moe/prepare_finalize.py:527`

### 5.2 Qwen3 MoE / A3 AllToAll

模板：`QwenMoEAlltoallTemplate`。

单层 hook 顺序：

```text
linear column:
  dbo_linear_column_hook(True)
  [QKV AllGather if needed]
  dbo_linear_column_hook(False)   # wait ATTN_PRE, yield

attention compute

linear row:
  dbo_linear_row_hook(True)       # record ATTN_POST
  [o_proj comm]
  dbo_linear_row_hook(False)      # yield, wait=False

MoE dispatch:
  dbo_moe_prepare_hook(True)      # record MOE_DISPATCH
  [Dispatch AllToAll + wait]
  dbo_moe_prepare_hook(False)     # wait MOE_DISPATCH, yield

expert compute

MoE combine:
  dbo_moe_finalize_hook(True)     # record ATTN_PRE
  [Combine AllToAll + wait]
  dbo_moe_finalize_hook(False)    # template 不做事
```

对应 overlap：

```text
ubatch0 o_proj comm        overlaps ubatch1 attention compute
ubatch0 dispatch AllToAll  overlaps ubatch1 o_proj / pre-MoE window
ubatch0 combine AllToAll   overlaps ubatch1 expert compute
ubatch1 finalize           prepares ubatch0 next layer ATTN_PRE
```


两 ubatch 的相对时序：

```text
时间 ─────────────────────────────────────────────────────────►

ubatch0:
  QKV AG | Attention | o_proj comm | Dispatch A2A | Expert | Combine A2A | next QKV
                       record ATTN_POST
                                     record/wait MOE_DISPATCH
                                                        record ATTN_PRE

ubatch1:
          wait ATTN_PRE | QKV AG | Attention | o_proj comm | Dispatch A2A | Expert | Combine A2A
                         ▲                 ▲              ▲
                         │                 │              │
          ubatch0 prev/finalize ATTN_PRE    overlaps       overlaps ubatch0 Expert/Combine window
```

A3 的 dispatch/combine 是 AllToAll，hook 落在 `token_dispatch()` 和 `token_combine()` 内部；因此对 trace 时应搜索 dispatch/combine 的 HCCL alltoall，而不是 AllGather/ReduceScatter。

源码校验：

- template: `vllm_ascend/dbo/overlap_templates/qwen3_moe.py:56`
- dispatch hook: `vllm_ascend/ops/fused_moe/token_dispatcher.py:486`
- combine hook: `vllm_ascend/ops/fused_moe/token_dispatcher.py:540`

### 5.3 first layer 特例

`ATTN_PRE` 通常由上一层 MoE finalize record。第一层没有上一层，所以 column hook 首次主动 record：

```text
if dbo_first_layer_sync:
  record ATTN_PRE
  dbo_first_layer_sync = False
```

源码：

- A2: `vllm_ascend/dbo/overlap_templates/qwen3_moe.py:30`
- A3: `vllm_ascend/dbo/overlap_templates/qwen3_moe.py:82`

## 6. compile / CUDAGraph 主线

### 6.1 实际组合关系

```text
torch.compile piecewise:
  编译的是 FX Graph piece。
  是否有 DBO，取决于 trace 时 DBO 分支是否进入 FX Graph。

CUDAGraphDispatcher:
  只返回当前 batch 的 CUDAGraphMode 和 BatchDescriptor。
  key 命中不代表 graph object 已捕获。

AscendUBatchWrapper:
  只在 runtime mode == FULL 时 capture/replay ubatch NPUGraph。
  runtime mode == PIECEWISE/NONE 时走 _run_ubatches()。
```

因此要避免一句话说“PIECEWISE CUDAGraph + DBO”。更准确是：

```text
PIECEWISE FX compile + eager DBO
FULL NPUGraph + DBO capture/replay
```

### 6.2 DBO + FULL NPUGraph

首次 FULL：

```text
AscendUBatchWrapper.__call__()
  -> _make_ubatch_metadata(..., cudagraph_runtime_mode=NONE)
  -> _capture_ubatches()
       -> start ubatch threads
       -> torch.npu.graph(...)
       -> join threads inside graph capture
       -> cat outputs
       -> save self.cudagraphs[num_tokens]
```

之后 FULL：

```text
AscendUBatchWrapper.__call__()
  -> self.cudagraphs[num_tokens].aclgraph.replay()
  -> return cached outputs tensor
```

源码：

- capture 分支: `vllm_ascend/worker/npu_ubatch_wrapper.py:397`
- replay 分支: `vllm_ascend/worker/npu_ubatch_wrapper.py:415`

### 6.3 DBO + PIECEWISE compile

PIECEWISE dispatch 下，wrapper 走：

```text
AscendUBatchWrapper.__call__()
  -> _make_ubatch_metadata(..., cudagraph_runtime_mode=NONE)
  -> _run_ubatches()
       -> thread0 model(...)
       -> thread1 model(...)
```

如果 model 已被 `torch.compile` / PiecewiseBackend 包过，子线程里调用的是 compiled pieces；但 DBO runtime 本身仍在 Python 线程调度层。

需要重点 dump：

1. compiled FX Graph 中是否包含 `_EXTRA_CTX.dbo_enabled` 分支。
2. DBO hook 是否进入 compiled region。
3. custom op fake impl 的 shape 是否与 runtime impl 一致。


### 6.4 compile、DBO、NPUGraph 的组合矩阵

```text
                       DBO off                         DBO on

FX compile NONE        eager model                     eager DBO

FX compile PIECEWISE   compiled pieces                 compiled pieces inside
                                                       _run_ubatches()
                                                       no DBO NPUGraph capture

CUDAGraph FULL         ACLGraphWrapper capture/replay  AscendUBatchWrapper
                                                       _capture_ubatches()/replay
```

这张矩阵是排查误解的关键：

- `PIECEWISE` 和 `FULL` 不是同一层的互斥“执行引擎”。
- vLLM v1 mixed batch 可以有 cudagraph，但默认是 PIECEWISE cudagraph，不是 DBO wrapper 当前实现的 FULL ubatch NPUGraph。
- piecewise compile 后，DBO 可以调度 compiled pieces。
- 但当前 DBO wrapper 只在 `CUDAGraphMode.FULL` 下做 ubatch NPUGraph。
- 因此“DBO + piecewise 编译”不等于“DBO + piecewise CUDAGraph”。

### 6.5 FULL capture 的代码级时序

对应 `_capture_ubatches()`：

```text
_capture_ubatches()
│
├─ define _capture_ubatch_thread()
│   ├─ set_device
│   ├─ init compute/comm stream blas handle
│   └─ with ubatch_context:
│       └─ model(ubatch inputs)
│
├─ with override_forward_context(None):
│   ├─ start thread0
│   ├─ start thread1
│   ├─ ready_barrier.wait()
│   │
│   ├─ cudagraph_metadata = AscendNPUGraphMetaData(torch.npu.NPUGraph())
│   │
│   └─ with torch.npu.graph(...):
│       ├─ ubatch0.cpu_wait_event.set()
│       ├─ thread0.join()
│       ├─ thread1.join()
│       ├─ torch.cat(outputs)
│       └─ save outputs tensor
│
└─ self.cudagraphs[num_tokens] = cudagraph_metadata
```

这里的 `join()` 在 capture 上下文内，是为了让两个子线程在 capture 生命周期内完成设备任务 enqueue。后续 replay 走 `aclgraph.replay()`，不会再跑这段 Python 调度。

## 7. FlashComm + compile 风险点

### 7.1 已确认风险：fake impl 读 live context

`maybe_all_gather_and_maybe_unpad` 的真实 impl 会读当前 forward context：

```text
_maybe_all_gather_and_maybe_unpad_impl()
  -> get_forward_context()
  -> _EXTRA_CTX.flash_comm_v1_enabled
  -> _EXTRA_CTX.pad_size
  -> dp_metadata / tp group / ep group
```

fake impl 也读 `_EXTRA_CTX.flash_comm_v1_enabled`：

```text
_maybe_all_gather_and_maybe_unpad_fake()
  if _EXTRA_CTX.flash_comm_v1_enabled and label and do_comm:
    return shape multiplied by TP size
  return x
```

源码：`vllm_ascend/ops/register_custom_ops.py:125`。

问题：

| 阶段 | `_EXTRA_CTX.flash_comm_v1_enabled` | fake shape |
|------|------------------------------------|------------|
| compile / fake trace | 可能是 `None` 或默认值 | 可能返回原 shape |
| runtime | 可能是 `True` | 真实通信后第一维按 TP/EP 变化 |

`maybe_pad_and_reduce` 同理：

- impl: `vllm_ascend/ops/register_custom_ops.py:80`
- fake: `vllm_ascend/ops/register_custom_ops.py:136`

### 7.2 DBO hook 是否被 prune 需要 dump 验证

编译时如果 `dbo_enabled=False`，Dynamo 可能：

1. 给 `_EXTRA_CTX.dbo_enabled` 建 guard，runtime 变化后 recompile。
2. 把 False 当常量剪掉 DBO 分支。
3. 因 live proxy / thread-local 读取导致 guard 不完整。

这不能靠文档断言，需要 dump 验证：

- FX code 里是否还有 `dbo_template.dbo_*_hook`
- Dynamo guard 里是否有 `dbo_enabled`
- runtime ubatch 子线程是否触发 recompile
- compiled piece 的输入 shape 是否等于 ubatch shape

### 7.3 长期边界

更稳的边界是：

```text
FX Graph 外:
  DBO scheduler
  CPU thread/event/yield
  stream 切换
  live forward_context dispatch

FX Graph 内:
  single-ubatch tensor compute
  shape 由显式参数决定的 opaque custom op

可选 NPUGraph:
  在更外层 capture 已经稳定的 callable 执行
```

短期至少要让 fake impl 纯 shape 推断：不要读 `_EXTRA_CTX`、`get_forward_context()`、`get_tp_group()` 这类 runtime 状态；把 `flash_comm_enabled`、`tp_size`、`pad_size`、`do_comm` 等作为显式参数传入。

## 8. 推荐 dump 顺序

按下面顺序 dump，最容易定位问题属于哪条主线。

### 8.1 batch / DBO 决策

看：

- `num_tokens_unpadded`
- `num_tokens_padded`
- `uniform_decode`
- `cudagraph_mode`
- `batch_descriptor`
- `moe_comm_type`
- `should_ubatch`
- `ubatch_slices`

入口：

- `vllm_ascend/worker/model_runner_v1.py:2661`
- `vllm_ascend/worker/ubatch_utils.py:68`
- `vllm_ascend/worker/ubatch_utils.py:96`

### 8.2 wrapper 运行路径

给三条路径加计数：

- `full_dbo_capture_count`
- `full_dbo_replay_count`
- `eager_dbo_count`

入口：`vllm_ascend/worker/npu_ubatch_wrapper.py:346`。

特别看：

- `forward_context.cudagraph_runtime_mode`
- `forward_context.ubatch_slices`
- `num_tokens` key
- `num_tokens in self.cudagraphs`

### 8.3 ubatch context

dump：

- ubatch id
- thread id
- token slice
- request slice
- compute stream
- comm stream
- `dbo_enabled`
- `flash_comm_v1_enabled`
- `moe_comm_type`
- `pad_size`

入口：

- `_make_ubatch_metadata`: `vllm_ascend/worker/npu_ubatch_wrapper.py:255`
- `AscendUBatchContext.__enter__`: `vllm_ascend/worker/ubatching.py:67`

### 8.4 hook 顺序

在 template hook 里 dump：

- layer index
- hook name
- `is_record`
- event key
- ubatch id
- current stream

优先看：

- `vllm_ascend/dbo/overlap_templates/qwen3_moe.py`
- `vllm_ascend/dbo/overlap_templates/deepseek.py`

### 8.5 MoE comm

dump：

- `moe_comm_type`
- `type(_EXTRA_CTX.moe_comm_method)`
- `prepare` 输入/输出 shape
- `fused_experts` 输入/输出 shape
- `finalize` 输入/输出 shape
- AllGather / AllToAll hook 前后 shape

入口：

- `vllm_ascend/ops/fused_moe/fused_moe.py:584`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py:369`
- `vllm_ascend/ops/fused_moe/token_dispatcher.py:466`

### 8.6 compile / fake impl

dump：

- FX Graph code
- Dynamo guards
- recompilation count
- fake impl 输入/输出 shape
- runtime impl 输入/输出 shape
- `_EXTRA_CTX.flash_comm_v1_enabled`
- `_EXTRA_CTX.pad_size`
- `do_comm`
- `label`

入口：

- `vllm_ascend/ops/register_custom_ops.py:40`
- `vllm_ascend/ops/register_custom_ops.py:80`
- `vllm_ascend/ops/register_custom_ops.py:125`
- `vllm_ascend/ops/register_custom_ops.py:136`

## 9. 一句话判断问题归属

| 现象 | 优先看 |
|------|--------|
| DBO 没启动 | `check_enable_ubatch()`、`moe_comm_type`、token 阈值、最后 ubatch 是否为空 |
| FULL graph 不 replay | `cudagraph_runtime_mode`、`num_tokens` key、`self.cudagraphs` |
| PIECEWISE 下以为有 graph | 先确认是不是 `_run_ubatches()`，PIECEWISE 通常是 eager DBO |
| FlashComm ON compile shape 错 | `register_custom_ops.py` fake impl 与 runtime impl |
| hook 顺序不对 | template + `linear_op.py` + `prepare_finalize.py` / `token_dispatcher.py` |
| 子线程 context 错 | `AscendUBatchContext.__enter__` 和 `_cpu_yield` |
| 性能没有 overlap | event key 是否成对、record/wait 是否落在通信前后、stream 是否交错 |

## 10. 最小阅读路线

只想快速跟一遍代码，按这个顺序：

1. `vllm_ascend/worker/model_runner_v1.py:2661`
2. `vllm_ascend/worker/ubatch_utils.py:68`
3. `vllm_ascend/worker/ubatch_utils.py:96`
4. `vllm_ascend/worker/npu_ubatch_wrapper.py:346`
5. `vllm_ascend/worker/ubatching.py:67`
6. `vllm_ascend/dbo/overlap_templates/qwen3_moe.py:7`
7. `vllm_ascend/ops/linear_op.py:439`
8. `vllm_ascend/ops/fused_moe/fused_moe.py:584`
9. `vllm_ascend/ops/fused_moe/prepare_finalize.py:369`
10. `vllm_ascend/ops/fused_moe/token_dispatcher.py:466`
11. `vllm_ascend/ops/register_custom_ops.py:125`
