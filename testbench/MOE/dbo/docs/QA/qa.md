# DBO、TorchDynamo 与 ACLGraph 捕获/回放 Q&A

本文结合当前 vLLM Ascend 源码，梳理 DBO（Dual Batch Overlap）与
`torch.compile`、TorchDynamo、piecewise compilation 和 ACLGraph 之间的关系。

## 1. 待澄清的问题

围绕 DBO 编译与图捕获，主要有以下疑问：

1. 第一次 dummy run 如何保证 TorchDynamo trace 捕获到 DBO 执行路径？
2. 如果第一次 trace 发生在 non-DBO 路径，后续打开 DBO 是否会自动补充 DBO
   hook？
3. DBO 有两个 ubatch，Dynamo 捕获的是单个 ubatch，还是同时捕获两个 ubatch？
4. DBO 切换点是 custom op，这些 custom op 是否会把两个 ubatch 拼接成一张大的
   FX graph？
5. DBO hook 的 fake implementation 是 identity，这对图捕获意味着什么？
6. ACLGraph 捕获的是单个 ubatch，还是两个 ubatch 的完整执行过程？
7. FULL 和 PIECEWISE 模式下，ACLGraph 的图粒度是否相同？
8. ACLGraph replay 时是否还会重新启动两个 DBO Python 线程？
9. 当前 graph cache key 是否能够正确描述两个 ubatch 的 shape？

## 2. 首先区分三种“图”

讨论这些问题前，必须区分三个层次：

```text
单个 ubatch 的 model.forward()
        │
        ▼
TorchDynamo FX graph
        │
        ├─ splitting op / custom op boundary
        ▼
piecewise compiled subgraph
        │
        ▼
ACLGraph capture
```

三种图的含义如下：

| 层次 | 捕获内容 | 典型粒度 |
|---|---|---|
| Dynamo FX graph | Python `model.forward()` 中可捕获的 Tensor 运算 | 一次 model forward |
| Piecewise compiled graph | FX graph 被 splitting ops 切分后的可编译区域 | 单个 FX partition |
| ACLGraph | callable 实际向 NPU stream 提交的设备任务 | 取决于 FULL/PIECEWISE 模式 |

DBO 的“双 batch”主要发生在 Python 线程和 NPU stream 调度层，不是 Dynamo
自动把两次 `model.forward()` 合并成一次 trace。

## 3. Q1：如何保证第一次 dummy run 捕获 DBO 路径？

### 3.1 当前调用链

当前 dummy run 的主要调用链为：

```text
ModelRunner._dummy_run()
  │
  ├─ _determine_batch_execution_and_padding()
  ├─ maybe_create_ubatch_slices()
  ├─ set_ascend_forward_context(...)
  └─ _model_forward(...)
       │
       ▼
     AscendUBatchWrapper.__call__()
       │
       ├─ _capture_ubatches()   # FULL capture
       └─ _run_ubatches()       # eager / PIECEWISE
            │
            ├─ ubatch thread 0 → model(ubatch0)
            └─ ubatch thread 1 → model(ubatch1)
```

源码位置：

- `vllm_ascend/worker/model_runner_v1.py::_dummy_run`
- `vllm_ascend/worker/npu_ubatch_wrapper.py::AscendUBatchWrapper.__call__`
- `vllm_ascend/worker/npu_ubatch_wrapper.py::_capture_ubatches`
- `vllm_ascend/worker/npu_ubatch_wrapper.py::_run_ubatches`

`_make_ubatch_metadata()` 会为每个 ubatch 创建独立 forward context。
`create_ascend_forward_context()` 在该 context 中设置：

```python
new_forward_context.dbo_enabled = True
```

因此，如果 compiled model 的第一次实际调用发生在 ubatch 子线程中，Dynamo trace
看到的是 `dbo_enabled=True` 的执行路径。

### 3.2 当前实现依赖的条件

第一次 dummy run 要捕获 DBO 分支，需要同时满足：

1. `parallel_config.enable_dbo=True`；
2. dummy token 数达到相应的 DBO threshold；
3. `allow_microbatching=True`；
4. DBO eligibility 检查最终返回 `should_ubatch=True`；
5. `maybe_create_ubatch_slices()` 确实生成两个有效 ubatch；
6. compiled model 此前没有被 non-DBO/profile forward 先行 trace；
7. 第一次进入 compiled model 时，当前子线程的 forward context 中
   `dbo_enabled=True`。

只满足 `enable_dbo=True` 并不充分。DBO 是每个 batch 动态判定的；如果第一次
forward 没达到 threshold，仍可能先捕获 non-DBO 路径。

### 3.3 non-DBO 首次 trace 的风险

模型内存在以下形式的 Python 控制流：

```python
if forward_context.dbo_enabled:
    torch.ops.vllm.dbo_linear_column_hook(x, is_record)
```

如果首次 trace 时 `dbo_enabled=False`，Dynamo 会沿当前路径捕获，DBO hook
分支可能被直接裁掉。后续仅修改 forward context，不会向已经生成的 FX graph
自动添加 hook。

因此，不能只用“dummy run 成功”判断 DBO 图已正确捕获。应检查导出的 FX graph
是否包含预期的 DBO custom op，例如：

```text
torch.ops.vllm.dbo_linear_column_hook
torch.ops.vllm.dbo_linear_row_hook
torch.ops.vllm.dbo_mla_preprocess_hook
torch.ops.vllm.dbo_moe_prepare_hook
torch.ops.vllm.dbo_moe_finalize_hook
```

### 3.4 更可靠的设计

从长期设计看，有两种可靠方案。

#### 方案 A：保持 FX 图拓扑不变

无条件调用 DBO hook：

```python
x = torch.ops.vllm.dbo_linear_column_hook(x, is_record)
```

real implementation 根据当前是否存在 DBO template/context 决定执行
record/wait/yield，或者直接 no-op。

这样 DBO 与 non-DBO 使用相同 FX 拓扑，不再依赖首次 trace 时的 Python bool。

#### 方案 B：显式编译两个 variant

将 DBO 状态纳入编译和 dispatch key：

```text
compile key =
    shape contract
    + dbo_enabled
    + communication mode
```

分别缓存 DBO 和 non-DBO graph。该方案更明确，但会增加编译数量和缓存管理复杂度。

## 4. Q2：Dynamo 捕获单个 ubatch 还是两个 ubatch？

结论：Dynamo 捕获的是单个 ubatch 的一次 `model.forward()`。

两个 DBO 子线程分别调用：

```python
model(
    input_ids=ubatch_metadata.input_ids,
    positions=ubatch_metadata.positions,
    ...
)
```

因此实际关系是：

```text
ubatch 0 ──┐
           ├── 调用同一个 compiled model / FX graph
ubatch 1 ──┘
```

而不是：

```text
一张 FX graph = forward(ubatch0) + forward(ubatch1)
```

通常第一个进入 model 的 ubatch 触发 Dynamo trace 和编译；另一个 ubatch 在 shape
contract 兼容时复用同一份 compiled artifact。

如果两个 ubatch shape 不兼容，则可能发生重新编译、range 不匹配或 graph cache
错误。因此“复用同一份图”隐含了两个 ubatch 满足相同编译契约的前提。

## 5. Q3：custom op 会把两个 ubatch 拼成一张 FX 大图吗？

不会。

DBO hook 注册成 custom op 的作用，是让单次 forward 中的 hook 成为 Dynamo 可以
保留的原子节点：

```text
单个 ubatch forward
  ├─ compute
  ├─ dbo hook custom op
  ├─ collective
  ├─ dbo hook custom op
  └─ compute
```

custom op boundary 不会：

- 跨越两个 Python 线程；
- 观察另一个线程正在执行的 `model.forward()`；
- 把两次 model 调用合并成一次 Dynamo trace；
- 自动产生包含两个 ubatch 的 FX graph。

两个 ubatch 的组合发生在 `AscendUBatchWrapper` 的线程调度和 FULL ACLGraph
capture 层。

## 6. Q4：DBO hook 的 fake implementation 为什么是 identity？

当前 DBO hook 的 fake implementation 为：

```python
def _dbo_hook_fake(x, is_record):
    return x
```

源码位于：

```text
vllm_ascend/ops/register_custom_ops.py
```

它表达的 shape contract 是：

```text
output.shape == input.shape
output.dtype == input.dtype
```

fake implementation 只服务于 Dynamo/FakeTensor 的 shape 和 metadata 传播。
它不会在编译阶段真的执行：

- Python thread yield；
- NPU event record/wait；
- stream 切换；
- DBO overlap template。

这些副作用由 custom op 的 real implementation 在实际执行或 ACLGraph capture
期间完成。

identity fake impl 保证 symbolic shape 可以穿过 hook，但它不能保证 DBO 分支一定
存在于 FX graph；分支是否被捕获仍由首次 trace 时的 Python 控制流决定。

## 7. Q5：FULL 模式下 ACLGraph 捕获单 batch 还是双 batch？

结论：DBO + FULL 模式捕获的是一张包含两个 ubatch 的完整设备执行图。

DBO 开启时，model runner 使用：

```python
AscendUBatchWrapper(
    self.model,
    self.vllm_config,
    CUDAGraphMode.FULL,
    self.device,
)
```

而不是普通的单 batch `ACLGraphWrapper`。

`_capture_ubatches()` 的逻辑是：

```python
with torch.npu.graph(aclgraph, stream=compute_stream):
    wake_up_ubatch_0()
    join(ubatch_thread_0)
    join(ubatch_thread_1)
    result = torch.cat(sorted_results, dim=0)
```

两个子线程会在各自的 compute/comm stream 上执行单-ubatch compiled forward。
由于整个提交过程处于同一个 `torch.npu.graph()` capture scope 内，最终
NPUGraph 包含：

```text
DBO FULL NPUGraph
  ├─ ubatch 0 compute stream tasks
  ├─ ubatch 0 comm stream tasks
  ├─ ubatch 1 compute stream tasks
  ├─ ubatch 1 comm stream tasks
  ├─ event/stream dependencies
  └─ 最终输出相关设备任务
```

所以：

- Dynamo FX graph 是单 ubatch；
- DBO FULL ACLGraph 是双 ubatch。

二者并不矛盾：FULL ACLGraph 记录的是两次调用同一 compiled forward 后产生的设备
任务序列。

## 8. Q6：FULL ACLGraph 如何 replay？

FULL capture 完成后，graph 按 token key 保存在：

```python
self.cudagraphs[num_tokens] = cudagraph_metadata
```

后续命中缓存时直接执行：

```python
cudagraph_metadata.aclgraph.replay()
return cudagraph_metadata.outputs
```

replay 时不会重新：

- 创建两个 DBO Python 子线程；
- 执行 Python overlap template；
- 重新调用 Dynamo；
- 重新运行 Python custom op implementation 来安排 yield。

capture 时已经形成的设备 task、stream 和 event dependency 由 NPUGraph
整体重放。

这也意味着 replay 必须满足 ACLGraph 的静态契约，包括：

- 输入 buffer 地址满足 capture/replay 要求；
- shape 与 capture descriptor 匹配；
- attention、KV cache 等 replay-time 参数已正确更新；
- 两个 ubatch 的切分和设备任务拓扑与 capture 时兼容。

## 9. Q7：PIECEWISE 模式下 graph 是单 batch 还是双 batch？

PIECEWISE 与 FULL 不同。

PIECEWISE 模式下，`AscendUBatchWrapper` 继续通过 `_run_ubatches()` 启动两个
Python 子线程。每个线程执行自己的单-ubatch model forward，forward 内的
piecewise `ACLGraphWrapper` 分别捕获或 replay 对应 partition。

因此：

```text
Python DBO scheduler
  ├─ ubatch 0
  │    ├─ replay piece 0
  │    ├─ eager/custom-op boundary
  │    └─ replay piece 1
  └─ ubatch 1
       ├─ replay piece 0
       ├─ eager/custom-op boundary
       └─ replay piece 1
```

PIECEWISE 模式下：

- 每张 ACLGraph 是单个 FX partition；
- 每次调用属于单个 ubatch；
- 两个 DBO Python 线程仍然参与运行；
- DBO hook/template 仍负责两个 ubatch 之间的 record、wait 和 yield；
- 不存在一张覆盖两个完整 forward 的双-ubatch大图。

## 10. Q8：当前 graph cache key 是否充分？

当前代码存在一个明确的等长 ubatch 假设。

查找 graph 时使用：

```python
num_tokens = ubatch0_size * 2
```

而 capture 保存 graph 时使用：

```python
num_tokens = ubatch0_size + ubatch1_size
```

两个表达式只有在：

```text
ubatch0_size == ubatch1_size
```

时才相等。

如果未来允许不等长 ubatch，单独使用 total token 数也不足以区分：

```text
(ubatch0=32, ubatch1=64)
(ubatch0=48, ubatch1=48)
```

两者可能有相同 total tokens，却具有不同的单-ubatch shape、通信任务和 stream
拓扑。

更完整的 DBO FULL graph key 应至少包含：

```text
(
    batch_descriptor,
    ubatch0_shape,
    ubatch1_shape,
    DBO topology,
    communication mode,
)
```

当前实现应明确维持“两个 ubatch 等长”的 invariant，或者改造 graph key 和
capture/replay contract。

## 11. Q9：piecewise compilation 的 range 是怎么定的？

piecewise compilation 的 range 不是运行时临时推出来的，而是在启动初始化阶段由
`CompilationConfig` 里的 `compile_ranges_endpoints` 决定，然后再被 Ascend 平台按融合
规则做一次补充。

源码里对应的入口是：

- `vllm_ascend/ascend_config.py::_get_compile_ranges()`：直接读取
  `compilation_config.compile_ranges_endpoints`
- `vllm_ascend/ascend_config.py::update_compile_ranges_split_points()`：在启用
  `fuse_allreduce_rms` 时，额外把 `ALLREDUCE_NORM_FUSE_THRESHOLD` append 进去
- `vllm_ascend/worker/worker.py::compile_or_warm_up_model()`：遍历这些 range，
  如果当前 warmup size / cudagraph size 没覆盖某个 range，就补一个
  `compile_range.end` 做 dummy run

也就是说，piecewise 的 range 决定因素是：

1. 用户或平台最终写入的 `compile_ranges_endpoints`
2. Ascend 为 matmul / allreduce 融合追加的阈值
3. worker 在 warmup 阶段为了覆盖未命中的 range 所做的补跑

这意味着：

- piecewise compile 的“range”是编译/预热用的离散区间；
- 它不是 ACLGraph 的 bucket；
- 它解决的是“哪些 FX partition 需要被编译到”，不是“运行时怎么把 size 归桶”。

## 12. Q10：ACLGraph 的 bucket 是怎么定的？

ACLGraph 的 bucket 不是连续值，而是由 `cudagraph_capture_sizes` 定义的一组离散边界。
真正起作用的映射表是在 `CudagraphDispatcher.initialize_cudagraph_keys()` 中预计算出来的。

关键逻辑是：

```python
self._bs_to_padded_graph_size = [0] * (max_size + 1)
for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
    for bs in range(start, end):
        if bs == start:
            self._bs_to_padded_graph_size[bs] = start
        else:
            self._bs_to_padded_graph_size[bs] = end
```

含义是：

- `capture_sizes` 里的数值本身就是 bucket 边界；
- 任意真实 `bs` 会先被映射到某个 padded graph size；
- `BatchDescriptor` 使用的是 padded 值，不是原始值；
- `ACLGraphWrapper` 以及 `FULL` 模式的 `npu_ubatch_wrapper` 都按这个 padded 值做
  exact match / replay。

因此 bucket 逻辑可以理解成：

```text
raw num_tokens
  -> _bs_to_padded_graph_size[num_tokens]
  -> BatchDescriptor(num_tokens=padded_value)
  -> ACLGraph key exact match
```

这和 piecewise compilation 的 range 不同：

- piecewise range 解决“FX 图怎么切、哪些区间需要编译”；
- ACLGraph bucket 解决“运行时 token size 怎么映射到可重放的 graph key”。

如果没有显式配置 `cudagraph_capture_sizes`，上游 vLLM 会用默认模式生成：

```text
[1, 2, 4] + range(8, 256, 8) + range(256, max_cudagraph_capture_size + 1, 16)
```

而 Ascend 侧默认 `max_cudagraph_capture_size` 还会跟 `max_num_seqs * decode_query_len`
绑定，并在没有显式配置时取较小者；如果 TP/SP 需要，还会进一步裁剪或重写。

## 13. 总结

| 图或执行层次 | 图的粒度 | 是否包含两个 ubatch | replay 是否依赖 DBO Python 线程 |
|---|---|---:|---:|
| Dynamo FX graph | 一次 `model.forward()` | 否 | 不适用 |
| Piecewise compiled graph | 单个 FX partition | 否 | 不适用 |
| PIECEWISE ACLGraph | 单 ubatch 的单个 partition | 否 | 是 |
| FULL non-DBO ACLGraph | 普通 batch 的完整 forward | 否 | 否 |
| FULL DBO ACLGraph | 完整 DBO 多 stream 设备调度 | 是 | 否 |

最终可以概括为：

1. Dynamo 不会自动捕获两个 ubatch；它捕获单次 model forward。
2. custom op 只在单次 forward 中建立可编译边界，不会跨线程拼接 FX graph。
3. 第一次 trace 时必须确保 DBO 分支存在，或者将图拓扑设计为与
   `dbo_enabled` 无关。
4. FULL DBO ACLGraph 会把两个 ubatch 的多 stream 设备任务捕获为一张图。
5. PIECEWISE ACLGraph 仍是单 ubatch、单 partition 粒度，DBO Python 调度继续存在。
6. 当前 FULL graph cache 隐含两个 ubatch 等长的假设，需要明确维护或改造。
