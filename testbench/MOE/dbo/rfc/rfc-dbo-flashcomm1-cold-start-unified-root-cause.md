# RFC: DBO + FlashComm1 冷启动问题统一根因、Shape Contract 与修复

## 状态

- 状态：Resolved for the validated TP=2 / EP=2 prefill workload
- 基准提交：`4fc82d042ab6b83d2891fcbce993ad7869d9f634`
- 日期：2026-07-04
- 适用范围：DeepSeek-V2 MLA、FlashComm1、DBO、`VLLM_COMPILE`、
  `FULL_AND_PIECEWISE` ACL graph

本文统一以下历史文档中互相覆盖、部分过时的结论：

- `rfc-dbo-cold-start-problem.md`
- `rfc-dbo-flashcomm1-compiled-reentrancy-and-shape-contract.md`
- `rfc-dbo-flashcomm1-compile-shape-contract.md`
- `rfc-custom-op-fakeimpl-compile-safe.md`
- `../bug/dbo-shape-inference.md`
- `../bug/flashcomm-dbo-compile-bugs.md`
- `../bug/dbo-flashcomm1-compiled-fix-log.md`

本文只把远端 trace、生成图和当前代码都能支持的结论视为最终结论。历史文档中
“AOT compiled callable 不可重入”“两个 rank collective 顺序不同”等判断已被后续
trace 证伪，不再作为根因。

## 1. 一张图说明完整故障链

```text
Python ForwardContext 中的 token 数 / pad_size
                    │
                    │ Dynamo trace 时读取 Python 值
                    ▼
           分支和整数被固化进 FX graph
                    │
                    │ torch.compile / AOT 生成固定 shape contract
                    ▼
        MLA / AllGather / ReduceScatter / residual
        对“逻辑长度”和“通信长度”理解不一致
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  output 4096 vs 8      ubatch 2051 % TP(2)
  residual 4096 vs 1026       AssertionError
          │                   │
          └─────────┬─────────┘
                    ▼
       ubatch 子线程只把异常存入 results
       另一线程仍在 Event / collective 中等待
                    │
                    ▼
       主线程卡在 join()，执行不到异常重抛
                    │
                    ▼
       外部表现为 “HCCL hang / engine 启动超时”
```

当前修复把 shape 控制信息分成两类：

1. 能由 tensor 表达的动态信息，从 `tensor.shape` 推导，保留为 symbolic shape；
2. 调度层必须保存的信息，明确区分 padded slice 与 logical slice。

代码入口：

- [model runner 同时保存 padded/logical ubatch slices](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2073-L2083)
  `[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2073]`
- [forward context 接收两类 slices](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2221-L2236)
  `[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2221]`
- [MLA output shape 从输入 shape 推导](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/mla.py#L155-L183)
  `[VERIFY: code:vllm_ascend/ops/mla.py:155]`
- [ReduceScatter padding 从当前 tensor rows 推导](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/linear_op.py#L556-L583)
  `[VERIFY: code:vllm_ascend/ops/linear_op.py:556]`

---

## 2. 问题一：生成图中出现未定义的 `Min`

### 2.1 问题描述

历史修复为解决 MLA 的 `4096 vs 8` output mismatch，曾在 MLA custom-op 入口按
logical token 数执行：

```python
hidden_states = hidden_states[:logical_num_tokens]
```

非 DBO compiled 路径中的 `logical_num_tokens` 不是普通固定整数。Dynamo/符号 shape
系统把“切片上界不能超过输入首维”表达为：

```text
Min(input_rows, logical_num_tokens)
```

随后 Ascend compiler 生成的 `computation_graph.py` 中出现：

```python
"bf16[Min(16384, ((s72 + 1)//2)), 2048]"
"Sym(Min(16384, ((s72 + 1)//2)))"
```

但生成文件只导入了 `torch`，没有导入 `sympy.Min`。执行生成代码时因此得到：

```text
NameError: name 'Min' is not defined
```

这是“生成代码缺少符号导入”与“不必要的动态切片”共同造成的编译产物错误，不是
NPU kernel、HCCL 或 DBO 调度错误。

历史证据记录于
[`rfc-dbo-cold-start-problem.md`](./rfc-dbo-cold-start-problem.md#min-的精确根因)。
当前代码不再执行该 logical-token 动态切片：

- [`_resolve_mla_forward_inputs` 只根据输入首维和静态 TP/VL 属性决定 output shape](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/mla.py#L174-L183)
  `[VERIFY: code:vllm_ascend/ops/mla.py:174]`

### 2.2 问题溯源

问题不是“torch.compile 无缘无故引入 `Min`”。它来自 Python 切片的准确数学语义：

```text
result_rows = min(input_rows, slice_stop)
```

当 `input_rows` 或 `slice_stop` 是 `SymInt` 时，Dynamo 必须在 FX graph 中保留这个
`min`。真正不合理的是：MLA output 本来可以从当前输入 tensor 的 symbolic shape
直接确定，却先从 runtime forward context 取一个 token 数，再拿它切输入。这额外
制造了一个跨生命周期的 shape 依赖。

当前 MLA wrapper 在构造时把 VL 首层这一模型静态属性固化为
`self.is_vl_first_layer`，避免运行时 shape 判断；普通路径直接返回
`hidden_states.shape[0]`，VL 首层返回 `hidden_states.shape[0] // tp_size`。

- [VL 首层属性在初始化阶段确定](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/mla.py#L141-L153)
  `[VERIFY: code:vllm_ascend/ops/mla.py:141]`
- [MLA output buffer 使用 `_resolve_mla_forward_inputs` 的结果](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/mla.py#L155-L170)
  `[VERIFY: code:vllm_ascend/ops/mla.py:155]`

### 2.3 系统解决

系统解决不是在生成文件里补一条 `from sympy import Min`。那只能修复代码生成器的
表面错误，仍然保留 runtime context 控制 compiled tensor shape 的错误架构。

当前真实代码：

```python
# vllm_ascend/ops/mla.py:155-183
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor | None = None,
    attn_metadata: AttentionMetadata | None = None,
) -> torch.Tensor:
    hidden_dim = self.hidden_size
    hidden_states, output_tokens, need_gather_q_kv = (
        _resolve_mla_forward_inputs(
            hidden_states,
            _EXTRA_CTX.flash_comm_v1_enabled,
            self.tp_size,
            self.is_vl_first_layer,
        )
    )

    output = torch.empty(
        (output_tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    torch.ops.vllm.mla_forward(
        hidden_states, need_gather_q_kv, output, self.prefix
    )
    return output.view(-1, hidden_dim)


def _resolve_mla_forward_inputs(
    hidden_states: torch.Tensor,
    flash_comm_v1_enabled: bool,
    tp_size: int,
    is_vl_first_layer: bool,
) -> tuple[torch.Tensor, int, bool]:
    if flash_comm_v1_enabled and tp_size > 1 and is_vl_first_layer:
        return hidden_states, hidden_states.shape[0] // tp_size, False
    return hidden_states, hidden_states.shape[0], flash_comm_v1_enabled
```

代码解读：

1. `hidden_states` 不再按 `forward_context.num_tokens` 做动态切片，所以不会额外生成
   `min(input_rows, logical_num_tokens)`。
2. 普通路径的 `output_tokens` 直接取 `hidden_states.shape[0]`。如果该维是
   `SymInt`，它会自然沿 FX graph 传播，而不是先转成 Python context 常量。
3. VL 第一层是唯一特殊分支；`is_vl_first_layer` 在模块初始化时确定，因此不是
   per-request 动态判断。
4. custom op 通过预分配的 `output` 明确声明写入 shape，fake impl 只需保持
   mutation contract，不再重新猜 token 数。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/mla.py#L155-L219)
`[VERIFY: code:vllm_ascend/ops/mla.py:155]`

正确约束是：

1. output shape 优先从输入 tensor symbolic shape 推导；
2. 模型结构分支在模块初始化时静态化；
3. 不在 compiled region 内用 Python context token 数切 tensor；
4. 如果切片确实是模型语义，必须让 compiler 正确支持完整的 symbolic expression，
   不能依赖偶然的 warmup 常量。

---

## 3. 问题二：token 维度被 trace 成静态值

### 3.1 问题描述

历史故障中出现过以下 shape：

```text
MLA:                 output 4096 vs o_proj_output 8
compiled MLP:        output 4096 vs residual 1026
AllGather unpad:     unpadded_length 被固化为 8192
DBO ubatch1:         2051 rows 进入 TP=2 ReduceScatter
```

它们不是四个独立问题。共同根因是：影响 tensor shape 的值来自 Python
`ForwardContext`，Dynamo trace 时把它们当作当前 forward 的普通整数或布尔值。

典型危险模式：

```python
num_tokens = forward_context.num_tokens
pad_size = forward_context.pad_size
if pad_size > 0:
    x = F.pad(x, ...)
output = torch.empty((num_tokens, hidden_dim), ...)
```

### 3.2 Dynamo、torch.compile 与 CUDAGraph 分别做了什么

#### Dynamo trace

Dynamo 执行 Python bytecode并构造 FX graph。普通 Python `int`/`bool` 会形成 guard，
或者被常量传播和死代码消除。因此：

- trace 时 `pad_size=0`，`if pad_size > 0` 分支可能不进入图；
- trace 时 `num_tokens=4096`，output allocation 可能固定为 4096；
- trace 时 `dbo_enabled=False`，DBO hook 分支可能被剪掉。

当前 DBO hooks 被注册为 custom ops，fake impl 只返回输入 tensor；这让 hook 对
compiler 是 shape-preserving opaque boundary，而 runtime impl 再读取当前
`dbo_template` 执行 event/yield。

- [DBO hook real/fake implementations](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L34-L62)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:34]`
- [DBO hooks 注册为 custom ops](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L357-L374)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:357]`

#### torch.compile / AOT / piecewise backend

torch.compile 消费 Dynamo 的 FX graph，并为 compile range 生成可执行图。它不会
自动知道某个 Python context 值下一请求或另一个 ubatch 会改变。错误常量进入 FX
graph 后，会继续进入：

- fake/meta shape propagation；
- compiled subgraph 的输入输出 contract；
- AOT cache；
- generated `computation_graph.py`。

Custom op 的 fake impl 必须与 runtime impl 声明相同 shape。当前
`maybe_unpad_after_all_gather` 将 `unpadded_length` 作为显式参数，并且 fake/real
都返回该首维；ReduceScatter fake 使用 ceil division。

- [AllGather 后 unpad 的 real/fake shape contract](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L125-L148)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:125]`
- [对应 fake impl](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L207-L217)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:207]`
- [ReduceScatter fake 使用 ceil division](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L216-L226)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:216]`

#### CUDAGraph / ACL graph

CUDAGraph 不是 token 静态化的最初来源。它要求 capture/replay 的地址和 shape
contract 稳定，因此会放大上游错误：

1. profile/warmup 用固定 capture size 构造 buffer；
2. compiled callable 按 FX/AOT contract 写入这些 buffer；
3. 一旦 logical、collective 或 graph-padded rows 不一致，错误在 warmup/capture
   阶段就暴露为 expand、fused-op tiling 或 collective assert；
4. 若错误 contract 被成功 capture，runtime replay 也无法通过改变 Python context
   修正图内已固化的 shape。

### 3.3 问题溯源

以 `4103` token、TP=2、两个 ubatch 为例：

```text
logical total             4103
scheduler/graph padded    4104
padded ubatches           2052 + 2052
logical ubatches          2052 + 2051
```

历史代码一度把 attention slices（2052+2051）、padded slices（2052+2052）和
`attn_metadata.num_actual_tokens` 混合用于 context、tensor 切片和 collective。
这造成两个方向的错误：

- 使用 logical rows 直接做 ReduceScatter：2051 不能被 TP=2 整除；
- 使用 padded/capture rows作为下游逻辑 output：4096/8192 等 warmup 常量泄漏。

当前 model runner 明确：

- `ubatch_slices_padded` 用于模型输入和 DBO wrapper；
- `ubatch_slices_attn` 保存 logical/attention 边界；
- 两者同时写入 Ascend forward context。

- [attention metadata 使用 `ubatch_slices_attn`](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2141-L2164)
  `[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2141]`
- [模型 context 使用 padded 与 logical 两套 slices](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2221-L2236)
  `[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2221]`

### 3.4 系统解决

1. **Shape-owning code 从 tensor 推导。**
   MLA output 使用 `hidden_states.shape[0]`；row-parallel collective padding 使用
   `(-x.shape[0]) % world_size`。
2. **Fake 与 runtime 共用同一公式。**
   ReduceScatter runtime 先 pad 再 scatter；fake 使用 ceil division。
3. **调度信息显式分层。**
   context 同时保存 padded rows 与 `num_tokens_logical`，不让一个 `num_tokens`
   承担所有语义。
4. **控制副作用变成 opaque custom op。**
   DBO hook fake 是 identity，real impl 在 runtime 查找 template。
5. **禁止用错误合流点掩盖上游问题。**
   不能只在 `maybe_chunk_residual` 处把两侧强行裁成相同长度；producer 必须先满足
   自己的 shape contract。

#### 3.4.1 padded 与 logical slices 的真实代码

```python
# vllm_ascend/worker/model_runner_v1.py:2073-2083
num_tokens_padded = batch_desc.num_tokens
num_reqs_padded = (
    batch_desc.num_reqs
    if batch_desc.num_reqs is not None
    else num_reqs
)
ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
    should_ubatch,
    num_scheduled_tokens_np,
    num_tokens_padded,
    num_reqs_padded,
    self.parallel_config.num_ubatches,
)

pad_attn = cudagraph_mode == CUDAGraphMode.FULL

# vllm_ascend/worker/model_runner_v1.py:2141
ubatch_slices_attn = (
    ubatch_slices_padded if pad_attn else ubatch_slices
)
```

代码解读：

- `ubatch_slices_padded` 描述实际模型输入 buffer 的切片，包含 scheduler/graph
  为固定 shape 加入的 rows。
- `ubatch_slices` 描述未补齐的请求语义。
- FULL graph 的 attention metadata 需要遵守 capture shape，所以选择 padded slices；
  非 FULL 模式选择 logical slices。
- 这一步只选择 attention 的视图，不能把另一套 slices 丢掉，因为 DBO collective
  和最终输出裁剪仍分别需要 padded 与 logical 信息。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2073-L2142)
`[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2073]`

两套 slices 随后同时进入 forward context：

```python
# vllm_ascend/worker/model_runner_v1.py:2221-2236
with set_ascend_forward_context(
    attn_metadata,
    self.vllm_config,
    num_tokens=num_tokens_padded,
    num_tokens_across_dp=num_tokens_across_dp,
    aclgraph_runtime_mode=cudagraph_mode,
    batch_descriptor=batch_desc,
    num_actual_tokens=scheduler_output.total_num_scheduled_tokens,
    model_instance=self.model,
    skip_compiled=has_encoder_input,
    input_ids=input_ids,
    ubatch_slices=ubatch_slices_padded,
    ubatch_slices_logical=ubatch_slices_attn,
):
    hidden_states = self._model_forward(...)
```

代码解读：

- `ubatch_slices` 明确是 padded slices，DBO wrapper 据此切实际 tensor。
- `ubatch_slices_logical` 单独保存 attention/logical 边界。
- `skip_compiled` 不再因为 DBO + FlashComm1 被强制打开；只有 encoder 输入路径回退，
  因此该修复真正覆盖 compiled DBO runtime。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2221-L2248)
`[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2221]`

#### 3.4.2 per-ubatch context 的真实代码

```python
# vllm_ascend/ascend_forward_context.py:273-296
new_forward_context.flash_comm_v1_enabled = (
    cur_forward_context.flash_comm_v1_enabled
)

slice_num_tokens = ubatch_slices[ubatch_num].num_tokens
if cudagraph_runtime_mode is CUDAGraphMode.FULL:
    new_forward_context.num_tokens = _get_actual_num_tokens(
        attn_metadata, slice_num_tokens
    )
else:
    new_forward_context.num_tokens = slice_num_tokens

tp_world_size = get_tensor_model_parallel_world_size()
if (
    new_forward_context.flash_comm_v1_enabled
    or new_forward_context.flashcomm_v2_enabled
):
    pad_size = (
        tp_world_size
        - (new_forward_context.num_tokens % tp_world_size)
    ) % tp_world_size
    new_forward_context.pad_size = pad_size

ubatch_slices_logical = getattr(
    cur_forward_context, "ubatch_slices_logical", None
)
if ubatch_slices_logical is not None:
    new_forward_context.num_tokens_logical = (
        ubatch_slices_logical[ubatch_num].num_tokens
    )
else:
    new_forward_context.num_tokens_logical = (
        new_forward_context.num_tokens
    )
```

代码解读：

- `num_tokens` 是当前 ubatch 实际参与模型/collective 的 rows。
- `num_tokens_logical` 是该 ubatch 最终应保留的 rows。
- FULL graph 的 token 来源允许从 attention metadata 解析，因为此时 metadata 与
  capture contract 绑定；非 FULL runtime 直接信任当前 ubatch slice。
- `pad_size` 仍用于部分 legacy FlashComm 路径，但关键 row-parallel
  ReduceScatter 已改为从 tensor shape 自行计算，避免该 Python int 被 trace 后跨
  ubatch 复用。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ascend_forward_context.py#L273-L305)
`[VERIFY: code:vllm_ascend/ascend_forward_context.py:273]`

---

## 4. 问题三：DBO 子线程异常被掩盖成死锁

### 4.1 问题描述

DBO 用两个 Python 线程和 CPU events 交替提交两个 ubatch。每个 context 进入时：

1. 注册 thread-id 到 context；
2. 等待所有线程和主线程到达 barrier；
3. 等待自己的 `cpu_wait_event`；
4. 恢复该 ubatch 的 forward context。

退出 context 时，无论是否带异常，当前实现都会 signal 下一线程，并返回 `False`
继续传播异常。

- [context enter/exit](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/ubatching.py#L86-L107)
  `[VERIFY: code:vllm_ascend/worker/ubatching.py:86]`
- [CPU yield：唤醒对端后等待自己再次被唤醒](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/ubatching.py#L126-L137)
  `[VERIFY: code:vllm_ascend/worker/ubatching.py:126]`

wrapper 捕获子线程异常后只执行：

```python
except Exception as e:
    results.append(e)
```

主线程则先逐个 `join()`，之后才调用 `_check_thread_exceptions()`。

- [runtime 子线程异常暂存](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L212-L258)
  `[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:212]`
- [异常检查发生在所有 join 之后](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L254-L258)
  `[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:254]`

如果 ubatch1 assert，ubatch0 可能继续进入下一个 event/collective wait；主线程又在
等待 ubatch0 退出。因此 `_check_thread_exceptions()` 永远没有机会执行。外部只能
看到 shared-memory timeout、HCCL watchdog 或 EngineCore cancel。

### 4.2 哪里 assert，为什么 assert

实际第一异常位于 FlashComm1 row-parallel down projection 的 ReduceScatter：

```text
input rows = 2051
TP world size = 2
assert input_tensor.shape[0] % world_size == 0
```

2051 来自 logical split `2052 + 2051`。ReduceScatter 要求参与通信的首维能均匀
分给两个 TP rank，因此这个 assert 是正确的；错误在调用方没有先满足 collective
shape contract。

当前 row-parallel 代码从实际 `x.shape[0]` 计算 padding：

```python
pad_size = (-x.shape[0]) % world_size
if pad_size > 0:
    x = F.pad(x, ...)
```

- [当前 row-parallel pad-before-ReduceScatter](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/linear_op.py#L579-L583)
  `[VERIFY: code:vllm_ascend/ops/linear_op.py:579]`

这保证 2051 先变为 2052，再进入 ReduceScatter。对应 fake impl 使用
`ceil(2051 / 2)=1026` 声明输出 rows，compile/runtime contract 一致。

### 4.3 当前异常反馈修到了什么程度

提交 `0228eced` 增加了：

- 子线程 `try/except`；
- results 中的 exception 检查；
- 主线程从首个子线程异常构造 `RuntimeError`。

- [`_check_thread_exceptions`](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L38-L49)
  `[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:38]`

它能处理“两线程都最终退出”的异常，但不能处理“一线程异常、另一线程永久等待”。
因此当前代码的异常反馈仍不是完整的 cooperative abort。

真实代码：

```python
# vllm_ascend/worker/npu_ubatch_wrapper.py:38-49
def _check_thread_exceptions(results: list, context: str) -> None:
    exceptions = [r for r in results if isinstance(r, Exception)]
    if exceptions:
        raise RuntimeError(
            f"{context}: {len(exceptions)} sub-thread(s) failed. "
            f"First error: {exceptions[0]}"
        ) from exceptions[0]


# vllm_ascend/worker/npu_ubatch_wrapper.py:212-258
def _ubatch_thread(results, model, ubatch_metadata):
    try:
        with ubatch_metadata.context:
            model_output = model(...)
            dbo_current_stream().synchronize()
            dbo_yield()
        results.append(
            (ubatch_metadata.context.id, model_output)
        )
    except Exception as e:
        results.append(e)

# Main thread
for thread in ubatch_threads:
    thread.join()
_check_thread_exceptions(results, "Ubatch runtime")
```

代码解读：

1. `raise ... from exceptions[0]` 能保留第一异常作为 cause，这比随后出现的
   EngineCore/HCCL 次级错误更有诊断价值。
2. `results` 同时保存 `(ubatch_id, output)` 与 `Exception`，所以只有
   `_check_thread_exceptions` 通过后才能排序和拼接。
3. 缺陷在执行顺序：`join()` 没有 abort 通道，而异常检查位于所有 `join()` 之后。
4. 子线程捕获异常后也没有设置共享 aborted flag 或唤醒全部等待者；因此这段代码只
   修复了“异常线程和对端都能自然退出”的情况。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L38-L49)
`[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:38]`

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L212-L258)
`[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:212]`

context 的 event 行为解释了为什么对端不会自动取消：

```python
# vllm_ascend/worker/ubatching.py:100-107, 126-137
def __exit__(self, exc_type, exc_val, exc_tb):
    _CURRENT_CONTEXTS[self.id] = None
    del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
    self.maybe_run_recv_hook()
    self.cpu_signal_event.set()
    self.cpu_wait_event.clear()
    return False

def _cpu_yield(self):
    assert forward_context._forward_context == self.forward_context
    assert dbo_current_stream() == self.current_stream
    assert not self.cpu_wait_event.is_set()
    self.cpu_signal_event.set()
    self.cpu_wait_event.wait()
    self.cpu_wait_event.clear()
    self._restore_context()
```

代码解读：

- `return False` 允许异常离开 `with`，随后被 wrapper 捕获。
- `__exit__` 即使收到异常仍执行 `cpu_signal_event.set()`，所以对端会继续运行，而
  不是进入统一取消流程。
- `_cpu_yield()` 使用无 timeout 的 `wait()`，并且没有检查共享异常。这正是需要
  cooperative abort 的位置。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/ubatching.py#L100-L137)
`[VERIFY: code:vllm_ascend/worker/ubatching.py:100]`

### 4.4 系统解决

完整机制必须引入每次 DBO run 独立的共享 abort state：

```text
first_exception
aborted event
all CPU wait events
barrier abort/broken state
```

任一子线程异常后必须：

1. 原子保存第一异常及 traceback；
2. 标记 aborted；
3. 唤醒所有 `cpu_wait_event`；
4. abort barrier；
5. 所有 `_cpu_yield()`、event wait 和 hook 边界在继续提交 collective 前检查 aborted；
6. 主线程 join 完成后以原异常作为 cause 重抛。

仅给 `join()` 加 timeout 不是解决方案：它只能更早发现卡住，不能阻止另一线程继续
提交 collective，也不能保证 NPU stream 进入可恢复状态。

---

## 5. 问题四：FlashComm1 + DBO shape 与 padding/unpad 全景

### 5.1 必须区分的四种长度

| 名称 | 含义 | 生命周期 |
|---|---|---|
| logical tokens | 请求真正需要计算和返回的 token | request / attention |
| scheduler/graph padded tokens | scheduler、DP 或 graph 为固定 batch shape 加的 rows | 整个 forward |
| collective input tokens | 当前 AG/RS 实际消费的 rows | 单次 collective |
| TP-local output tokens | collective 后每个 TP rank 持有的 rows | 至下一次 AG/merge |

这四种长度不能都塞进 `forward_context.num_tokens`。

### 5.2 调度层 padding

`maybe_create_ubatch_slices` 同时生成 logical slices 和 padded slices。当前代码把
padded slices 交给模型输入切分，把 logical slices存入
`ubatch_slices_logical`，并在每个 ubatch context 中记录
`num_tokens_logical`。

- [padded/logical slices 的创建](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/model_runner_v1.py#L2073-L2081)
  `[VERIFY: code:vllm_ascend/worker/model_runner_v1.py:2073]`
- [per-ubatch logical count](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ascend_forward_context.py#L291-L296)
  `[VERIFY: code:vllm_ascend/ascend_forward_context.py:291]`

### 5.3 FlashComm1 column-parallel：AllGather

DBO 路径：

```text
TP-local input
  → dbo_linear_column_hook(record)
  → explicit TP AllGather
  → dbo_linear_column_hook(wait/yield)
  → GEMM
```

非 DBO 路径由 `maybe_all_gather_and_maybe_unpad` 执行通信和必要裁剪。

- [SequenceColumnParallelOp 的 DBO/non-DBO 分支](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/linear_op.py#L467-L501)
  `[VERIFY: code:vllm_ascend/ops/linear_op.py:467]`

历史错误是在 DBO compiled 分支对已经满足 collective contract 的 tensor 又使用
Python `num_tokens` unpad，令生成图出现固定的 8192/4096。当前 DBO column path
不再在显式 AllGather 后按 forward-context token 数裁剪。

真实代码：

```python
# vllm_ascend/ops/linear_op.py:467-501
class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_: torch.Tensor):
        bias = self.bias if not self.skip_bias_add else None
        need_all_gather = not (
            extract_layer_index(self.layer.prefix) == 0
            and is_vl_model()
            and "attn" in self.prefix
        )

        forward_context = get_forward_context()
        if forward_context.dbo_enabled:
            torch.ops.vllm.dbo_linear_column_hook(
                input_, is_record=True
            )
            if (
                forward_context.flash_comm_v1_enabled
                and need_all_gather
            ):
                input_ = tensor_model_parallel_all_gather(input_, 0)
            torch.ops.vllm.dbo_linear_column_hook(
                input_, is_record=False
            )
        else:
            input_ = (
                torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    input_, label=need_all_gather
                )
            )

        output_parallel = self.quant_method.apply(
            self.layer, input_, bias
        )
        return output_parallel, (
            self.bias if self.skip_bias_add else None
        )
```

代码解读：

- DBO 路径把通信显式放在两个 hook 之间，确保通信提交点可被 overlap template
  控制。
- hook 自身是 custom op；compiler 看见 shape-preserving fake impl，runtime 才执行
  event/yield。
- AllGather 后没有再使用 `forward_context.num_tokens` 切 `input_`，因此不会把某次
  warmup 的 8192 固化为所有 ubatch 的 unpad 长度。
- 非 DBO 路径仍可使用复合 custom op，因为它不需要在通信中间切换 ubatch。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/linear_op.py#L467-L501)
`[VERIFY: code:vllm_ascend/ops/linear_op.py:467]`

### 5.4 FlashComm1 row-parallel：ReduceScatter

```text
GEMM input rows N
  → collective_pad = (-N) % TP
  → pad 到 N + collective_pad
  → ReduceScatter
  → 每 rank rows = ceil(N / TP)
```

这里的 padding 只属于该次 ReduceScatter，必须从当前 tensor shape 计算，不能从
attention logical count 或 graph capture count 猜。

- [runtime collective padding](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/linear_op.py#L579-L583)
  `[VERIFY: code:vllm_ascend/ops/linear_op.py:579]`
- [fake output rows 使用 ceil division](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L309-L317)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:309]`

真实 runtime 代码：

```python
# vllm_ascend/ops/linear_op.py:579-595
world_size = self.layer.tp_size
pad_size = (-x.shape[0]) % world_size
dsa_cp_attn_out = enable_dsa_cp() and (
    "o_proj" in self.layer.prefix
    or "wo_b" in self.layer.prefix
)
if pad_size > 0 and not dsa_cp_attn_out:
    x = F.pad(x, (0, 0, 0, pad_size))

comm_mode = "aiv"
hcom_name = (
    get_tp_group()
    .device_group
    ._get_backend(torch.device("npu"))
    .get_hccl_comm_name(self.layer.tp_rank)
)
```

对应 fake shape：

```python
# vllm_ascend/ops/register_custom_ops.py:309-317
def _matmul_and_reduce_impl_fake(
    input_parallel: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _FLASH_COMM_V1_SNAPSHOT:
        num_tokens = _get_reduce_scatter_num_tokens(
            num_tokens, self.tp_size
        )
    return torch.empty(
        (num_tokens, self.output_size_per_partition),
        device=input_parallel.device,
        dtype=input_parallel.dtype,
    )
```

代码解读：

- `(-N) % TP` 给出 `[0, TP-1]` 范围内的最小 padding；N=2051、TP=2 时结果为 1。
- padding 基于实际 tensor 首维，因此不同 ubatch 可以分别得到正确结果。
- fake 侧的 `_get_reduce_scatter_num_tokens` 是 ceil division，声明
  `ceil(N/TP)` rows，与 runtime 的 `pad → ReduceScatter` 等价。
- `_FLASH_COMM_V1_SNAPSHOT` 只表达 compile-stable 的功能开关；动态 N 仍来自
  `input_parallel.size(0)`，两者职责不同。

### 5.5 Residual 合流

AddRmsNormBias 要求主分支和 residual 首维一致。FlashComm1 后主分支可能已经是
TP-local rows，而 residual 仍是 global rows。当前代码以主分支 `x.size(0)` 为
local contract，必要时把 residual 补到 `local_rows * TP`，再按 TP rank 切出等长
local residual。

- [TP-local residual slicing](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L65-L86)
  `[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:65]`

这修复了早期 `AddRmsNormBias` 的 `x1/x2 shape invalid`。它不是任意截断：
目标长度由已经完成通信的主分支 local rows 决定。

真实代码：

```python
# vllm_ascend/ops/register_custom_ops.py:65-86
def _slice_tp_local_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    local_num_tokens = x.size(0)
    global_num_tokens = local_num_tokens * tp_size
    if residual.size(0) < global_num_tokens:
        residual = F.pad(
            residual,
            (0, 0, 0, global_num_tokens - residual.size(0)),
        )
    start = tp_rank * local_num_tokens
    end = start + local_num_tokens
    return residual[start:end]


def _maybe_chunk_residual_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    try:
        get_forward_context()
    except AssertionError:
        return residual

    if x.size(0) != residual.size(0):
        residual = _slice_tp_local_residual(x, residual)
    return residual
```

代码解读：

- `x` 是通信后主分支，`x.size(0)` 因而是 fused residual op 必须遵守的 local
  rows。
- residual 不足 `local_rows * TP` 时只补到最低全局长度，不按旧 `pad_size`
  再次推算。
- `[tp_rank * local_rows : (tp_rank + 1) * local_rows]` 保证每个 rank 得到与 `x`
  完全相同的首维。
- 只有两侧 rows 不同时才切片；相同时不引入额外复制或 shape 变化。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/register_custom_ops.py#L65-L86)
`[VERIFY: code:vllm_ascend/ops/register_custom_ops.py:65]`

### 5.6 MLA output

MLA custom op 的 output buffer 必须与输入 tensor 的当前 symbolic rows 保持同一
contract。当前普通路径 output rows 等于 `hidden_states.shape[0]`；只有静态确定
的 VL 第一层使用 `rows // TP`。

- [MLA shape resolver](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/ops/mla.py#L174-L183)
  `[VERIFY: code:vllm_ascend/ops/mla.py:174]`

这消除了从 `_EXTRA_CTX.num_tokens` 分配 4096-row buffer、而实际 o_proj 只有
8-row output 的 contract 分叉。

本路径的真实代码和逐行解读见 §2.3。关键不是在写回点执行
`o_proj_output[:output.shape[0]]`，而是让 `output` 从创建时就拥有正确的 symbolic
rows；否则裁剪只会把上游错误延迟到下一个 residual/MLP 合流点。

### 5.7 DBO 最终输出 unpad

每个 ubatch 的模型输出先进行 TP AllGather，再删除
`num_tokens - num_tokens_logical` 对应的 scheduler/graph padding，最后拼接两个
ubatch。这里删除的是调度层 padding，不是单次 ReduceScatter 的临时 padding。

- [DBO output gather/unpad/concat](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L259-L275)
  `[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:259]`

真实代码：

```python
# vllm_ascend/worker/npu_ubatch_wrapper.py:259-275
sorted_results = [
    value for position, value in sorted(results)
]
if (
    get_forward_context().flash_comm_v1_enabled
    and get_pp_group().is_last_rank
):
    for i in range(2):
        sorted_results[i] = tensor_model_parallel_all_gather(
            sorted_results[i], 0
        )

        ctx = ubatch_metadata[i].context.forward_context
        num_padded = ctx.num_tokens
        num_logical = getattr(
            ctx, "num_tokens_logical", num_padded
        )
        output_pad = num_padded - num_logical
        if output_pad > 0:
            sorted_results[i] = sorted_results[i][:-output_pad, :]

result = torch.cat(sorted_results, dim=0)
```

代码解读：

- 先按 ubatch id 排序，避免线程完成顺序改变输出 token 顺序。
- 每个 TP-local output 先 AllGather 恢复该 ubatch 的完整 rows。
- `output_pad` 使用同一个 ubatch 的 padded/logical 差值，不能使用整个 batch 的
  `pad_size`。
- 两个 ubatch 分别 unpad 后再 concat，避免 padding row 落入最终 hidden states。

[源码与行号](https://github.com/micropuma/vllm-ascend/blob/4fc82d042ab6b83d2891fcbce993ad7869d9f634/vllm_ascend/worker/npu_ubatch_wrapper.py#L259-L275)
`[VERIFY: code:vllm_ascend/worker/npu_ubatch_wrapper.py:259]`

### 5.8 Shape ownership 总表

| 边界 | pad 的负责人 | unpad 的负责人 | shape 来源 |
|---|---|---|---|
| scheduler / graph | model runner | DBO wrapper 最终输出 | padded/logical slices |
| column AllGather | column-parallel op | 同一通信路径 | input rows + explicit length |
| row ReduceScatter | row-parallel op | 后续 logical merge | current `x.shape[0]` |
| residual global→local | residual helper | 不适用 | main-branch local rows |
| MLA output buffer | MLA wrapper | 不适用 | `hidden_states.shape[0]` |

原则是：**谁引入 padding，谁负责记录和移除；任何层都不能借用另一层的
`pad_size` 来完成自己的 shape contract。**

---

## 6. 最终修复与验证状态

基准提交 `4fc82d04` 的核心修复包括：

1. DBO hooks 注册为 compile-safe custom ops；
2. model runner 同时保留 padded 与 logical ubatch slices；
3. MLA output 从输入 symbolic shape 推导；
4. row ReduceScatter 从当前 tensor rows 计算 padding；
5. residual 按主分支 TP-local rows 切分；
6. DBO 最终输出按 per-ubatch logical rows unpad。

2026-07-04 在远端以全新 `VLLM_CACHE_ROOT` 验证：

```text
FlashComm1=1
FlashComm2=0
DBO=1
TP=2 / EP=2
VLLM_COMPILE
FULL_AND_PIECEWISE
input=4096, output=16, requests=500, concurrency=96
```

结果：

```text
compile range (1, 16384): 10.57 s
torch.compile total:      20.32 s
PIECEWISE capture:        completed
FULL capture:             completed
successful requests:      500
failed requests:          0
test exit code:           0
```

仍需明确的边界：

- 当前异常收集不是完整 cooperative abort，见 §4.3；
- 单测仍有 3 个 MLA mock 基础设施失败，不能记录为全量 UT 通过；
- decode DBO、DP/PP、TP>2 和 A3 未由本轮验证覆盖。

## 7. 回归测试要求

### 7.1 冷编译必须真的隔离缓存

`VLLM_COMPILE_CACHE_PATH` 不是当前 vLLM 0.22.1 的 torch compile cache 根目录。
必须使用全新的：

```bash
export VLLM_CACHE_ROOT=/path/that/does/not/exist
```

并从日志确认出现：

```text
Saved compiled graph to cache
```

不能出现：

```text
Loaded npugraph_ex compilation cache
Directly load AOT compilation
```

### 7.2 最小矩阵

| Case | 目的 |
|---|---|
| FC1=0, DBO=1 | DBO control |
| FC1=1, DBO=0 | FlashComm compile control |
| FC1=1, DBO=1, odd ubatch | RS ceil/padding contract |
| FC1=1, DBO=1, 4K single | compiled dynamic shape |
| FC1=1, DBO=1, 4K concurrent | 多 ubatch/thread/collective |

每个 case 都必须检查：

1. 最早 Python/device error；
2. compile cache 是 save 还是 load；
3. KV cache 与 PIECEWISE/FULL capture 完成；
4. API 监听；
5. 成功/失败请求数；
6. server/test 退出码。

## 8. Review checklist

- [ ] 影响 tensor shape 的值是否来自 tensor symbolic shape或显式参数？
- [ ] FakeImpl 与 runtime 是否使用同一 shape 公式？
- [ ] logical、scheduler-padded、collective、TP-local rows 是否分开命名？
- [ ] 每一层 padding 是否由同一层 unpad？
- [ ] DBO hook 是否通过 compile-safe boundary，而非 trace Python thread state？
- [ ] 子线程异常是否能主动中止并唤醒另一 ubatch？
- [ ] 冷编译验证是否确认没有命中旧 cache？
