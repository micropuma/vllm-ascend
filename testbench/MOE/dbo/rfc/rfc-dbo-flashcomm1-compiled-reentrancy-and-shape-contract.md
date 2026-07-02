# RFC: DBO + FlashComm1 compiled runtime 可重入性与 MLA 动态 shape contract

## 状态

- 状态：Draft / root cause localized
- 日期：2026-07-02
- vLLM commit：`7977da779ad8a2725a0ad58ac167af0a2278964a`
- vLLM Ascend commit：`fcb84f5c223d9d4559ced6bca30f30d9d2939b98`
- 模型：`/data/models/DeepSeek-V2-Lite-Chat`
- 硬件：2 x Ascend 910B3，TP=2，EP=2
- 关联 RFC：
  - `rfc-dbo-cold-start-problem.md`
  - `rfc-custom-op-fakeimpl-compile-safe.md`
  - `rfc-dbo-flashcomm1-compile-shape-contract.md`

## 摘要

在 DeepSeek-V2、TP=2、FlashComm1、DBO、AI_CPU、vLLM compile
同时开启时，系统呈现两个彼此独立、但会连续暴露的缺陷：

1. 冷启动 graph warmup 使用 16 个逻辑 token，但模型输入 buffer 的 token
   维是 4096。MLA output 的 shape contract 在 logical token、sequence-parallel
   local token 和 graph padded token 之间不一致，最初表现为 8/4096 mismatch。
2. 修复冷启动 shape mismatch 后，真实请求一旦触发 DBO，两个 ubatch 线程会并发
   调用同一个 AOT compiled callable。两个 TP rank 随后在第一层出现不同的
   hook/collective 推进顺序并永久等待。相同代码切到 eager 后请求正常完成。

进一步实验表明：

- AI_CPU 和 AIV 都会复现，排除单一 HCCL expansion backend。
- 将 DBO hooks 注册为 tensor identity custom ops 或 mutating custom ops
  都会复现。
- 将全部 DBO hooks 加入 piecewise `splitting_ops` 仍会复现，排除“仅仅是
  Dynamo 把 hook 节点重排”。
- DBO piecewise 路径本来就给 ubatch context 设置
  `CUDAGraphMode.NONE`，因此不是两个线程并发 replay 同一个 ACL Graph。
- 仅在真实 DBO runtime 设置 `skip_compiled=True` 后，4K DBO 请求成功。
  这把死锁边界定位到两个 ubatch 线程并发复用同一个 AOT compiled callable。
- 随后的 103-token 非 DBO 请求又暴露 8/52 mismatch，证明当前基于
  `_EXTRA_CTX.num_tokens` 分配 MLA output 的补丁把 warmup 值固化进了图，
  不是合法的动态 shape 修复。

本 RFC 建议把问题拆成两个 contract 分别修复：

1. **Execution contract**：DBO 双线程不得并发复用未声明为 reentrant 的 compiled
   execution instance。短期对真实 DBO runtime 使用 eager fallback；长期给两个
   ubatch 建立独立 compiled execution instance。
2. **Shape contract**：MLA logical token count 必须成为可符号化、可参与 guard/cache
   key 的显式 graph input，禁止通过 Python forward context 决定 compiled tensor
   的第一维。

---

## 1. 背景

### 1.1 DBO 的执行模型

vLLM upstream 用 `ParallelConfig.enable_dbo`、prefill/decode token threshold
决定是否启用 microbatch：

```python
# /data/workspace/vllm/vllm/v1/worker/ubatch_utils.py:38-46
def check_ubatch_thresholds(config, num_tokens, uniform_decode):
    if not config.use_ubatching:
        return False
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold
    return num_tokens >= config.dbo_prefill_token_threshold
```

切分时默认按 padded token 数等分：

```python
# /data/workspace/vllm/vllm/v1/worker/ubatch_utils.py:63-77
if split_point is None:
    split_point = int(num_tokens_padded) // num_ubatches
token_split_points = [
    split_point * i for i in range(1, num_ubatches)
]
```

vLLM Ascend 在 `_run_ubatches()` 中创建两个 Python 线程。两个线程共享传入的
`model` callable，但分别进入自己的 `AscendUBatchContext`：

```python
# vllm_ascend/worker/npu_ubatch_wrapper.py:212-258
def _run_ubatches(self, ubatch_metadata, model):
    def _ubatch_thread(results, model, ubatch_metadata):
        with ubatch_metadata.context:
            model_output = model(...)
            dbo_current_stream().synchronize()
            dbo_yield()

    for metadata in ubatch_metadata:
        thread = threading.Thread(
            target=_ubatch_thread,
            args=(results, model, metadata),
        )
        thread.start()
```

这里存在一个关键前提：

> `model` 及其内部 execution backend 必须支持两个 host thread 在不同 NPU
> stream 上并发调用，或者 DBO 必须为两个 ubatch 提供独立 execution instance。

eager PyTorch op 基本满足这一调用方式：每个线程在自己的 current stream/context
中逐个提交算子。AOT compiled callable 是否满足同样的可重入性，当前 vLLM 和
vLLM Ascend 均没有显式 contract 或保护。

### 1.2 compiled model 的调用入口

vLLM 的 compiled model decorator 在 forward context 的 `skip_compiled` 为
false 时复用 `self.aot_compiled_fn`：

```python
# /data/workspace/vllm/vllm/compilation/decorators.py:502-520
def __call__(self, *args, **kwargs):
    if self.do_not_compile or torch.compiler.is_compiling():
        return self.forward(*args, **kwargs)

    if (
        is_forward_context_available()
        and get_forward_context().skip_compiled
    ):
        return self.forward(*args, **kwargs)

    if getattr(self, "aot_compiled_fn", None) is not None:
        with maybe_use_cudagraph_partition_wrapper(self.vllm_config):
            return self.aot_compiled_fn(self, *args, **kwargs)
```

cache wrapper 只是无锁转发给同一个 `optimized_call`：

```python
# /data/workspace/vllm/vllm/compilation/caching.py:216-217
def __call__(self, *args, **kwargs):
    return self.optimized_call(*args, **kwargs)
```

代码中没有：

- per-thread compiled instance；
- per-ubatch execution state；
- reentrancy capability check；
- 保护 compiled invocation 的锁；
- 把 ubatch id 加入 compiled cache key。

锁本身也不是 DBO 的正确长期方案，因为串行化整个 forward 会消除计算通信重叠。

### 1.3 DeepSeek A2 overlap hook

DeepSeek A2 模板通过两个 event key 协调两个 ubatch：

```python
# vllm_ascend/dbo/overlap_templates/deepseek.py
def dbo_mla_preprocess_hook(self, is_record):
    if is_record:
        if get_forward_context().dbo_first_layer_sync:
            dbo_record_current_stream(event=UBatchEventKey.ATTN_PRE)
            get_forward_context().dbo_first_layer_sync = False
    else:
        dbo_wait_current_stream_and_yield(
            event=UBatchEventKey.ATTN_PRE
        )

def dbo_linear_row_hook(self, is_record):
    if is_record:
        dbo_record_current_stream(event=UBatchEventKey.ATTN_POST)

def dbo_moe_prepare_hook(self, is_record):
    if not is_record:
        dbo_wait_current_stream_and_yield(
            event=UBatchEventKey.ATTN_POST
        )
```

hook 并不是普通数值计算。它改变：

- 当前 ubatch 的 NPU event；
- 当前 stream 的 wait dependency；
- 两个 CPU worker thread 的交替执行顺序。

因此 hook 的执行次数、相对 collective 的位置和 host 可见顺序都属于 correctness
contract，而不是可自由优化的 instrumentation。

---

## 2. 用户可见表象

### 2.1 冷缓存 compilation failure

最初 fresh cache 启动在 MLA/o_proj 附近失败。诊断输出为：

```text
MLA output mismatch:
dbo_enabled=False,
fc1=True,
forward_num_tokens=16,
hidden_states=(8, 2048),
output=(4096, 2048),
o_proj_output=(8, 2048)
```

旧代码：

```python
output[...] = o_proj_output[: output.shape[0]]
```

等价于把 `[8, 2048]` 写入 `[4096, 2048]`，无法广播。

### 2.2 冷启动成功但请求永久等待

将 output 改为清零并写 valid prefix 后：

```python
output.zero_()
output[: o_proj_output.shape[0]] = o_proj_output
```

fresh-cache server 可以完成：

- AOT compile；
- piecewise/full graph capture；
- API startup。

但第一条触发 DBO 的请求永久等待：

- 4096 input / 1 output / concurrency 1：timeout；
- 96 input / 1 output / 32 requests / concurrency 16：
  第一条未触发 DBO 的请求成功，聚合 batch 触发 DBO 后停止。

服务没有最早 device-side exception，也没有 HTTP error；两个 worker 保持存活，
说明这是同步/collective progress deadlock，而不是普通 Python exception。

### 2.3 runtime-only eager 后 4K 请求成功

同一 server 配置只加 `--enforce-eager`：

```text
Successful requests: 1
Failed requests: 0
Total input tokens: 4103
Mean TTFT: 278.68 ms
Mean E2EL: 278.73 ms
```

在 compiled server 中，仅对真实 `_run_ubatches()` 创建的 forward context 设置
`skip_compiled=True`：

```text
Successful requests: 1
Failed requests: 0
Total input tokens: 4103
Mean TTFT: 184.02 ms
Mean E2EL: 184.07 ms
```

注意：两个 TTFT 都是单请求诊断数据，不能当作稳定性能结论。

### 2.4 随后暴露 8/52 mismatch

runtime-only fallback 通过 4K DBO 后，下一轮 103-token 非 DBO warmup 失败：

```text
RuntimeError:
The expanded size of the tensor (8) must match
the existing size (52) at non-singleton dimension 0.
Target sizes: [8, 2048].
Tensor sizes: [52, 2048].
```

调用点：

```python
# vllm_ascend/attention/mla_v1.py
output[: o_proj_output.shape[0]] = o_proj_output
```

这证明 output 的 8 rows 是 warmup 的 `16 / TP2` 被固化后的结果，而新的
103-token request 经过 sequence parallel 后产生 52 rows。

---

## 3. 实验矩阵

| 实验 | Compile | Hook 表达 | HCCL | 结果 |
|---|---:|---|---|---|
| fresh cold start，原 shape | 开 | compile guard/custom hook | AI_CPU | 8/4096 compilation failure |
| prefix-write shape patch | 开 | identity-return custom op | AI_CPU | 启动成功，4K DBO timeout |
| expansion 对照 | 开 | identity-return custom op | AIV | 4K DBO timeout |
| 移除误加 FC1 row hook | 开 | identity-return custom op | AI_CPU | 4K DBO timeout |
| 恢复 mutating hook | 开 | `None` + `mutates_args=["x"]` | AI_CPU | fresh start 成功，4K DBO timeout |
| hooks 加入 splitting ops | 开 | mutating splitting op | AI_CPU | fresh start 成功，4K DBO timeout |
| 完整 eager 对照 | 关 | Python hook | AI_CPU | 4K DBO 成功 |
| compiled server，DBO runtime skip compiled | server 开，DBO eager | Python hook | AI_CPU | 4K DBO 成功 |
| 同 server 后续 103-token non-DBO | 开 | compiled path | AI_CPU | 8/52 MLA mismatch |

关键日志：

```text
/data/workspace/logs/fc1_dbo_mutating_fresh_server.log
/data/workspace/logs/fc1_dbo_mutating_fresh_test.log
/data/workspace/logs/fc1_dbo_hookmap_aiv_server.log
/data/workspace/logs/fc1_dbo_split_hooks_fresh_server.log
/data/workspace/logs/fc1_dbo_split_hooks_4k_test.log
/data/workspace/logs/fc1_dbo_eager_control_server.log
/data/workspace/logs/fc1_dbo_eager_control_4k_test.log
/data/workspace/logs/fc1_dbo_skip_runtime_fresh_server.log
/data/workspace/logs/fc1_dbo_skip_runtime_4k_test.log
```

---

## 4. Trace 还原

### 4.1 正常的 eager 目标顺序

每个 rank 上，两个 ubatch 应保持相同的 host-level collective submission order：

```text
ubatch0 MLA preprocess AG
  record/wait/yield
ubatch1 MLA preprocess AG
ubatch0 o_proj RS
  record/wait/yield
ubatch1 o_proj RS
ubatch0 MoE prepare AG
...
```

TP rank 0 与 rank 1 不要求时间戳完全一致，但必须看到相同的 collective 序列。

### 4.2 compiled timeout 的最后进度

在 hook/yield 诊断中，两 rank 都能完成 28 层 capture/warmup。真实 DBO 请求开始后：

```text
ubatch0: mla T/F -> yield count 1
ubatch1: mla T/F -> yield count 1
ubatch0: column T/F -> yield count 2
ubatch1: column T/F -> yield count 2
ubatch0: mla T/F -> yield count 3
...
```

随后 rank 0 和 rank 1 的 ubatch 推进不同：

- 一个 rank 的 ubatch 已进入下一 hook；
- 另一个 rank 仍阻塞在前一个 FC1/collective 区间；
- 对端 rank 不再提交匹配 collective；
- 两个 ubatch CPU thread 均无法完成；
- `_run_ubatches()` 主线程永久阻塞在 `thread.join()`。

这一 trace 排除了：

- output merge；
- `torch.cat`；
- HTTP server；
- sampling；
- 最终 stream synchronize；

因为这些步骤都发生在两个 ubatch thread join 之后：

```python
# vllm_ascend/worker/npu_ubatch_wrapper.py:254-272
for thread in ubatch_threads:
    thread.join()
_check_thread_exceptions(results, "Ubatch runtime")
...
result = torch.cat(sorted_results, dim=0)
```

### 4.3 为什么不是 ACL Graph replay

真实 DBO piecewise 分支在创建 ubatch metadata 时已经显式传入：

```python
# vllm_ascend/worker/npu_ubatch_wrapper.py:441-453
ubatch_metadata = self._make_ubatch_metadata(
    ...
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
)
return self._run_ubatches(ubatch_metadata, self.model)
```

`ACLGraphWrapper` 在 runtime mode 为 `NONE` 时直接调用 runnable：

```python
# vllm_ascend/compilation/acl_graph.py:152-164
if (
    aclgraph_runtime_mode == CUDAGraphMode.NONE
    or aclgraph_runtime_mode != self.runtime_mode
):
    return self.runnable(*args, **kwargs)
```

因此 DBO timeout 不能归因于两个线程直接并发执行
`entry.aclgraph.replay()`。仍然共享的是 wrapper 下面的 AOT compiled runnable。

### 4.4 为什么 splitting ops 没有解决

vLLM piecewise compilation 用 `splitting_ops` 把 FX graph 分段。upstream 文档也强调：
piecewise CUDA graph 依赖 VLLM_COMPILE 和非空 splitting ops，而 CUDA graph mode
与 compilation mode 在设计上是正交维度。

实验把以下 ops 加入 splitting list：

```text
vllm::dbo_linear_column_hook
vllm::dbo_linear_row_hook
vllm::dbo_mla_preprocess_hook
vllm::dbo_moe_prepare_hook
vllm::dbo_moe_finalize_hook
```

EngineCore 日志确认配置生效，但 4K DBO 仍 timeout。这说明：

- 让 hook 从 compiled subgraph 中逃逸是必要的设计清理；
- 但不能令左右两个 compiled subgraph execution instance 自动变成 per-thread；
- 两个 ubatch 仍从同一个 `aot_compiled_fn` 入口进入共享 compiled execution。

---

## 5. 根因分析

## 5.1 Root cause A：compiled execution contract 未定义 DBO 可重入性

DBO 的并发模型要求：

```text
thread 0 -> model instance -> stream 0
thread 1 -> same model instance -> stream 1
```

vLLM compiled model 当前实现是：

```text
same model
  -> same aot_compiled_fn
    -> same PiecewiseBackend cache wrapper
      -> same optimized_call
```

没有任何代码保证 `optimized_call` 能被两个 host threads 并发调用。

远端实验建立了以下因果链：

1. compiled shared callable：稳定 timeout；
2. 只改 AI_CPU→AIV：仍 timeout；
3. 只改 hook identity→mutating：仍 timeout；
4. hooks 变成 splitting ops：仍 timeout；
5. 只改完整 eager：成功；
6. compiled server 中只让 runtime DBO bypass compiled：成功。

因此当前可确认的根因边界是：

> vLLM Ascend DBO 把 upstream compiled callable 当作 thread-safe/reentrant
> execution instance 使用，但 upstream 和 Ascend backend 没有提供这一 contract。

还不能仅凭 Python trace 断言 backend 内部具体是哪一个对象发生竞争。候选包括：

- AOT functionalization 对 mutable custom op 的 invocation state；
- compiled submodule 的 workspace/output buffer；
- backend execution plan 的 current-stream state；
- communication op 的 host submission state；
- generated execution function 的非 thread-local temporary state。

后续必须在 compiler/backend 入口记录：

```text
rank
ubatch id
host thread id
current NPU stream id
compiled callable id
subgraph id
collective sequence number
workspace/output address
```

只有拿到这些字段，才能把“shared compiled callable 不可重入”继续细分到具体 backend
对象。RFC 不把尚未观测的 backend 内部状态写成已确认事实。

## 5.2 Root cause B：Python runtime context 被误用为 compiled shape input

当前实验补丁增加了：

```python
def _resolve_mla_forward_inputs(
    hidden_states,
    flash_comm_v1_enabled,
    tp_size,
    is_vl_first_layer,
):
    logical_num_tokens = _EXTRA_CTX.num_tokens
    if logical_num_tokens is None:
        logical_num_tokens = get_forward_context().num_tokens
    local_num_tokens = (
        logical_num_tokens + tp_size - 1
    ) // tp_size
    return hidden_states, local_num_tokens, True
```

这一逻辑在 eager 中成立，但在 tracing 中存在生命周期错误：

1. `_EXTRA_CTX.num_tokens` 是 Python forward-context 值；
2. tracing warmup 时它等于 16；
3. Python 计算得到 `local_num_tokens=8`；
4. `torch.empty((8, hidden_dim))` 被记录进 compiled graph；
5. 之后 103-token request 复用同一图；
6. runtime o_proj 产生 `ceil(103/2)=52` rows；
7. 8-row output 无法容纳 52-row结果。

这不是 dynamic shape。dynamic shape 必须来自：

- tensor 的 symbolic dimension；
- 显式 `torch.SymInt` graph input；
- 能进入 guard/cache key 的 scalar input。

Python thread-local context 既不是 FX input，也不会自动加入 compiled cache key。

## 5.3 为什么 8/4096 和 8/52 是同一个 shape contract 的两面

旧实现按 graph padded input 分配：

```text
hidden graph buffer = 4096
output = 4096
runtime logical local = 8
```

补丁按 tracing 时 Python context 分配：

```text
warmup logical local = 8
output = 8
next runtime logical local = 52
```

前者过度采用 graph buffer shape，后者过度采用一次 tracing 的 runtime value。
正确 contract 必须同时表达：

```text
graph storage capacity
current logical token count
TP-local logical token count = ceil(logical / TP)
valid output length
```

并且 valid length 必须在每次 invocation 中可变。

---

## 6. 修复方案

## 6.1 Phase 0：恢复安全、可测试的 baseline

在继续开发前：

1. 保留已经验证必要的 FakeImpl/runtime shape 职责拆分；
2. 撤销 identity-return hook 实验；
3. hook 使用明确 side-effect schema，不能伪装成数值 identity；
4. 撤销通过一次 Python context 值静态分配 MLA output 的补丁；
5. 增加 regression tests，先让错误稳定、快速暴露。

## 6.2 Phase 1：DBO runtime 安全 fallback

最初实验只在 `_run_ubatches()` 的两个真实 runtime context 中设置：

```python
create_ascend_forward_context(
    ...,
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
    skip_compiled=True,
)
```

但必须限制作用域：

- graph capture/warmup 的 `_capture_ubatches()` 不设置；
- 非 DBO 请求不设置；
- DBO 被 abort 后的单 batch 不设置；
- 只对真正由两个 host thread 并发执行的 `_run_ubatches()` 设置。

建议 API：

```python
def _make_ubatch_metadata(
    ...,
    skip_compiled: bool = False,
):
    ...
```

capture call-site：

```python
skip_compiled=False
```

runtime dual-thread call-site：

```python
skip_compiled=True
```

该方案的意义：

- server 仍可完成 cold compile 和 graph capture；
- 非 DBO batch 继续使用 compiled path；
- DBO 保留双 stream eager overlap；
- correctness 不再依赖未声明的 compiled reentrancy。

该 fallback 已通过单条 4K DBO 请求，但在合入前必须先修复 MLA shape contract，
否则后续 non-DBO compiled 请求仍可能发生 8/52 mismatch。

在 MLA symbolic shape contract 完成前，可采用更保守但可立即工作的 fallback：

```python
use_dbo_runtime_eager_fallback = (
    self.parallel_config.enable_dbo
    and enable_sp(self.vllm_config)
)

set_ascend_forward_context(
    ...,
    skip_compiled=(
        has_encoder_input
        or use_dbo_runtime_eager_fallback
    ),
)
```

同时，`create_ascend_forward_context()` 必须向两个 ubatch context 传播外层值：

```python
skip_compiled=cur_forward_context.skip_compiled
```

该版本的边界是：

- cold compile/capture 不变；
- DBO + FlashComm1 server 的真实 runtime 全部 eager；
- DBO 与 non-DBO 请求均不会复用已固化 8-row output 的 compiled graph；
- 不启用 DBO 或 FlashComm1 的配置不受影响。

远端连续验证结果：

| Case | Result | Mean TTFT |
|---|---:|---:|
| 4096/1, 1 request, DBO | 1/1 | 242.39 ms |
| 96/1, 1 request, non-DBO after DBO | 1/1 | 178.08 ms |
| 96/1, 32 requests, concurrency 16 | 32/32 | 587.40 ms |

对应日志：

```text
/data/workspace/logs/fallback_v2_dbo4k_test.log
/data/workspace/logs/fallback_v2_non_dbo_test.log
/data/workspace/logs/fallback_v2_multi32_test.log
/data/workspace/logs/fc1_dbo_global_runtime_fallback_v2_server.log
```

这一版本适合作为短期 correctness fallback，但它会暂时放弃 DBO+FlashComm1
真实请求的 compiled 性能，不能替代 per-ubatch compiled execution instance。

## 6.3 Phase 2：per-ubatch compiled execution instance

长期方案不能用全局锁串行化 compiled forward，因为那会取消 DBO overlap。

建议为每个 ubatch 建立独立 execution slot：

```text
model weights / parameters       shared, read-only
FX graph                         shared, immutable
compiled execution instance[0]   owned by ubatch0
compiled execution instance[1]   owned by ubatch1
workspace/output buffers[0]      owned by ubatch0
workspace/output buffers[1]      owned by ubatch1
stream binding[0]                compute stream
stream binding[1]                comm/secondary stream
```

接口方向：

```python
compiled_model = compile_model(...)

ubatch_runners = [
    compiled_model.create_execution_instance(slot=0),
    compiled_model.create_execution_instance(slot=1),
]
```

如果 backend 不支持 clone execution instance，应在 capability 中显式声明：

```python
supports_concurrent_invocation: bool
supports_per_stream_execution_instance: bool
```

DBO enable check 应包含该能力：

```python
if enable_dbo and compilation_enabled:
    if not compiler_backend.supports_concurrent_invocation:
        use_runtime_dbo_eager_fallback()
```

不能默认所有 `torch.compile` backend 都具备同一并发语义。

## 6.4 Phase 3：hook 作为 control-plane boundary

即使有 per-ubatch compiled instance，hook 仍不应作为普通 tensor identity op。

建议二选一：

### 方案 A：piecewise control op

- hook custom op 标记为 cudagraph unsafe / splitting op；
- compiled subgraph 在 hook 前后切分；
- hook 在 host eager control-plane 执行；
- hook 输入只用于建立分区边界，不通过假 alias 表达 dependency。

### 方案 B：编译器原生 side-effect token

引入显式 control token：

```text
subgraph A
  -> DBO event record/wait/yield
  -> subgraph B
```

compiler IR 必须保留 token 顺序，backend 不能删除、合并或跨越该 token 调度
collective。

在 upstream 没有 side-effect token contract 前，方案 A 更现实。

仅把 hook 名字加入 `splitting_ops` 不足以解决当前 bug，因为它没有解决两个线程
共享同一个 compiled execution instance，但它仍应作为长期架构的一部分。

## 6.5 Phase 4：MLA 显式 logical-length contract

MLA custom op 需要显式区分 capacity 和 valid rows。推荐接口：

```python
def mla_forward(
    hidden_states: Tensor,
    logical_num_tokens: SymInt,
    tp_size: int,
    need_gather_q_kv: bool,
    prefix: str,
) -> tuple[Tensor, SymInt]:
    ...
```

或者 caller 分配 capacity buffer，custom op 返回 valid length：

```python
output_capacity = hidden_states.shape[0]
output = torch.empty(
    (output_capacity, hidden_dim),
    ...
)
valid_rows = ceil_div(logical_num_tokens, tp_size)
torch.ops.vllm.mla_forward(..., output, valid_rows)
return output[:valid_rows]
```

关键要求：

1. `logical_num_tokens` 必须是 graph input 或从 symbolic tensor shape 推导；
2. 不能在 traced function 内读取 `_EXTRA_CTX` 决定 allocation shape；
3. `ceil_div` 公式 runtime/FakeImpl 完全一致；
4. logical length 参与 graph guard/cache key；
5. prefix slicing 不得在生成代码中产生未定义 `Sym(Min(...))`；
6. graph storage capacity 与 model-visible logical shape 必须分离。

如果 Ascend backend 暂时不能支持 dynamic output，备选方案是按 logical-token
range 编译多个 graph：

```text
[1, 16]
[17, 128]
[129, 1024]
[1025, 4096]
[4097, 16384]
```

每个 range 的 output capacity 固定，但 runtime 必须携带 valid rows；不能把
warmup 的单点值 16 当成整个 range 的 output shape。

---

## 7. 测试计划

## 7.1 Unit tests

### compiled dispatch

- `skip_compiled=False` 调用 `aot_compiled_fn`；
- `skip_compiled=True` 调用 eager `forward`；
- capture metadata 和 runtime metadata 分别验证；
- DBO abort 后不得遗留 `skip_compiled=True`。

### MLA shape matrix

至少覆盖：

| logical tokens | TP | local rows |
|---:|---:|---:|
| 1 | 2 | 1 |
| 8 | 2 | 4 |
| 16 | 2 | 8 |
| 103 | 2 | 52 |
| 4096 | 2 | 2048 |
| 4103 | 2 | 2052 |

对每组检查：

- runtime output；
- FakeImpl output；
- compiled output；
- odd-token padding/unpadding；
- residual shape；
- o_proj input/output shape。

### hook sequence

用 recording template 记录：

```text
(rank, ubatch, layer, hook, record/wait, sequence)
```

断言：

- 两 rank collective sequence 一致；
- 每个 event key record/wait 配对；
- 每层 hook 次数一致；
- capture 与 runtime trace 分开统计。

## 7.2 System tests

### 必须组合

```text
DBO on/off
FlashComm1 on/off
compile on/off
AI_CPU/AIV
fresh/warm cache
single long request/many concurrent requests
odd/even token count
```

### 最小 correctness cases

1. 4096/1，1 request，concurrency 1；
2. 96/1，32 requests，concurrency 16；
3. 4096/16，500 requests，concurrency 96；
4. 103/1，验证 8/52 regression；
5. 服务启动后交替发送 DBO 与 non-DBO batch。

### 通过标准

- fresh cache startup 成功；
- 所有请求 HTTP 200；
- 无 worker timeout；
- 无 rank collective sequence divergence；
- 输出 shape 与 eager baseline 一致；
- 至少运行三轮并报告 TTFT/QPS 方差。

---

## 8. Upstream 参考

### vLLM DBO 配置与切分

- `ParallelConfig.enable_dbo` 和 threshold：
  <https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py>
- upstream ubatch slicing：
  <https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/ubatch_utils.py>

### vLLM compilation / CUDAGraph

- CompilationConfig API：
  <https://docs.vllm.ai/en/latest/api/vllm/config/compilation/>
- CompilationConfig/CUDAGraph overhaul RFC：
  <https://github.com/vllm-project/vllm/issues/20283>
- 该 RFC 明确把 compilation mode、splitting ops 和 cudagraph mode 作为可独立
  组合的维度，并提出 debug mode、shape/address checks。
- full CUDA graph 与 piecewise graph 相关实现讨论：
  <https://github.com/vllm-project/vllm/pull/20059>

### 可借鉴的 `skip_compiled` contract

upstream compiled model decorator 已提供 `ForwardContext.skip_compiled`，用于 shape/type
无法安全复用单一 compiled graph 的场景：

```python
if get_forward_context().skip_compiled:
    return self.forward(*args, **kwargs)
```

本 RFC 的 Phase 1 不引入新的全局开关，而是复用该 per-forward contract，并把作用域
限制在真实 DBO dual-thread runtime。

### custom op / graph partition

upstream CompilationConfig 支持：

- `splitting_ops`；
- cudagraph-unsafe custom op partition；
- `use_inductor_graph_partition`。

DBO hook 可参考 attention/cache update ops 作为 piecewise boundary 的设计，但需要
额外解决 per-ubatch execution instance；仅分图不提供并发可重入性。

---

## 9. 不推荐方案

### 9.1 回退整个 server 到 eager

能保证 correctness，但会无差别关闭 non-DBO compiled/graph 优化。只适合作为
诊断对照，不是最终修复。

### 9.2 用全局锁保护 compiled callable

可避免 concurrent invocation，但会把两个 ubatch 完全串行化，DBO 的 overlap
收益消失，并可能与 hook yield 形成新的锁等待。

### 9.3 identity-return hook

```python
def hook(x):
    side_effect()
    return x
```

如果 schema 没有声明 alias/mutation，runtime 返回输入 alias 与 compiler contract
不一致；如果声明 mutation，又会引入 functionalization 语义。它不是可靠的纯
control dependency。

### 9.4 FakeImpl 或 allocation 读取 forward context

FakeTensor replay 和 runtime forward 的生命周期不同。context 不存在时 fallback、
捕获 AssertionError 或读取 snapshot 都不能表达 per-request dynamic length。

### 9.5 `min(output_rows, o_proj_rows)` 静默裁剪

这会隐藏 shape contract 错误并丢弃 token。此前 symbolic `Min` 还曾在生成图中
导致未定义名称错误。correctness 修复必须让 producer/consumer 共享相同 valid
length，而不是在最后赋值处截断。

---

## 10. 建议实施顺序

1. 增加 103-token regression，证明当前静态 8-row output 不合法。
2. 重构 MLA API，使 logical length 成为显式 symbolic contract。
3. 验证 eager、non-DBO compiled、fresh capture 全部通过。
4. 引入 runtime-only DBO `skip_compiled` fallback。
5. 完成 4K single 和 32x96 concurrent correctness。
6. 跑 500x4096/concurrency96 soak test。
7. 在 compiler backend 增加 execution-instance capability。
8. 实现两个 per-ubatch compiled runners。
9. 将 hooks 正式迁移为 control-plane splitting boundaries。
10. 对比 eager fallback 与 per-ubatch compiled 的 TTFT、QPS 和 overlap trace。

---

## 11. 结论

本问题不是单一的“冷启动 shape bug”，也不是单一的“HCCL 卡死”：

```text
问题 A：MLA logical shape 没有成为 compiled graph 的显式输入
  -> 8/4096
  -> 临时补丁后变成 8/52

问题 B：DBO 双线程并发复用同一个 AOT compiled callable
  -> TP ranks collective progress divergence
  -> thread.join 永久等待
```

eager 对照、splitting-op 对照和 runtime-only `skip_compiled` 对照共同把问题定位到：

> 当前 compiled model execution contract 不覆盖 DBO 所需的 concurrent,
> per-stream, reentrant invocation；同时当前 MLA patch 把 runtime Python
> context 错当成 dynamic graph shape。

正确修复必须同时提供：

- per-ubatch 独立 compiled execution state；
- graph 外的 DBO control-plane hooks；
- 显式 symbolic logical-length；
- runtime/FakeImpl/compiled 一致的 sequence-parallel shape contract。

在这些 contract 完成前，最安全的产品策略是：保留 server compile 能力和 non-DBO
compiled path，但对真实 DBO dual-thread runtime 使用严格限域的 eager fallback。
