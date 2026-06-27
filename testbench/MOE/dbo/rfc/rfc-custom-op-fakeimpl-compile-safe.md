# RFC: vLLM-Ascend custom op 的 compile-safe FakeImpl 拆分

## 状态

- 状态：核心方案已实现；建议扩展为 custom-op 设计规范
- 关联提交：`98446bf8`
- 日期：2026-06-27

## 摘要

旧版 `register_custom_ops.py` 将通信、padding/unpadding 和运行时配置判断封装在
同一个 custom op 中。其 FakeImpl 为了决定输出 shape，直接读取
`_EXTRA_CTX`/`get_forward_context()`。

最新 vLLM piecewise compiler 会在 backend 构造阶段执行 `compile_all_ranges()`。
FakeTensor 重放发生时不保证存在 active forward context，因此 FakeImpl 抛出：

```text
AssertionError: Forward context is not set
```

本 RFC 将问题归纳为两个独立约束：

1. FakeImpl 必须是 compile-time 可执行的纯 shape function；
2. DBO 需要在通信前后插入 hook，因此通信和 tensor shape 变换必须拆分。

解决方案是把复合 op 拆成显式通信与纯 tensor 变换，并为 compile 所需配置建立
compile-safe snapshot。runtime 继续读取 per-request context，FakeImpl 不再读取
runtime-only state。

## 背景

相关 custom ops：

```text
maybe_all_gather_and_maybe_unpad
maybe_unpad_after_all_gather
maybe_pad_and_reduce
maybe_prepare_for_reduce
matmul_and_reduce
```

FlashComm1 关闭时，部分 op 的 runtime 语义接近 identity/all-reduce，错误的 fake
shape 可能恰好不暴露。FlashComm1 开启后，同一个 op 变成条件性 shape 变换：

```text
AllGather + unpad
pad + ReduceScatter
```

FakeImpl 若不知道 compile-time 开关，或读取不到 runtime context，就会崩溃或声明
错误 shape。

## 问题描述

典型启动错误：

```text
Dynamo failed to run FX node with fake tensors
torch.ops.vllm.maybe_pad_and_reduce(...)
AssertionError: Forward context is not set
```

调用链：

```text
torch.compile
  -> VllmBackend
  -> PiecewiseCompileInterpreter
  -> PiecewiseBackend.compile_all_ranges
  -> inductor / FakeTensorMode
  -> custom op FakeImpl
  -> _EXTRA_CTX.flash_comm_v1_enabled
  -> get_forward_context()
  -> AssertionError
```

该错误发生在 compile 阶段，不是模型 runtime forward。

## 为什么新编译链稳定暴露问题

旧版 piecewise backend 更接近 lazy compile：运行到某一 range 时再编译，编译时
经常仍处于 forward context 生命周期内。

新版 backend 在构造阶段执行所有 range 的编译。compile 与 runtime forward
生命周期解耦，以下假设不再成立：

```text
FakeImpl 执行时一定存在 forward context
```

新版编译链没有引入 custom-op bug，而是稳定暴露了原有的不安全依赖。

## 根因

### FakeImpl 不是纯 shape function

旧设计：

```python
def fake(x, ...):
    if _EXTRA_CTX.flash_comm_v1_enabled:
        ...
```

`_EXTRA_CTX` 是 runtime forward-context proxy，不是 compile context。

FakeImpl 的正确职责仅包括：

- 根据输入 shape、显式参数和 compile-safe 常量推导输出 shape；
- 创建 fake/meta tensor；
- 不执行通信；
- 不读取 request、stream、DP metadata 或 runtime context。

### 一个 op 混合了两种生命周期

旧复合 op 同时承担：

```text
HCCL collective
tensor pad/unpad
runtime configuration branch
```

DBO 需要在 collective 前后执行 record/wait/yield。通信被封装在 custom op 内部时，
hook 无法插入正确边界，于是又引入 `do_comm=True/False`，令同一个 op 拥有多种
shape 语义，FakeImpl 条件组合进一步复杂化。

### Fake 和 runtime shape contract 不一致

例如 reduce-scatter runtime 会先 pad 到 TP 整数倍：

```python
runtime_tokens = ceil(num_tokens / tp_size)
```

旧 FakeImpl 使用：

```python
fake_tokens = num_tokens // tp_size
```

奇数 token 时稳定差一。

## 设计目标

1. FakeImpl 不读取 `_EXTRA_CTX` 或 `get_forward_context()`；
2. 通信操作和 tensor shape 操作可分别建图；
3. DBO hook 能覆盖显式 collective；
4. runtime impl 与 FakeImpl 使用同一 shape 公式；
5. 无 active forward context 时也能完成所有 compile ranges；
6. eager、compile、DBO/non-DBO 共享明确 contract。

## 解决方案

### 1. 拆分通信与 shape 变换

DBO 路径：

```text
hook(record)
explicit all_gather
hook(wait/yield)
maybe_unpad_after_all_gather
```

finalize 路径：

```text
maybe_prepare_for_reduce
hook(record)
explicit reduce_scatter
hook(wait/yield)
```

新增/独立后的 tensor-only ops：

- `maybe_unpad_after_all_gather`
- `maybe_prepare_for_reduce`

它们的关键长度通过显式参数传入，因此 FakeImpl 可以直接构造输出：

```python
empty((unpadded_length, *x.shape[1:]))
empty((prepared_length, *x.shape[1:]))
```

### 2. compile-safe 配置快照

对于仍需在 FakeImpl 判断的模型级配置，使用在 compile 前写入的普通模块状态：

```python
_FLASH_COMM_V1_SNAPSHOT = False

def set_flash_comm_v1_snapshot(value):
    global _FLASH_COMM_V1_SNAPSHOT
    _FLASH_COMM_V1_SNAPSHOT = value
```

`set_ascend_forward_context()` 在进入编译前更新 snapshot。FakeImpl 只读取 snapshot，
runtime impl 仍读取当前 forward context。

该 snapshot 只适合模型/编译期稳定配置。per-ubatch token 数、DP metadata、stream
等动态信息不能快照，必须作为显式参数或从输入 symbolic shape 推导。

### 3. 统一 shape 公式

ReduceScatter FakeImpl 使用：

```python
def reduce_scatter_tokens(num_tokens, tp_size):
    return (num_tokens + tp_size - 1) // tp_size
```

runtime 和 fake 必须共享等价公式，禁止一侧 floor、另一侧 pad+ceil。

### 4. runtime fallback 与 compile contract 分离

runtime impl 可以在没有 forward context 的工具/测试场景执行保守路径；FakeImpl
不能通过 `try/except get_forward_context()` 猜配置。猜测会令同一 FX graph 在不同
编译时机得到不同 shape。

## 为什么不推荐其他方案

### FakeImpl 读不到 context 时默认 False

这会把 compile 结果绑定到偶然的 context 生命周期。FlashComm runtime 为 True 时，
fake 仍按 identity 推导，错误会延迟到图执行阶段。

### 在 fake 中捕获 AssertionError

只能避免启动崩溃，不能保证 fake/runtime shape 一致。

### 把所有 DBO hook 放进 compiled region

DBO hook 包含 event、stream 和 CPU yield，其生命周期不是普通 tensor graph。
应让 compiled tensor graph 与调度控制边界清晰分离。

### 仅关闭 compile cache

cache 可能隐藏旧 FakeImpl，但不是代码修复。验证时还必须同时隔离：

```text
CompilationConfig.cache_dir
VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile
```

## 实现范围

提交 `98446bf8` 包含：

- compile-safe FlashComm1 snapshot；
- all-gather 与 unpad 的拆分；
- reduce 前 prepare op；
- MLA、linear、MoE prepare/finalize call-site 调整；
- FakeImpl 与 runtime impl 的职责分离。

后续 odd-ubatch 修复进一步要求：

- ubatch logical tokens 来自 attention metadata；
- FakeImpl reduce-scatter 使用 ceil division；
- RoPE、unpad 和 collective padding 使用同一逻辑长度。

详见：

- `rfc-dbo-flashcomm1-compile-shape-contract.md`

## 测试策略

### 单元测试

1. 无 active forward context 调用所有 FakeImpl；
2. FlashComm1 snapshot on/off；
3. `do_comm`、`label`、EP/TP 参数组合；
4. odd/even token；
5. fake/runtime 输出 shape 对比；
6. 显式 unpadded/prepared length。

### 编译测试

1. eager；
2. torch.compile，无 DBO；
3. torch.compile + DBO；
4. FlashComm1 on/off；
5. fresh piecewise cache 和 fresh AOT cache；
6. 多 compile ranges。

### 验收标准

- compile 阶段不读取 runtime forward context；
- 无 `Forward context is not set`；
- fake/runtime shape 全矩阵一致；
- fresh AOT 可完成启动和 graph capture；
- DBO + FC1 + compile 压力测试 100% 成功。

## 风险与后续工作

1. 模块级 snapshot 必须限定为 compile-stable 配置，避免线程间动态覆盖；
2. 审计所有 `direct_register_custom_op` FakeImpl；
3. 为 custom-op 增加统一 fake/runtime shape contract 测试工具；
4. code review 必须检查 FakeImpl 是否读取 runtime singleton/context；
5. 将 odd-ubatch 和无-forward-context compile 加入 nightly。

## 参考资料

- `../bug/dbo-shape-inference.md`
- `../bug/flashcomm-dbo-compile-bugs.md`
- `../bug/vllm-ascend-compilation.md`
- commit `98446bf8`

