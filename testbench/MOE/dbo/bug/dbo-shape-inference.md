# FlashComm1 在最新 vLLM 编译链路下崩溃分析

## 1. Bug 描述

### 1.1 现象

开启 FlashComm1：

```bash
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
```

启动阶段会在 `torch.compile` 期间崩溃，典型报错是：

```text
Dynamo failed to run FX node with fake tensors
torch.ops.vllm.maybe_pad_and_reduce(...)
AssertionError: Forward context is not set
```

关闭 FlashComm1 时，这个问题通常不会出现。

### 1.2 直接原因

崩溃发生在 fake impl 里读取了运行时 `forward context`：

- [vllm_ascend/ops/register_custom_ops.py](../vllm_ascend/ops/register_custom_ops.py)
  - `_maybe_all_gather_and_maybe_unpad_fake`
  - `_maybe_pad_and_reduce_fake`
- [vllm_ascend/ascend_forward_context.py](../vllm_ascend/ascend_forward_context.py)
  - `_EXTRA_CTX`
- [../../vllm-dly/vllm/forward_context.py](../../vllm-dly/vllm/forward_context.py)
  - `get_forward_context()`

其中 fake impl 会访问：

```python
_EXTRA_CTX.flash_comm_v1_enabled
```

而 `_EXTRA_CTX` 最终会走到 upstream 的：

```python
get_forward_context()
```

当 `_forward_context is None` 时，upstream 直接报：

```python
AssertionError: Forward context is not set
```

### 1.3 调用链

这次不是运行时 forward 崩，而是 compile 阶段崩：

```text
model __call__
  -> torch.compile wrapper
    -> VllmBackend.__call__
      -> split_graph(...)
      -> PiecewiseCompileInterpreter.run(...)
        -> PiecewiseCompileInterpreter.call_module(...)
          -> PiecewiseBackend(...)
            -> PiecewiseBackend.compile_all_ranges()
              -> CompilerManager.compile(...)
                -> inductor.compile(...)
                  -> FakeTensorMode 重放 FX graph
                    -> torch.ops.vllm.maybe_pad_and_reduce(...)
                      -> _maybe_pad_and_reduce_fake(...)
                        -> _EXTRA_CTX.flash_comm_v1_enabled
                          -> get_forward_context()
                            -> AssertionError
```

### 1.4 为什么 FlashComm1 会触发

> :cry: remain check

这里要区分两件事：

1. 某个 custom op 是否会进入 FX graph
2. 哪个 subgraph 会在 compile 阶段实际触发崩溃

并不是所有相关 custom op 都是“开启 FlashComm1 后才第一次进图”。
例如在 `DBO + MLA preprocess` 路径中，`maybe_all_gather_and_maybe_unpad(..., do_comm=False)` 本身就可能进入图。

真正的问题在于：

- 最新 vLLM 的 piecewise backend 会在 `PiecewiseBackend.__init__` 中更早地执行 `compile_all_ranges()`
- compile 过程中，inductor 会用 fake tensor 重放 FX graph
- Ascend 的 fake impl 会访问 `_EXTRA_CTX.flash_comm_v1_enabled`
- `_EXTRA_CTX` 依赖 runtime `forward context`，而 compile 阶段并不保证这个 context 仍然存在

因此，FlashComm1 更准确的作用不是“让某个 op 首次进入图”，而是：

- 改变了 subgraph 的分支形态、覆盖范围或编译时机
- 使依赖 `_EXTRA_CTX` 的 fake impl 更容易在无 active forward context 的 compile 阶段被执行
- 从而把这个 compile-time 不安全依赖暴露出来

另外，即使某些调用点传入的是 `do_comm=False`，fake impl 也仍然可能先求值：

```python
_EXTRA_CTX.flash_comm_v1_enabled
```

所以问题不在于“这次 runtime 是否真的做 comm”，而在于 fake impl 在 compile-time 读取了 runtime-only 状态。

## 2. v0.13.0 和当前版本的区别

这里最关键的是 `PiecewiseBackend` 的行为变了。

### 2.1 v0.13.0：懒编译

`v0.13.0` 的 [piecewise_backend.py](../../vllm-dly/vllm/compilation/piecewise_backend.py) 行为是：

- `__init__` 只初始化 `range_entries`
- 不会立即编译
- 真正编译发生在：

```text
PiecewiseBackend.__call__()
  -> _maybe_compile_for_range_entry(...)
```

也就是：运行到某个 shape，才编译该 shape/range。

这意味着编译时机更接近真实 forward 运行期，很多情况下仍处于：

```python
with set_forward_context(...):
```

或 Ascend 对应的：

```python
with set_ascend_forward_context(...):
```

因此虽然 fake impl 依赖 runtime context 这件事本身也不合理，但旧链路下更不容易立即暴露。

### 2.2 当前版本：构造时全编译

当前版本的 [piecewise_backend.py](../../vllm-dly/vllm/compilation/piecewise_backend.py) 变成了：

- `PiecewiseBackend.__init__` 中如果 `graph is not None`
- 直接调用：

```python
self.compile_all_ranges()
```

也就是：

- subgraph 一旦被 `PiecewiseCompileInterpreter` 识别为要编译
- backend 构造时就同步完成所有 compile range 的编译
- 编译完成后才返回 runtime callable

这比 `v0.13.0` 明显更“激进”。

### 2.3 这个变化为什么会导致现在挂

因为当前版把 compile 时机从“运行到时再编”改成了“构造 backend 时先全部编完”，导致：

1. compile 和 runtime forward 生命周期更彻底地解耦
2. fake impl 被调用时，不再能假设 `forward context` 一定存在
3. 任何 compile-time 读取 runtime-only context 的代码都会更容易炸

你的 fake impl 恰好就是这种情况：

- 它不是纯 shape function
- 它依赖 `_EXTRA_CTX.flash_comm_v1_enabled`
- 而 `_EXTRA_CTX` 是 runtime proxy，不是 compile-safe snapshot

所以当前版本不是“引入了新 bug”，而是“把原本潜伏的不安全依赖稳定放大了”。

## 3. Piecewise Backend 需要重点理解的代码

如果要彻底理解这类问题，建议按下面顺序读。

### 3.1 编译入口

- [../../vllm-dly/vllm/compilation/decorators.py](../../vllm-dly/vllm/compilation/decorators.py)
  - `support_torch_compile`
  - `__call__`
  - `monitor_torch_compile`

这里负责：

- 包装模型 `forward`
- 首次调用时触发 `torch.compile`
- 组织 compile 与 warmup/profiling run

### 3.2 torch.compile wrapper

- [../../vllm-dly/vllm/compilation/wrapper.py](../../vllm-dly/vllm/compilation/wrapper.py)
  - `TorchCompileWithNoGuardsWrapper`

这里负责：

- 创建 `self._compiled_callable = torch.compile(...)`
- 真正把模型 forward 接到 Dynamo

### 3.3 vLLM 自定义 backend 主入口

- [../../vllm-dly/vllm/compilation/backends.py](../../vllm-dly/vllm/compilation/backends.py)
  - `VllmBackend.__call__`

重点看这里做了什么：

1. `split_graph(...)`
2. `PiecewiseCompileInterpreter(...).run(*fake_args)`
3. `generate_execution_code(...)`
4. 拼最终 runtime callable

### 3.4 subgraph 编译调度

- [../../vllm-dly/vllm/compilation/backends.py](../../vllm-dly/vllm/compilation/backends.py)
  - `PiecewiseCompileInterpreter.call_module`

这里是 piecewise compile 的核心调度点。  
它会在访问某个 submodule 时创建：

```python
PiecewiseBackend(...)
```

### 3.5 piecewise backend 本体

- [../../vllm-dly/vllm/compilation/piecewise_backend.py](../../vllm-dly/vllm/compilation/piecewise_backend.py)

重点看：

- `PiecewiseBackend.__init__`
- `compile_all_ranges`
- `get_fake_args_from_graph`
- `create_concrete_args`
- `__call__`

当前问题最关键的逻辑就在：

```python
if self.graph is not None:
    self.compile_all_ranges()
```

### 3.6 真正调用 inductor 的位置

- [../../vllm-dly/vllm/compilation/backends.py](../../vllm-dly/vllm/compilation/backends.py)
  - `CompilerManager.compile`

这里会继续往 backend adaptor 走，最终进入 inductor。

### 3.7 forward context 生命周期

- [../../vllm-dly/vllm/forward_context.py](../../vllm-dly/vllm/forward_context.py)
  - `set_forward_context`
  - `override_forward_context`
  - `get_forward_context`

- [vllm_ascend/ascend_forward_context.py](../vllm-ascend/vllm_ascend/ascend_forward_context.py)
  - `set_ascend_forward_context`
  - `_ExtraForwardContextProxy`
  - `_EXTRA_CTX`

理解这两个文件后，你就会明白：

- upstream `forward_context` 是 runtime 对象
- Ascend `_EXTRA_CTX` 只是对它的代理
- fake impl 不应该依赖它

### 3.8 FlashComm1 fake impl

- [vllm_ascend/ops/register_custom_ops.py](../vllm-ascend/vllm_ascend/ops/register_custom_ops.py)

重点看：

- `_maybe_all_gather_and_maybe_unpad_fake`
- `_maybe_pad_and_reduce_fake`

这两个函数应该被当成：

- compile-time shape function
- 不能依赖 runtime forward context

## 4. 如果要基于最新 vLLM + DBO + FlashComm，该怎么做

### 4.1 基本判断

如果你要长期基于：

- 最新 vLLM
- Ascend DBO
- FlashComm1

那么需要接受一个前提：

- 不能再假设 compile 阶段能安全读取 runtime forward context

也就是说，所有 fake impl / compile-time metadata 逻辑都必须 compile-safe。

### 4.2 设计原则

建议按下面原则改：

1. fake impl 只做 shape 推断
2. fake impl 不读取 `_EXTRA_CTX`
3. compile 需要的开关值在 compile 前做快照
4. runtime context 和 compile-time context 分离

### 4.3 建议落地方向

最直接的做法是：

1. 为 `flash_comm_v1_enabled` 建 compile-time snapshot
2. 在进入 `set_ascend_forward_context(...)` 之前或刚进入时写入这个 snapshot
3. fake impl 只读 snapshot，不读 `_EXTRA_CTX`

也就是说：

- runtime 路径继续用 `forward_context.flash_comm_v1_enabled`
- fake impl 路径改成读普通模块级状态或显式 compile-safe 状态

这样能同时兼容：

- 最新 vLLM 的 eager piecewise compile
- DBO 运行时上下文
- FlashComm1 custom op shape 推断

### 4.4 不建议的做法

不建议继续沿着下面方向补洞：

1. 在 fake impl 里 try/except `get_forward_context`
2. 读不到就默认为 False
3. 或者用 runtime context 是否存在来猜逻辑

原因是这会把 shape 推断变成不稳定行为，后面还会出别的问题。

### 4.5 推荐的调试/学习路径

如果你准备自己继续跟代码，建议这样做：

1. 先读 `decorators.py -> wrapper.py -> backends.py`
2. 再读 `PiecewiseCompileInterpreter -> PiecewiseBackend`
3. 再读 `forward_context.py` 和 `ascend_forward_context.py`
4. 最后只盯 `register_custom_ops.py` 里的 fake impl

你真正要建立的心智模型是：

- 运行时 forward context 是谁创建的
- 生命周期覆盖到哪里
- compile 阶段谁在重放 graph
- fake impl 为什么会在没有 forward context 的情况下被调用

### 4.6 当前建议

如果目标是“基于最新 vLLM + DBO + FlashComm 持续开发”，建议优先做这件事：

- 把 Ascend fake impl 全面排查一遍
- 所有 compile-time fake impl 禁止依赖 `_EXTRA_CTX`

| 信息类型                 | 例子                                                            |             fake impl 能不能读 | 正确做法                                      |
| -------------------- | ------------------------------------------------------------- | -------------------------: | ----------------------------------------- |
| 静态配置                 | `flash_comm_v1_enabled`, `tp_world_size`, 后端开关                | 可以，但不要从 `ForwardContext` 读 | 从 env/config/module attr 建 snapshot，或显式传参 |
| 输入 shape 派生信息        | `pad_size = f(x.shape[0], tp_world_size)`                     |                         可以 | fake impl 内部根据 `x.shape` 推                |
| batch-dependent 信息   | `dp_metadata`, `num_tokens_across_dp_cpu`, `padded_length`    |            不应从 context 隐式读 | 显式传入 custom op，或让该 op 不进入 compile         |
| runtime group/stream | `get_ep_group()`, `get_dp_group()`, HCCL group, stream switch |              fake impl 不能读 | 只放 real impl                              |
| runtime 特例           | `is_draft_model`, `is_vl_model`, per-step 状态                  |            fake impl 不能隐式读 | 显式参数化，或静态化为 module attr                   |
| 真实通信副作用              | `all_gather`, `reduce_scatter`, `all_reduce`                  |             fake impl 不能执行 | fake 只返回 empty tensor metadata            |

