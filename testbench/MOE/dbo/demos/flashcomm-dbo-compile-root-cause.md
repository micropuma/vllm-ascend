# FlashComm1 + DBO + torch.compile 不兼容根因分析

> 基于 `bug.log` 的实际崩溃调用栈，结合 `bug/flashcomm-dbo-compile-bugs.md` 中的已知 bug 模式进行深入分析

---

## 一、崩溃现场还原

```
错误: torch._dynamo.exc.Unsupported: failed to find name in frame builtins
      Explanation: Failed to find name `get_forward_context` in frame's builtins.
```

**完整调用链**（从用户代码到崩溃点）：

```
profile_run()                                              # model_runner_v1.py:3594
  → _dummy_run()                                           # model_runner_v1.py:3531
    → _model_forward()                                     # model_runner_v1.py:2753
      → run_model()                                        # npu_ubatch_wrapper.py:366
        → self.runnable(*args, **kwargs)                   # npu_ubatch_wrapper.py:366
          → DeepseekV2ForCausalLM.forward()                # deepseek_v2.py:1692
            → DeepseekV2Model.forward()                    # deepseek_v2.py:1315
              → layer(hidden_states)                       # deepseek_v2.py:1315
                → DeepseekV2DecoderLayer.forward()         # deepseek_v2.py:1191
                  → self.mlp(hidden_states)                 # DeepseekV2MLP.forward()
                    → self.gate_up_proj(x)                  # deepseek_v2.py:239
                      → custom_op.apply(input_)             # linear_op.py:124
                        → SequenceColumnParallelOp.apply_impl()  # linear_op.py:454
                          → get_forward_context()           # ← 💥 CRASH
```

## 二、根因：`get_forward_context()` 在 Dynamo fullgraph 中被 trace

### 2.1 触发条件

这是一个 **三重条件同时满足** 的 bug：

| 条件 | 如何满足 |
|------|---------|
| `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | 脚本明确开启 |
| `--enable-dbo` | 脚本明确开启 |
| vllm 的 torch.compile fullgraph capture | `profile_run()` 时由 vllm 的 `@compiled` decorator 触发 |

**缺一不可**：

- 关闭 FlashComm1 → `enable_sp()` 返回 `False` → 不使用 `SequenceColumnParallelOp` → 不走 `get_forward_context()` 代码路径 → OK
- 关闭 DBO → DBO 迁移代码不会添加这些 `get_forward_context()` 调用 → OK
- 不使用 compile → Dynamo 不会 trace → OK

### 2.2 精确的因果链

**第 1 步：FlashComm1=1 → `enable_sp()=True`**

```python
# utils.py:864
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    ...
    _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1  # = 1 → True
```

**第 2 步：`enable_sp()=True` → DeepSeekV2 MLP 层使用 `SequenceColumnParallelOp`**

```python
# linear_op.py:667 _get_column_parallel_op()
def _get_column_parallel_op(prefix, layer):
    ...
    # line 670: 条件跳过（mlp_tp_enable() → False，因为没开启 finegrained TP）
    # line 675: enable_sp() → True!
    if enable_sp():
        if "shared_expert" in prefix:  # 这是 dense MLP，不是 shared expert
            return None
        sp_column_prefix = ["gate_up_proj", ...]  # ← gate_up_proj 命中!
        for a_prefix in sp_column_prefix:
            if a_prefix in prefix:
                return SequenceColumnParallelOp(layer)  # ← 返回 SequenceColumnParallelOp
```

**第 3 步：DBO 迁移在 `SequenceColumnParallelOp.apply_impl()` 中添加了无条件 `get_forward_context()`**

```python
# linear_op.py:440-464 (迁移后)
def apply_impl(self, input_):
    ...
    # dbo overlap for qwen3 moe with flashcomm1
    forward_context = get_forward_context()  # ← 无条件调用！在 if dbo_enabled 之前！
    if forward_context.dbo_enabled:
        _dbo_call_linear_column_hook(forward_context, is_record=True)
        ...
    else:
        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(...)
```

**第 4 步：Dynamo fullgraph capture 时无法 trace `get_forward_context()`**

`get_forward_context()` 内部使用 `contextvars` 读取当前 forward context：

```python
# vllm/vllm/forward_context.py
_forward_context: contextvars.ContextVar = ...

def get_forward_context() -> ForwardContext:
    return _forward_context.get()  # ← contextvar lookup
```

Dynamo 在 fullgraph 模式下无法 trace `contextvars.ContextVar.get()`。它把这个函数当作"未解析的 builtin"，抛出 `Unsupported: failed to find name in frame builtins`。

## 三、为什么现有的 `compile_guard.py` 不够？

迁移中新增的 `dbo/compile_guard.py` 只保护了 **hook 调用本身**：

```python
# dbo/compile_guard.py — 有保护 ✅
@torch.compiler.disable()
def _dbo_call_linear_column_hook(forward_context, is_record):
    forward_context.dbo_template.dbo_linear_column_hook(is_record=is_record)
```

但 `get_forward_context()` 在 **hook 调用之前** 就被执行了，而且 **不在 `@torch.compiler.disable()` 的保护范围内**：

```python
# linear_op.py:454 — 无保护 ❌
forward_context = get_forward_context()          # ← 第 454 行，无条件执行，无 compile 保护
if forward_context.dbo_enabled:                  # ← 第 455 行，分支条件在 compile region 内
    _dbo_call_linear_column_hook(...)            # ← 第 456 行，hook 本身有 compile_guard 保护
    ...
```

**受影响的代码位置不止一处**（所有 DBO 迁移中新增 `get_forward_context()` 的地方）：

| 文件 | 行号 | Op 类 | 触发条件 |
|------|------|-------|---------|
| `linear_op.py` | 198 | `MLPColumnParallelOp` | `mlp_tp_enable()=True` 时 |
| `linear_op.py` | 454 | `SequenceColumnParallelOp` | `enable_sp()=True` 时（**当前命中**） |
| `linear_op.py` | 543 | `SequenceRowParallelOp.matmul_and_reduce` | `enable_sp()=True` + FlashComm1=OFF 时 |
| `prepare_finalize.py` | 389 | `PrepareAndFinalizeWithAllGather` | DBO 子线程中 |
| `token_dispatcher.py` | 490 | `TokenDispatcherWithAll2AllV` | DBO 子线程中 + A3 模式 |
| `mla_v1.py` | 1669 | `AscendMLAImpl._forward` | DBO 子线程中 |

> 注意：`prepare_finalize.py`、`token_dispatcher.py`、`mla_v1.py` 中的 `get_forward_context()` 只在 `dbo_enabled=True` 时执行（运行时 DBO 子线程），compile 期 `dbo_enabled=False` → 走 else 分支，所以它们实际上在 compile 期不会被 trace。**但 `linear_op.py` 中的 3 处调用是无条件执行的**，无论 DBO 是否启用。

## 四、为什么看起来"只"是 FlashComm1 的问题？

对比两种配置下的代码路径：

### FlashComm1=1 → 崩溃

```
enable_sp()=True
  → gate_up_proj 使用 SequenceColumnParallelOp
    → apply_impl() 第 454 行: forward_context = get_forward_context()
      → Dynamo trace → 💥 Unsupported
```

### FlashComm1=0 → 当前不崩溃（但隐患存在）

```
enable_sp()=False
  → gate_up_proj 不使用任何 custom op (返回 None)
    → 走标准 vllm ColumnParallelLinear.forward()
      → 无 get_forward_context() 调用
        → Dynamo trace ✅
```

**但这只是运气好。** 如果后续有人开启了 `mlp_tp_enable()`（finegrained TP），即使 FlashComm1=0，`MLPColumnParallelOp` 也会被使用，那个类在第 198 行有同样的 `get_forward_context()` 调用，会触发同样的崩溃。

## 五、修复建议

### 方案 A（推荐）：将 `get_forward_context()` 也纳入 compile_guard

```python
# linear_op.py:440-464
@torch.compiler.disable()  # ← 整个方法禁用 compile
def _get_dbo_forward_context_and_maybe_all_gather(input_, need_all_gather):
    forward_context = get_forward_context()
    if forward_context.dbo_enabled:
        _dbo_call_linear_column_hook(forward_context, is_record=True)
        if get_forward_context().flash_comm_v1_enabled and need_all_gather:
            input_ = tensor_model_parallel_all_gather(input_, 0)
        _dbo_call_linear_column_hook(forward_context, is_record=False)
        return torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, do_comm=False, label=need_all_gather)
    else:
        return torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, label=need_all_gather)

class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_):
        ...
        need_all_gather = not (...)
        input_ = _get_dbo_forward_context_and_maybe_all_gather(input_, need_all_gather)
        ...
```

但这有一个副作用：整个 all_gather 逻辑都会被排除在 compile 之外，可能损失部分优化。

### 方案 B：使用 `torch._dynamo.graph_break()` 或 `torch._dynamo.assume_constant_result()`

```python
forward_context = torch._dynamo.graph_break()(get_forward_context)()
```

但这在 fullgraph 模式下不可用（`graph_break` 需要允许 graph break）。

### 方案 C（根本解决）：将 DBO hook 整个逻辑提升到 compiled region 外部

在 `AscendUBatchWrapper._run_ubatches()` 层（compiled region 之前）完成所有 DBO 状态的读取和 hook 调度，让 compiled graph 内部完全看不到 `get_forward_context()` 调用。这需要较大的架构重构。

### 当前 workaround

```bash
# 在脚本中已注释：
# TODO(leon)：目前flashcomm1/flashcomm2在dbo下和torch compile不兼容，先默认关闭
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
```
