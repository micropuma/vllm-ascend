# 方案C：DBO dispatch 提升到 compiled region 外部

> **核心原则**：黑盒只包 `get_forward_context → hook → 通信 → custom_op`，**不包 matmul / attention compute**。

---

## 一、边界定义

### 1.1 每个注入点的代码解剖

```
apply_impl(input_)
  │
  ├── bias = ...                                      ← Python 常量，无影响
  ├── need_all_gather = ...                           ← Python 常量，无影响
  │
  ├── ╔═══════════ DBO dispatch 边界 ═══════════╗
  │   ║ forward_context = get_forward_context()  ║   ← contextvar 读
  │   ║ if forward_context.dbo_enabled:          ║   ← 分支判断
  │   ║     hook(record)                         ║   ← NPU event
  │   ║     all_gather(input_, 0)                ║   ← HCCL 通信
  │   ║     hook(wait)                           ║   ← NPU event
  │   ║     custom_op(input_, do_comm=False)     ║   ← custom op (unpad only)
  │   ║ else:                                    ║
  │   ║     custom_op(input_, do_comm=True)      ║   ← custom op (通信+unpad)
  │   ╚══════════════════════════════════════════╝
  │
  ├── output_parallel = self.quant_method.apply(...)  ← 🔥 MATMUL (必须编译)
  ├── output = self.comm_group.all_gather(...)         ← 后续通信（非 DBO 管辖）
  └── return output
```

**DBO dispatch 段和 compute 段天然分离**：AllGather 发生在 matmul 之前，matmul 不知道也不关心 AllGather 是通过 DBO 路径还是非 DBO 路径完成的。

### 1.2 7 个注入点的边界一览

| # | 文件 | 注入点 | DBO dispatch 段（入黑盒） | Compute 段（留在编译区） |
|---|------|--------|--------------------------|------------------------|
| 1 | `linear_op.py:198` | `MLPColumnParallelOp` | `get_forward_context → hook → all_gather` | `quant_method.apply(matmul)` |
| 2 | `linear_op.py:454` | `SequenceColumnParallelOp` | `get_forward_context → hook → all_gather → maybe_all_gather_unpad` | `quant_method.apply(matmul)` |
| 3 | `linear_op.py:260` | `OProjRowParallelOp` | `hook → all_to_all → hook → reduce_scatter` | 纯通信 op，无 compute |
| 4 | `linear_op.py:543` | `SequenceRowParallelOp` | `get_forward_context → hook → all_reduce` | 前面 flashcomm matmul |
| 5 | `prepare_finalize.py:389` | `moeprepare_v2` | `hook → all_gather → maybe_all_gather_unpad(do_comm=False)` | 后续 router/gate 计算 |
| 6 | `prepare_finalize.py:548` | `moefinalize_v2` | `maybe_pad_and_reduce(do_comm=False) → hook → reduce_scatter` | MoE compute 已在前 |
| 7 | `mla_v1.py:1669` | `AscendMLAImpl._forward` | `hook → all_gather q/kv → maybe_all_gather_unpad(do_comm=False)` | 全部 attention compute |

**结论**：7 个注入点中，没有任何一个会在黑盒内丢失 matmul 或 attention compute。

---

## 二、目标架构

```
DeepseekV2Model.forward(@compiled)
  │
  ├── embedding(...)                         ← Dynamo trace ✅
  ├── layer.forward()                        ← Dynamo trace ✅
  │     ├── attn(...)                        ← Dynamo trace ✅
  │     ├── gate_up_proj.forward()
  │     │     └── _dbo_dispatch_col_allgather(input_)  ┌─────────────────────────┐
  │     │                                               │ @torch.compiler.disable │
  │     │                                               │   get_forward_context() │ ← Dynamo 跳过
  │     │                                               │   if dbo_enabled: ...   │
  │     │                                               │   custom_op(...)        │
  │     │                                               └─────────────────────────┘
  │     │     └── quant_method.apply(...)   ← Dynamo 从这里恢复 trace ✅
  │     ├── down_proj.forward()             ← 同理
  │     └── ...
  └── lm_head(...)                          ← Dynamo trace ✅
```

**compile 期**（`_dummy_run`）：Dynamo trace → 遇到 `_dbo_dispatch_*` → "@torch.compiler.disable，跳过黑盒" → 黑盒内部 eager 执行（`dbo_enabled=False`，走 else 分支）→ Dynamo 从黑盒后继续 trace compute → 正常编译。

**DBO runtime**：同一份编译图 replay → 黑盒内部 eager 执行（`dbo_enabled=True`，走 if 分支）→ hook + comm + `do_comm=False` → 后面的编译 compute 正常执行。

---

## 三、文件改动

### 3.1 新建 `dbo/dispatch.py`（替代 compile_guard.py）

```python
"""DBO dispatch functions — eager-only wrappers for DBO hook + comm injection.

Every function in this module is decorated with @torch.compiler.disable().
Dynamo treats each call site as an opaque box and never traces inside.
The compute (matmul / attention) happens AFTER these functions return,
in the compiled region.
"""

import torch
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context

from vllm_ascend.dbo.compile_guard import (
    _dbo_call_linear_column_hook,
    _dbo_call_linear_row_hook,
    _dbo_call_moe_prepare_hook,
    _dbo_call_moe_finalize_hook,
    _dbo_call_mla_preprocess_hook,
)


# ── Linear Column Parallel ──────────────────────────────────────

@torch.compiler.disable()
def dbo_dispatch_column_allgather(
    op, input_: torch.Tensor, need_all_gather: bool
) -> torch.Tensor:
    """DBO-aware AllGather for MLPColumnParallelOp / SequenceColumnParallelOp.

    Dispatches between DBO (hook + explicit comm + do_comm=False)
    and non-DBO (do_comm=True) paths.  The MATMUL happens *after* this
    function returns, inside the compiled region.
    """
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_linear_column_hook(fc, is_record=True)
        if fc.flash_comm_v1_enabled and need_all_gather:
            input_ = tensor_model_parallel_all_gather(input_, 0)
        _dbo_call_linear_column_hook(fc, is_record=False)
        return torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, do_comm=False, label=need_all_gather
        )
    else:
        return torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, label=need_all_gather
        )


# ── Linear Row Parallel ─────────────────────────────────────────

@torch.compiler.disable()
def dbo_dispatch_row_allreduce(
    op, output_parallel: torch.Tensor
) -> torch.Tensor:
    """DBO-aware AllReduce for SequenceRowParallelOp.matmul_and_reduce."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_linear_row_hook(fc, is_record=True)
        output = tensor_model_parallel_all_reduce(output_parallel)
        _dbo_call_linear_row_hook(fc, is_record=False)
        return output
    else:
        return tensor_model_parallel_all_reduce(output_parallel)


@torch.compiler.disable()
def dbo_dispatch_oproj_alltoall(
    op, input_parallel: torch.Tensor, send_buf, recv_buf, total_batch_size,
    chunk_size, output_parallel, output_shape
) -> torch.Tensor:
    """DBO-aware all_to_all + reduce_scatter for OProjRowParallelOp."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_linear_row_hook(fc, is_record=True)
    dist.all_to_all_single(recv_buf, send_buf, group=op.comm_group.device_group)
    input_parallel = recv_buf.view(total_batch_size, chunk_size)
    output_parallel = op.quant_method.apply(op.layer, input_parallel, None)
    output = op.comm_group.reduce_scatter(output_parallel, dim=0)
    if fc.dbo_enabled:
        _dbo_call_linear_row_hook(fc, is_record=False)
    output = output.view(output_shape)
    return output


# ── MoE Prepare / Finalize (AllGather path) ─────────────────────

@torch.compiler.disable()
def dbo_dispatch_moe_prepare_allgather(
    prepare_obj, hidden_states, router_logits, pertoken_scale
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """DBO-aware MoE prepare for PrepareAndFinalizeWithAllGather."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_moe_prepare_hook(fc, is_record=True)
        if fc.flash_comm_v1_enabled:
            if fc.dp_metadata is None:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                router_logits = tensor_model_parallel_all_gather(router_logits, 0)
                if pertoken_scale is not None:
                    pertoken_scale = tensor_model_parallel_all_gather(pertoken_scale, 0)
            else:
                from vllm.distributed.parallel_state import get_ep_group
                hidden_states = get_ep_group().all_gather(hidden_states, 0)
                router_logits = get_ep_group().all_gather(router_logits, 0)
                if pertoken_scale is not None:
                    pertoken_scale = get_ep_group().all_gather(pertoken_scale, 0)
            _dbo_call_moe_prepare_hook(fc, is_record=False)
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, True, True, do_comm=False
        )
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            router_logits, True, True, do_comm=False
        )
        if pertoken_scale is not None:
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                pertoken_scale, True, True, do_comm=False
            )
    else:
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            hidden_states, True, True
        )
        router_logits = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            router_logits, True, True
        )
        if pertoken_scale is not None:
            pertoken_scale = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                pertoken_scale, True, True
            )
    return hidden_states, router_logits, pertoken_scale


@torch.compiler.disable()
def dbo_dispatch_moe_finalize_allgather(
    prepare_obj, hidden_states: torch.Tensor
) -> torch.Tensor:
    """DBO-aware MoE finalize for PrepareAndFinalizeWithAllGather."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(
            hidden_states, True, do_comm=False
        )
        _dbo_call_moe_finalize_hook(fc, is_record=True)
        if fc.flash_comm_v1_enabled:
            if fc.dp_metadata is None:
                hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            else:
                from vllm.distributed.parallel_state import get_ep_group
                hidden_states = get_ep_group().reduce_scatter(hidden_states, 0)
        else:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        _dbo_call_moe_finalize_hook(fc, is_record=False)
        return hidden_states
    else:
        return torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True)


# ── MoE Token Dispatch (All2All path) ───────────────────────────

@torch.compiler.disable()
def dbo_dispatch_moe_dispatch_all2all(
    dispatcher, hidden_states, topk_ids, ...
):
    """DBO-aware MoE dispatch for TokenDispatcherWithAll2AllV (A3)."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_moe_prepare_hook(fc, is_record=True)
    # ... original dispatch logic ...
    if fc.dbo_enabled:
        _dbo_call_moe_prepare_hook(fc, is_record=False)
    return ...


@torch.compiler.disable()
def dbo_dispatch_moe_combine_all2all(
    dispatcher, hidden_states, combine_metadata, ...
):
    """DBO-aware MoE combine for TokenDispatcherWithAll2AllV (A3)."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_moe_finalize_hook(fc, is_record=True)
    # ... original combine logic ...
    if fc.dbo_enabled:
        _dbo_call_moe_finalize_hook(fc, is_record=False)
    return ...


# ── MLA Preprocess ──────────────────────────────────────────────

@torch.compiler.disable()
def dbo_dispatch_mla_preprocess(
    q_c: torch.Tensor,
    kv_no_split: torch.Tensor,
    need_gather_q_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DBO-aware MLA preprocess AllGather."""
    fc = get_forward_context()
    if fc.dbo_enabled:
        _dbo_call_mla_preprocess_hook(fc, is_record=True)
        if fc.flash_comm_v1_enabled:
            q_c = tensor_model_parallel_all_gather(q_c.contiguous(), 0)
            kv_no_split = tensor_model_parallel_all_gather(kv_no_split.contiguous(), 0)
        _dbo_call_mla_preprocess_hook(fc, is_record=False)
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c, need_gather_q_kv, do_comm=False
        )
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split, need_gather_q_kv, do_comm=False
        )
    else:
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c.contiguous(), need_gather_q_kv
        )
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split.contiguous(), need_gather_q_kv
        )
    return q_c, kv_no_split
```

### 3.2 修改 `ops/linear_op.py` — 回退 DBO 内联代码

**以 `SequenceColumnParallelOp` 为例**：

```python
# 改造后
from vllm_ascend.dbo.dispatch import dbo_dispatch_column_allgather

class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        need_all_gather = not (
            extract_layer_index(self.layer.prefix) == 0
            and is_vl_model() and "attn" in self.prefix
        )

        # 只有这一行：DBO dispatch + 通信（黑盒，Dynamo 不 trace）
        input_ = dbo_dispatch_column_allgather(self, input_, need_all_gather)

        # 下面全部是 compute，正常编译
        output_parallel = self.quant_method.apply(self.layer, input_, bias)
        if self.gather_output:
            output = self.comm_group.all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

**同样地改造 `MLPColumnParallelOp`**：

```python
class MLPColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None

        input_parallel = dbo_dispatch_column_allgather(self, input_, need_all_gather=True)

        output = self.quant_method.apply(self.layer, input_parallel, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

**`SequenceRowParallelOp.matmul_and_reduce`**：

```python
from vllm_ascend.dbo.dispatch import dbo_dispatch_row_allreduce

class SequenceRowParallelOp(CustomRowParallelOp):
    def matmul_and_reduce(self, input_parallel, bias_):
        ...
        if not flash_comm_v1_enabled:
            output_parallel = self.layer.quant_method.apply(
                self.layer, x, bias=bias_)
            return dbo_dispatch_row_allreduce(self, output_parallel)
        # ... flashcomm path unchanged ...
```

**`OProjRowParallelOp`**：

```python
from vllm_ascend.dbo.dispatch import dbo_dispatch_oproj_alltoall

class OProjRowParallelOp(CustomRowParallelOp):
    def apply_impl(self, input_):
        # ... prep ...
        return dbo_dispatch_oproj_alltoall(
            self, input_parallel, send_buf, recv_buf,
            total_batch_size, chunk_size, output_parallel, output_shape)
```

### 3.3 修改 `ops/fused_moe/prepare_finalize.py`

```python
from vllm_ascend.dbo.dispatch import (
    dbo_dispatch_moe_prepare_allgather,
    dbo_dispatch_moe_finalize_allgather,
)

class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
    def moeprepare_v2(self, hidden_states, router_logits, pertoken_scale=None):
        # ... conditions unchanged ...
        if self.multistream_overlap_gate:
            # ... unchanged ...
        else:
            hidden_states, router_logits, pertoken_scale = (
                dbo_dispatch_moe_prepare_allgather(
                    self, hidden_states, router_logits, pertoken_scale
                )
            )
        self.num_tokens = hidden_states.shape[0]
        # ... rest unchanged ...

    def moefinalize_v2(self, hidden_states):
        # ... pcp logic unchanged ...
        hidden_states = dbo_dispatch_moe_finalize_allgather(self, hidden_states)
        return hidden_states
```

### 3.4 修改 `ops/fused_moe/token_dispatcher.py`

```python
from vllm_ascend.dbo.dispatch import (
    dbo_dispatch_moe_dispatch_all2all,
    dbo_dispatch_moe_combine_all2all,
)

class TokenDispatcherWithAll2AllV(MoETokenDispatcher):
    def dispatch(self, hidden_states, topk_ids, ...):
        # ... preprocess unchanged ...
        result = dbo_dispatch_moe_dispatch_all2all(
            self, hidden_states, topk_ids, ...)
        # ... postprocess unchanged ...

    def combine(self, hidden_states, combine_metadata, ...):
        # ... preprocess unchanged ...
        result = dbo_dispatch_moe_combine_all2all(
            self, hidden_states, combine_metadata, ...)
        # ... postprocess unchanged ...
```

### 3.5 修改 `attention/mla_v1.py`

```python
from vllm_ascend.dbo.dispatch import dbo_dispatch_mla_preprocess

class AscendMLAImpl(MLAAttentionImpl):
    def _forward(self, hidden_states, ...):
        # ... kv_a_proj_with_mqa unchanged ...
        q_c, kv_no_split = dbo_dispatch_mla_preprocess(
            q_c, kv_no_split, need_gather_q_kv)
        # ... attention compute unchanged ...
```

### 3.6 修改 `worker/npu_ubatch_wrapper.py`

**无需 `_patch_model_for_dbo()`！** 

因为 DBO dispatch 函数已经在各 op 类的 `apply_impl()` 中直接调用，不需要 runtime patching。`compile_guard.py` 的 hook wrapper 被 `dispatch.py` 内部引用。

`AscendUBatchWrapper.__init__` 保持不变。

### 3.7 修改 `dbo/compile_guard.py`

保留 5 个 `_dbo_call_*_hook` wrapper（它们功能正确），被 `dispatch.py` 引用。不再独立暴露。

### 3.8 其余文件不变

| 文件 | 改动 |
|------|------|
| `ascend_forward_context.py` | 不变 |
| `worker/model_runner_v1.py` | 不变 |
| `ops/register_custom_ops.py` | 不变（`do_comm` + snapshot 已就绪） |
| `dbo/overlap_templates/*` | 不变 |
| `worker/ubatching.py` | 不变 |
| `worker/ubatch_utils.py` | 不变 |

---

## 四、文件改动汇总

| 文件 | 改动 | 行数 |
|------|------|------|
| `dbo/dispatch.py` | **新建** | +250 |
| `dbo/compile_guard.py` | 保留（被引用） | 0 |
| `ops/linear_op.py` | 回退 DBO 内联，改为调用 dispatch | -80 / +20 |
| `ops/fused_moe/prepare_finalize.py` | 同上 | -50 / +15 |
| `ops/fused_moe/token_dispatcher.py` | 同上 | -15 / +10 |
| `attention/mla_v1.py` | 同上 | -19 / +8 |

**总改动量**：~300 行新增，~160 行删除。无 monkey-patch，无模型遍历。

---

## 五、为什么这个方案能工作

### 5.1 `_dummy_run` 时

```
1. model_runner.profile_run()
2.   → _dummy_run()
3.     → _model_forward()
4.       → AscendUBatchWrapper.__call__()
5.         → self.runnable(*args) [模型 forward，尚未编译]
6.           → vllm @compiled decorator 触发 Dynamo fullgraph trace
7.             → layer.forward()
8.               → gate_up_proj.forward()
9.                 → custom_op.apply(input_)
10.                  → SequenceColumnParallelOp.apply_impl()
11.                    → dbo_dispatch_column_allgather(input_)  ← @torch.compiler.disable
12.                      [Dynamo: "跳过黑盒"]
13.                      [Eager 执行: fc = get_forward_context()
14.                       fc.dbo_enabled = False (set_ascend_forward_context 设置的)
15.                       → 走 else: torch.ops.vllm.maybe_all_gather_and_maybe_unpad(do_comm=True)
16.                       → 正常通信，返回 tensor]
17.                    [Dynamo: 从黑盒后恢复 trace]
18.                    → quant_method.apply(matmul)  ← 编译 ✅
19.            → layer 2, 3, ...
```

**关键**：
- 第 11 行：Dynamo 看到 `dbo_dispatch_column_allgather` → `@torch.compiler.disable` → 跳过
- 第 13-16 行：在 eager 模式执行，`get_forward_context()` 正常工作（此时有 forward context）
- 第 18 行：matmul 正常编译

### 5.2 DBO runtime 时

```
1. AscendUBatchWrapper._run_ubatches()
2.   → 子线程: with ubatch_context:
3.       → model.forward()  [compiled graph replay]
4.         → gate_up_proj.forward()
5.           → custom_op.apply(input_)
6.             → SequenceColumnParallelOp.apply_impl()
7.               → dbo_dispatch_column_allgather(input_)  ← @torch.compiler.disable
8.                 [compiled graph: "call this eager function"]
9.                 [Eager 执行: fc = get_forward_context()
10.                  fc.dbo_enabled = True (create_ascend_forward_context 设置的)
11.                  → 走 if: hook(record) → all_gather → hook(wait) → custom_op(do_comm=False)
12.                  → DBO overlap 正常触发 ✅]
13.               → quant_method.apply(matmul)  ← compiled graph replay ✅
```

**关键**：
- 编译图 replay 到 `dbo_dispatch_column_allgather` 调用点时，执行这个 eager 函数
- eager 函数内部读到的 `forward_context.dbo_enabled = True`（子线程 context）
- DBO hook + 通信正常执行
- 返回后继续 replay 编译好的 matmul

---

## 六、与 flashcomm-dbo-compile-bugs.md 中 8 个 bug 的对照

| Bug | 方案C 是否解决 | 说明 |
|-----|--------------|------|
| Bug 1: snapshot 未设置 | ✅ 无关 | snapshot 在编译期由 `set_flash_comm_v1_snapshot` 设置，fake impl 读 snapshot 而非 context |
| Bug 2: matmul_and_reduce fake 读 context | ✅ 已解决 | fake impl 读 snapshot |
| Bug 3: all_reduce fake 读 context | ✅ 已解决 | 同上 |
| **Bug 4: DBO 分支被剪枝** | ✅ **根除** | DBO 分支在 `@torch.compiler.disable` 黑盒内，永不进入 FX graph |
| Bug 5: pad_size 不一致 | ⚠️ 独立问题 | 与 compile 无关，需单独修复 |
| Bug 6: matmul_and_reduce 路径 | ✅ 已解决 | dispatch 函数接管 |
| Bug 7: _EXTRA_CTX 编译期读 | ✅ 已解决 | 所有 context 读取在黑盒内 |
| Bug 8: threading.get_ident 被 trace | ✅ 已解决 | 黑盒内操作不被 trace |

---

## 七、风险评估

| 风险 | 概率 | 缓解 |
|------|------|------|
| PIECEWISE 下 `torch.ops.vllm.*` 在黑盒内导致 splitting 边界偏移 | 低 | custom op 在 eager 模式仍走 ACL Graph capture，通信本身不受编译影响 |
| 黑盒内 `torch.ops.vllm.*` 的 eager 调用比 compiled 调用慢 | 极低 | 通信是 I/O bound，HCCL 延迟主导，compile vs eager 无差异 |
| 新 dispatch 函数签名与 op 内部状态耦合 | 中 | 显式传入 `op` 引用访问 `comm_group` 等属性 |
