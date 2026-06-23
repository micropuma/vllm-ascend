# FlashComm + DBO + torch.compile 冲突根因分析

## 执行摘要

当前脚本 `deepseek-v2-dbo-server.sh` 显式启用了 FlashComm1/FlashComm2（`VLLM_ASCEND_ENABLE_FLASHCOMM1=1`），但同时启用了 DBO + torch.compile（PIECEWISE 模式，从 `platform.py:379` 可知 `FULL_AND_PIECEWISE` 被降级为 `PIECEWISE`）。

FlashComm 与 torch.compile 冲突的根因可以归结为三类问题，按严重程度排序：

1. **fake impl 与 runtime impl 的 shape 不一致**：compile 阶段的 fake tensor 推断出的 shape 与 runtime 真实通信后的 shape 不匹配。
2. **DBO 分支被 Dynamo 内联/剪枝**：compile 阶段 `dbo_enabled=False`，Dynamo 可能把 DBO hook 分支剪掉；runtime 子线程 `dbo_enabled=True` 时触发 recompile 或 DBO hook 被跳过。
3. **compile 阶段读取无 forward_context 的动态状态**：custom op 的 fake impl 通过 `_EXTRA_CTX` / `get_forward_context()` 读取 runtime 状态，但 piecewise compile 发生在没有任何 forward context 的环境中。

---

## 根因分析：为什么只有 FlashComm 开启才出问题？

这是本文最关键的问题。答案需要从**两条代码路径的差异**来理解。

### 触发链

```
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
  → enable_sp() == True                                  # utils.py:929
    → flash_comm_v1_enabled = True                       # ascend_forward_context.py:124/131
      → 选择 SequenceColumnParallelOp / SequenceRowParallelOp  # linear_op.py:672/711
        → 使用 torch.ops.vllm.maybe_all_gather_and_maybe_unpad   # 自定义 op
        → 使用 torch.ops.vllm.maybe_pad_and_reduce               # 自定义 op
          → fake impl 读 _FLASH_COMM_V1_SNAPSHOT          # register_custom_ops.py:138/147
            → 永远是 False（从未被设置）                    # Bug 1
              → fake shape = 原始 shape（identity）
                → runtime shape ≠ fake shape → 💥
```

### 对比：FlashComm OFF vs ON

#### 第一条岔路：Op 类选择 (`linear_op.py:662-724`)

`_get_column_parallel_op()` 和 `_get_row_parallel_op()` 根据 `enable_sp()` 选择不同的 op 类：

| 层类型 | FlashComm OFF (`enable_sp()=False`) | FlashComm ON (`enable_sp()=True`) |
|--------|-------------------------------------|-----------------------------------|
| Column (gate_up_proj, qkv_proj) | `MLPColumnParallelOp` | `SequenceColumnParallelOp` |
| Row (down_proj, o_proj) | `MLPRowParallelOp` / `MatmulAllreduceRowParallelOp` | `SequenceRowParallelOp` |

#### 第二条岔路：通信实现方式

**FlashComm OFF — `MLPColumnParallelOp.apply_impl` (line 191-208):**

```python
def apply_impl(self, input_):
    forward_context = get_forward_context()
    if forward_context.dbo_enabled:
        # DBO hook + 标准 PyTorch all_gather
        forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
        input_parallel = self.comm_group.all_gather(input_, 0)   # ← torch.distributed
        forward_context.dbo_template.dbo_linear_column_hook(is_record=False)
    else:
        input_parallel = self.comm_group.all_gather(input_, 0)   # ← torch.distributed
    output = self.quant_method.apply(self.layer, input_parallel, bias)
```

- 通信使用 `self.comm_group.all_gather()` — **标准 PyTorch distributed op**
- Dynamo 原生理解 `all_gather` → FakeTensor 自动正确传播 shape：`[N, H] → [N*TP, H]`
- **不需要 fake impl** → 没有 fake/real 不一致的问题

**FlashComm OFF — `MLPRowParallelOp.apply_impl` (line 211-228):**

```python
def apply_impl(self, input_):
    # ...
    output = self.comm_group.reduce_scatter(output_parallel, 0)  # ← torch.distributed
```

- 同样使用标准 PyTorch distributed op
- Dynamo 原生理解 `reduce_scatter` → shape 自动正确

**FlashComm ON — `SequenceColumnParallelOp.apply_impl` (line 439-474):**

```python
def apply_impl(self, input_):
    forward_context = get_forward_context()
    if forward_context.dbo_enabled:
        forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
        if flash_comm_v1_enabled and need_all_gather:
            input_ = tensor_model_parallel_all_gather(input_, 0)  # 显式 AG
        forward_context.dbo_template.dbo_linear_column_hook(is_record=False)
        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, do_comm=False, ...)   # ← 自定义 op (仅 unpad)
    else:
        input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            input_, label=need_all_gather) # ← 自定义 op (AG + unpad)
```

- 通信使用 `torch.ops.vllm.maybe_all_gather_and_maybe_unpad` — **自定义 op**
- Dynamo **不理解**这个 op → **依赖 fake impl 推断 shape**
- Fake impl 检查 `_FLASH_COMM_V1_SNAPSHOT` → 永远是 `False` → 返回原 shape

**FlashComm ON — `SequenceRowParallelOp.apply_impl` (line 510-527):**

```python
def apply_impl(self, input_):
    # ...
    if self.tp_size == 1 or not self.reduce_results:
        output = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
    else:
        output = torch.ops.vllm.matmul_and_reduce(
            input_parallel, self.unique_prefix)  # ← 自定义 op
```

- 使用 `torch.ops.vllm.matmul_and_reduce` — **自定义 op**
- Fake impl 读 `get_forward_context()` 和 `_EXTRA_CTX.flash_comm_v1_enabled`（Bug 2）

### 根因本质：条件性 shape 变换

两个关键自定义 op 的行为取决于 `flash_comm_v1_enabled`：

```
maybe_all_gather_and_maybe_unpad:
  flash_comm_v1_enabled=False → identity（shape 不变）
  flash_comm_v1_enabled=True  → AllGather（×TP） + unpad（−pad）→ shape 改变！

maybe_pad_and_reduce:
  flash_comm_v1_enabled=False → AllReduce（shape 不变）
  flash_comm_v1_enabled=True  → Pad（+pad） + ReduceScatter（÷TP）→ shape 改变！
```

**FlashComm OFF 时**：real impl 走 identity/AllReduce 路径，shape 不变；fake impl 也返回 identity → **恰好一致**。

**FlashComm ON 时**：real impl 走 AG+unpad / pad+RS 路径，shape 改变；fake impl 仍返回 identity → **不一致**。

这就是整个问题的根因：**同样的自定义 op，在不同配置下有不同的 shape 语义，而 fake impl 缺乏正确的配置感知**。

### 具体 shape 差异示例

假设 `N=2048, TP=2`，`pad_size = (2 - 2048%2) % 2 = 0`：

| 操作 | FlashComm OFF (real+ fake 一致) | FlashComm ON (real) | FlashComm ON (fake) |
|------|-------------------------------|---------------------|---------------------|
| `maybe_all_gather_and_maybe_unpad` | `[2048, H]` → `[2048, H]` | `[2048, H]` → `[4096, H]` | `[2048, H]` → `[2048, H]` ❌ |
| `maybe_pad_and_reduce` | `[4096, H]` → `[4096, H]` | `[4096, H]` → `[2048, H]` | `[4096, H]` → `[4096, H]` ❌ |

如果 `N=2049, TP=2`，`pad_size = (2 - 2049%2) % 2 = 1`：

| 操作 | FlashComm ON (real) | FlashComm ON (fake) |
|------|---------------------|---------------------|
| `maybe_all_gather_and_maybe_unpad` | `[2049, H]` → AG → `[4098, H]` → unpad(1) → `[4097, H]` | `[2049, H]` → `[2049, H]` ❌ |
| `maybe_pad_and_reduce` | `[4097, H]` → pad(1) → `[4098, H]` → RS → `[2049, H]` | `[4097, H]` → `[4097, H]` ❌ |

fake 和 real 的 shape 差了将近一倍，compile 时推导出的所有下游 tensor shape 全部错误。

### 为什么 DBO 让它更糟

即使没有 DBO，flashcomm + compile 单独组合也存在 fake impl 问题（Bug 1、2）。但 DBO 叠加了两个额外问题：

1. **编译期 vs 运行期分支不同**：编译时 `dbo_enabled=False`，Dynamo trace 到非 DBO 分支；运行时 DBO 子线程 `dbo_enabled=True`，但 compiled graph 里是非 DBO 的代码 → 要么 guard fail 触发 recompile，要么 DBO hook 被跳过导致 overlap 失效（Bug 4）。

2. **`pad_size` 在 ubatch 上下文中不同**：外层 context 的 `pad_size` 基于总 token 数计算（如 4096），ubatch context 的 `pad_size` 基于 ubatch token 数重新计算（如 2048）。如果 Dynamo 成功为 DBO 路径重新编译，编译时的 pad_size（基于 DBO=False 时的 token 数）可能和运行时子线程的 pad_size 不一致（Bug 5）。

### 补充纠正：FlashComm OFF 时自定义 op 也会被调用

自定义 op（`maybe_all_gather_and_maybe_unpad`、`maybe_pad_and_reduce`）有 **40+ 个 call site** 分布在 9 个文件中。并非所有 call site 都被 `enable_sp()` 守卫：

| Call site 分类 | 文件 | FlashComm OFF 时是否调用 | 为何正常 |
|---------------|------|------------------------|---------|
| MoE topk (AllGatherCommImpl) | `fused_moe.py:633-634` | ✅ **会调用** | real: identity / fake: identity → 一致 |
| MLA attention | `mla_v1.py:1605-1692` | ✅ 会调用 | real: identity / fake: identity → 一致 |
| DSA attention | `dsa_v1.py:1402` | ✅ 会调用 | 同上 |
| Rotary embedding | `rotary_embedding.py:249` | ❌ 被 `if flash_comm_v1_enabled:` 守卫 | 同上 |
| Eagle proposer | `eagle_proposer.py:*` | ✅ 会调用 | 同上 |
| Vocab embedding | `vocab_parallel_embedding.py:204` | ✅ 会调用 | 同上 |
| Graph fusion passes | `norm_quant_fusion_pass.py:*` | ✅ 会调用 | 同上 |
| MoE prepare/finalize (EP group) | `prepare_finalize.py:388-542` | ❌ 被 `enable_sp()` 守卫 | 不调用 |
| Linear (Sequence*ParallelOp) | `linear_op.py:462-490` | ❌ 被 `enable_sp()` 守卫 | 不调用 |

**结论**：自定义 op **在 flashcomm OFF 时也会被调用**，但因为 `flash_comm_v1_enabled=False`，real impl 走 identity 路径。Fake impl 的 `_FLASH_COMM_V1_SNAPSHOT` 也是 `False`，也走 identity 路径。**两者恰好一致，所以不报错。**

问题的本质不是"flashcomm 引入了自定义 op"，而是"flashcomm 让自定义 op 从 identity 变为 shape-transforming，而 fake impl 没有同步感知到这一变化"。

### 一句话总结

> 两个自定义 op `maybe_all_gather_and_maybe_unpad` / `maybe_pad_and_reduce` 是"**条件性 shape 变换**"的——`flash_comm_v1_enabled=False` 时是 identity，`=True` 时改变 shape。**无论 flashcomm 开关，这些 op 都可能被调用**。不开启时 real 和 fake 都是 identity → 一致；开启时 real 变换 shape 但 fake 仍是 identity → 不一致。Fake 通过 `_FLASH_COMM_V1_SNAPSHOT` 感知配置，但这个 snapshot **从未被设置**。

---

## 深层设计分析：为什么标准 op 天然兼容 compile 而自定义 op 不兼容

前面的分析解释了"出了什么 bug"，这一节解释"为什么设计上会出这些 bug"。核心问题是：**标准 PyTorch op 和 Ascend 自定义 op 在 Dynamo 面前的待遇完全不同。**

### 标准 op 的 fake tensor 机制

以 `all_gather` 为例。当 Dynamo 在 fake tensor 模式下遇到：

```python
input_parallel = self.comm_group.all_gather(input_, 0)
```

PyTorch 的 dispatch 流程是：

```
FakeTensor mode
  → dispatch to "Meta" key
  → TORCH_META_FUNC(all_gather)(input_tensor, world_size, dim)
  → output_shape = input.shape; output_shape[dim] *= world_size
  → return empty_strided(output_shape, ...)
```

标准 op 的 meta kernel 是**写在 PyTorch C++ 核心里的**，不是 vllm-ascend 手写的。它具备三个关键属性：

| 属性 | 说明 |
|------|------|
| **无条件性** | 同样的输入 shape 永远产生同样的输出 shape，不存在 `if flashcomm_enabled` 分支 |
| **纯函数** | 不读 `_EXTRA_CTX`、不读 `get_forward_context()`、不读任何全局状态 |
| **普适性** | 由 PyTorch 团队维护，所有后端（CUDA/ROCm/NPU）共用同一套 meta kernel |

所以 Dynamo trace 到标准 op 时，FakeTensor 自动获得正确的输出 shape。**不需要任何人写 fake impl。**

### Ascend 自定义 op 的 fake tensor 机制

当 Dynamo 在 fake tensor 模式下遇到：

```python
input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, label=True)
```

dispatch 流程是：

```
FakeTensor mode
  → dispatch to "PrivateUse1" key
  → 没有 Meta key 注册（自定义 op）
  → 调用 hand-written fake_impl 函数
  → _maybe_all_gather_and_maybe_unpad_fake(x, label, ...)
  → if _FLASH_COMM_V1_SNAPSHOT and label and do_comm:  ← 手写的条件判断
        return shape_transformed
      return x
```

自定义 op 没有 PyTorch 内置的 meta kernel，**必须由 vllm-ascend 提供手写的 fake impl**。而这个 fake impl：

| 属性 | 说明 |
|------|------|
| **有条件性** | 行为取决于 `_FLASH_COMM_V1_SNAPSHOT`、`label`、`do_comm` 等多个 flag |
| **依赖全局状态** | 虽然已去掉了 `_EXTRA_CTX` 直接读取（是你修的），但 snapshot 仍然是一个全局变量 |
| **易出错** | snapshot 忘了设置 → 永远返回 identity；条件组合遗漏 → shape 错误 |

### 为什么自定义 op 需要"条件性"

这是最核心的设计问题。逐层解开。

#### 第一层：为什么需要这个自定义 op？

FlashComm（sequence parallelism）相比标准 TP，多了一个 **pad/unpad** 步骤：

```
标准 TP (FlashComm OFF):
  AllGather → [N, H] → [N*TP, H]       # 一步完成，没有 padding

FlashComm TP (FlashComm ON):
  pad → AllGather → unpad               # 三步：先对齐、再通信、再去掉对齐
  [N, H] → [N+pad, H] → [(N+pad)*TP, H] → [N*TP, H]
```

为了封装这个 pad→AllGather→unpad 的逻辑，创建了 `maybe_all_gather_and_maybe_unpad` 自定义 op。它做了**两件事**：通信（AllGather）+ 数据变换（unpad）。

#### 第二层：DBO 如何改变需求

DBO 需要在**通信前后**插入 hook（record event / wait event / yield）。如果通信和数据变换在同一个 op 里，hook 就无法插在"通信之后、数据变换之前"：

```
非 DBO（op 内部一把梭）:
  ┌─────────────────────────────────────────┐
  │ maybe_all_gather_and_maybe_unpad        │
  │   Step 1: AllGather (通信)              │  ← hook 无法介入
  │   Step 2: unpad (数据变换)              │
  └─────────────────────────────────────────┘

DBO 需要的（拆开，hook 介入）:
  ┌──────────────────┐
  │ AllGather (通信) │  ← 显式调用标准 all_gather
  └──────────────────┘
  hook(record)                          ← hook 夹在中间
  hook(wait, yield)
  ┌──────────────────┐
  │ unpad (数据变换) │  ← 自定义 op，do_comm=False（只 unpad）
  └──────────────────┘
```

这就催生了 `do_comm` 参数：
- `do_comm=True`：通信 + unpad（非 DBO，一步做完）
- `do_comm=False`：仅 unpad（DBO，通信已在外部完成）

#### 第三层：两路径对比

把标准路径和 FlashComm 路径的 DBO 处理并排比较：

```
                标准路径 (MLPColumnParallelOp)         FlashComm路径 (SequenceColumnParallelOp)
                ─────────────────────────────         ──────────────────────────────────────

DBO:            hook(record)                          hook(record)
                all_gather(...)       ← 标准op         all_gather(...)           ← 标准op（通信）
                hook(wait, yield)                      hook(wait, yield)
                                                       maybe_ag_and_unpad(      ← 自定义op
                                                         do_comm=False)           仅 unpad，不通信

非DBO:          all_gather(...)       ← 同一个标准op    maybe_ag_and_unpad(      ← 同一个自定义op
                                                         label=True)              通信+unpad 一起
```

**标准路径**：两个分支做的是**完全一样的事**——都是调用标准 `all_gather`。DBO 只是给这同一个调用包了一层 hook 包装纸。`all_gather` 永远是 `all_gather`，不需要条件性。Dynamo 看到的是同一个标准 op 在两个分支里，meta kernel 一次性给出正确 shape。

**FlashComm 路径**：DBO 分支把"通信+unpad"拆成两步（显式 AG + 自定义 op with `do_comm=False`），非 DBO 分支一步做完（自定义 op with `do_comm=True`）。**同一个自定义 op，DBO 和非 DBO 调用时参数不同、语义不同**。这就是条件性的来源。

#### 为什么标准路径不需要 pad/unpad

标准 all_gather 不对 token 数做对齐要求。TP group 内部天然处理任意 token 数的数据。FlashComm 为了实现 sequence parallelism 的优化（融合通信与计算），需要 token 数对齐到 TP world size，所以引入 pad/unpad。

**一句话**：标准路径的通信 op 是"纯通信"（无数据变换），不需要条件性。FlashComm 的自定义 op 是"通信 + 数据变换"的组合体，DBO 要求把它拆开 → 同一个 op 需要两种运行模式 → 条件性 → fake impl 复杂度爆炸。

#### 这条设计链的全景

```
FlashComm 需要 pad/unpad（对齐 TP world size）
  → 创建 maybe_all_gather_and_maybe_unpad 封装 pad+AG+unpad
    → op 内部包含"通信"和"数据变换"两个步骤
      → DBO 需要在通信前后插入 hook
        → 必须把通信从 op 中拆出，只留 unpad 在 op 内
          → 引入 do_comm 参数控制 op 的行为模式
            → op 行为变为条件性的
              → fake impl 也必须条件性
                → 条件依赖 _FLASH_COMM_V1_SNAPSHOT
                  → snapshot 从未被设置 → 💥
```

---

## 系统性修复方案

针对上述 8 个 bug 的根因——**自定义 op 的行为取决于 runtime context，而 fake impl 在 compile 阶段无法访问正确的 context**——需要从三个层面系统性修复。

### 方案总览

```
                现状                          目标
        ┌─────────────────┐          ┌─────────────────┐
        │ 自定义 op 内部读    │          │ op 行为由显式参数   │
        │ _EXTRA_CTX 决定   │   →      │ 决定，不读 context │
        │ identity / 变换   │          │                  │
        └─────────────────┘          └─────────────────┘

        ┌─────────────────┐          ┌─────────────────┐
        │ DBO hook 混在      │          │ DBO hook 在        │
        │ compiled region  │   →      │ compiled region   │
        │ 内部             │          │ 外部              │
        └─────────────────┘          └─────────────────┘

        ┌─────────────────┐          ┌─────────────────┐
        │ fake impl 读      │          │ fake impl 纯      │
        │ forward_context  │   →      │ shape 推断        │
        └─────────────────┘          └─────────────────┘
```

### 第一层：修复 fake impl（本周可完成）

**原则：fake impl 不读任何 runtime 状态。**

当前两个 fake impl 的问题是：shape 变换与否取决于 `_FLASH_COMM_V1_SNAPSHOT`（全局变量，未设置）。

修复方式：**把 `flash_comm_enabled` 做成自定义 op 的显式参数**，fake impl 从参数直接读取。

```python
# 改造前（现状）：
def _maybe_all_gather_and_maybe_unpad_fake(x, label, is_ep_comm=False, do_comm=True):
    if (_FLASH_COMM_V1_SNAPSHOT or ...) and label and do_comm:
        return shape_transformed
    return x  # identity

# 改造后：
def _maybe_all_gather_and_maybe_unpad_fake(x, label, is_ep_comm=False, do_comm=True, 
                                            flash_comm_enabled=False, tp_size=1):
    if flash_comm_enabled and label and do_comm:
        return torch.empty(
            (x.shape[0] * tp_size, *x.shape[1:]), device=x.device, dtype=x.dtype)
    return x
```

**所有 call site 需要同步修改**，把 `flash_comm_v1_enabled` 和 `tp_size` 作为显式参数传入：

```python
# 改造前：
hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, True, True)

# 改造后：
hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
    hidden_states, True, True, 
    flash_comm_enabled=_EXTRA_CTX.flash_comm_v1_enabled,
    tp_size=get_tensor_model_parallel_world_size()
)
```

**优点**：
- fake impl 不需要任何全局状态，直接从参数推导 shape
- Dynamo 会为 `flash_comm_enabled` 参数生成 guard → 正确追踪 shape 变化
- 不存在 snapshot 忘记设置的问题

**缺点**：
- 需要修改 40+ 个 call site 的签名
- `flash_comm_enabled` 的取值仍然来自 `_EXTRA_CTX`（但在 caller 侧，不是在 fake impl 内，Dynamo 可以正确处理）

**注意**：`pad_size` 比 `flash_comm_enabled` 更棘手——它根据 `num_tokens` 动态计算。需要额外处理：
- 方案 A：把 `pad_size` 也作为显式参数传入 fake impl
- 方案 B：fake impl 忽略 padding（只按 `×TP` / `÷TP` 计算 shape）——因为 padding 通常为 0（TP 对齐时），即使不为 0，fake shape 有一点误差也比完全错（差一倍）好

### 第二层：DBO 与 compile 的边界（本月可完成）

**原则：DBO hook 不出现在 compiled FX graph 中。**

当 `dbo_enabled` 在 compile 阶段和 runtime 阶段值不同时，Dynamo 的 guard 机制不能可靠工作。需要确保 Dynamo 看到的代码路径与 runtime 一致。

**推荐方案：用 `torch.compiler.disable()` 包裹 DBO 分支**

```python
# 改造后的 SequenceColumnParallelOp.apply_impl：
def apply_impl(self, input_):
    forward_context = get_forward_context()
    if forward_context.dbo_enabled:
        # DBO 分支：用 disable 标记，Dynamo 不 trace 这部分
        return self._apply_impl_dbo(input_)
    else:
        # 非 DBO 分支：正常编译
        return self._apply_impl_compiled(input_)

@torch.compiler.disable()
def _apply_impl_dbo(self, input_):
    # DBO hook + 通信逻辑
    forward_context = get_forward_context()
    forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
    if forward_context.flash_comm_v1_enabled and need_all_gather:
        input_ = tensor_model_parallel_all_gather(input_, 0)
    forward_context.dbo_template.dbo_linear_column_hook(is_record=False)
    input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, do_comm=False, ...)
    output_parallel = self.quant_method.apply(self.layer, input_, bias)
    return output_parallel, output_bias

def _apply_impl_compiled(self, input_):
    # 纯计算 + 通信，可以被编译
    input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, label=need_all_gather, ...)
    output_parallel = self.quant_method.apply(self.layer, input_, bias)
    return output_parallel, output_bias
```

**效果**：
- Dynamo trace 时只看到 `_apply_impl_compiled`（`dbo_enabled=False` 分支）
- 运行时 DBO 子线程走 `_apply_impl_dbo`（`dbo_enabled=True` 分支），完全不被 compile 影响
- 没有 guard fail → 没有 recompile
- DBO hook 正确执行

### 第三层：架构收敛到 upstream vLLM 模式（长期）

**原则：MoE 通信整体放入 opaque custom op。**

这是 upstream vLLM 的做法（参见 `vllm-ascend.md` 第 3 节）：
- 把整个 MoE `prepare → expert MLP → finalize` 注册为一个 `torch.ops.vllm.ascend_moe_forward` custom op
- fake impl 返回 `(global_hidden_states, output)` shape
- real impl 内部处理 AllGather/AllToAll + DBO hook + FlashComm

这可以一次性解决当前所有的 fake impl / DBO 分支 / context 读取问题，但工程量大，需要重构 MoE 模块的全部调用路径。

### 修复优先级

```
优先级 P0（本周）：
  ├── 第一层：给两个 custom op 加 explicit flash_comm_enabled 参数（修复 Bug 1/2/7）
  └── 第一层：给 matmul_and_reduce 的 fake impl 加 explicit 参数（修复 Bug 2/6）

优先级 P1（本月）：
  ├── 第二层：用 torch.compiler.disable 包裹所有 DBO 分支（修复 Bug 4）
  └── 第二层：修复 ubatch pad_size 不一致（修复 Bug 5）

优先级 P2（下个 release）：
  └── 第三层：AscendMoEForward custom op（根除所有边界问题）
```

### 为什么不能只设一下 snapshot 就完事

只调用 `set_flash_comm_v1_snapshot(True)` 存在以下问题：

1. **DBO 场景不工作**：snapshot 是全局变量。外层 context `flash_comm_v1_enabled=True` 但两个 ubatch 子线程各有不同的 `pad_size`。全局 snapshot 只能存一个值。

2. **warmup 与 runtime 不一致**：warmup 阶段 `flash_comm_v1_enabled` 的 `num_tokens` 条件可能不满足（token 数太少），但正式运行时满足 → snapshot 需要在每次 forward 前更新，极易遗漏。

3. **thread safety**：DBO 有两个线程，`_FLASH_COMM_V1_SNAPSHOT` 是模块级变量，没有线程隔离。虽然两个 ubatch 的 `flash_comm_v1_enabled` 通常相同，但 `pad_size` 不同时会有问题。

4. **生命周期**：snapshot 何时 reset？server 重启前一直保持？还是每个 step 更新？没有明确的所有权。

**显式参数方案**从根本上避免了这些问题：每个 call site 显式传入当前 context 的值，Dynamo 自然地为每个不同的参数值生成不同的 guard/compilation。

---

## Bug 1（严重）：`_FLASH_COMM_V1_SNAPSHOT` 从未被设置

**文件**：`vllm_ascend/ops/register_custom_ops.py:26-31`

```python
_FLASH_COMM_V1_SNAPSHOT: bool = False  # 永远是 False!

def set_flash_comm_v1_snapshot(value: bool) -> None:
    global _FLASH_COMM_V1_SNAPSHOT
    _FLASH_COMM_V1_SNAPSHOT = value
```

**现象**：`set_flash_comm_v1_snapshot()` 在整个代码库中**没有任何调用点**。这个函数被设计了但从未接入。

**影响链**：

1. `_maybe_all_gather_and_maybe_unpad_fake`（line 135-143）检查 `_FLASH_COMM_V1_SNAPSHOT`，永远是 `False`。
2. `_maybe_pad_and_reduce_fake`（line 146-152）同理，永远是 `False`。
3. 因此 fake impl 总是返回原始 shape，不做 TP 维度变换。

**Runtime 对比**：

| 函数 | Compile 时 (fake) | Runtime 时 (real) |
|------|-------------------|-------------------|
| `maybe_all_gather_and_maybe_unpad` | shape 不变 | `flash_comm_v1_enabled=True` 时第一维 `×TP` 后再 unpad |
| `maybe_pad_and_reduce` | shape 不变 | `flash_comm_v1_enabled=True` 时先 pad 再 reduce_scatter，第一维 `÷TP` |

**错误表现**：
- FakeTensor shape mismatch → `torch.compile` 重新触发（动态 shape guard 失败）
- 或更严重：graph capture 时记录的是错误 shape，replay 时输入 shape 不对 → 静默错误或 HCCL 报错
- 典型错误信息：`RuntimeError: The expanded size of the tensor must match the existing size`

**修复方向**：在 `set_ascend_forward_context()` 和 `create_ascend_forward_context()` 中，设置完 `flash_comm_v1_enabled` 后立即调用 `set_flash_comm_v1_snapshot()`。同时在 DBO wrapper 的 `_make_ubatch_metadata()` 中也需要为每个 ubatch context 更新 snapshot。

---

## Bug 2（严重）：`_matmul_and_reduce_impl_fake` 直接读取 live forward context

**文件**：`vllm_ascend/ops/register_custom_ops.py:198-208`

```python
def _matmul_and_reduce_impl_fake(input_parallel: torch.Tensor, layer_name: str) -> torch.Tensor:
    forward_context = get_forward_context()       # ← 直接读 live context!
    self = forward_context.no_compile_layers[layer_name]
    num_tokens = input_parallel.size(0)
    if _EXTRA_CTX.flash_comm_v1_enabled:          # ← 又读 _EXTRA_CTX!
        num_tokens = num_tokens // self.tp_size
    ...
```

**问题**：

- Compile/fake 阶段**没有** active forward context → `get_forward_context()` 抛出 `AssertionError`
- 即使有 context，compile 阶段的 `_EXTRA_CTX.flash_comm_v1_enabled` 可能与 runtime 不一致
- `self.tp_size` 依赖 layer object，而 layer object 在 compile 时可能不可用

**错误表现**：
- `AssertionError: No forward context is set` 在 compile 阶段
- 或 `AttributeError` 如果 `no_compile_layers` 为空

**修复方向**：改为纯 shape 推断——`num_tokens // tp_size` 中的 `tp_size` 通过显式参数传入（如 `get_tensor_model_parallel_world_size()` 是静态可用的），但 `flash_comm_v1_enabled` 必须通过 snapshot 或显式参数传入。

---

## Bug 3（严重）：`_maybe_all_reduce_tensor_model_parallel_impl` 读取 live context，fake 为 identity

**文件**：`vllm_ascend/ops/register_custom_ops.py:177-185, 292`

```python
def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states):
    moe_comm_type = _EXTRA_CTX.moe_comm_type        # ← 读 live context
    if moe_comm_type in {..., ...} or _EXTRA_CTX.flash_comm_v1_enabled:
        return final_hidden_states                   # skip all_reduce
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)

# fake impl:
lambda x: x   # 永远是 identity，不改变 shape
```

**问题**：fake impl 是纯 identity（shape 不变），这是**恰好正确的**——因为 real impl 在 MoE/FlashComm 场景下也确实不改变 shape（跳过 all_reduce）。

**但是**：real impl 读取 `_EXTRA_CTX.moe_comm_type` 和 `_EXTRA_CTX.flash_comm_v1_enabled`。如果 Dynamo 或 FX graph 把这个函数**不当作 opaque custom op** 而是 inline 进去，guard 会捕获到 live context 的值。这在 DBO 子线程中可能导致 guard 失败 → recompile。

**风险等级**：中等（如果作为 opaque custom op 处理则安全，但如果被 inlined 则危险）

---

## Bug 4（高风险）：DBO 分支在 compile 阶段被剪枝

### 4.1 linear_op.py 中的 DBO 分支

**文件**：`vllm_ascend/ops/linear_op.py`

涉及至少 4 个 op 类：

| 类 | 行号 | DBO 分支条件 | compile 时值 | runtime 子线程值 |
|----|------|-------------|-------------|-----------------|
| `MLPColumnParallelOp` | 198-204 | `forward_context.dbo_enabled` | `False` | `True` |
| `Flashcomm2OProjRowParallelOp` | 349-387 | `forward_context.dbo_enabled` | `False` | `True` |
| `SequenceColumnParallelOp` | 454-462 | `forward_context.dbo_enabled` | `False` | `True` |
| `SequenceRowParallelOp.matmul_and_reduce` | 604-608 | `forward_context.dbo_enabled` | `False` | `True` |

**编译阶段**：`set_ascend_forward_context()` 中 `forward_context.dbo_enabled = False`（line 150）。Dynamo trace 模型 forward 时，这些 `if dbo_enabled:` 分支走 else 路径（无 DBO hook 的普通通信）。

**运行时 DBO 子线程**：`create_ascend_forward_context()` 中 `new_forward_context.dbo_enabled = True`（line 299）。子线程的 forward context 中 `dbo_enabled=True`。

**两种可能的错误模式**：

- **模式 A（有 guard）**：Dynamo 为 `dbo_enabled` 生成了 guard。第一次 decode 在非 DBO 路径（`dbo_enabled=False`），graph 被编译。后续 prefill 进入 DBO 子线程（`dbo_enabled=True`），guard 失败 → 触发 recompile。每次 batch mode 切换都 recompile → 性能灾难。
- **模式 B（无 guard/inlined）**：Dynamo 把 `dbo_enabled=False` 当作常量内联，DBO hook 代码被 DCE（死代码消除）。runtime DBO 子线程调用 compiled graph 时，DBO hook 永远不会执行 → **两个 ubatch 不再交替**，退化为串行执行，DBO 失效。

**实际更可能的是模式 B**，因为 `forward_context.dbo_enabled` 是一个普通 Python bool 属性，Dynamo 在 trace 时会将其视为常量。

### 4.2 prepare_finalize.py 中的 DBO 分支

**文件**：`vllm_ascend/ops/fused_moe/prepare_finalize.py:390-414`

```python
if forward_context.dbo_enabled:
    # DBO 路径: 拆成 hook + 显式通信 + custom op (do_comm=False)
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)
    if get_forward_context().flash_comm_v1_enabled:
        hidden_states = tensor_model_parallel_all_gather(...)
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False)
    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(..., do_comm=False)
else:
    # 非 DBO 路径: custom op 内部完成通信
    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, True, True)
```

**关键差异**：
- DBO 路径：`do_comm=False`，通信由显式的 `all_gather` 完成，custom op 只做 unpad
- 非 DBO 路径：`do_comm=True`（默认），custom op 内部完成 AllGather + unpad

**编译期**：走 else 分支，`do_comm=True`。fake impl 检查 `_FLASH_COMM_V1_SNAPSHOT`（永远是 `False`），返回原始 shape。

**运行时**：走 if 分支，`do_comm=False`。fake impl 在 compile 时没考虑 TP 扩展，但 runtime 时 real impl 做了 `do_comm=False`（只 unpad 不通信）——这**恰好** shape 是一致的（因为 `do_comm=False` 时没有 TP 维度变化）。

**BUT**：这里又回到 Bug 4.1 的问题——DBO 分支本身会被编译期剪枝。最终效果是 DBO 子线程中 MoE prepare/finalize 的 hook 不执行 → DBO overlap 失效。

---

## Bug 5（中等）：DBO 子线程中 FlashComm 的 `pad_size` 不一致

**文件**：`vllm_ascend/ascend_forward_context.py:252-254`

在 `create_ascend_forward_context()` 中：

```python
if new_forward_context.flash_comm_v1_enabled or new_forward_context.flashcomm_v2_enabled:
    pad_size = (tp_world_size - (new_forward_context.num_tokens % tp_world_size)) % tp_world_size
    new_forward_context.pad_size = pad_size
```

- `new_forward_context.num_tokens` = ubatch 的 token 数（约总 token 数的一半）
- `new_forward_context.flash_comm_v1_enabled` = 从外层 context 继承（基于总 token 数判断的）

**问题**：
- 外层 context：总 token 数可能是 4096，`flash_comm_v1_enabled=True`（因为 >1000），`pad_size` 基于 4096 计算
- 子线程 ubatch0：token 数 = 2048，`flash_comm_v1_enabled=True`（继承），但 `pad_size` 基于 2048 重新计算

这导致两个 ubatch 的 `pad_size` 可能不正确——它们各自独立计算了基于 ubatch token 数的 padding，但实际通信（如 AllGather、ReduceScatter）是基于整个 TP group 的。

**影响**：如果 `tp_size=2` 且 ubatch token 数为偶数，则 `pad_size=0`，看似没问题。但如果 token 数不完全被 TP 整除，pad_size 在两个 ubatch 之间可能不一致，且与编译期的 fake shape 不一致。

---

## Bug 6（中等）：`matmul_and_reduce` 通过 `torch.ops.vllm.matmul_and_reduce` 调用但 fake 不安全

**文件**：`vllm_ascend/ops/linear_op.py:524, 529-611`

`SequenceRowParallelOp.apply_impl()` 中：

```python
output = torch.ops.vllm.matmul_and_reduce(input_parallel, self.unique_prefix)
```

这个 custom op 的 real impl（`_matmul_and_reduce_impl`）通过 `layer_name` 查找 layer object 并调用 `matmul_and_reduce`，后者又读取 `_EXTRA_CTX.flash_comm_v1_enabled` 和 `_EXTRA_CTX.mmrs_fusion`（line 532-536）。

**问题链**：
1. Compile 时 fake impl 读取 `get_forward_context()`（Bug 2）→ 可能失败
2. Runtime 时 real impl 读取 `_EXTRA_CTX` → 在 DBO 子线程中读到子线程 context
3. `pad_size` 基于子线程 num_tokens 计算（Bug 5）→ 可能不一致

---

## Bug 7（低-中）：`_EXTRA_CTX` 在 compile 阶段读取会抛异常

**文件**：`vllm_ascend/ascend_forward_context.py:487-503`

```python
class _ExtraForwardContextProxy:
    @staticmethod
    def _ctx():
        return get_forward_context()   # ← 在 compile 阶段可能抛 AssertionError

    def __getattr__(self, name):
        ...
        ctx = self._ctx()              # ← 抛异常点
```

任何在 compile/FX trace 阶段（Dynamo fake tensor propagation, AOTAutograd, PiecewiseBackend.compile_all_ranges()）读取 `_EXTRA_CTX.*` 的操作都会触发 `get_forward_context()` → `AssertionError`。

**受影响路径**：
- `register_custom_ops.py:41,59,66,78,91,107,119,120,178,181,189,199,202` — 大量 real impl 和 fake impl 读取 `_EXTRA_CTX`
- `linear_op.py:318,380,532,533,544` — 读取 `_EXTRA_CTX.pad_size`, `flash_comm_v1_enabled`, `mmrs_fusion`

**实际风险**：

- **real impl**：只在 runtime 调用（不在 compile 阶段），有 forward context，安全。
- **fake impl**：在 compile 阶段调用，没有 forward context，**不安全**。
- **被 Dynamo inlined 的非 opaque 代码**：如果 Dynamo 尝试 trace 进这些读 `_EXTRA_CTX` 的函数 → compile 阶段失败。

---

## Bug 8（系统性）：`dbo_template` 函数通过 `threading.get_ident()` 查找 context

**文件**：`vllm_ascend/worker/ubatching.py:172-179`

```python
def _register_ubatch_function(func):
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)
    return wrapper
```

所有的 `dbo_record_current_stream`、`dbo_wait_current_stream_and_yield` 等函数都通过 `threading.get_ident()` 查找当前线程对应的 ubatch context。

**与 compile 的冲突**：如果这些函数出现在 compiled FX graph 中（例如 DBO 分支没有被剪枝而是被 trace），Dynamo 会尝试 trace `threading.get_ident()` 和全局 dict 查找。这会导致：
- Dynamo 无法建立正确的 guard（thread local 状态不可靠）
- 或者产生不正确的 FX graph（把 thread ID 当作常量）

**现状**：这更多是一个"不应该发生"的问题——因为 DBO hook 应该始终在 compiled region 外部。但如果 Bug 4 中的"DBO 分支被 trace 而非剪枝"发生，这个问题就会出现。

---

## 问题汇总矩阵

| Bug | 严重度 | 症状 | 是否一定触发 | 检测方法 |
|-----|--------|------|-------------|---------|
| 1: Snapshot 未设置 | **严重** | shape mismatch, HCCL error | **是**（flashcomm=1 时） | 在 fake impl 中加 assert |
| 2: matmul_and_reduce fake 读 context | **严重** | AssertionError in compile | 是（使用此 op 的模型） | 检查 compile 日志 |
| 3: all_reduce fake 读 context | 中 | guard failure → recompile | 可能 | 检查 recompile 计数 |
| 4: DBO 分支被剪枝 | **严重** | DBO overlap 失效 | **是**（DBO+compile） | dump FX graph code |
| 5: pad_size 不一致 | 中 | shape 错误 | 非整除 TP 时 | 检查 pad_size 值 |
| 6: matmul_and_reduce 路径 | 中 | fake 不安全 | 使用此 op 的模型 | dump fake 调用栈 |
| 7: _EXTRA_CTX 编译期读 | 中 | 各种崩溃 | fake impl 路径 | 检查 compile 错误 |
| 8: threading.get_ident 被 trace | 低 | Dynamo error | 仅 DBO 分支被 trace 时 | FX graph dump |

---

## 推荐的修复路线

### 第一阶段：止血（紧急）

1. **修复 Bug 1**：在 `set_ascend_forward_context()` yield 之前和 `create_ascend_forward_context()` 中调用 `set_flash_comm_v1_snapshot(flash_comm_v1_enabled)`。

   ```python
   # 在 ascend_forward_context.py 的 set_ascend_forward_context 中，设置完 flash_comm_v1_enabled 后：
   from vllm_ascend.ops.register_custom_ops import set_flash_comm_v1_snapshot
   set_flash_comm_v1_snapshot(flash_comm_v1_enabled)
   ```

2. **修复 Bug 2**：重写 `_matmul_and_reduce_impl_fake`，不读 live context：

   ```python
   def _matmul_and_reduce_impl_fake(input_parallel, layer_name):
       # 纯 shape 推断：如果 flashcomm 启用，输出第一维 ÷ tp_size
       num_tokens = input_parallel.size(0)
       tp_size = get_tensor_model_parallel_world_size()
       if _FLASH_COMM_V1_SNAPSHOT:
           num_tokens = num_tokens // tp_size
       return torch.empty(
           (num_tokens, input_parallel.size(1) // tp_size), 
           device=input_parallel.device, dtype=input_parallel.dtype
       )
   ```

   **注意**：上面的 `output_size_per_partition` 信息来自 layer，需要改为通过显式参数传入，或者根据 weight shape 推断。

### 第二阶段：修复 DBO + compile 语义鸿沟（核心）

3. **修复 Bug 4**：确保 DBO hook 不出现在 compiled FX graph 中。

   **方案 A（推荐）**：把 DBO 插桩点全部封成 custom op。例如：
   - `torch.ops.vllm.ascend_dbo_linear_column_allgather(input, ...)` — 内部处理 DBO hook + AllGather
   - 使 Dynamo 看到的是 opaque op，不 trace 进 DBO 分支

   **方案 B**：用 `torch._dynamo.disable` 或 `torch.compiler.disable()` 标记 DBO hook 函数，让 Dynamo 跳过它们。

   **方案 C**：在 `AscendUBatchWrapper._run_ubatches()` 中，临时设置 `forward_context.dbo_enabled = True` **在** compiled pieces 被调用之前，让 Dynamo 在 warmup/trace 阶段就见到 `dbo_enabled=True` 的路径。

4. **修复 Bug 5**：在 `create_ascend_forward_context()` 中，`pad_size` 应该基于外层 context 的 padded num_tokens 重新计算（考虑 TP 对齐），而不是基于单个 ubatch 的 num_tokens。

### 第三阶段：架构加固（长期）

5. **纯化所有 fake impl**：建立 lint 规则——fake impl 中不允许出现 `get_forward_context()`、`_EXTRA_CTX.*`、`threading.get_ident()`。

6. **建立 compile safety test**：对每个 custom op，在无 forward context 的环境中调用其 fake impl，确保不抛异常。

7. **引入 DBO-safe compile mode**：如果 DBO + compile 确实需要在 compile 时就确定分支，可以：
   - 编译两份 graph：`dbo=True` 和 `dbo=False` 版本
   - 或者在 DBO wrapper 层做 dispatch：根据 `dbo_enabled` 选择不同的 compiled callable

---

## 当前脚本的直接建议

在 `deepseek-v2-dbo-server.sh` 中：

```bash
# 当前配置（有 bug）：
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}    # → bug

# 临时 workaround（关闭 FlashComm 以优先保证 DBO + compile 稳定）：
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-0}
```

**原因**：当前代码的 fake impl 基础设施（snapshot 机制）尚未就绪，FlashComm + DBO + compile 三者同时启用会触发 Bug 1-4 的多个问题。建议先在 `flashcomm=0` 下验证 DBO + compile 的稳定性，然后按上述第一阶段修复后再逐步打开 FlashComm。

---

## 调试建议

### 快速确认 Bug 1（snapshot 未设置）

在 `register_custom_ops.py` 的 `_maybe_all_gather_and_maybe_unpad_fake` 入口加：

```python
import logging
logging.warning(f"[FAKE] all_gather_unpad: FLASH_SNAPSHOT={_FLASH_COMM_V1_SNAPSHOT}, "
                f"label={label}, do_comm={do_comm}, input_shape={x.shape}")
```

如果看到 `FLASH_SNAPSHOT=False` 但 script 中设置了 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`，则确认 Bug 1。

### 快速确认 Bug 4（DBO 被剪枝）

在 compiled model 第一次 forward 后，dump FX graph code：

```python
# 在 model_runner_v1.py 的 load_model 之后
if hasattr(self.model, 'runnable'):
    gm = self.model.runnable
    if hasattr(gm, 'code'):
        with open('/tmp/fx_graph_code.py', 'w') as f:
            f.write(gm.code)
```

搜索 `dbo_linear_column_hook`、`dbo_moe_prepare_hook`、`dbo_*`——如果完全找不到，则 DBO 分支已被剪枝。

### 检查 recompile 计数

```python
import torch._dynamo.utils as dynamo_utils
# 每个 step 后
print(f"Recompile count: {dynamo_utils.get_dynamo_compile_time()}")
```

如果 recompile 计数持续增长，说明有 guard failure（可能与 Bug 3/4 相关）。

---

## 参考文件索引

| 文件 | 关键行 | 关联 Bug |
|------|--------|---------|
| `vllm_ascend/ops/register_custom_ops.py` | 26-31, 135-152, 198-208 | 1, 2, 3, 7 |
| `vllm_ascend/ops/linear_op.py` | 198-204, 318, 349-387, 454-464, 529-611 | 4, 5, 6, 7 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | 390-414, 527-543 | 4, 7 |
| `vllm_ascend/ascend_forward_context.py` | 100-199, 244-299, 487-507 | 5, 7 |
| `vllm_ascend/worker/npu_ubatch_wrapper.py` | 255-316, 346-431 | 4, 5 |
| `vllm_ascend/worker/ubatching.py` | 67-169, 172-196 | 8 |
| `vllm_ascend/platform.py` | 379-381 | PIECEWISE 降级 |
| `vllm_ascend/dbo/overlap_templates/deepseek.py` | 7-68 | DBO hook 定义 |
