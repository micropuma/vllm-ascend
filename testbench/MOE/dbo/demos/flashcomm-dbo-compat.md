# FlashComm + DBO + torch.compile 兼容性分析

> 回答：为什么 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 会导致 DBO + torch.compile 崩溃，以及如何系统性地解决。

---

## 一、问题本质：vllm 执行范式切换

### v0.13 时代（一切正常）

```
vllm v0.13 执行模型:

  model.forward()               ← 普通 Python 调用
    → layer.forward()           ← 逐行 eager 执行
      → custom_op.apply()       ← 直接调用 ACL 实现
        → get_forward_context() ← 普通函数调用 ✅
        → if dbo_enabled: ...   ← 普通 Python if ✅
        → hook(record)          ← NPU event record ✅
        → all_gather(...)       ← HCCL 通信 ✅
```

**没有 Dynamo，没有 FakeTensor，没有 fullgraph capture。**

### v0.20.2+ 时代（崩溃开始）

```
vllm v0.20.2+ 执行模型:

  model.forward()  [@compiled]
    → Dynamo fullgraph trace       ← 第一次调用触发
      → FX Graph
        → FakeTensor propagation   ← 需要 fake impl
          → get_forward_context()  ← 💥 contextvars 无法 trace
          → if dbo_enabled:        ← guard 生成，编译期 False → DCE
          → hook(record)           ← 被 DCE 剪枝
```

**核心矛盾**：DBO 代码假设在 eager 模式运行（读 context、分支、hook），但 vllm v0.20+ 默认走 compile-first 路径。

---

## 二、FlashComm1 的角色：触发器而非根因

### FlashComm1 做了什么

```
VLLM_ASCEND_ENABLE_FLASHCOMM1=1
  → enable_sp() = True
    → DeepSeek V2 的 gate_up_proj 使用 SequenceColumnParallelOp
      → SequenceColumnParallelOp.apply_impl()
        → 第 454 行: forward_context = get_forward_context()
          → Dynamo trace → 💥
```

### FlashComm1=0 为什么不崩

```
VLLM_ASCEND_ENABLE_FLASHCOMM1=0
  → enable_sp() = False
    → gate_up_proj 不使用任何 custom op
      → 走标准 vllm ColumnParallelLinear.forward()
        → 没有 get_forward_context() 调用
          → Dynamo trace ✅
```

**但这只是运气好**。如果开启 `mlp_tp_enable()`（finegrained TP），`MLPColumnParallelOp` 会被使用，它也有同款 `get_forward_context()` 调用（line 198）。

### DBO 的所有注入点都有隐患

| 注入点 | 触发条件 | 当前是否崩溃 |
|--------|---------|------------|
| `MLPColumnParallelOp` (line 198) | `mlp_tp_enable()=True` | 否（未触发） |
| `SequenceColumnParallelOp` (line 454) | `enable_sp()=True` | **是（FlashComm1 触发）** |
| `SequenceRowParallelOp` (line 543) | `enable_sp()=True` + 特定 flashcomm 路径 | 否（flashcomm 路径分支避开） |
| `PrepareAndFinalizeWithAllGather` | DBO 子线程中 | 否（`dbo_enabled` 守卫，但 Bug 4） |
| `TokenDispatcherWithAll2AllV` | DBO 子线程 + A3 | 否（同上） |
| `AscendMLAImpl._forward` | DBO 子线程中 | 否（同上） |

---

## 三、完整的因果链

```
                    ┌────────────────────┐
                    │  vllm v0.20+ 升级   │
                    │  @compiled 默认开启  │
                    └────────┬───────────┘
                             │
                    ┌────────▼───────────┐
                    │  Dynamo fullgraph   │
                    │  trace 所有 forward  │
                    └────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────────┐
     │ FlashComm1=0  │ │FlashComm1=1│ │ DBO 迁移代码   │
     │ enable_sp=Off │ │enable_sp=On│ │ 添加了无条件    │
     │ 不用 SP op    │ │用 SP op    │ │ get_fc() 调用   │
     └───────┬───────┘ └─────┬──────┘ └─────┬──────────┘
             │               │              │
             ▼               ▼              ▼
       不触发 ✅      SequenceColumn    DBO hook
                     ParallelOp        在 compiled
                     被选择            区域内
                         │              │
                         └──────┬───────┘
                                │
                     ┌──────────▼──────────┐
                     │  get_forward_context()│
                     │  在 Dynamo trace 中   │
                     │  → Unsupported 💥    │
                     └─────────────────────┘
```

---

## 四、所有受影响的代码路径

### 4.1 会导致编译期崩溃（`get_forward_context()` 在 compiled 路径中）

| 文件 | 行号 | 代码 | 触发条件 |
|------|------|------|---------|
| `linear_op.py` | 198 | `forward_context = get_forward_context()` | `mlp_tp_enable()=True` |
| `linear_op.py` | 454 | `forward_context = get_forward_context()` | `enable_sp()=True`（**当前命中**） |
| `linear_op.py` | 543 | `forward_context = get_forward_context()` | `enable_sp()=True` + FlashComm1=OFF 时 |

### 4.2 不会崩溃但 DBO 路径被 DCE 剪枝（Bug 4）

| 文件 | 行号 | 代码 | 说明 |
|------|------|------|------|
| `prepare_finalize.py` | ~389 | `if forward_context.dbo_enabled:` | compile 期 `False`，DBO 分支被 DCE |
| `prepare_finalize.py` | ~548 | `if forward_context.dbo_enabled:` | 同上 |
| `token_dispatcher.py` | ~490 | `if forward_context.dbo_enabled:` | 同上 |
| `mla_v1.py` | ~1669 | `if forward_context.dbo_enabled:` | 同上 |

### 4.3 已修复的问题（通过 FlashComm Snapshot + do_comm）

| 文件 | 问题 | 状态 |
|------|------|------|
| `register_custom_ops.py` | `_maybe_all_gather_and_maybe_unpad_fake` 读 `_EXTRA_CTX` | ✅ 改为读 `_FLASH_COMM_V1_SNAPSHOT` |
| `register_custom_ops.py` | `_maybe_pad_and_reduce_fake` 同上 | ✅ 同上 |
| `register_custom_ops.py` | `do_comm` 参数支持 DBO 通信拆分 | ✅ 已实现 |

---

## 五、修复矩阵

| 阶段 | 措施 | 解决的问题 | 工作量 |
|------|------|-----------|--------|
| **Workaround** | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=0` | 暂时避开 FlashComm1 路径 | 0 |
| **Phase 1** | 对 3 个 `linear_op.py` 注入点加 `@torch.compiler.disable()` | 编译期崩溃 | 小（~30行） |
| **Phase 2** | 实施 [方案C](plan-c-dbo-compile-arch.md)：将全部 7 个注入点提升到 dispatch 层 | 编译期崩溃 + Bug 4（DBO 被 DCE） | 中（~300行） |
| **Phase 3** | 建立 compile safety test：每个注入点的 fake impl 在无 forward context 环境中验证 | 防止回归 | 中 |

---

## 六、推荐迁移路径

1. **当前**：`FlashComm1=0` workaround，在非 FlashComm 模式下验证 DBO overlap 正确性和性能
2. **短期**：实施方案C Phase 1+2，让 FlashComm1 + DBO + compile 三者兼容
3. **验证**：`FlashComm1=1 + --enable-dbo` 启动 server，确认：
   - 不崩溃
   - `grep "should_ubatch: True"` 确认 DBO 触发
   - profiler trace 中确认 compute-comm overlap 正常

---

## 七、参考

- [方案C 详细设计](plan-c-dbo-compile-arch.md)
- [FlashComm+DBO+Compile 8 个 Bug 分析](bug/flashcomm-dbo-compile-bugs.md)
- [DBO 迁移报告](migration-report.md)
- [FlashComm1 崩溃根因](flashcomm-dbo-compile-root-cause.md)
