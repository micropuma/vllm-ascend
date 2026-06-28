# 综合分析：DBO + FlashComm + Torch Compile 兼容性及启动性能

## 文档元信息

- 日期：2026-06-28
- 硬件：Ascend 910B (A2), TP=2
- 软件：vLLM 0.22.1, CANN 9.0.0
- 关联 RFC：
  - [[rfc-dbo-flashcomm2-aicpu-alltoall]]
  - [[rfc-dbo-flashcomm1-compile-shape-contract]]
  - [[rfc-dbo-flashcomm2-aicpu-alltoall-analysis]]

---

## 一、DBO + FlashComm2 AICPU 复现与分析

### 1.1 复现结果

| RUN | DBO | FC2 | 结果 | 日志关键特征 |
|-----|-----|-----|------|------------|
| 1 | on | off | ✅ 通过 | `should_ubatch: True` for prefill (120/232 tokens) |
| 2 | on | on | ❌ AICPU crash | `error 507018`, `aicpu exception`, HCCL watchdog terminated |
| 3 | off | on | ❌ AICPU crash | **DBO=off 仍崩溃！**`kernelName=RunAicpuRpcSrvLaunchV2_alltoall, errorCode=0x2a` |
| 4 | off | off | ⏭ 未跑 | 与 RUN=3 资源冲突，但基线应通过 |

### 1.2 RUN=2 故障时序

```
09:25:19 - Profiling PIECEWISE graph memory (should_ubatch, captures acL graph)
09:25:20 - NPU graph warning: "Waiting for pending HCCL work to finish"
09:25:21 - Estimated PIECEWISE graph memory
09:25:22 - Both rank 0 and rank 1 crash with AICPU error 507018
```

关键观察：错误发生在 `profile_cudagraph_memory` 阶段的 **graph capture** 中，不是在正式推理阶段。这说明即使是 dummy_run（仅为 graph capture 和内存估算），FlashComm2 + DBO 组合也会触发 AICPU 错误。

完整 Python 调用栈：

```
profile_cudagraph_memory (model_runner_v1.py:4866)
  → _dummy_run (model_runner_v1.py:3531)
    → _model_forward (model_runner_v1.py:2753)
      → npu_ubatch_wrapper.__call__ (npu_ubatch_wrapper.py:369)
        → acl_graph.__call__ (acl_graph.py:206)
          → torch.npu.synchronize()  ← 在此发现异步 AICPU 错误
```

### 1.3 设备侧错误

```
[Rank 0]: rtEventQueryStatus execution failed, reason=aicpu exception
          runtime result = 507018
          ERR00100 PTA call acl api failed

[Rank 1]: rtEventQueryStatus execution failed, reason=aicpu exception
          runtime result = 507018
          HCCL watchdog thread terminated
          ERR02005 DIST internal error
```

两个 rank **同时**报告 AICPU 异常，符合"两个线程在两条 stream 上向同一 communicator 无序提交 collective 导致跨 rank 配对破坏"的根因假设。

### 1.3 ⚠️ 新发现：FC2 在 DBO=off 时也崩溃

**RUN=3（DBO=off, FC2=on）也触发了完全相同的 AICPU 错误**：

```
kernelName=RunAicpuRpcSrvLaunchV2_alltoall
errorCode=0x2a
runtime result = 507018
soName=libccl_kernel.so
funcName=RunAicpuRpcSrvLaunchV2
```

但 RUN=3 的调用栈中**没有 `npu_ubatch_wrapper`**，确认 DBO 未触发：

```
profile_cudagraph_memory
  → _dummy_run → _model_forward
    → acl_graph.__call__         ← 直接走 graph capture，无 ubatch wrapper
      → torch.npu.synchronize()  ← 在此发现异步 AICPU 错误
```

vs RUN=2（DBO on）：

```
profile_cudagraph_memory
  → _dummy_run → _model_forward
    → npu_ubatch_wrapper.__call__  ← DBO 线程包装
      → acl_graph.__call__
        → torch.npu.synchronize()
```

**这意味着**：`Flashcomm2OProjRowParallelOp` 的 ODP AlltoAll 在 910B 上即使**单线程**也有问题。可能有以下原因：

1. **ODP group 初始化问题**：`get_flashcomm2_odp_group()` 返回的 communicator 在 910B 上的 AlltoAll 实现存在 bug
2. **Tensor shape/stride 问题**：`otp_maybe_quant_comm()` 中 tensor reorganization（`chunked = chunked[self.group_indices]`）可能导致不连续的 tensor，HCCL AlltoAll 对此处理有缺陷
3. **HCCL_OP_EXPANSION_MODE=AI_CPU 与 FC2 AlltoAll 不兼容**：910B 的 AICPU AlltoAll kernel 本身有问题
4. **Group indices 错误**：`get_flashcomm2_reorgnized_batch_ids()` 返回的 batch reordering 在特定 TP/DP 配置下产生无效映射

**需要进一步排查**：
- 关闭 `HCCL_OP_EXPANSION_MODE=AI_CPU` 后 RUN=3 是否通过？（如果通过，则 AICPU AlltoAll kernel 本身是问题）
- 使用 `deepep_high_latency` 后 RUN=3 是否通过？
- 检查 `group_indices` tensor 的具体值和 device 放置

### 1.4 NPU Graph 警告的证据价值

```
Warning: Waiting for pending HCCL work to finish before starting graph capture.
```

这个警告在 graph capture 启动前出现，说明 **前一个 operation 留下了未完成的 HCCL 通信**。在 DBO 场景下，这是两个 ubatch 线程交错提交 collective 的直接证据 — graph capture 开始时，另一个 ubatch 的通信还在 stream 上 pending。

---

## 二、已确认的根因

代码级根因为 `Flashcomm2OProjRowParallelOp.apply_impl()` 缺少 DBO row hook。

### 2.1 关键代码对比

**正确路径：OProjRowParallelOp** (`linear_op.py:260-276`)：

```python
forward_context = get_forward_context()
if forward_context.dbo_enabled:
    _dbo_call_linear_row_hook(forward_context, is_record=True)   # ← record ATTN_POST

dist.all_to_all_single(recv_buf, send_buf, ...)                  # all2all
output_parallel = self.quant_method.apply(...)                   # matmul
output = self.comm_group.reduce_scatter(output_parallel)         # reduce_scatter

if forward_context.dbo_enabled:
    _dbo_call_linear_row_hook(forward_context, is_record=False)  # ← yield CPU
```

**故障路径：Flashcomm2OProjRowParallelOp** (`linear_op.py:315-392`)：

```python
# 无 dbo_enabled 检查，无 hook 调用
input_parallel = otp_maybe_quant_comm(input_parallel)  # ODP AlltoAll (行 367)
output_parallel = self.quant_method.apply(...)          # MatMul (行 375)
output = self.comm_group.reduce_scatter(output_parallel) # OTP ReduceScatter (行 379)
# 行 383-387: 条件性 TP all_gather
```

### 2.2 为什么在 A2（910B）上仍出问题

910B 是 A2 设备，选择 `DeepseekAllgatherTemplate`。但该模板的 `dbo_linear_row_hook(is_record=True)` 仍会 record `ATTN_POST` event（`deepseek.py:20`），而 `dbo_linear_column_hook(is_record=False)` 会 wait 这个 event 并 yield（`deepseek.py:25`）。

Flashcomm2OProjRowParallelOp 缺少 hook 导致：
1. `ATTN_POST` event 未 record → 后续 `dbo_linear_column_hook` wait 立即通过（stale event）
2. CPU yield 被跳过 → 两个线程并发在两条 stream 上提交 HCCL collective
3. `odp_group` AlltoAll 的跨 rank 配对被破坏 → AICPU 0x2a

### 2.3 dispatcher 触发条件

`_get_row_parallel_op()` (`linear_op.py:709-715`) 中，`oproj_tp_enable()` 检查（规则 #3）在 `flashcomm2_enable()`（规则 #5）之前。因此：

- `oproj_tp_enable() = True` → 走 `OProjRowParallelOp`（有 hook）
- `oproj_tp_enable() = False` & `flashcomm2_enable() = True` → 走 `Flashcomm2OProjRowParallelOp`（无 hook）

即 DBO + FC2 仅在 **oproj_tp 关闭 + flashcomm2 开启** 时触发 bug。

---

## 三、Torch Compilation Cache 及 Shape 问题

### 3.1 Cache Hash 不完整（潜在 bug）

**文件**：`vllm_ascend/compilation/compiler_interface.py:236-249`

```python
def compute_hash(self, vllm_config: VllmConfig) -> str:
    factors = {
        "torch_npu_version": torch_npu.__version__,
        "enable_npugraph_ex": ascend_compilation_config.enable_npugraph_ex,
        "enable_static_kernel": ascend_compilation_config.enable_static_kernel,
    }
    return sha256(str(factors).encode()).hexdigest()[:10]
```

**问题**：
1. Hash 仅 10 字符（40 bits），存在碰撞风险
2. 不包含 flash_comm_v1_enabled, dbo_enabled, tp_size, model_arch
3. **RUN=1 和 RUN=2 的实际日志确认了相同的 cache hash `55b0c8f3df`**，但两个运行的 flashcomm 配置完全不同
4. 代码中有显式 TODO：`# TODO(wxs): add passes related to compilation in compute_hash`

### 3.2 DBO 分支在 compile 中的处理（已修复的部分）

**文件**：`vllm_ascend/dbo/compile_guard.py`

5 个 `@torch.compiler.disable()` wrapper 保护了 DBO hook 调用不被 Dynamo trace。但 `get_forward_context()` 调用（读取 `dbo_enabled` 属性）仍在 compile 区域内：

| 文件 | 行号 | Op Class | 风险 |
|------|------|----------|------|
| `linear_op.py` | 199 | `MLPColumnParallelOp` | `get_forward_context()` 在 `@torch.compiler.disable()` 之外 |
| `linear_op.py` | 262 | `OProjRowParallelOp` | 同上 |
| `linear_op.py` | 455 | `SequenceColumnParallelOp` | 同上，且当前被 FC1 触发 |
| `linear_op.py` | 545 | `SequenceRowParallelOp` | 同上 |

### 3.3 Shape Contract 修复（已完成）

**Commit `9143d287`**：修复了 pad/unpad mismatch。引入 `_get_actual_num_tokens()` 从 attention metadata 读取逻辑 token 数而非 scheduler padded token 数。

**Commit `3cc7fbe3`**：修复了 shape mismatch。在 fake/runtime 路径统一使用 ceiling division `_get_reduce_scatter_num_tokens()`。

**当前状态**：DBO + FlashComm1 的 shape mismatch 问题已修复。DBO + FlashComm2 仍有待修复（本文档主题）。

### 3.4 `_FLASH_COMM_V1_SNAPSHOT` 机制

**文件**：`vllm_ascend/ops/register_custom_ops.py:26-31`

```python
_FLASH_COMM_V1_SNAPSHOT: bool = False

def set_flash_comm_v1_snapshot(value: bool) -> None:
    global _FLASH_COMM_V1_SNAPSHOT
    _FLASH_COMM_V1_SNAPSHOT = value
```

在 `ascend_forward_context.py:142` 处设置。FakeImpl 读取此 snapshot 而非 runtime context。这是在 torch.compile graph 中安全获取 flashcomm 状态的机制。

---

## 四、FlashComm1 启动慢分析

### 4.1 根因：每个 `_dummy_run` 都执行 HCCL collective

FlashComm1 本身没有复杂的初始化逻辑，它只是一个配置标志。启动慢的根本原因是：

**在 `profile_cudagraph_memory` 和 warmup 阶段，每个 `_dummy_run` 都执行了所有 HCCL collective**。

关键代码路径（`model_runner_v1.py:2758-2759`）：

```python
if forward_context.flash_comm_v1_enabled and not get_forward_context().dbo_enabled
   and not isinstance(hidden_states, IntermediateTensors):
    hidden_states = self._all_gather_hidden_states_and_aux(hidden_states)
```

此操做在 **每个 `_dummy_run` 末尾** 执行一次完整的 TP all-gather。

### 4.2 量级估算

- 默认 `cudagraph_capture_sizes`：35 个 size（1 到 256）
- PIECEWISE warmup：35 个
- FULL warmup：35 个
- **总计 ~105 个 `_dummy_run`**

每个 `_dummy_run` 执行：
- 1 次 post-model all-gather（~225MB/4096 tokens）
- MoE prepare all-gather（每 MoE 层）
- MoE finalize reduce-scatter（每 MoE 层）
- Sequence parallel linear collectives

**FlashComm1 关闭时：以上所有 collective 都不执行。**

### 4.3 优化建议（优先级排序）

#### 高优先级：跳过 warmup 阶段的 post-model all-gather

```python
# model_runner_v1.py:2758
if (forward_context.flash_comm_v1_enabled 
    and not forward_context.in_profile_run    # ★ 新增
    and not get_forward_context().dbo_enabled 
    and not isinstance(hidden_states, IntermediateTensors)):
```

`profile_cudagraph_memory` 的 `_dummy_run` 仅需估算内存和 graph capture，不需要完整的 post-model all-gather。

#### 中优先级：warmup 阶段全局跳过 flashcomm1 collective

在 `set_ascend_forward_context()` 中，如果检测到 `in_profile_run=True`，将 `flash_comm_v1_enabled` 设置为 `False`。

#### 长期：分离 compile-time graph 和 SP graph

如 `plan-c-dbo-compile-arch.md` 中所述，将 sequence parallelism 决策从 compiled graph 中分离。

### 4.4 NPU Graph Sync 开销的证据

RUN=2 日志中观察到的：
```
[NPUGraph.cpp:223] Warning: Waiting for pending NCCL work to finish before starting graph capture.
```

这表明 **每次 graph capture 之前都需要等待前序 HCCL 通信完成**，进一步放大了 warmup 阶段的开销。

---

## 五、次生问题：异常传播

`npu_ubatch_wrapper.py` 的 `_ubatch_thread` 函数（`npu_ubatch_wrapper.py:196-213`）没有 try/except。

RUN=2 的实际行为证实了问题：
1. AICPU 错误在子线程中异步发生
2. Python 侧在 `torch.npu.synchronize()` 处发现
3. 主线程接收到正确的 RuntimeError（因为 `synchronize()` 在主线程调用）
4. 但如果 AICPU 错误更早被触发且子线程先崩溃，错误会被 `IndexError` 掩盖

---

## 六、总结与行动项

| 优先级 | 问题 | 状态 | 行动 |
|--------|------|------|------|
| 🔴 P0 | DBO + FC2 AICPU crash | **已复现** | 实施方案 A：为 Flashcomm2OProjRowParallelOp 添加 DBO row hook |
| 🔴 P0 | **FC2 AICPU crash（DBO=off）** | **新发现** | 排查 ODP group / AICPU AlltoAll kernel / group_indices；尝试关闭 AI_CPU 验证 |
| 🔴 P0 | compile cache hash 不完整 | **已确认** | 扩展 hash 包含 flash_comm, dbo, tp_size, model_arch |
| 🟡 P1 | FC1 启动慢（warmup 中执行 HCCL） | **已分析** | 在 profile_run 中跳过 post-model all-gather |
| 🟡 P1 | `get_forward_context()` 在 `@torch.compiler.disable()` 之外 | **已分析** | 将 context 读取移入 guarded 函数内 |
| 🟢 P2 | `_ubatch_thread` 异常传播 | **已分析** | 添加 try/except 收集子线程异常 |
| 🟢 P2 | npugraph_ex monkey-patch 竞态 | **已分析** | 添加线程锁 |
