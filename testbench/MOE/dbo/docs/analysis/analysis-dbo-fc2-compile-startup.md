# 综合分析：DBO + FlashComm + Torch Compile 兼容性及启动性能

## 文档元信息

- 日期：2026-06-28
- 硬件：Ascend 910B (A2), TP=2
- 软件：vLLM 0.22.1, CANN 9.0.0
- 关联 RFC：
  - [[rfc-dbo-flashcomm2-aicpu-alltoall]]
  - [[rfc-dbo-flashcomm1-compile-shape-contract]]
  - [[rfc-dbo-flashcomm2-aicpu-alltoall-analysis]]

> **阅读口径（2026-07-04 更新）**
>
> 本文前六章保留 2026-06-28 到 2026-06-30 的故障发现过程，属于“历史现场”；
> 第七章以后以远端当前未提交工作区、2026-06-30 配置矩阵、2026-07-01 至
> 2026-07-03 回归日志为准。此前“FC2 在 DBO=off 必然崩溃”“compile_guard.py
> 是当前方案”等表述已经被后续 AIV 对照实验和当前 custom-op 方案修正，不能脱离
> 时间线单独引用。[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo0_fc10_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]
> [VERIFY: vllm_ascend/ops/register_custom_ops.py:34]

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

---

## 七、2026-07-04 当前结论：稳定粒度到底到哪里

### 7.1 “稳定支持”必须拆成四级证据

本文不把“服务进程仍在”“API ready”“单请求通过”直接等同于稳定。DBO 是线程、
stream、collective、compile、graph capture 和动态 shape 的交叉功能，因此稳定性
按以下四级定义：

| 等级 | 判定条件 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| L0 启动 | 模型、KV cache、compile/graph warmup 完成，API ready | 配置至少能初始化 | 不证明 DBO 被触发 |
| L1 单请求 | 4K prefill、输出 1 token、`completed=1 failed=0` | 真实 forward 可完成一次 | 不证明并发 shape 和重复调度 |
| L2 并发回归 | 32 请求、并发 16，全部完成 | 多 batch/多轮 ubatch 基本正确 | 不证明长稳与性能 |
| L3 压测 | 500 请求、并发 96，全部完成 | 当前模型与固定配置具备较强运行证据 | 不等于跨模型、跨 SoC、跨 TP 的产品级支持 |

该分级来自现有实验规模：单请求结果、32×16 回归、500×96 配置矩阵均真实存在。
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_compiled_4k_20260703_in4096_out1_np1_c1.json:1]
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_compiled_step3_multi32_in4096_out1_np32_c16.json:1]
[VERIFY: testbench/MOE/dbo/results/fc1_fc2_aiv_dbo_default_full_in4096_out16_np500_c96.json:1]

### 7.2 当前有强证据支持的固定边界

当前证据最强的范围不是“DBO 任意配置”，而是：

```text
模型        DeepSeek-V2-Lite-Chat
硬件        2 × Ascend 910B3（A2）
并行        TP=2, EP enabled, DP=1, PP=1
DBO         prefill enabled, threshold=1024
decode DBO  实际关闭（threshold=1_000_000_000）
ubatch      固定切成 2 份
compile     VLLM_COMPILE
graph       FULL_AND_PIECEWISE
MoE backend deepep_low_latency
负载        4096 input / 16 output / 500 prompts / concurrency 96
```

服务脚本明确设置 TP=2、EP、DBO、prefill/decode threshold 和
`deepep_low_latency`。[VERIFY: testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:60]
[VERIFY: testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:178]
ubatch 数在插件侧仍硬编码默认为 2。[VERIFY: vllm_ascend/worker/ubatching.py:15]

因此本文中“稳定”只应写成：

> 在 DeepSeek-V2-Lite、A2、TP2、prefill-only DBO、双 ubatch、固定 compile/graph
> 组合上，若使用已经跑通的通信配置，存在 L3 级证据。

不能扩展为：

- decode DBO 稳定；
- DP>1 或 PP>1 稳定；
- TP4/TP8 稳定；
- A3 AllToAll 模板稳定；
- O-Shard 与 DBO 组合稳定；
- 所有动态 token shape 稳定；
- VL first layer 稳定。

这些维度没有与当前 500×96 结果等价的实验覆盖；模板选择还会在 A3 切换到
`DeepseekAlltoallTemplate`，其事件序列与 A2 的 `DeepseekAllgatherTemplate`
不同。[VERIFY: vllm_ascend/dbo/utils.py:14]

### 7.3 2026-06-30 九组矩阵：当前最可靠的配置证据

以下九组均为 4096 输入、16 输出、500 请求、并发 96，结果文件均记录
`completed=500, failed=0`：

| DBO | FC1 | FC2 | HCCL mode | completed/failed | req/s | 结论级别 |
|---:|---:|---:|---|---:|---:|---|
| 0 | 0 | 0 | AIV | 500/0 | 6.207 | L3 基线 |
| 0 | 1 | 0 | AIV | 500/0 | 6.232 | L3 |
| 0 | 0 | 1 | AIV | 500/0 | 6.396 | L3 |
| 1 | 0 | 0 | AIV | 500/0 | 6.403 | L3 |
| 1 | 0 | 0 | AI_CPU | 500/0 | 6.590 | L3 |
| 1 | 1 | 0 | AIV | 500/0 | 6.555 | L3 |
| 1 | 1 | 0 | AI_CPU | 500/0 | 7.609 | L3 |
| 1 | 0 | 1 | AIV | 500/0 | 6.506 | L3 |
| 1 | 1 | 1 | AIV | 500/0 | 6.619 | L3 |

逐项证据位于九个 `matrix_dsv2_*.json` 文件首行。[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo0_fc10_fc20_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo0_fc11_fc20_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo0_fc10_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc10_fc20_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc10_fc20_aicpu_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc11_fc20_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc11_fc20_aicpu_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc10_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc11_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]

这里必须区分“正确性矩阵”和“性能结论”。九组只各跑一次，冷启动、cache、
进程放置和运行时噪声没有通过多轮统计消除，因此 `7.609 req/s` 不能据此宣称
AI_CPU 或 FC1 带来确定加速；它只能证明该组合完成了当前压测。

### 7.4 AICPU 与 AIV：原始现象为何不再等于当前结论

早期 FC2 + AI_CPU 日志出现 `RunAicpuRpcSrvLaunchV2_alltoall`、507018 和
0x2a，说明失败点确实落在 AICPU AllToAll kernel，而不是 Python 最后一层异常。
但后续 AIV 配置下，DBO=0/FC2=1 和 DBO=1/FC2=1 均完成 500×96。
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo0_fc10_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]
[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc10_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]

所以系统结论应改为：

1. **FC2 本身不是“DBO=off 也必崩”**；AIV 已给出反例。
2. **早期失败至少受 HCCL expansion mode 影响**；失败 kernel 明确是 AICPU。
3. **DBO hook 缺失仍是真问题**；因为两个 ubatch 必须保证同一 communicator
   上各 rank 的 collective 提交顺序一致。
4. **不能把 AIV 的通过外推到 AI_CPU + FC2**；当前矩阵没有
   `DBO=1, FC2=1, AI_CPU` 的 L3 结果。

### 7.5 当前建议配置

在只追求“按已有证据复现稳定”的场景：

```bash
HCCL_OP_EXPANSION_MODE=AIV
VLLM_ASCEND_ENABLE_DBO=1
VLLM_ASCEND_ENABLE_FLASHCOMM1=1       # 0 也有 L3 证据
VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1 # 0 也有 L3 证据
DBO_PREFILL_TOKEN_THRESHOLD=1024
DBO_DECODE_TOKEN_THRESHOLD=1000000000
TP=2
```

此建议严格绑定上述 DeepSeek-V2-Lite/A2/TP2 条件，不是全局默认值。
脚本当前默认 `HCCL_OP_EXPANSION_MODE=AI_CPU`，与 FC2 的保守建议不一致；
如果 FC2 开启，应显式覆盖为 AIV 并记录日志。[VERIFY: testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:60]

---

## 八、DBO 的真实执行粒度：不是“算子开关”，而是五层协议

### 8.1 调度粒度：一个 scheduler batch 是否切成两个 ubatch

上游 wrapper 根据 token threshold 决定是否 ubatch；插件内部默认 ubatch 数为 2。
切分后不是一个 Python 线程顺序调用两次，而是创建两个线程、两个
`AscendUBatchContext`，并用 barrier 和成对 CPU event 交替推进。
[VERIFY: vllm_ascend/worker/ubatching.py:15]
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:241]

```text
主线程
  ├─ 构造 ubatch0 metadata/context
  ├─ 构造 ubatch1 metadata/context
  ├─ 启动 thread0、thread1
  ├─ ready_barrier 等待两线程 ready
  ├─ 唤醒 ubatch0
  └─ join 两线程并合并结果

thread0 / stream0                 thread1 / stream1
  compute A0                        等待
  record event + CPU yield   ───▶   comm/compute B0
  等待                       ◀───   record event + CPU yield
  compute A1                        等待
```

线程上下文通过 `_THREAD_ID_TO_CONTEXT` 映射，进入 context 时写入当前线程，
退出时清理；因此 hook 必须在 ubatch 线程内调用才会生效。
[VERIFY: vllm_ascend/worker/ubatching.py:86]
[VERIFY: vllm_ascend/worker/ubatching.py:191]

### 8.2 CPU 粒度：任意时刻只能有一个 ubatch Python 线程提交工作

`_cpu_yield()` 的注释直接声明 correctness 要求：一次只能运行一个线程。
函数在唤醒另一个线程后等待自己的 event，再恢复该线程的 forward context。
[VERIFY: vllm_ascend/worker/ubatching.py:126]

这不是普通性能 hint。若 FC2 或 MoE collective 路径漏掉 hook，两个 Python 线程
可能在不同 NPU stream 上交错向同一 HCCL communicator 提交 collective。
即使两个 rank 各自“调用次数相同”，只要顺序不同，也会造成跨 rank 配对错误。

### 8.3 stream 粒度：compute stream 与 comm stream 通过 NPU event 建依赖

每个 ubatch context 同时保存 compute stream 和 comm stream，并为
`ATTN_PRE`、`ATTN_POST`、`MOE_DISPATCH`、`MOE_COMBINE`、`DEFAULT` 分配
compute-done 与 comm-done event。[VERIFY: vllm_ascend/worker/ubatching.py:40]
[VERIFY: vllm_ascend/worker/ubatching.py:245]

`record_current_stream()` 在当前计算流记录 compute-done；
`wait_current_stream_and_yield()` 让当前流等待 event 后执行 CPU yield。
[VERIFY: vllm_ascend/worker/ubatching.py:149]

因此 hook 的两个阶段含义不是简单的 begin/end 日志：

- `is_record=True`：在正确 stream 上建立“前序计算完成”事件；
- `is_record=False`：建立等待关系并把 Python 执行权交给另一个 ubatch。

### 8.4 模型粒度：A2 与 A3 使用不同 overlap template

DeepSeek 在 A3 使用 `DeepseekAlltoallTemplate`，其他设备使用
`DeepseekAllgatherTemplate`。[VERIFY: vllm_ascend/dbo/utils.py:20]

A2 当前测试机是 910B3，所以矩阵只验证 AllGather template。A3 template 中的
MoE dispatch/combine 使用 AllToAll，collective 数量、communicator 和 event
依赖均不同；不能用 A2 结果替代。

### 8.5 算子粒度：五类 hook 是同一个调度协议的五个切点

当前 custom-op 注册表包含：

| custom op | 语义位置 |
|---|---|
| `dbo_linear_column_hook` | column-parallel all-gather 前后 |
| `dbo_linear_row_hook` | row-parallel OProj/reduce-scatter 前后 |
| `dbo_mla_preprocess_hook` | MLA q/kv gather 前后 |
| `dbo_moe_prepare_hook` | MoE dispatch/prepare 通信前后 |
| `dbo_moe_finalize_hook` | MoE combine/finalize 通信前后 |

五个 op 统一由 `_run_dbo_hook()` 在 runtime 读取 forward context 和 template，
fake impl 只原样返回 tensor。[VERIFY: vllm_ascend/ops/register_custom_ops.py:34]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:357]

---

## 九、这次未提交修改的总因果链

### 9.1 现象不是一个 bug，而是三个 contract 同时失效

当前 11 个 tracked 文件的 150 行新增、90 行删除可归为三条主线：

```text
主线 A：compile 可见性
Python 动态 template hook
  → Dynamo 无法稳定 trace
  → compile_guard 只 graph-break，不能保留图内顺序语义
  → 改为 torch custom op：fake 图内占位，runtime 执行 hook

主线 B：shape 口径
logical token / scheduler padded token / TP padded token 混用
  → all-gather、residual、MLA output、最终 concat 的 shape contract 不一致
  → 同时携带 padded slices 与 logical slices
  → 各边界按自己的 contract pad/unpad

主线 C：上下文传播
outer forward context 的 skip_compiled / graph mode / token metadata
  → 未完整复制到每个 ubatch context
  → ubatch 走错 compiled/eager 分支或使用错 token 数
  → 显式传播 skip_compiled 和双份 slice metadata
```

主线 A 的证据是删除 `dbo/compile_guard.py` 并新增五个 custom op；
主线 B/C 的证据集中在 `ascend_forward_context.py`、`model_runner_v1.py`、
`npu_ubatch_wrapper.py`、`mla.py` 和 `linear_op.py`。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:34]
[VERIFY: vllm_ascend/ascend_forward_context.py:231]
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:280]

### 9.2 为什么原 `torch.compiler.disable()` 方案不够

旧方案把每个 Python hook 包在 `@torch.compiler.disable()` 函数中。它能阻止
Dynamo 进入 template 对象，但代价是 graph break；更关键的是，图内 tensor
依赖并没有把 hook 固定在通信前后，compiler 对其顺序约束不可见。

新方案把一个真实 tensor `x` 传入 custom op，并原样返回：

```python
def _run_dbo_hook(x, hook_name, is_record):
    forward_context = get_forward_context()
    if forward_context.dbo_template is not None:
        getattr(forward_context.dbo_template, hook_name)(is_record=is_record)
    return x
```

[VERIFY: vllm_ascend/ops/register_custom_ops.py:34]

图构建时 fake impl 不读取 forward context，只返回 `x`；runtime dispatch 到
PrivateUse1 implementation，再读取 thread-local forward context。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:61]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:368]

tensor 依赖形成如下顺序边：

```text
input tensor
   │
   ▼
dbo_*_hook(input, record=True)
   │
   ▼
HCCL collective / matmul
   │
   ▼
dbo_*_hook(output, record=False)
   │
   ▼
downstream op
```

这解释了为何调用点必须传“通信前真实输入”和“通信后真实输出”，而不能使用
无 tensor 参数的纯 Python side effect。

### 9.3 仍需审查的 custom-op 语义风险

当前实现声明 `mutates_args=[]`，fake 和 runtime 都返回输入 tensor 本身。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:368]

提交前必须确认 `direct_register_custom_op` 对“output alias input”的 schema
生成是否允许，并用 compile/export 测试验证 custom op 不被 DCE 或重排。
现有未跟踪 UT 没有覆盖五个 hook 的 runtime 调用次数、record/false 顺序、
fake tensor、Dynamo fullgraph 或 compiled graph 中是否保留节点。
[VERIFY: tests/ut/ops/test_register_custom_ops.py:1]

建议最少增加：

1. 每个 hook `record=True/False` 各调用一次；
2. `dbo_template=None` 时为 identity；
3. fake tensor propagation shape/dtype 不变；
4. `torch.compile(fullgraph=True)` 后 graph 中仍有 hook op；
5. runtime template 收到严格的 `True → False` 顺序；
6. 两个 ubatch thread 各自读取自己的 forward context。

---

## 十、逐文件代码追踪

### 10.1 `ascend_forward_context.py`：建立双 token 口径

#### 现象

同一 ubatch 同时存在：

- scheduler/graph 为 capture shape 准备的 padded slice；
- attention metadata 表示的 actual token；
- FlashComm 为 TP 整除增加的 pad；
- 最终 API 输出需要的 logical token。

只用 `num_tokens` 与 `pad_size` 两个字段无法区分这些来源，最终 unpad 容易删多
或删少。

#### 当前修改

outer context 新增 `ubatch_slices_logical`，同时继续保存 `ubatch_slices`。
[VERIFY: vllm_ascend/ascend_forward_context.py:60]

创建每个 ubatch context 时：

- FULL graph：优先从 attention metadata 取 actual token；
- 非 FULL：使用传入 slice 的 token 数；
- 再单独写入 `num_tokens_logical`。

[VERIFY: vllm_ascend/ascend_forward_context.py:275]
[VERIFY: vllm_ascend/ascend_forward_context.py:291]

#### 原理

设：

```text
L = logical token count
S = scheduler/graph slice token count
T = tensor-parallel world size
P = ceil(S / T) * T
```

TP 对齐 padding 为：

```text
pad_tp = (-S) mod T
```

最终模型输出若按 `S` 保留，而 API 只需要 `L`，则输出 padding 为：

```text
pad_output = S - L
```

这两个 padding 不是同一个量。`pad_tp` 只保证 collective 可均分；
`pad_output` 用于恢复业务 token 数。当前 wrapper 正是使用
`num_padded - num_logical` 做最终裁剪。
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:265]

#### 风险

`_get_actual_num_tokens()` 的 docstring 声称 attention metadata “always reflects”
真实 token，但实现对 dict 取第一个带字段的 metadata 后立即返回。
[VERIFY: vllm_ascend/ascend_forward_context.py:212]

如果多个 attention group 的 `num_actual_tokens` 不一致，该函数没有一致性断言。
系统修改应把隐含前提变成显式校验，至少在 debug/UT 中确认所有 metadata 值相同。

### 10.2 `model_runner_v1.py`：在 outer context 注入 padded/logical 两套 slices

#### 正式 execute path

模型 runner 现在把 `ubatch_slices_padded` 作为 `ubatch_slices`，
把 `ubatch_slices_attn` 作为 `ubatch_slices_logical`。
[VERIFY: vllm_ascend/worker/model_runner_v1.py:2231]

其目的不是重复存储，而是让：

- compiled/graph tensor 使用稳定 padded shape；
- attention 与最终输出恢复使用 logical shape。

#### dummy/warmup path

dummy run 同样显式传 padded slices，并根据 FULL mode 选择 logical slices。
[VERIFY: vllm_ascend/worker/model_runner_v1.py:3551]

这点决定启动阶段是否与 runtime 使用同一 shape contract。若 dummy capture 与
runtime 对 logical/padded 的解释不同，即使 capture 成功，也会在 replay 时触发
shape 或地址 contract 错误。

### 10.3 `npu_ubatch_wrapper.py`：上下文传播、异常传播和最终合并

#### `skip_compiled` 传播

每个 ubatch context 现在继承 outer context 的 `skip_compiled`。
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:301]

这修复了一个控制面缺口：outer runner 因 encoder input 等条件决定跳过 compiled
路径时，两个 ubatch 不能重新落回 compiled model。

#### 最终 unpad

FlashComm1 且 last PP rank 时，两个结果先 TP all-gather，再分别裁剪
`num_tokens - num_tokens_logical`，最后 concat。
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:260]

裁剪必须逐 ubatch 执行，不能 concat 后只删总 pad；因为 ubatch0 的尾部 pad 位于
concat 中间，若最后统一裁剪，会保留 ubatch0 pad 并误删 ubatch1 的真实 token。

#### 异常传播

当前 runtime thread 已经用 try/except 把异常对象放入 results，join 后调用
`_check_thread_exceptions()`，因此前文第五章关于“完全没有 try/except”的描述
已过时。[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:212]

这里仍需保证 results 中异常与 `(id, output)` 不会直接进入 `sorted()`；
当前检查发生在排序之前，顺序正确。[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:254]

### 10.4 `register_custom_ops.py`：compile/runtime 边界与 residual shard

#### 五个 DBO hook custom op

这是本次 compile 方案核心，负责把动态 Python template 调度放到
PrivateUse1 runtime，同时让 fake propagation 保持 identity。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:34]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:357]

#### residual local shard

旧逻辑依赖 `_EXTRA_CTX.pad_size` 后 `torch.chunk`。新逻辑以当前 local tensor
行数反推 global 所需行数：

```text
local_num_tokens  = x.size(0)
global_num_tokens = local_num_tokens * tp_size
```

residual 不足时补零，再按 `tp_rank * local_num_tokens` 切连续区间。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:65]

这一修改解决 scheduler padding 与 TP padding 来源混杂时，`pad_size` 无法精确
描述 residual 应补到何处的问题。

#### 边界例子

TP=2，local `x` 为 5 行，global residual 只有 8 行：

```text
目标 global shape = 5 × 2 = 10
residual: [0..7] + [zero8, zero9]
rank0: [0..4]
rank1: [5..7, zero8, zero9]
```

现有 UT 对 rank0/rank1 和 shape 相等路径均有覆盖并通过。
[VERIFY: tests/ut/ops/test_register_custom_ops.py:9]

### 10.5 `linear_op.py`：四条 linear 路径统一 hook，但 shape 策略仍分叉

#### MLP column parallel

DBO 开启时，column hook 包围 all-gather，输入 tensor 用于 record，gather 后 tensor
用于 wait/yield。[VERIFY: vllm_ascend/ops/linear_op.py:197]

#### 普通 OProj row parallel

row hook 包围 AllToAll → matmul → reduce-scatter 整段通信计算链。
[VERIFY: vllm_ascend/ops/linear_op.py:260]

`record=False` 放在 reduce-scatter 后而不是 AllToAll 后，意味着 template 把整个
OProj 通信段视为一个不可拆的 overlap phase。

#### FlashComm2 OProj

FC2 路径现在也加入相同 row hook，修复早期缺口。
[VERIFY: vllm_ascend/ops/linear_op.py:384]

其内部仍有两种路径：

- 非 W8A8：先 ODP communication，再 matmul；
- W8A8：通过 quant method 注入 communication；
- 随后 OTP reduce-scatter，必要时 TP all-gather 和 unpad。

hook 必须覆盖两种路径，当前放在 quant 分支之前和所有输出处理之后，覆盖范围完整。
[VERIFY: vllm_ascend/ops/linear_op.py:389]

#### Sequence column parallel

DBO 分支删除了 column op 内的 `maybe_unpad_after_all_gather`。
[VERIFY: vllm_ascend/ops/linear_op.py:481]

这是一项 contract 迁移：该层不再擅自恢复 logical shape，而让 compiled graph
持有 padded shape，最终由更外层处理有效前缀。必须用端到端 shape 测试证明每个
consumer 都接受 padded rows，不能只测该函数局部。

#### Sequence row parallel

reduce-scatter 前 padding 改为按实际 `x.shape[0]` 计算：

```python
pad_size = (-x.shape[0]) % world_size
```

[VERIFY: vllm_ascend/ops/linear_op.py:579]

这比使用全局 `_EXTRA_CTX.pad_size` 更局部、更符合该 collective 的真实输入。
但 `not self.reduce_results` 分支仍直接调用动态 template hook，没有改成 custom op。
[VERIFY: vllm_ascend/ops/linear_op.py:569]

提交前需要确认该分支一定处于 custom-op runtime/compile graph 外；否则仍保留
原 Dynamo 风险。

### 10.6 `mla.py`：custom op 输出 buffer 的 shape 所有权

wrapper 先调用 `_resolve_mla_forward_inputs()`，得到：

- 实际传给 `mla_forward` 的 hidden states；
- custom op 输出 buffer 行数；
- MLA 内部是否 gather q/kv。

[VERIFY: vllm_ascend/ops/mla.py:160]

当前 helper 的规则非常简单：

1. FC1 开、TP>1、VL first layer：输出行数为 input/TP，不 gather q/kv；
2. 其他情况：输出行数等于 input，`need_gather_q_kv=flash_comm_v1_enabled`。

[VERIFY: vllm_ascend/ops/mla.py:174]

这里必须指出：未跟踪 UT 的预期与当前实现冲突。UT 期望普通 FC1 路径根据
forward context 把 4096 行解析为 4 或 8 行，但 helper 完全不读取 context。
[VERIFY: tests/ut/ops/test_register_custom_ops.py:46]
[VERIFY: vllm_ascend/ops/mla.py:174]

### 10.7 `attention/mla_v1.py`：padded output buffer 只写有效前缀

MLA runtime 中，DBO preprocessing hook 仍直接调用 template；它位于
`mla_forward` custom op 的 runtime implementation 内，因此与 linear call site
所处 compile 层级不同。[VERIFY: vllm_ascend/attention/mla_v1.py:1668]

OProj 返回后不再要求 shape 与 output buffer 完全相等，而是：

```python
output.zero_()
output[:o_proj_output.shape[0]] = o_proj_output
```

[VERIFY: vllm_ascend/attention/mla_v1.py:1807]

其 contract 是“graph 拥有固定 padded buffer，runtime 只生产有效前缀”。
先 `zero_()` 很重要：graph replay 会复用 buffer，若不清零，无效尾部可能保留
上一次 replay 数据并污染后续 residual/concat。

风险是如果 `o_proj_output.shape[0] > output.shape[0]`，slice assignment 会失败；
当前没有显式断言。建议在 eager/debug 路径增加 shape assertion，并为
`<`、`==`、`>` 三种关系写 UT。

### 10.8 `prepare_finalize.py`：MoE AllGather 路径接入 custom hooks

prepare hook 包围 MoE prepare communication，finalize hook 包围 reduce-scatter
或 all-reduce。[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:402]
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:587]

这里存在一个需重点复核的分支：prepare 的 `record=False` 当前缩进在
`if flash_comm_enabled` 内，而 `record=True` 在外层 DBO 分支内。
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:405]

若能出现 `dbo_enabled=True` 且 `flash_comm_enabled=False`，则 hook 序列可能只有
record 没有 wait/yield。需要根据 op dispatcher 证明该类只在 flash comm 路径使用，
或把成对 hook 的条件统一。

### 10.9 `token_dispatcher.py`：AllToAllV dispatch/combine 接入 custom hooks

dispatch：

```text
record MOE_PREPARE
  → quant/async_all_to_all
  → handle.wait()
  → wait/yield MOE_PREPARE
```

[VERIFY: vllm_ascend/ops/fused_moe/token_dispatcher.py:488]

combine：

```text
record MOE_FINALIZE
  → async_all_to_all
  → handle.wait()
  → wait/yield MOE_FINALIZE
```

[VERIFY: vllm_ascend/ops/fused_moe/token_dispatcher.py:545]

与 FC2 一样，这里的核心正确性条件是所有 rank、两个 ubatch 对相同 communicator
提交 collective 的顺序一致。custom hook 负责 CPU 交替，HCCL handle 的 wait
负责 device completion，两者不可相互替代。

### 10.10 `ubatching.py`：新增 trace 但尚未接入

`dbo_debug_trace()` 能打印 rank、ubatch、每 context sequence、thread、stream 和
label。[VERIFY: vllm_ascend/worker/ubatching.py:21]

当前仓库中没有调用点，只有定义本身，因此它不会自动产生任何诊断信息。
提交前应二选一：

- 若仅为临时诊断，删除该函数，避免新增未使用生产代码；
- 若作为长期能力，接入受环境变量控制的 hook 边界，不能无条件 print 热路径。

根据项目规范，若新增环境变量，必须集中定义在 `vllm_ascend/envs.py` 并经过评审。

---

## 十一、shape contract 的系统推导

### 11.1 四种 token 数必须命名分离

建议后续代码统一使用以下概念，避免继续把所有量命名为 `num_tokens`：

| 名称 | 符号 | 生产者 | 消费者 |
|---|---:|---|---|
| logical tokens | `L` | scheduler/attention metadata | API output、最终 concat |
| graph tokens | `G` | compile/cudagraph capture size | fixed output buffer |
| TP padded tokens | `P` | collective input alignment | all-gather/reduce-scatter |
| TP local tokens | `Q=P/T` | sequence parallel shard | local linear/MLA |

当前修改的 `num_tokens_logical` 是向该方向迈进，但 `num_tokens` 在 FULL 与非 FULL
模式下仍可能代表不同概念。[VERIFY: vllm_ascend/ascend_forward_context.py:275]

### 11.2 为什么“统一提前 unpad”不成立

compile/graph 需要稳定 shape，而业务语义需要 logical shape。若每个 linear op
通信后立即 unpad：

```text
graph expects G rows
runtime returns L rows
→ downstream compiled add/norm/MLA output contract 改变
```

反之，若全程保留 padding 而最终不逐 ubatch 裁剪：

```text
concat([ubatch0 logical + pad0], [ubatch1 logical + pad1])
→ pad0 位于整体中间，无法只从尾部删除
```

因此当前设计选择：

1. 图内尽量保留 padded buffer；
2. 只写有效前缀，其余清零；
3. 每个 ubatch 输出独立 all-gather；
4. 每个 ubatch 独立按 `S-L` 裁剪；
5. 最后 concat。

[VERIFY: vllm_ascend/attention/mla_v1.py:1807]
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:260]

### 11.3 奇数 token 的 TP2 示例

假设 batch logical token 为 9，拆成：

```text
ubatch0: L0=5
ubatch1: L1=4
TP=2
```

若 scheduler 把两份都 pad 到 6/4：

```text
S0=6, Q0=3, output_pad0=1
S1=4, Q1=2, output_pad1=0
```

正确合并：

```text
gather(u0) -> 6 rows -> remove 1 -> 5
gather(u1) -> 4 rows -> remove 0 -> 4
concat -> 9
```

错误的整体尾部裁剪：

```text
concat -> [u0(5), pad0(1), u1(4)] -> 10
remove last 1 -> [u0(5), pad0(1), u1(3)]
```

当前逐 ubatch 裁剪逻辑正是为避免这个错误。[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:260]

---

## 十二、compile 与 graph startup 的完整调用链

### 12.1 启动阶段不是“只编译不通信”

服务启动包含：

```text
模型加载
  → KV cache profile
  → torch compile / range compile
  → PIECEWISE graph memory profile
  → PIECEWISE graph capture sizes
  → FULL decode graph capture sizes
  → API ready
```

现有矩阵日志显示默认 capture size 多达 34/35 个，并分别执行 PIECEWISE 与 FULL
capture；因此任何位于 model forward 内的 HCCL collective 都会被重复触发。
这解释 FC1/FC2 开启后启动时间为何显著受通信同步影响。

### 12.2 custom op 如何跨越三个执行世界

```text
世界 1：Dynamo trace
  看见 torch.ops.vllm.dbo_*_hook(tensor, bool)
  不进入 Python template

世界 2：FakeTensor / shape propagation
  调用 _dbo_hook_fake
  identity，shape/dtype 不变

世界 3：NPU runtime / graph capture
  PrivateUse1 impl → get_forward_context()
  → 当前 ubatch template → event + CPU yield
```

[VERIFY: vllm_ascend/ops/register_custom_ops.py:34]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:61]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:357]

这套设计的优势是 compile 不再解析动态 template；代价是必须证明 side-effect op
在编译、序列化、cache load、graph capture 和 replay 中都不会被消除或错误复用。

### 12.3 compile cache 风险仍未被本次 diff 修复

前文指出 compiler hash 未包含 DBO、FC1、FC2、TP 和 model architecture。
本次 11 文件 diff 没有修改 compiler interface，所以该问题仍是开放项。

系统修改应把所有会改变 graph 结构、custom op 节点或 shape contract 的配置纳入
cache key，至少包括：

- DBO enable 与 ubatch 数；
- prefill/decode threshold 若影响 capture 路径；
- FC1/FC2 enable；
- TP/DP/EP size；
- model architecture；
- HCCL expansion mode（若生成图或 kernel 选择受影响）；
- custom op schema/version；
- graph mode 与 compile range。

### 12.4 启动优化不能破坏 capture/runtime 等价性

“profile 阶段跳过 collective”虽然能缩短启动，但必须区分：

- 纯内存 profile：可考虑抽象通信结果；
- graph capture：不能随意跳过 runtime 必需 op，否则 capture 图不等价；
- compile fake propagation：只应执行 fake impl；
- warmup replay：必须验证真实 collective 顺序。

因此不能简单以 `in_profile_run` 全局关闭 FlashComm。正确方案应明确每个 startup
阶段的 contract，并为跳过后的 tensor shape、buffer ownership 和 graph replay
建立等价证明。[VERIFY: vllm_ascend/ascend_forward_context.py:108]

---

## 十三、现有实验对修改演进的解释

### 13.1 step1：失败，说明第一版修复不足

`fc1_dbo_compiled_step1` 为 `completed=0, failed=1`。
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_compiled_step1_in4096_out1_np1_c1.json:1]

该结果只能证明当时版本失败，必须结合对应 server log 判断具体根因，不能用最终
代码反推 step1 的失败原因。

### 13.2 step2：单请求通过、并发失败

step2 单请求 1/1 通过，但 multi32 只有 17 完成、15 失败。
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_compiled_step2_in4096_out1_np1_c1.json:1]
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_compiled_step2_multi32_in4096_out1_np32_c16.json:1]

这直接证明 L1 不能当作稳定：单一 4K shape 能通过，不代表 scheduler 形成多轮、
非均匀 ubatch 后仍满足 shape contract。

### 13.3 step3：multi32 全通过

step3 multi32 为 `completed=32, failed=0`。
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_compiled_step3_multi32_in4096_out1_np32_c16.json:1]

结合 diff，step3 所在演进阶段聚焦 padded/logical slice 分离和逐 ubatch unpad；
这与“单请求通过、并发才暴露中间 pad”现象一致，但精确 commit 对应关系仍需用
实验脚本或 git commit 固化，不能只靠文件名推断。

### 13.4 最终 500×96

FC1 DBO default full 为 500/0；FC1+FC2+AIV+DBO default full 也为 500/0。
[VERIFY: testbench/MOE/dbo/results/fc1_dbo_default_full_in4096_out16_np500_c96.json:1]
[VERIFY: testbench/MOE/dbo/results/fc1_fc2_aiv_dbo_default_full_in4096_out16_np500_c96.json:1]

这给当前 DeepSeek/A2/TP2/prefill-only 场景提供 L3 证据，但还缺：

- 输出准确性对照，而不只是 HTTP completion；
- 多轮重复的方差；
- 8K、混合长度、奇偶 token、chunked prefill；
- decode DBO；
- AI_CPU + FC2；
- DP/PP/A3；
- 冷 cache 与热 cache 各自验证。

---

## 十四、当前修改的阻断项与系统修改建议

### 14.1 P0：未跟踪 UT 当前失败

2026-07-04 在远端执行：

```bash
source /data/workspace/.venv-dbo/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd /data/workspace/vllm-ascend
pytest -q tests/ut/ops/test_register_custom_ops.py
```

结果：

```text
3 failed, 3 passed
```

三个失败均首先发生在 mock `_EXTRA_CTX._ctx`：proxy 拒绝该非 extra attribute；
因此测试甚至没有执行到预期的 output token 断言。[VERIFY: tests/ut/ops/test_register_custom_ops.py:46]

系统修复：

1. 不要 patch proxy 私有实现；
2. helper 若设计为纯函数，UT 只传参数并直接断言；
3. 若 helper 应读取 context，就把 token 数作为显式参数传入，避免隐藏全局依赖；
4. 先决定当前实现还是 UT 代表正确 contract，再统一二者；
5. 加入 odd token、VL first layer、FC1 off、TP1/TP2、padded input 的参数化测试。

### 14.2 P0：验证 custom hooks 在 compiled graph 中没有被消除

当前 UT 完全没有测试新注册的五个 op。[VERIFY: tests/ut/ops/test_register_custom_ops.py:1]

需要新增一个最小 template recorder：

```text
calls = []
record(True)  -> append(("row", True))
record(False) -> append(("row", False))
```

分别在 eager、`torch.compile(fullgraph=True)`、graph capture/replay 下验证：

```text
calls == [("row", True), ("row", False)]
```

并导出 graph 检查 custom op 节点仍位于 collective 两侧。

### 14.3 P0：FC2 + AI_CPU 尚未形成稳定结论

早期 crash 与 AIV L3 通过同时存在；矩阵缺少
`DBO=1, FC2=1, AI_CPU`。[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc10_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]

应做严格单因子实验：

```text
固定：同一代码、冷 cache、DeepSeek-V2-Lite、TP2、DBO1、FC10、FC21
变量：HCCL_OP_EXPANSION_MODE=AIV / AI_CPU
请求：L1 → L2 → L3 逐级推进
日志：每次独立 server/test log
判定：最早 device error，不以 EngineCore 后续异常为根因
```

### 14.4 P1：统一 paired hook 条件

每组 `record=True/False` 必须拥有相同控制条件。重点审查
`PrepareAndFinalizeWithAllGather` 中 `record=False` 是否可能因
`flash_comm_enabled=False` 被跳过。[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:402]

推荐封装成结构化 context manager 或单一 helper，避免未来分支修改只移动其中一半。

### 14.5 P1：移除或产品化 `dbo_debug_trace`

函数目前无调用点且无日志级别控制。[VERIFY: vllm_ascend/worker/ubatching.py:21]

若保留，必须：

- 使用集中 env 配置；
- 默认关闭；
- 避免每 token/每层无条件 print；
- 对 rank、ubatch、sequence 使用结构化字段；
- 明确仅用于诊断，不作为同步机制。

### 14.6 P1：扩大正确性测试，而非只看 benchmark 完成数

HTTP `completed=500 failed=0` 证明服务完成请求，不证明输出与非 DBO baseline
数值一致。建议加入：

1. 固定 prompt 与 greedy decode；
2. DBO off/on、FC1 off/on、FC2 off/on；
3. 比较 token IDs，而不是文本模糊匹配；
4. 覆盖 1、2、3、31、32、33、4095、4096、4097 token；
5. 覆盖单请求和混合长度并发；
6. 对 graph replay 连续运行多轮，检查 buffer 尾部清零。

### 14.7 P2：代码质量与提交前清理

当前工作区还包含一个异常未跟踪文件 `tatus --short --branch`，应先确认来源再删除；
不能把它带入提交。`dbo/compile_guard.py` 删除后仍有 `__pycache__`，后者未被 Git
跟踪，不应纳入提交。

tracked 修改应按逻辑拆分审查：

1. compile-safe DBO custom hooks；
2. FC2 hook coverage；
3. padded/logical token contract；
4. MLA output buffer contract；
5. residual/row padding；
6. UT；
7. 临时 trace 清理。

---

## 十五、建议的验证矩阵与完成标准

### 15.1 最小功能矩阵

| 维度 | 取值 |
|---|---|
| DBO | off/on |
| FC1 | off/on |
| FC2 | off/on |
| HCCL | AIV/AI_CPU |
| execution | eager/compile |
| graph | none/PIECEWISE/FULL_AND_PIECEWISE |
| load | 1×1、32×16、500×96 |

不要求一次做笛卡尔积，但每个主修改点必须至少有正向和反向对照，且一次只改变一个
变量。

### 15.2 shape 专项矩阵

```text
token counts: 1, 2, 3, 7, 8, 9, 31, 32, 33, 4095, 4096, 4097
TP: 1, 2（有资源后扩 TP4）
ubatch split:
  even/even
  odd/even
  odd/odd
  scheduler padded
  graph padded
```

每项检查：

- local linear input shape；
- gather 后 shape；
- MLA output buffer shape；
- residual shard shape；
- 每 ubatch logical output；
- concat 后总 logical output。

### 15.3 collective 顺序专项

启用受控 trace 后，每个 rank 应能得到同构序列：

```text
rank0 ub0: MLA_PRE record
rank0 ub1: MLA_PRE record
...
rank0 ub0: ROW record
rank0 ub1: ROW record
...
rank1 ub0: MLA_PRE record
rank1 ub1: MLA_PRE record
...
```

验证的不是 wall-clock 行号完全一致，而是每个 communicator 上 collective 的逻辑
序列一致。trace 中必须同时包含 rank、ubatch、stream、hook、record/wait 和
collective group 标识。

### 15.4 提交门禁

只有以下条件全部满足，才能称这次修改可提交：

- [ ] 新增 UT 全部通过，现有相关 UT 无回归；
- [ ] 五个 custom hook 有 eager + compile 测试；
- [ ] MLA helper 实现与 UT contract 一致；
- [ ] 4K 单请求、multi32、500×96 均通过；
- [ ] DBO 确实触发，有 `should_ubatch=True` 或等价 trace；
- [ ] 输出 token IDs 与 baseline 一致；
- [ ] AIV/AI_CPU 的支持边界被明确记录；
- [ ] 无新增未审查环境变量；
- [ ] 无热路径无条件 print；
- [ ] compile cache 隔离已验证；
- [ ] 文档引用对应当前代码，而不是历史行号。

---

## 十六、最终结论

当前远端已经不是“DBO + FC1/FC2 完全不能与 compile 共存”的状态。现有矩阵证明，
在 DeepSeek-V2-Lite、A2、TP2、prefill-only DBO、双 ubatch、
VLLM_COMPILE + FULL_AND_PIECEWISE、AIV 的固定范围内，FC1、FC2 及其组合均有
500×96 全完成证据。[VERIFY: testbench/MOE/dbo/results/matrix_dsv2_dbo1_fc11_fc21_aiv_in4096_out16_np500_c96_20260630.json:1]

这次未提交修改的核心也不只是“给 FC2 补两个 hook”。它在同时重建：

1. **调度 contract**：两个 ubatch 线程必须通过成对 hook 串行提交、异步执行；
2. **compile contract**：动态 Python template 通过 custom op 留在 runtime；
3. **shape contract**：logical、graph padded、TP padded、TP local 分层；
4. **context contract**：每个 ubatch 继承 graph mode、skip_compiled 和双 slice；
5. **output contract**：固定 graph buffer + 有效前缀 + 逐 ubatch unpad。

但是当前工作区仍不能视为提交完成：新增 UT 实测 3/6 失败，custom hook 缺少直接
测试，MLA helper 与测试预期冲突，FC2+AI_CPU 边界未闭环，debug trace 未接入且未
清理。下一步应先完成第十四、十五章的 P0 门禁，再讨论性能优化和 commit 拆分。
[VERIFY: tests/ut/ops/test_register_custom_ops.py:46]
