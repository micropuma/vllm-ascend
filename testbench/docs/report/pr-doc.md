# vLLM-Ascend DBO + FlashComm PR 文档

## 1. DBO 参数支持

### 1.1 顶层开关

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `enable_dbo` | `ParallelConfig` (bool) | `False` | DBO 总开关，通过 `LLM(enable_dbo=True)` 传入 |
| `dbo_prefill_token_threshold` | int | 512 | 触发 DBO 的最小 token 数 |
| `dbo_decode_token_threshold` | int | 32 | Decode 触发阈值（当前未使用） |

### 1.2 通信流 Core 分配参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_ASCEND_DBO_COMM_AIC_NUM` | `-1`（全部） | 通信 stream 独占的 AI Cube 核数 |
| `VLLM_ASCEND_DBO_COMM_AIV_NUM` | `-1`（全部） | 通信 stream 独占的 AI Vector 核数，HCCL 要求 ≥16 |

> 代码位置: `vllm_ascend/envs.py:113-119`, `vllm_ascend/worker/ubatching.py:78-84`

### 1.3 触发条件（全部满足才启用）

1. `enable_dbo=True`
2. `num_tokens >= dbo_prefill_token_threshold`（默认 512）
3. MoE 通信类型非 MC2（EP≥16 时触发 MC2，与 DBO 互斥）
4. Padding 后第二个 micro-batch 非空

> 代码位置: `vllm_ascend/worker/ubatch_utils.py:68 check_enable_ubatch()`

### 1.4 支持模型架构

| 架构 | A2 (AllGather) | A3 (AllToAll) |
|---|---|---|
| DeepSeek-V2/V3 (MLA+MoE) | `DeepseekAllgatherTemplate` | `DeepseekAlltoallTemplate` |
| Qwen3 MoE | `QwenMoEAllgatherTemplate` | `QwenMoEAlltoallTemplate` |
| GLM-4 MoE | `Glm4MoEAllgatherTemplate` | `Glm4MoEAlltoallTemplate` |
| Bailing MoE V2.5 | `BailingMoEV25AllgatherTemplate` | `BailingMoEV25AlltoallTemplate` |
| Qwen3 Dense | `QwenDenseAllgatherTemplate` | `QwenDenseAlltoallTemplate` |
| GLM MoE DSA (MLA+MoE) | `GlmMoeDsaAllgatherTemplate` | `GlmMoeDsaAlltoallTemplate` |

> 代码位置: `vllm_ascend/dbo/utils.py select_dbo_templates()`, `vllm_ascend/dbo/overlap_templates/`

---

## 2. 测试覆盖

### 2.1 单元测试

| 测试文件 | 覆盖内容 |
|---|---|
| `tests/ut/worker/a2/test_model_runner_v1_with_device.py` | DBO 触发条件判断 (`should_ubatch`)、padding 后 ubatch 非空检查 |
| `tests/ut/ops/test_prepare_finalize.py` | MoE prepare/finalize 中 DBO hook 调用的正确性 |
| `tests/ut/ops/test_flashcomm2_oshard_manager.py` | FC2 O-Shard 管理器的 DBO 兼容性 |
| `tests/ut/ops/test_register_custom_ops.py` | 5 个 DBO hook 自定义算子的注册和调用 |

### 2.2 E2E 测试

| 测试配置 | 覆盖维度 | 状态 |
|---|---|---|
| TP=2 (单机 2 卡) | 基础 TP + DBO + FlashComm 组合 | ✅ |
| DP=2 (单机 2 卡, TP=1) | DP + DBO 一致性（`_sync_metadata_across_dp`） | ✅ |
| TP=2 + compile fullgraph | DBO 多线程 + torch.compile 兼容性 | ✅ |
| TP=2 + ACLGraph capture | DBO + cudagraph 兼容性 | ✅ |
| FC1=1 + DBO=1 | FlashComm1 通信与 DBO overlap 的集成 | ✅ |
| FC2=1 + DBO=1 (AIV) | FlashComm2 AllToAll 与 DBO 的集成 | ✅ |

> 测试入口: `tests/e2e/pull_request/two_card/test_flashcomm_distributed.py`

### 2.3 Benchmark 测试（testbench）

| 脚本 | 功能 |
|---|---|
| `testbench/MOE/dbo/demos/auto_benchmark.sh` | 全自动 10 组配置矩阵 benchmark |
| `testbench/MOE/dbo/demos/bench.sh` | 单次 A/B 对比 benchmark |
| `testbench/MOE/dbo/demos/deepseek-v2-dbo-test.sh` | DBO 客户端压测脚本 |

---

## 3. 与 FlashComm 算子库的集成

### 3.1 FlashComm1

| 集成点 | 文件 |
|---|---|
| Column-parallel linear (AllGather 前后 hook) | `vllm_ascend/ops/linear_op.py` |
| Row-parallel linear (ReduceScatter 前后 hook) | `vllm_ascend/ops/linear_op.py` |
| MLA preprocess (MLA AllGather 前后 hook) | `vllm_ascend/attention/mla_v1.py` |

### 3.2 FlashComm2

| 集成点 | 文件 |
|---|---|
| MoE prepare/finalize AllGather/ReduceScatter (A2) | `vllm_ascend/ops/fused_moe/prepare_finalize.py` |
| MoE dispatch/combine AllToAll (A3) | `vllm_ascend/ops/fused_moe/token_dispatcher.py` |

---

## 4. torch.compile + 通信优化的兼容性工作

DBO 通过两个 CPU 线程交替执行两个 micro-batch 的 forward。`torch.compile(fullgraph=True)` 的 Dynamo trace 发生在**第一次调用 `model(...)` 时** — 此时恰好在 DBO 子线程内部，runtime 状态（forward context、分布式 group 等）处于不稳定状态。**不开 FlashComm 时无此问题**（Dynamo 将通信分支裁剪掉），**开了 FlashComm 后 Dynamo 进入通信代码路径，大量 runtime 状态读取全部炸开。**

针对此冲突，识别出三类 bug 并分别修复：

### 4.1 类型 A：FakeImpl 读取 runtime 状态（compile_all_ranges 阶段）

**触发时机**：`PiecewiseBackend.compile_all_ranges()` 用 FakeTensor 重放 FX graph 推断 output shape 时，FakeImpl 内部读取了 `_EXTRA_CTX` / `get_forward_context()`，但此时无 active forward context。

**炸点**：
- `register_custom_ops.py` FakeImpl 中 `get_forward_context()` → AssertionError
- `get_tensor_model_parallel_world_size()` → `get_tp_group()` → assert `_TP is not None`

**修复**：建立 compile-safe snapshot 机制。在 `set_ascend_forward_context` 时将 runtime 变量（`flash_comm_v1_enabled`、`tp_world_size` 等）写入 `_SNAPSHOT`，FakeImpl 读取 snapshot 而非 runtime context。

> 详见 `rfc/rfc-custom-op-fakeimpl-compile-safe.md`，关联 commit `98446bf8`

### 4.2 类型 B：Real forward 读取 runtime 状态（fullgraph trace 阶段）

**触发时机**：Dynamo trace 遍历 forward 函数时，遇到 runtime 状态读取。

**炸点**：
- `mla.py:forward` 中 `_EXTRA_CTX.flash_comm_v1_enabled` → context 为空
- `linear_op.py:apply_impl` 中 `get_forward_context()` → assert
- `deepseek_v2.py:forward` 中 `get_pp_group()` → assert `_PP is not None`

**修复**：
- `mla.py`：`flash_comm_v1_enabled` 静态化为 `self.flash_comm_v1_enabled = enable_sp()`
- `linear_op.py`：`apply()` 加 `@torch.compiler.disable`，跳过 Dynamo trace
- `deepseek_v2.py`：upstream 代码，`get_pp_group()` 在 trace 时 `_PP is None`（PP 未启用时此分支不应进入，需 upstream 适配）

### 4.3 类型 C：DBO Hook 中的线程操作（fullgraph trace 阶段）

**触发时机**：`forward_context.dbo_enabled = True`（profiler run 的残留值），Dynamo trace 进入 DBO hook（`dbo_linear_column_hook` 等），hook 内部调用 `threading.get_ident()` — Dynamo 无法 trace 线程操作。

**修复**：所有 DBO hook 的注册函数加 `@torch.compiler.disable`，使 Dynamo 完全跳过 DBO 同步代码。

### 4.4 修复总结

| Bug 类型 | 触发阶段 | 修复方式 | 文件 |
|---|---|---|---|
| A: FakeImpl 读 runtime | `compile_all_ranges` | Compile-safe snapshot | `register_custom_ops.py` |
| B: Forward 读 runtime | fullgraph trace | 静态属性 + `@torch.compiler.disable` | `mla.py`, `linear_op.py` |
| C: Hook 线程操作 | fullgraph trace | `@torch.compiler.disable` | `ubatching.py` |

### 4.5 当前状态

| 组合 | compile fullgraph | compile piecewise | eager | 备注 |
|---|---|---|---|---|
| DBO + FC1 | ✅ 可用 | ✅ 可用 | ✅ 可用 | 存在已知 shape contract 问题（compile range 边界） |
| DBO + FC2 (AIV) | ✅ 可用 | ✅ 可用 | ✅ 可用 | AIV 模式正常 |
| DBO + FC2 (AI_CPU) | ❌ crash | ❌ crash | ❌ crash | CANN 9.0.0 AllToAll bug，非 compile 问题 |
| DBO only (无 FC) | ✅ 可用 | ✅ 可用 | ✅ 可用 | 无已知问题 |

> 详细分析见 `testbench/MOE/dbo/dbo-torch-compile-compat.md`

---

## 5. 尚未完成

### 4.1 TP + DP + PP 组合

- [ ] TP + DP 组合：DP 下 `_sync_metadata_across_dp` 已验证基础功能，但大规模 DP(>4) + TP + DBO 的稳定性待验证
- [ ] PP (Pipeline Parallel) + DBO：当前 DBO 仅作用于单层 forward，PP 下各 stage 的 DBO 触发一致性待验证
- [ ] TP + DP + PP 全组合：无测试覆盖

### 4.2 AI_CPU + FlashComm2

- [ ] `HCCL_OP_EXPANSION_MODE=AI_CPU` + FC2 AllToAll 已知 crash（CANN 9.0.0 bug）
- [ ] 根因：`Flashcomm2OProjRowParallelOp` 的 ODP AllToAll 未接入 DBO row hook
- [ ] 修复方案：待实现和 NPU 验证（详见 `rfc/rfc-dbo-flashcomm2-aicpu-alltoall.md`）
- [ ] 当前 workaround：FC2 必须搭配 `HCCL_OP_EXPANSION_MODE=AIV`

### 4.3 DeepSeek Shared Expert 融合

- [ ] Shared Expert 的 AllReduce 通信尚未接入 DBO hook
- [ ] 可能通过新增 `dbo_shared_expert_hook` 实现 overlap
- [ ] 预估收益：Shared Expert 通信 ~1200μs/层，overlap 后可进一步改善 prefill latency

---

## 6. 参考资料

- `testbench/MOE/dbo/dbo.md` — DBO 机制详细说明
- `testbench/MOE/dbo/dbo-torch-compile-compat.md` — DBO + torch.compile 兼容性分析
- `testbench/MOE/dbo/rfc/` — 各 RFC 文档
- `testbench/MOE/dbo/docs/` — FlashComm / Compilation / Profiling 分析文档
- `vllm_ascend/dbo/` — DBO 源码
- `vllm_ascend/worker/ubatching.py` — 同步原语
- `vllm_ascend/worker/npu_ubatch_wrapper.py` — Stream 管理
