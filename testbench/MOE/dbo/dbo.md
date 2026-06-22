# DBO（Dual Batch Overlap）机制说明

DBO 将一个大 batch 拆成两个 microbatch，分别运行在两条 NPU stream 上，通过 GPU event 同步点让一个 microbatch 的通信（AllGather / AllToAll）与另一个 microbatch 的计算（Attention / MoE 矩阵乘）在硬件层面并发执行，从而隐藏通信延迟。

---

## 调用链

```
LLM(enable_dbo=True)
│
├─ [配置解析]
│   vllm/config/parallel.py:194  ParallelConfig.enable_dbo
│   vllm/config/parallel.py:492  ParallelConfig.use_ubatching (property)
│   vllm/config/parallel.py:496  ParallelConfig.num_ubatches  (property, 固定为 2)
│
├─ [每次 forward 入口]
│   vllm_ascend/worker/model_runner_v1.py:2583  _determine_batch_execution_and_padding()
│   │   内部调用 _sync_metadata_across_dp() (line 562) 计算 should_ubatch
│   │
│   ├─ [触发检查] ──────────────────────────────────────────────────────────────────┐
│   │   vllm_ascend/worker/ubatch_utils.py:68    check_enable_ubatch()             │
│   │   vllm/v1/worker/ubatch_utils.py:38        check_ubatch_thresholds()         │
│   │       条件 1: parallel_config.enable_dbo == True                             │
│   │       条件 2: num_tokens >= dbo_prefill_token_threshold (默认 512)            │
│   │       条件 3: moe_comm_type != MC2                                            │
│   │       条件 4: padding 后第二个 ubatch 非空                                    │
│   │                                                                              │
│   └─ [切分 batch] ─────────────────────────────────────────────────────────────── ┘
│       vllm_ascend/worker/ubatch_utils.py:96    maybe_create_ubatch_slices()
│           按 token 数对半切成 ubatch_slices[0], ubatch_slices[1]
│
├─ [构造 UBatch 上下文]
│   vllm_ascend/worker/npu_ubatch_wrapper.py:255 _make_ubatch_metadata()
│   │
│   ├─ [选择重叠模版]
│   │   vllm_ascend/dbo/utils.py:14              select_dbo_templates()
│   │       A2 + DeepSeek → DeepseekAllgatherTemplate
│   │       A3 + DeepSeek → DeepseekAlltoallTemplate
│   │       (其他模型见 dbo/overlap_templates/)
│   │
│   └─ [创建双 stream 上下文]
│       vllm_ascend/worker/ubatching.py:208      make_ubatch_contexts()
│           comm_stream  ← 已有的辅助流
│           compute_stream ← torch.npu.current_stream()
│           每个 AscendUBatchContext 持有一对 cpu_event + gpu_event 字典
│
├─ [并发执行]
│   vllm_ascend/worker/npu_ubatch_wrapper.py:194 _run_ubatches()
│       Thread 0 → ubatch[0].context (compute_stream)  → model forward
│       Thread 1 → ubatch[1].context (comm_stream)     → model forward
│       两线程通过 Barrier 同步后交替唤醒，由 GPU event 保证正确的硬件依赖
│
└─ [层内重叠钩子] (以 A2 DeepSeek 为例)
    vllm_ascend/dbo/overlap_templates/deepseek.py:7  DeepseekAllgatherTemplate
    │
    ├─ dbo_moe_finalize_hook(is_record=True)   → record ATTN_PRE event
    │   ubatch[0] MoE finalize AllGather 入队后打标
    │
    ├─ dbo_mla_preprocess_hook(is_record=False) → wait ATTN_PRE, then yield
    │   ubatch[1] 等 ubatch[0] 的 finalize 入队，随即自己推进 MLA preprocess AllGather
    │   → 两者在 NPU 上并发
    │
    ├─ dbo_linear_row_hook(is_record=True)      → record ATTN_POST event
    │   ubatch[0] post-MLA linear_row 结束后打标
    │
    └─ dbo_moe_prepare_hook(is_record=False)    → wait ATTN_POST, then yield
        ubatch[1] 等 ubatch[0] 的 ATTN_POST，随即自己推进 MoE prepare AllGather
        → 两者在 NPU 上并发

    底层同步原语:
    vllm_ascend/worker/ubatching.py:130  AscendUBatchContext.record_current_stream()
    vllm_ascend/worker/ubatching.py:138  AscendUBatchContext.wait_current_stream_and_yield()
```

---

## 触发条件

DBO 在每次 forward 时动态决定是否启动，需要**同时满足**以下 4 个条件：

### 条件 1：启用开关

```python
# LLM() 或 vllm serve 参数
enable_dbo=True
```

对应配置字段：[`ParallelConfig.enable_dbo`](../../../vllm/config/parallel.py#L194)

---

### 条件 2：batch token 数超过阈值

```python
# vllm/v1/worker/ubatch_utils.py:38
def check_ubatch_thresholds(config, num_tokens, uniform_decode):
    if uniform_decode:
        return num_tokens >= config.dbo_decode_token_threshold   # 默认 32
    else:
        return num_tokens >= config.dbo_prefill_token_threshold  # 默认 512
```

**注意**：当前 ascend model runner 中 `uniform_decode` 始终传 `False`，因此无论 prefill 还是 decode 阶段，门槛都是 **512 tokens**。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `dbo_prefill_token_threshold` | 512 | 含 prefill 的 batch 触发门槛 |
| `dbo_decode_token_threshold` | 32 | 纯 decode batch 触发门槛（当前实际未走此分支） |

调低门槛用于调试：
```python
LLM(
    ...,
    enable_dbo=True,
    override_neuron_config={
        "dbo_prefill_token_threshold": 64,
    },
)
```

---

### 条件 3：MoE 通信模式不为 MC2

```python
# vllm_ascend/worker/ubatch_utils.py:84
if moe_comm_type == MoECommType.MC2:
    return False  # DBO 与 MC2 互斥
```

A2 上 MC2 触发条件（满足则 DBO 不可用）：

```
enable_expert_parallel=True
AND ep_world_size >= 16
AND num_experts_per_device <= 24
AND num_tokens <= mc2_tokens_capacity
```

A2 小规模 TP=2、EP=2 时走 ALLGATHER，DBO 可以共存。

---

### 条件 4：padding 后第二个 ubatch 非空

```python
# vllm_ascend/worker/ubatch_utils.py:13
def is_last_ubatch_empty(orig_num_tokens, padded_num_tokens, num_ubatches=2):
    return (padded_num_tokens // num_ubatches) * (num_ubatches - 1) >= orig_num_tokens
```

当 padding 过多导致所有真实 token 都落入第一个 ubatch 时，DBO 自动退化为单 batch。

---

## 重叠生效的额外要求

仅触发 DBO 不够，要在 profiler 里看到真正的硬件并发还需要：

1. **`HCCL_OP_EXPANSION_MODE=AI_CPU`**：让 HCCL 通信 kernel 跑在 AI_CPU 核而非 AI Core 上，否则通信和计算会争同一批 AI Core，无法真正并发。

2. **核数分配**（可选调优）：
   ```bash
   VLLM_ASCEND_DBO_COMM_AIC_NUM=N   # 给通信分配的 AI Cube 核数
   VLLM_ASCEND_DBO_COMM_AIV_NUM=N   # 给通信分配的 AI Vector 核数（HCCL 需要 >= 16）
   ```
   对应代码：[`vllm_ascend/worker/ubatching.py:60`](../../../vllm_ascend/worker/ubatching.py#L60)

3. **profile 工具需抓取多流**：用 `torch_npu.profiler` 时确认 `experimental_config` 开启了所有 stream 的追踪。

---

## 快速验证 DBO 是否触发

在日志中搜索：

```bash
# 启动时开启 DEBUG 日志
VLLM_LOGGING_LEVEL=DEBUG python your_script.py 2>&1 | grep -E "should_ubatch|dbo_enabled|ubatch_slices"
```

`should_ubatch: True` 且 `ubatch_slices` 非 `None` 说明 DBO 已触发。
