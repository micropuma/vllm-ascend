# `kv_cache_memory_bytes` 最小化测试集

这个目录提供了一组最小化的手工测试，用来验证 Ascend 单卡场景下
`kv_cache_memory_bytes` fast path 的关键行为。

## 为什么说它是“最小化”

这 3 个 case 已经覆盖了最核心、最有信号的行为，不需要引入更大的模型矩阵：

1. `kv_cache_memory_bytes` 会忽略 `gpu_memory_utilization`。
2. 更大的 `kv_cache_memory_bytes` 会带来更大的 KV token 容量。
3. 当 `kv_cache_memory_bytes` 太小且 `max-model-len` 很大时，会在启动早期失败。

这已经足够验证 fast path 本身，避免把多实例 profiling、TP/PP 配比、KV cache dtype
覆盖等无关因素混进来。

## Ascend 910B 64GB 推荐模型

这个测试场景下，模型不是越大越好。

建议选择同时满足下面条件的模型：

- 足够大，能覆盖真实初始化流程；
- 足够小，在 64GB 的 910B 卡上还能留出明确的 KV cache 空间。

默认推荐：

- `Qwen/Qwen3-8B`

原因：

- 7B/8B 级别的 dense 模型，通常是 64GB 卡上验证 `kv_cache_memory_bytes`
  的最佳平衡点。
- 模型太大时，显存大多被权重占用，KV sizing 的信号会变弱。
- 模型太小时，测试依然有效，但对真实部署的代表性会降低。

## 快速开始

`run_simple.py` 直接使用 `from vllm import LLM, SamplingParams` 做一次 offline inference，
不再启动 `vllm serve`：

```bash
python tutorial/kv-memory-bytes/run_simple.py \
  --model Qwen/Qwen3-8B \
  --dtype bfloat16
```

本地模型目录示例：

```bash
python tutorial/kv-memory-bytes/run_simple.py \
  --model /data/models/Qwen3-8B \
  --dtype bfloat16
```

如果要直接跑 3 个手工 case，用：

```bash
bash tutorial/kv-memory-bytes/run_three_cases.sh
```

## 测试项

### Case 1：固定 KV bytes，忽略 `gpu_memory_utilization`

`run_three_cases.sh` 会调用两次 `run_simple.py`，`kv_cache_memory_bytes` 相同，
`gpu_memory_utilization` 不同。

预期结果：

- 两次日志都包含 `skipping memory profiling`；
- 两次日志里的 `GPU KV cache size: ... tokens` 完全一致。

### Case 2：更大的 KV bytes 带来更大的 token 容量

`run_three_cases.sh` 会调用两次 `run_simple.py`，分别设置两个不同的
`kv_cache_memory_bytes`。

预期结果：

- 更大的配置会输出更大的 `GPU KV cache size: ... tokens`。

### Case 3：KV bytes 太小时，大 `max-model-len` 会提前失败

`run_three_cases.sh` 会调用一次 `run_simple.py`，使用较小的
`kv_cache_memory_bytes` 和较大的 `max-model-len` 做离线推理初始化。

预期结果：

- 在正常推理前初始化失败；
- 日志中能看到 KV cache 内存不足的报错信息。

## 说明

- `kv_cache_memory_bytes` 是按卡计算，不是按节点计算。
- `kv_cache_memory_bytes` 会绕过基于 `gpu_memory_utilization` 的 KV sizing，
  但不会绕过启动阶段的分配检查和模型预热。
- 实际分配的 KV bytes 会按 cache block 向下取整，因此应以日志中的 token
  容量为准，不要假设字节数会精确相等。
