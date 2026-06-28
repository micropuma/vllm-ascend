# RFC: vLLM-Ascend + FlashComm1 启动时延分析与治理

- 状态：Draft
- 日期：2026-06-28
- 适用版本：vLLM 0.22.1 + 当前 vLLM-Ascend main
- 实测硬件：2 x Ascend 910B3
- 模型：DeepSeek-V2-Lite-Chat

## 1. 摘要

在 TP=2、EP、DBO、torch.compile、ACL graph FULL_AND_PIECEWISE 配置下，开启 FlashComm1 后，服务从命令启动到 API ready 约从 190 秒增加到 508 秒。

关键卡点不是模型加载，也不是 torch.compile cache miss，而是 ACL graph capture：

| 配置 | compile | graph capture | Engine 初始化 | 启动到 API ready |
|---|---:|---:|---:|---:|
| FC1 off | 16.84 s，冷编译 | 71 s | 105.15 s | 约 190 s |
| FC1 on | 3.18 s，cache hit | 407 s | 430.25 s | 约 508 s |

FC1-on 即使命中 compile cache，仍比 FC1-off 慢约 318 秒；graph capture 本身多 336 秒。结论是：当前启动慢的 P0 瓶颈为“FlashComm1 通信路径被对每个 graph shape 重复 warmup 和 capture”，不是 Dynamo/Inductor 编译。

这也不是 DBO ubatch 自身导致的启动开销。本次 decode DBO threshold 设置为 1,000,000,000，capture size 最大为 256，启动 capture 不触发 decode ubatch；日志中没有 `should_ubatch: True`。DBO 是部署配置的一部分，但直接慢点是 FC1 + ACL graph 多档 capture。

## 2. 实测配置与复现方法

两组只切换 FC1，显式关闭 FC2，避免把 FC2 AllToAll/AICPU 问题混入：

```bash
# FC1 off
VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0 VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0 PORT=8011 LOG_STATS=0 bash testbench/MOE/dbo/demos/deepseek-v2-dbo-server.sh

# FC1 on
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0 VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0 PORT=8012 LOG_STATS=0 bash testbench/MOE/dbo/demos/deepseek-v2-dbo-server.sh
```

公共参数：

- TP=2，DP=1，EP enabled
- `--enable-dbo`
- `--all2all-backend deepep_low_latency`
- `--dbo-prefill-token-threshold 1024`
- `--dbo-decode-token-threshold 1000000000`
- `--max-model-len 8192`
- `--max-num-batched-tokens 16384`
- `--max-num-seqs 256`
- `enforce_eager=False`
- graph mode `FULL_AND_PIECEWISE`
- graph warmup count 1

原始日志：

- `/data/workspace/logs/startup-tp2-dbo-compile-fc1off-20260628.log`
- `/data/workspace/logs/startup-tp2-dbo-compile-fc1on-20260628.log`

说明：FC1-off 是该配置的首次冷编译，FC1-on 已有缓存。因此不能用这两次运行比较 FC1 对 compile 时间的影响；但这组条件反而更有力地排除了 compile 是 FC1-on 启动慢主因。

## 3. 启动时间线

### 3.1 FC1 off

```text
15:27:02 wrapper start
15:28:03 worker starts loading model       +61 s
15:28:23 weights loaded                    17.30 s
15:28:42 torch.compile finished            16.84 s (cold)
15:28:49 graph memory profiling starts
15:28:54 graph memory estimate finished     5 s
15:30:07 graph capture finished            71 s
15:30:09 engine initialization finished   105.15 s
15:30:12 API server starts
```

capture 细分：

- PIECEWISE：35 档，约 34 秒，约 1 秒/档
- FULL：35 档，约 35 秒，约 1 秒/档

### 3.2 FC1 on

```text
15:31:20 wrapper start
15:32:21 worker starts loading model       +61 s
15:32:35 weights loaded                    11.15 s
15:32:40 torch.compile finished             3.18 s (cache hit)
15:32:49 graph memory profiling starts
15:32:56 graph memory estimate finished     7 s
15:39:45 graph capture finished           407 s
15:39:46 engine initialization finished   430.25 s
15:39:49 API server starts
```

capture 细分：

- PIECEWISE：34 档，78 秒；前几档约 1.4 秒，后续约 3 秒/档
- FULL：34 档，约 327 秒；首档约 5.5 秒，随 shape 增大升到约 12 秒/档
- EngineCore 每 60 秒打印一次 shared-memory broadcast 等待提示，FC1-on 出现 6 次，FC1-off 仅 1 次。这是长 capture 的结果，不是根因。

FC1 启用后 graph size 从 35 变为 34，是 `platform.py` 调用 `update_sizes_for_sequence_parallelism()` 后只保留 TP=2 可整除的 size，移除了 size=1。虽然少 capture 两张图，时间仍增加 5.7 倍。

## 4. 代码级根因推导

### 4.1 每个 MoE forward 都启用 FC1

`vllm_ascend/ascend_forward_context.py`：

```python
if is_context_moe_model:
    flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
```

DeepSeek-V2 是 MoE。与 dense 模型的 `num_tokens > 1000` 不同，MoE 路径没有 token threshold。只要 FC1 配置打开且 dummy batch 有 token，每个 capture shape（2 到 256）都进入 FC1 路径。

### 4.2 FC1 改变线性层和 MoE 通信拓扑

`vllm_ascend/ops/linear_op.py` 的 dispatcher 在 `enable_sp()` 时将相关层替换为：

- `SequenceColumnParallelOp`：必要时做 TP all-gather
- `SequenceRowParallelOp`：使用 reduce-scatter 或融合 MM+RS，而不是普通 all-reduce

`vllm_ascend/ops/fused_moe/prepare_finalize.py` 在 FC1 下执行：

- MoE prepare：TP/EP all-gather，再 unpad
- MoE finalize：pad/prepare 后 TP/EP reduce-scatter

这些不是一次性的 communicator 初始化，而是模型每层 forward 的实际通信。DeepSeek-V2 多层 MoE 会把单次 dummy forward 中的 collective 数量放大。

### 4.3 vLLM 对每个 shape 执行一次 warmup 加一次 capture

上游 `vllm/v1/worker/gpu_model_runner.py::_warmup_and_capture()`：

```python
for _ in range(cudagraph_num_of_warmups):  # 当前为 1
    self._dummy_run(..., cudagraph_runtime_mode=NONE)
self._dummy_run(..., is_graph_capturing=True)
```

当前有两类 graph：

- mixed prefill-decode PIECEWISE：34 档
- uniform decode FULL：34 档

因此正式 capture 阶段至少执行：

```text
68 shapes x (1 warmup forward + 1 capture forward) = 136 full model forwards
```

此外 `profile_cudagraph_memory()` 对每种 mode 的前两张图再次执行 warmup + capture，随后清空临时图；这部分本次仅 7 秒，不是主项，但它确实是重复工作。

### 4.4 FULL graph 是最大热点

FC1-off 时每档约 1 秒且基本恒定；FC1-on 时 FULL 从约 5.5 秒/档增长到约 12 秒/档。该 shape-dependent 增长与通信 tensor 的 token 维增长一致：all-gather/reduce-scatter 搬运量随 batch token 数增加。

因此，407 秒不是 Python 循环本身造成，而是 34 个 FULL shape 上反复执行包含 FC1 collectives 的完整模型 warmup/capture。PIECEWISE 也受影响，但只贡献 78 秒；FULL 贡献约 80%，是首要治理对象。

### 4.5 为什么不是 torch.compile cache

- FC1-off 冷编译：16.84 秒
- FC1-on cache hit：3.18 秒
- FC1-on graph capture：407 秒

即使把 compile 降到 0，FC1-on 仍需约 505 秒启动；优化 compile cache 最多节省个位到十几秒，不能解决该问题。

### 4.6 为什么不是 DBO capture

`_capture_cudagraphs()` 只有同时满足以下条件才允许 microbatching：

```text
use_ubatching
and mode == FULL
and uniform_decode
and check_ubatch_thresholds(...)
```

本次 decode threshold 为 1e9，capture size 最大 256，因此 `allow_microbatching=False`。启动日志也没有实际 `should_ubatch: True`。所以不要把 407 秒归因于两个 ubatch 线程或 ubatch shape contract。

## 5. 问题定义

当前行为在正确性上不一定错误：graph 必须捕获与运行时一致的通信语义，不能简单在 capture 时关闭 FC1，否则 replay graph 与真实执行拓扑不同。

真正的问题是默认策略的启动成本失控：

1. 默认生成 34 个 PIECEWISE 加 34 个 FULL shape。
2. 每个 shape 都有 warmup 和 capture 两次完整 forward。
3. MoE 的 FC1 对所有 token size 生效。
4. FULL graph capture 对 FC1 通信开销高度敏感，并随 shape 增长。
5. ACL graph 当前不能像 compile artifact 一样跨进程直接复用，每次服务启动都重复 407 秒。

## 6. 修复与优化方案

### P0：先提供可控的 graph shape 策略

允许部署显式缩减 `cudagraph_capture_sizes`，并针对 FC1 给出推荐档位，而不是默认捕获 2..256 的 34 档。比如根据线上并发选择少量高频 size，并依赖 padding/回退覆盖其他 batch。

验收必须同时包含：

- API ready 时间
- decode/prefill 吞吐与 TPOT
- 未命中 graph shape 时的 fallback 比例
- 输出准确性
- graph memory

这是当前最可落地、风险最低的优化。若从 34 档缩到 8 档，按实测单档成本，FULL capture 可近似线性下降，但具体收益必须实测。

### P0：评估 FC1 下禁用 FULL、保留 PIECEWISE

本次 FULL 约 327 秒，PIECEWISE 约 78 秒。对 mixed/prefill 主导服务，可评估 `PIECEWISE` 模式；对 decode 性能敏感服务，需要用 benchmark 判断失去 FULL replay 的代价。

不能直接把该方案设为全局默认，因为它可能牺牲 decode 性能。建议形成按 workload 选择的启动参数，而不是硬编码。

### P1：减少重复的 graph memory profiling capture

`profile_cudagraph_memory()` 对每种 mode 的前两档先捕获临时图，估算后全部清空，随后正式 capture 再做一遍。本次仅约 7 秒，但在更大模型或更多通信时可能放大。

可评估：

- 复用 profiling 阶段捕获的图，而不是 clear 后重捕获；
- 允许已知部署配置关闭 graph memory estimate；
- 对 FC1 使用保守静态估算。

需保证 KV cache 容量估算不回归，不能只为启动速度跳过内存安全检查。

### P1：让 capture size 由运行负载驱动

固定 34 档是通用策略，不等于业务最优。建议支持：

- 从历史 batch-size 分布生成 capture sizes；
- 只捕获覆盖主要流量的分位点；
- 服务 ready 后低优先级 lazy capture 其余 shape；
- lazy capture 必须处理多 rank 一致性和请求并发隔离。

### P2：研究 ACL graph 持久化

compile graph 已能 cache hit，但 ACL/NPU graph 每次重建。若 CANN/torch_npu 支持安全序列化或稳定重建元数据，可按以下因素建立 cache key：

- 模型与权重版本
- TP/DP/EP 拓扑
- FC1/FC2/DBO 配置
- graph mode 和 capture sizes
- CANN、torch_npu、vLLM、vLLM-Ascend 版本
- dtype、quantization、max model len 等

这是收益最大的长期方案，也是实现和兼容风险最高的方案。

### 不建议：capture 时临时关闭 FC1

这会让捕获图缺少运行时需要的 all-gather/reduce-scatter，改变 tensor shape、分片语义和 collective 顺序，属于正确性风险，不是合法优化。

### 不建议：优先优化 compile hash/cache

compile cache 完整性需要单独治理，但它只占本次 FC1-on 的 3.18 秒，不能解释 407 秒 graph capture。不要把两类问题绑定成一个修复。

## 7. 建议实施顺序

1. 增加启动基准脚本，固定记录 wrapper、load、compile、graph profile、PIECEWISE、FULL、API ready 七段时间。
2. 用 34、16、8、4 档 capture sizes 做启动与吞吐曲线。
3. 对比 `FULL_AND_PIECEWISE`、`FULL`、`PIECEWISE` 三种模式。
4. 选出 DeepSeek-V2 + TP2 + FC1 的默认推荐配置。
5. 再设计 profiling graph 复用或 lazy capture。
6. ACL graph 持久化作为长期项目评估。

## 8. 验收标准

建议第一阶段目标：

- FC1-on 服务 API ready 小于 180 秒；
- 典型线上负载吞吐下降不超过 3%；
- TPOT/P99 延迟回归不超过 5%；
- 所有未捕获 shape 能正确 fallback；
- DBO on/off、FC1 on/off、compile cold/hit 均有独立结果；
- FC2 必须保持关闭，另行验证，防止测试变量污染。

## 9. 尚未证明的事项

以下不能从本次两组实验直接下结论：

- FC1 是否增加 torch.compile 冷编译时间；需要分别清理隔离 cache 后重复多次。
- DBO 打开是否对 FC1 graph capture 有额外固定成本；本次 capture 没有触发 ubatch，需要 DBO on/off 独立对照。
- 缩减 capture sizes 对真实吞吐的影响；必须压测。
- ACL graph 是否可跨进程持久化；取决于 torch_npu/CANN 能力。

本 RFC 已确认的是：在当前部署配置中，启动慢的关键卡点为 FC1-on 的 FULL ACL graph 多 shape capture，且其量级远大于 compile、模型加载和 graph memory profiling。
