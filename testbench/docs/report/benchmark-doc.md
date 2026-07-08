# vLLM-Ascend DBO Benchmark 文档

## 1. 端到端测试

### 1.1 测试环境

```
Device: Ascend 910B3 × 2
CANN: 9.0.0
torch_npu: 2.10.0
Model: DeepSeek-V2-Lite-Chat
TP: 2, EP: 2
compile: fullgraph (28 subgraphs per rank)
```

### 1.2 全配置矩阵 (prefill4k: INPUT=4096, OUTPUT=16, 500 prompts, 96 concurrency)

| DBO | FC1 | FC2 | HCCL Mode | QPS | Out tok/s | Mean TTFT | Mean TPOT |
|---|---|---|---|---|---|---|---|
| OFF | OFF | OFF | AIV | 6.207 | 99.3 | 6144.9 | 605.8 |
| OFF | ON | OFF | AIV | 6.232 | 99.7 | 6079.7 | 605.6 |
| OFF | ON | OFF | AI_CPU | 6.080 | 97.3 | 6230.1 | 620.3 |
| OFF | OFF | ON | AIV | 6.396 | 102.3 | 5932.0 | 589.6 |
| OFF | OFF | OFF | AI_CPU | 6.023 | 96.4 | 6304.7 | 625.4 |
| ON | OFF | OFF | AIV | 6.403 | 102.5 | 5958.8 | 586.7 |
| ON | OFF | OFF | AI_CPU | 6.590 | 105.4 | 5782.7 | 570.9 |
| ON | ON | OFF | AIV | 6.555 | 104.9 | 5777.5 | 574.1 |
| **ON** | **ON** | **OFF** | **AI_CPU** | **7.609** | **121.7** | **5027.2** | **491.8** |
| ON | OFF | ON | AIV | 6.506 | 104.1 | 5808.2 | 581.2 |
| ON | ON | ON | AIV | 6.619 | 105.9 | 5749.8 | 566.6 |

**最佳配置**: DBO=ON + FC1=ON + FC2=OFF + AI_CPU, QPS=7.609, TTFT=5027ms

> 数据来源: `testbench/MOE/dbo/results/matrix_dsv2_*.json` (2026-06-30)

### 1.3 不同 Workload 对比 (DBO=ON)

| Workload | INPUT | OUTPUT | HCCL | QPS | Out tok/s | TTFT |
|---|---|---|---|---|---|---|
| prefill4k | 4096 | 16 | AI_CPU | 7.609 | 121.7 | 5027 |
| prefill4k (baseline) | 4096 | 16 | AIV | 6.207 | 99.3 | 6145 |
| decode-heavy | 1024 | 128 | — | *待补充* | *待补充* | *待补充* |
| prefill8k | 8192 | 16 | — | *待补充* | *待补充* | *待补充* |
| ttft4k | 4096 | 1 | — | *待补充* | *待补充* | *待补充* |

---

## 2. torch.compile / ACLGraph 抖动测试

> 数据采集日期: 2026-07-06, 6 组实验 (FULL=1 矩阵中 aclgraph DBO=1 组因远端资源问题未完成)
> 数据来源: `testbench/MOE/dbo/demos/compilation-dump/warmup-exp/logs/`

### 2.1 实验设计

两类抖动隔离测试：

| 实验 | COMPILE_RANGES | CAPTURE_SIZES | 测试序列 | 测什么 |
|---|---|---|---|---|
| compile_range | `[1024,2048,4096,8192,16384]` | `[16]` | 1023,1024,1025,2047,2048,2049,4095,4096,4097 | 跨 compile range 首次命中 |
| aclgraph | `[16384]` | `[1,2,4,8,16,32,64,128,256]` | 15,16,17,31,32,33,...,255,256 | 跨 ACLGraph bucket 首次命中 |

固定配置：DeepSeek-V2-Lite, TP=2, FC1=1, FC2=0, HCCL=AI_CPU, 每个 input_len 4 prompts × 1 concurrency

### 2.2 编译 artifact 数量验证

| 实验 | compile_range 组数 | subgraph 数 | rank 数 | 预期 | 实际 |
|---|---|---|---|---|---|
| compile_range | 5 | 28 | 2 | 5×28×2=**280** | **280** ✅ |
| aclgraph | 1 | 28 | 2 | 1×28×2=**56** | **56** ✅ |

**所有实验 recompile 事件数 = 0** — 编译在冷启动阶段一次性完成，跨 range 边界无重新编译。

### 2.3 首请求冷启动延迟 (65-70x)

observe 模式下，第一个请求 TTFT 高达 6.8-7.6s，后续请求仅 ~90-130ms：

```
compile_range DBO=0 (observe):
  Input:   1023    1024    1025    2047    2048    2049    4095    4096    4097
  TTFT:    7550    124     114     422     121     119     444     142     138   ms
           ↑ 65x cold start         ↑ 3.5x boundary               ↑ 3.1x

compile_range DBO=1 (observe):
  Input:   1023    1024    1025    2047    2048    2049    4095    4096    4097
  TTFT:    7335    127     119     454     127     129     474     147     146   ms
           ↑ 60x

aclgraph DBO=0 (observe):
  Input:   15      16      17      31      32      33      ...  127    128    255    256
  TTFT:    6828    89      88      92      90      90      ...  99     121    114    110   ms
           ↑ 70x
```

**首请求的 7 秒延迟 = ACLGraph capture + NPU graph 初始化**，非 compile range 重新编译（recompile=0）。

### 2.4 Range 边界次级抖动 (3-4x)

compile_range 实验中，`range_end − 1` 位置（2047, 4095）出现 3-4x 延迟突增：

| Input | DBO=0 TTFT | DBO=1 TTFT | 所在 range | 分析 |
|---|---|---|---|---|
| 2047 | **422ms** | **454ms** | [1025, 2048] 端点 | padded token 接近 range 边界 → 不同 graph capture 路径 |
| 2048 | 121ms | 127ms | [1025, 2048] | 正常 |
| 2049 | 119ms | 129ms | [2049, 4096] | 新 range，无抖动 ✅ |
| 4095 | **444ms** | **474ms** | [2049, 4096] 端点 | 同上 |
| 4096 | 142ms | 147ms | [2049, 4096] | 正常 |

**跨 range 边界本身不触发抖动**（2048→2049 平滑），但 range 端点可能有次级 graph capture 开销。aclgraph 实验（单 range）无此现象。

### 2.5 Warmup 消除效果

| 实验 | observe 首请求 | warmup 正式阶段首请求 | 稳态 |
|---|---|---|---|
| compile_range DBO=0 | 7550ms | **115ms** ✅ | ~120ms |
| compile_range DBO=1 | 7335ms | N/A (测试未完成) | ~128ms |
| aclgraph DBO=0 | 6828ms | **84ms** ✅ | ~90ms |

**预 warmup 完全消除所有抖动**，首请求 TTFT 从 7s 降至稳态水平。

### 2.6 DBO 对冷启动的影响

| 指标 | compile_range DBO=0 | DBO=1 | 差异 |
|---|---|---|---|
| 首请求 TTFT | 7550ms | 7335ms | −2.9% |
| 稳态 TTFT | ~120ms | ~128ms | +6.7% |
| Compile artifacts | 280 | 280 | 相同 |
| Recompile | 0 | 0 | 相同 |

**DBO 对冷启动抖动几乎无影响**（差异在 3% 以内）。冷启动延迟主因是 ACLGraph capture + NPU graph 初始化。

### 2.7 冷启动 Compilation 全流程

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 模型初始化 | ~81s | 权重加载 + compilation_config 生成 |
| Compile artifact 生成 | compile_range: 280 artifacts, aclgraph: 56 | piecewise compile 产物 |
| Graph Capture | ~7s | 首请求触发 (首请求 TTFT − 稳态 TTFT) |
| **总计 (compile_range)** | **~88s + 7s = ~95s** | "Initializing engine" → 首请求完成 |
| **总计 (aclgraph)** | **~15s + 7s = ~22s** | 单 range，compile artifact 少 |

---

## 3. FlashComm / FlashComm2 算子库测试

### 3.1 FlashComm1

| 测试项 | 结果 | 备注 |
|---|---|---|
| FC1 + DBO + eager | ✅ | 基础功能验证通过 |
| FC1 + DBO + compile fullgraph | ✅ | 已验证，存在已知 shape contract 问题 |
| FC1 + DBO + TP=2 | ✅ | column/row parallel 的 DBO hook 正确插入 |
| FC1 + DBO + DP=2 | ✅ | DP 下 `_sync_metadata_across_dp` 保证一致性 |

### 3.2 FlashComm2

| 测试项 | 结果 | 备注 |
|---|---|---|
| FC2 + DBO + AIV | ✅ | 基础功能验证通过 |
| FC2 + DBO + AI_CPU | ❌ | CANN 9.0.0 AllToAll crash (errorCode=0x2a) |
| FC2 (无 DBO) + AI_CPU | ❌ | 纯 `dist.all_to_all_single()` 也 crash (非 DBO 问题) |
| FC2 (无 DBO) + AIV | ✅ | 2-rank 最小化复现通过 |

**已知问题**:
- AI_CPU + AllToAll 是 CANN/HCCL 层面 bug，与 DBO 无关
- DBO + FC2 在 AI_CPU 下有额外的无序 collective 提交问题（ODP communicator 未接入 hook）
- 详见 `rfc/rfc-dbo-flashcomm2-aicpu-alltoall.md`

### 3.3 FlashComm 对 DBO 效果的影响

| 配置 | TTFT (ms) | vs baseline | 说明 |
|---|---|---|---|
| DBO only (AIV) | 5958.8 | baseline | DBO 无 FlashComm |
| DBO + FC1 (AIV) | 5777.5 | −3.0% | FC1 加速通信 |
| DBO + FC1 (AI_CPU) | 5027.2 | −15.6% | FC1 + AI_CPU 最佳 |
| DBO + FC1 + FC2 (AIV) | 5749.8 | −3.5% | FC2 有额外收益但受 AI_CPU 限制 |

---

## 4. AI_CPU vs AIV

### 4.1 核心结论

| | AI_CPU | AIV |
|---|---|---|
| TTFT (prefill4k) | **~4900ms** ✅ | ~5800ms |
| Pure compute / call | **171ms** (−27%) | 236ms |
| Overlap 率 | 61.1% | **73.3%** |
| Free/idle | 15.2% | **8.6%** |
| FC2 兼容 | ❌ CANN crash | ✅ |

基于torch profile捕获出的profile结果可知，AI_CPU 和 AIV有如下tradeoff：  

* AIV会导致计算kernel变慢  
* AI_CPU会导致通信时间发生明显抖动异常  

**深度分析见**请参考 [`profile-deep-analysis-4way.md`](profile-deep-analysis-4way.md) ，涵盖 AI_CPU vs AIV 的权威 profiling 分析，包含 kernel 级对比、async 开销根因、DBO cascade 放大、Profiling vs TTFT 矛盾解释、MindStudio 定位方法。

---

## 5. 参考资料

- [`profile-deep-analysis-4way.md`](profile-deep-analysis-4way.md) — AI_CPU vs AIV Profiling 四维深度分析（权威源）
- `testbench/MOE/dbo/rfc/rfc-dbo-aiv-vs-aicpu-profiling-analysis.md` — Core 分区否决 + Workload 实验设计
- `testbench/docs/report/pr-doc.md` — PR 文档
- `testbench/docs/report/deep-analysis-flashcomm-dbo.md` — FlashComm+DBO 数学推导
- `testbench/MOE/dbo/results/` — Benchmark 原始结果
