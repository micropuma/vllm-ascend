# DBO Serving 起始抖动与 Shape 敏感性 Profiling 计划

## 1. 目标

本文给出一套可重复、可归因的 profiling 方案，用来回答：

1. DBO serving 在服务 ready 后是否存在首请求或首批次抖动？
2. 抖动发生在 Dynamo、piecewise compile、ACLGraph capture，还是 DBO
   线程/通信调度阶段？
3. 抖动是否对实际 batch token shape、DBO threshold、compile range 和 ACLGraph
   capture key 敏感？
4. DBO/non-DBO 状态切换时，是否因为无法 replay 已有 FULL graph 而退化？
5. 冷编译缓存、热编译缓存和进程内 graph cache 分别贡献多少启动及在线延迟？

本文是实验计划，不预设“DBO 一定有抖动”。所有结论必须由日志、请求级延迟和
NPU timeline 三类证据共同支持。

## 2. 背景模型

vLLM Ascend 的执行链可以分为三层：

```text
TorchDynamo
  捕获带 SymInt 的动态 FX graph
        │
        ▼
Piecewise compilation / AscendCompiler
  按 compile range 生成或加载 compiled callable
        │
        ▼
ACLGraph
  按有限的 BatchDescriptor/capture size 捕获固定设备任务图
```

需要避免把三层的 shape contract 混为一谈：

- Dynamo graph 可以包含 symbolic token dimension；
- piecewise backend 可以为同一 FX graph 的不同 range 生成不同执行产物；
- ACLGraph replay 要命中已经捕获的固定 descriptor、输入地址和设备任务拓扑。

DBO 又增加了第四层：

```text
一个 scheduler batch
  → 两个 ubatch
  → 两个 Python 线程
  → compute/comm stream 交替提交
```

Dynamo 捕获一次单-ubatch `model.forward()`；DBO FULL ACLGraph 可以在外层记录两个
ubatch 的完整多 stream 设备任务。[VERIFY:
testbench/MOE/dbo/docs/QA/qa.md:162] [VERIFY:
vllm_ascend/worker/npu_ubatch_wrapper.py:128]

## 3. 源码给出的风险假设

以下是假设，不是实验结论。

### H1：首次 Dynamo/compile 发生在启动期，不一定发生在 16-token graph warmup

worker 在确定 KV cache 可用内存时会调用 `profile_run()`；即使用户显式配置 KV
cache 大小，代码仍保留该调用以编译模型。[VERIFY:
vllm_ascend/worker/worker.py:474] [VERIFY:
vllm_ascend/worker/worker.py:492]

因此，必须从日志确定真正的第一次 compiled model forward，不能把
`cudagraph_capture_sizes=[16]` 等同于“Dynamo 首次只看到 16 tokens”。

### H2：现有 benchmark warmup 会隐藏起始抖动

`deepseek-v2-dbo-test.sh` 默认在正式 benchmark 前发送 16 个 warmup 请求，并使用
与正式 benchmark 相同的 input/output shape。[VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh:83] [VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh:173]

因此，正式 benchmark 的 TTFT/TPOT 不能直接回答“第一个 shape 是否抖动”。起始
抖动实验必须令 `WARMUP_PROMPTS=0`，或使用不会覆盖目标 shape 的独立 warmup。

### H3：DBO 是否触发取决于实际 scheduler batch，不只取决于 prompt length

运行时使用 `num_tokens`、padding 后 token 数、DBO threshold、MoE 通信类型和空
ubatch 检查共同决定 `should_ubatch`。[VERIFY:
vllm_ascend/worker/model_runner_v1.py:2897] [VERIFY:
vllm_ascend/worker/ubatch_utils.py:68]

所以实验自变量不能只记录 `INPUT_LEN`。必须记录每一步的：

```text
logical_num_tokens
padded_num_tokens
batch_descriptor
num_reqs
should_ubatch
ubatch0_tokens
ubatch1_tokens
cudagraph_runtime_mode
```

### H4：同一 token key 在 DBO/non-DBO 间切换可能失去 FULL replay

当当前 batch 没有 ubatch slices，但相同 token size 已存在 DBO FULL graph 时，
`AscendUBatchWrapper` 将 runtime mode 改为 `NONE`，避免错误 replay 双-ubatch
graph。[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:378]

这保证执行拓扑正确，但可能使该次 non-DBO batch 失去 FULL ACLGraph replay，形成
稳定的性能台阶或状态切换抖动。

### H5：DBO graph key 隐含等长 ubatch 假设

DBO FULL graph capture 保存 key 时使用两个 ubatch token 数之和，而运行查找路径
使用第一个 ubatch token 数乘二。[VERIFY:
vllm_ascend/worker/npu_ubatch_wrapper.py:172] [VERIFY:
vllm_ascend/worker/npu_ubatch_wrapper.py:397]

实验必须保存两个 ubatch 的独立 shape，不能只按 total tokens 聚合。

### H6：`allow_microbatching` 当前不能作为可靠的实验开关

`_dummy_run()` 将 `allow_microbatching` 传入
`_determine_batch_execution_and_padding()`，但当前函数仍直接执行
`check_enable_ubatch()`，没有使用该参数屏蔽 DBO。[VERIFY:
vllm_ascend/worker/model_runner_v1.py:2815] [VERIFY:
vllm_ascend/worker/model_runner_v1.py:2897]

实验应以最终的 `should_ubatch` 和 `ubatch_slices` 为事实来源，不能根据调用参数
推断 DBO 是否执行。

## 4. “抖动”和“shape 敏感”的可测定义

### 4.1 起始抖动

对固定 shape 连续执行至少 20 次，定义：

```text
first-hit penalty = latency(first) / median(latency[6:20])
```

分别计算：

- 请求级 TTFT；
- 请求级 E2E；
- engine step CPU wall time；
- model forward wall time；
- NPU device task duration。

建议将以下任一情况标为显著起始抖动：

- first-hit penalty 大于 1.20；
- 首次额外延迟大于 20 ms；
- 首次执行出现 compile/capture，而后续为 replay；
- P99 明显由前 1～3 个请求主导。

阈值必须在报告中固定，不能看到结果后再修改。

### 4.2 Shape 敏感性

对相邻 shape `s1`、`s2`，在执行模式相同且都进入稳态后定义：

```text
normalized latency = step_latency / scheduled_tokens
shape jump = normalized_latency(s2) / normalized_latency(s1)
```

当相邻 shape 跨越以下边界时单独标记：

- DBO prefill/decode threshold；
- compile range endpoint；
- ACLGraph capture size/BatchDescriptor；
- padding bucket；
- MoE 通信方式切换点；
- DBO/non-DBO 状态切换；
- ubatch 等长/不等长边界。

不能将正常的 token 数增长误判为 shape 抖动，因此必须同时报告绝对延迟和
per-token 归一化延迟。

## 5. 实验输入与现有 demos 的复用

### 5.1 基础 server

第一阶段只使用两个脚本：

- `deepseek-v2-server.sh`：non-DBO baseline；
- `deepseek-v2-dbo-server.sh`：DBO server。

DBO server 默认使用 TP=2、EP、prefill threshold 1024，并将 decode threshold
设为极大值以关闭 decode DBO。[VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:80] [VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:96]

这样首先隔离 prefill DBO。不要在第一轮同时打开 decode DBO。

### 5.2 工作负载生成

复用 `deepseek-v2-dbo-test.sh` 的 `vllm bench serve` 参数和结果格式。该脚本已支持
固定 random input length、固定 output length、并发、request rate、request-level
detailed records 和 profiler API。[VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh:136]

但需要增加一个专用 shape-sequence driver，按确定顺序发送 workload，而不是只输出
整轮聚合值。建议新增：

```text
testbench/MOE/dbo/demos/profile_shape_jitter.py
```

该 driver 至少支持：

```text
--sequence 512,1023,1024,1025,2048
--repeat 20
--concurrency 1,2,4,8,16,32,64,96
--output-len 1|16
--ordering grouped|ascending|alternating|randomized
--save-request-timestamps
```

注意：prompt token length 不是 scheduler batch token 数。driver 的职责是稳定控制
请求输入；server instrumentation 才是实际 shape 的权威来源。

### 5.3 配置扩展

第二阶段复用 `auto_benchmark.sh` 的配置矩阵。该脚本已经定义 baseline、DBO、
FC1、FC2 和 AI_CPU/AIV 的组合。[VERIFY:
testbench/MOE/dbo/demos/auto_benchmark.sh:77]

不要一开始运行全部十组。推荐顺序：

1. `baseline`；
2. `dbo`；
3. `fc1`；
4. `dbo_fc1`；
5. 有明确证据表明 FC2 相关时，再运行 `fc2`、`dbo_fc2`；
6. 最后比较 AI_CPU 与 AIV。

这样可以避免将 DBO、FlashComm 和 HCCL 执行模式的影响混在一起。

### 5.4 冷启动与 compilation dump

复用 `compilation-dump/server.sh` 控制独立 cache 目录、`TORCH_TRACE`、compile
artifact 和 capture sizes。该脚本会检查 compile cache 是否为空，并支持显式设置
`cudagraph_capture_sizes` 与 cache directory。[VERIFY:
testbench/MOE/dbo/demos/compilation-dump/server.sh:220] [VERIFY:
testbench/MOE/dbo/demos/compilation-dump/server.sh:253]

## 6. 必须补充的 instrumentation

现有 INFO/DEBUG 日志不足以把请求延迟与 graph 状态一一对应。建议增加结构化 JSONL
事件，默认关闭，通过一个现有集中式环境变量或 debug 配置启用；如果引入新环境
变量，必须按仓库要求在 `vllm_ascend/envs.py` 集中定义并评审。

### 6.1 每个 engine step 的事件

在 `_determine_batch_execution_and_padding()` 返回后记录：

```json
{
  "event": "batch_dispatch",
  "step_id": 1001,
  "logical_tokens": 1023,
  "padded_tokens": 1024,
  "num_reqs": 8,
  "dbo_enabled_config": true,
  "should_ubatch": false,
  "ubatch_shapes": null,
  "runtime_mode": "FULL",
  "batch_descriptor": "...",
  "moe_comm_type": "...",
  "timestamp_ns": 0
}
```

### 6.2 Dynamo/piecewise compile 事件

至少记录：

- 首次 Dynamo capture 开始/结束；
- FX graph id/hash；
- symbolic input shape；
- DBO hook nodes 是否存在；
- compile range；
- subgraph index；
- cache hit/miss；
- AscendCompiler 分支：`npugraph_ex` 或 `fusion_pass`；
- compile wall time。

现有 compilation-dump 已提供 `TORCH_TRACE` 和 artifact 检查入口，应优先复用，而
不是通过 profiler 猜测编译发生时间。[VERIFY:
testbench/MOE/dbo/demos/compilation-dump/server.sh:229]

### 6.3 ACLGraph 事件

分别为普通 `ACLGraphWrapper` 和 `AscendUBatchWrapper` 记录：

```text
graph_kind = full_non_dbo | full_dbo | piecewise
graph_key
batch_descriptor
ubatch_shapes
action = capture | replay | fallback_none
capture_duration_ns
replay_duration_ns
input_addresses
```

DBO FULL graph 的 cache、普通 FULL graph cache和 piecewise graph cache必须分别
计数，不能只统计总 capture 次数。

### 6.4 DBO 调度事件

记录：

- 两个 ubatch 线程创建、ready、开始和结束时间；
- 每次 hook 的 `hook_name`、record/wait、ubatch id；
- compute/comm stream id；
- event record/wait时间；
- 子线程异常；
- 最终 concat/unpad时间。

profiling 日志中禁止对 NPU tensor 调用新的 `.item()`；所有 shape 和状态优先使用
已有 CPU metadata。

## 7. 实验矩阵

### 7.1 Cache 状态

每种关键配置至少覆盖：

| 状态 | 进程 | compile cache | 目的 |
|---|---|---|---|
| C0 真冷启动 | 新进程 | 空 | Dynamo、compile、ACL capture 全成本 |
| C1 AOT/compile 热启动 | 新进程 | 保留 | 隔离 artifact load 与 ACL capture |
| C2 进程内首次 shape | 同进程 | 已加载 | 测在线 lazy capture/dispatch |
| C3 进程内重复 shape | 同进程 | 已加载 | 稳态 replay baseline |

每组 C0/C1 至少重复 3 次；C2/C3 每个 shape 至少重复 20 次。

### 7.2 Graph/compile 模式

最低矩阵：

| 编号 | DBO | graph mode | `enable_npugraph_ex` |
|---|---:|---|---:|
| M0 | 0 | NONE | 0 |
| M1 | 1 | NONE | 0 |
| M2 | 0 | PIECEWISE | 0 |
| M3 | 1 | PIECEWISE | 0 |
| M4 | 0 | FULL_AND_PIECEWISE | 1 |
| M5 | 1 | FULL_AND_PIECEWISE | 1 |

解释：

- M0/M1 给出没有 ACLGraph 的 CPU/runtime基线；
- M2/M3 观察单-ubatch piecewise replay 和 DBO Python 调度；
- M4/M5 代表生产路径；
- 如 M5 出现异常，再增加 `FULL_AND_PIECEWISE +
  enable_npugraph_ex=0`，区分 backend compile 与 ACL capture。

### 7.3 Shape 序列

#### S0：固定 shape 重复

```text
同一 input length、concurrency、output length，连续 20 次
```

目的：测 first-hit penalty 和稳态方差。

#### S1：DBO threshold 两侧

以实际 threshold `T` 为中心，构造 scheduler token 数尽量接近：

```text
T-2, T-1, T, T+1, T+2
```

必须根据 server 记录的 `logical_tokens` 分组，不能只按 prompt length分组。

#### S2：单调 shape sweep

建议 input lengths：

```text
256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192
```

concurrency：

```text
1, 2, 4, 8, 16, 32, 64, 96
```

先使用 `OUTPUT_LEN=1` 隔离 prefill/TTFT，再用 `OUTPUT_LEN=16` 观察短 decode
干扰。

#### S3：capture size边界

从实际 `cudagraph_capture_sizes` 读取 capture size `C`，针对每个关键值测：

```text
C-1, C, C+1
```

目标是观察 padding/descriptor切换、首次 capture和后续 replay。

#### S4：compile range边界

从 `CompilationConfig.get_compile_ranges()` 读取 endpoint `R`，测：

```text
R-1, R, R+1
```

比较 compiled artifact key、compile time和稳态设备执行时间。

#### S5：状态交替

在同一进程内执行：

```text
non-DBO shape A
DBO shape B
non-DBO shape A
DBO shape B
...
```

然后构造尽可能相同的 padded token key但不同 `should_ubatch` 的序列，验证
`fallback_none` 假设。

#### S6：顺序效应

对相同 shape集合分别按以下顺序运行：

```text
ascending
descending
randomized(seed 固定)
```

如果结果依赖顺序而不是只依赖 shape，说明存在 cache/capture状态污染。

## 8. 分阶段执行计划

### Phase 0：正确性与事实校验

1. 固定代码 commit、vLLM commit、CANN、torch、torch_npu、npugraph_ex版本。
2. 固定模型、TP/DP/EP、设备、CPU affinity和 HCCL mode。
3. 打印完整 compilation config、compile ranges和 capture sizes。
4. 用小规模请求确认输出与 non-DBO baseline一致。
5. 确认日志能关联 request、engine step和 graph action。
6. 确认实际 `should_ubatch=True`，不能仅确认 `--enable-dbo`。

退出条件：同一输入逐 token结果一致，并能为每个高延迟请求找到对应 step。

### Phase 1：冷启动分解

分别运行 C0 和 C1，记录：

```text
process start
model weights loaded
profile_run begin/end
first Dynamo capture begin/end
compile_all_ranges begin/end
KV cache ready
ACLGraph profile begin/end
ACLGraph capture begin/end
HTTP ready
```

对比：

- DBO off/on；
- compile cache cold/hit；
- capture size只有 16 与包含大 shape；
- `enable_npugraph_ex` off/on。

输出：启动阶段 waterfall、每阶段耗时和 artifact/cache inventory。

### Phase 2：服务 ready 后的 zero-warmup首请求

1. server ready 后不运行现有 16-request warmup。
2. 执行 S0，每次只改变一个 shape。
3. 保存每请求详细时间和每 step graph action。
4. 重启进程重复至少 3 次。
5. 再执行现有 warmup后重复同一实验，量化 warmup消除了多少 penalty。

输出：first-hit penalty 表和请求时间序列。

### Phase 3：Shape sweep

1. 执行 S1～S4；
2. 每个点先运行一次 cold-in-process，再重复 20 次；
3. 按实际 scheduler shape重新分桶；
4. 画 latency/token、TTFT、capture/replay次数与 shape 的关系；
5. 标记 threshold、range endpoint和 capture size。

输出：shape-response curve和所有不连续点。

### Phase 4：DBO/non-DBO切换

1. 执行 S5；
2. 记录每步 `should_ubatch`；
3. 验证是否出现 `full_dbo replay → fallback_none → full_dbo replay`；
4. 对 fallback step比较 CPU launch、device idle gap和 TTFT；
5. 使用 graph mode NONE作为控制，排除模型计算本身的 shape差异。

输出：状态机时间线和 FULL replay缺失的增量成本。

### Phase 5：定点 NPU profiling

只有 Phase 2～4 找到异常点后才开启 torch_npu profiler。全矩阵开启 profiler会显著
扰动时序并产生过大 trace。

每次 trace只覆盖：

```text
1 次首次 shape
+ 3 次稳态同 shape
+ 1 次状态切换
```

利用 server 脚本已有 profiler配置采集 stack、shape和 memory。[VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:108] [VERIFY:
testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh:173]

timeline重点检查：

- 首次是否出现额外 kernel build/graph capture；
- NPU stream是否存在 device idle gap；
- 两个 ubatch compute/comm是否真正 overlap；
- event wait是否产生长空洞；
- HCCL collective时长是否随 ubatch shape突变；
- replay与非 replay的 kernel launch数量差异。

### Phase 6：FlashComm与 HCCL扩展

确认基础 DBO因果链后，再用 `auto_benchmark.sh` 的配置逐步加入 FC1、FC2和 AIV。

每次只改变一个维度：

```text
baseline → dbo
fc1 → dbo_fc1
AI_CPU → AIV
fc1 → fc2
dbo_fc1 → dbo_fc2
```

输出：配置交互效应，而不只是十组配置的吞吐排名。

## 9. 数据与图表规范

每次 run建立独立目录：

```text
results/<run_id>/
  manifest.json
  server.log
  requests.jsonl
  engine_steps.jsonl
  graph_events.jsonl
  compilation_events.jsonl
  benchmark.json
  profiler/
  artifacts/
```

`manifest.json` 至少包含：

- git commit和 dirty status；
- vLLM commit；
- 软件/固件版本；
- server完整命令和环境变量白名单；
- compile cache状态；
- graph mode、compile ranges、capture sizes；
- DBO thresholds；
- TP/DP/EP与通信后端；
- workload sequence和随机 seed。

必需图表：

1. 冷启动 waterfall；
2. 每 shape前 20 次请求/step latency折线；
3. latency/token 对 actual scheduled tokens散点图；
4. threshold/range/capture boundary局部放大图；
5. capture、replay、fallback_none次数堆叠图；
6. DBO on/off TTFT和TPOT箱线图；
7. NPU compute/comm stream timeline；
8. ubatch overlap ratio随 shape变化曲线。

## 10. 归因判定规则

| 观测 | 归因 |
|---|---|
| 首次出现 Dynamo/compile事件，后续消失 | 编译 first-hit |
| compile cache hit但首次出现 graph capture | ACLGraph first-hit |
| 无 compile/capture，但 `fallback_none` 且 launch增多 | FULL replay缺失 |
| graph action相同，仅 threshold后线程/event增多 | DBO调度成本 |
| NPU任务时间稳定但TTFT抖动 | scheduler/CPU/RPC侧 |
| per-token设备时间在 compile range边界突变 | range-specific compiled code |
| 相同 padded shape因执行顺序不同而变化 | graph/cache状态污染 |
| 只在 FC1/FC2开启时出现 | FlashComm交互，不归因于纯 DBO |

不得仅凭一张 profiler timeline归因。每个结论至少需要：

1. 可重复的请求级现象；
2. 对应的 graph/compile结构化事件；
3. 一个只改变单一变量的控制组。

## 11. 最小可执行首轮

资源有限时先运行以下六组：

```text
1. baseline, graph NONE, fixed 4K, zero warmup
2. DBO, graph NONE, fixed 4K, zero warmup
3. baseline, FULL_AND_PIECEWISE, fixed 4K, zero warmup
4. DBO, FULL_AND_PIECEWISE, fixed 4K, zero warmup
5. DBO, FULL_AND_PIECEWISE, threshold boundary sweep
6. DBO, FULL_AND_PIECEWISE, alternating DBO/non-DBO sequence
```

每组执行：

```text
3 个新进程
× 每 shape 20 次
× cold compile cache和warm compile cache各一次
```

首轮完成后必须能够回答：

- 首次请求是否比稳态慢；
- 慢点对应 compile、capture还是 fallback；
- 相同 shape第二次是否恢复；
- DBO threshold两侧是否出现不连续；
- DBO/non-DBO切换是否丢失 FULL replay。

## 12. 通过标准

若目标是证明“DBO serving没有不可接受的起始抖动”，建议同时满足：

1. warm compile cache下 first-hit penalty不超过 1.20；
2. 已捕获 shape的重复请求不发生新 compile/capture；
3. DBO threshold两侧没有无法解释的 per-token延迟尖峰；
4. DBO/non-DBO切换不会长期落入 `fallback_none`；
5. 同一 shape的稳态 latency变异系数小于 5%；
6. DBO输出与 baseline逐 token一致；
7. profiler显示目标 shape下存在预期的通信/计算 overlap；
8. 所有异常点均能映射到明确的 compile range、graph key或 scheduler状态。

如果第 4 条不满足，应优先评估为同一 descriptor分别缓存 DBO FULL graph和
non-DBO FULL graph，而不是通过扩大请求 warmup掩盖问题。
