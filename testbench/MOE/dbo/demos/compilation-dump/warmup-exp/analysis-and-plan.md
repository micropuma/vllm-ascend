# Warmup Experiments: Compile-Range Jitter 与 ACLGraph First-Capture Jitter

## 1. 目标

这个实验目录要回答两个问题：

1. `piecewise compilation` 的 `compile range` 在真实 workload 下，是否会因为 shape 落点变化而产生明显的编译抖动。
2. `ACLGraph` 的 `cudagraph_capture_sizes` 分桶是否会因为首次命中 capture 而带来明显的首回合抖动。

这里的“抖动”不是笼统的慢，而是指：

- 首个落点触发了额外编译或 capture；
- 后续同一 bucket / 同一 range 已命中缓存，延迟明显下降；
- 延迟差异能在日志或 profiler timeline 中找到明确证据。

## 2. 先说结论：两类问题可以分开测

这两个问题必须拆开测，因为它们位于不同层：

- compile range 属于 `piecewise compilation` 层，关注的是 FX graph 被切到哪些 range；
- ACLGraph 分桶属于 graph capture 层，关注的是实际 token size 怎么映射到 padded bucket。

源码里这两者的职责是分离的：

- `compile_ranges_endpoints` 决定 piecewise 的 range；
- `cudagraph_capture_sizes` 决定 ACLGraph 的 bucket。

这意味着实验设计上应该分别构造：

- `compile-range jitter` 实验；
- `ACLGraph first-capture jitter` 实验。

## 3. 现有 `compilation-dump` 能复用什么

现有 `compilation-dump/` 已经提供了三类基础能力：

- 冷启动 cache 隔离；
- `DUMP_MODE=dynamo|compile|backend|all` 的分层 dump；
- `ENABLE_PROFILER=1` 的 torch_npu profiler 采集。

可直接复用的脚本能力：

- `server.sh` 用于启动并控制 dump/profiler 开关；
- `test.sh` 用于压测、warmup、保存结果、可选 profile；
- `logs/` / `cache/` / `profile/` 用于存放证据。

所以 warmup-exp 不需要重新发明一套完整框架，重点是补两个实验矩阵和对应的观测口径。

## 4. 实验一：compile range 导致的编译抖动

### 4.1 观察对象

观察 `piecewise compilation` 在不同 shape 落点下的行为：

- 是否触发新的 `torch.compile` / AOT 过程；
- 是否触发新的 `compile_range` 子图编译；
- 是否出现 `recompile`、guard 变化、或新的 compile cache artifact；
- 首次触发是否比稳态明显更慢。

### 4.2 适合的证据

日志证据：

- `torch.compile took`
- `recompile`
- `graph_break`
- `compile_range`
- `AscendCompiler hash factors`
- `artifact_compile_range_*`

Profiler 证据：

- `create_aot_dispatcher_function`
- `aot_collect_metadata`
- `aot_trace_joint_graph`
- `compile_fx`

### 4.3 推荐 workload 方式

用固定 shape 序列，而不是随机分布。

建议形状围绕 range 边界做扫描，例如：

- `range.end - 1`
- `range.end`
- `range.end + 1`

如果当前模型的 compile ranges 是多个区间，就挑 2 到 4 个边界做测试，不需要一次扫全量。

### 4.4 推荐实验矩阵

#### A 组：真实抖动组

- `server`: 使用当前默认或接近线上默认的 `compile_ranges_endpoints`
- `test`: 按边界附近的 shape 序列连续发请求
- `profile`: 开 `ENABLE_PROFILER=1`，并保留 `DUMP_MODE=compile` 或 `DUMP_MODE=all`

目标是看到：

- 第一次跨入新 range 时，编译/预热延迟突然上升；
- 后续同 range 重复请求明显更快。

#### B 组：对照组 1，手动缩小 range 空间

- `server`: 手动把 `compile_ranges_endpoints` 合并成更少的区间，或者只保留单个大 range
- `test`: 同样的 shape 序列

如果抖动明显减弱，说明问题主要来自 range 切分，而不是别的路径。

#### C 组：对照组 2，提前 warmup 边界

- `server`: 保持原 range
- `test` 或 `server warmup`: 先把所有边界值各跑一轮

如果后续正式请求的抖动消失，说明问题可通过 warmup 覆盖，不一定需要改编译策略。

### 4.5 判定标准

如果满足以下任一条件，就可以认为 compile range 造成了真实抖动：

- 某个边界第一次触发 compile 的 wall time 明显高于稳态；
- 日志中出现新的 compile artifact 或重新编译；
- profiler 中 `compile_fx` / `aot_trace_joint_graph` 明显占时；
- 抖动只发生在跨 range 首次命中时，而不是稳定重复阶段。

## 5. 实验二：ACLGraph 首次命中 capture 导致的抖动

### 5.1 观察对象

观察 `cudagraph_capture_sizes` 对应的 bucket 是否在首次命中时产生 capture 抖动：

- 第一次遇到某个 bucket 时是否发生 `capture`；
- 后续相同 bucket 是否直接 `replay`；
- 首次 capture 的成本是否显著高于 replay；
- 这个成本是否只在 bucket 首次出现时发生。

### 5.2 适合的证据

日志证据：

- `Capturing CUDA graphs`
- `Graph capturing finished`
- `Capturing a aclgraph`
- `Replaying aclgraph`
- `batch_descriptor`
- `should_ubatch`

Profiler 证据：

- `torch.npu.graph`
- `aclgraph capture`
- `aclgraph replay`

### 5.3 推荐 workload 方式

仍然使用固定 shape 序列，但这次重点不是 compile range，而是 bucket 边界。

建议围绕 capture sizes 的边界做请求序列，例如：

- `capture_size - 1`
- `capture_size`
- `capture_size + 1`

最好同时覆盖：

- 一个小 bucket；
- 一个中等 bucket；
- 一个接近 `max_cudagraph_capture_size` 的 bucket。

### 5.4 推荐实验矩阵

#### A 组：真实抖动组

- `server`: 使用当前默认 `cudagraph_capture_sizes`
- `test`: 按 bucket 边界附近的 shape 序列连续发请求
- `profile`: 开 `ENABLE_PROFILER=1`

目标是看到：

- 每个新 bucket 的第一次请求会触发 capture；
- 同 bucket 的第二次及以后请求变成 replay；
- 首次 capture 在日志和 profiler 上都明显更慢。

#### B 组：对照组 1，手动减少 bucket 数量

- `server`: 手动缩减 `cudagraph_capture_sizes`
- `test`: 同样的 shape 序列

如果首回合抖动次数变少，说明问题主要来自 bucket 数量和分布。

#### C 组：对照组 2，提前 warmup 全部 bucket

- `server`: 保持原 bucket
- `test`: 在正式 benchmark 前，把 bucket 序列全部跑一遍

如果正式阶段几乎只看到 replay，没有新的 capture，说明抖动可以通过 warmup 消掉。

### 5.5 判定标准

如果满足以下任一条件，就可以认为 ACLGraph 首次命中导致了真实抖动：

- bucket 第一次命中时，日志出现 capture，后续出现 replay；
- 第一次命中的延迟显著高于 replay；
- profiler 时间轴里 capture 阶段明显长于 replay；
- 抖动仅在新 bucket 首次出现时发生，而不是持续发生。

## 6. 两组实验的关系

这两组实验的差别在于“桶”定义的位置不同：

- compile range 的桶，由 `compile_ranges_endpoints` 决定；
- ACLGraph 的桶，由 `cudagraph_capture_sizes` 决定。

因此实验时不要混在一起跑，否则会把两类抖动叠加起来，难以归因。

建议顺序：

1. 先做 compile-range jitter；
2. 再做 ACLGraph first-capture jitter；
3. 最后做混合场景，验证两者是否会叠加。

## 7. 是否能只靠 log 或 profiling 做完

可以，而且这是推荐方式。

### 7.1 只看 log 能看到什么

log 足以判断：

- 是否触发了新的 compile；
- 是否触发了新的 ACLGraph capture；
- 是首次 capture 还是 replay；
- 是否存在 guard/recompile。

### 7.2 profiling 能补什么

profiling 能补：

- 把首回合额外延迟具体归到编译、capture 还是普通执行；
- 看 CPU wall time 和 NPU timeline 的重叠关系；
- 验证“首回合慢”是不是 CPU-side 还是 device-side。

### 7.3 最稳妥的组合

最稳妥的是：

- log 做事件归因；
- profiler 做时间归因；
- benchmark 结果做端到端归因。

三者一起，才足够判断“是不是抖动”。

## 8. 建议的目录产物

建议在 `warmup-exp/` 下保留下面几类产物：

- `README.md`：快速入口；
- `analysis-and-plan.md`：本文件；
- `logs/`：server/test 日志；
- `cache/`：冷启动 cache；
- `profile/`：torch_npu profiler 结果；
- `results/`：benchmark JSON。

## 9. 建议的执行步骤

### Step 1：先做 compile-range 实验

- 开 `DUMP_MODE=compile` 或 `DUMP_MODE=all`
- 固定 shape 序列
- 先跑真实组，再跑两个对照组

### Step 2：再做 ACLGraph bucket 实验

- 开 `ENABLE_PROFILER=1`
- 固定 shape 序列
- 比较首次 capture 和 replay

### Step 3：把两组实验合并到一个小矩阵

只保留最能说明问题的 4 组：

- compile-range 真实组
- compile-range warmup 对照组
- ACLGraph 真实组
- ACLGraph warmup 对照组

## 10. 预期输出

最终应该能得到下面几类结论：

- compile range 是否真的造成了首次编译抖动；
- ACLGraph bucket 是否真的造成了首次 capture 抖动；
- 两类抖动分别能否通过预 warmup 缓解；
- 哪一类抖动更值得优先治理；
- 哪些 shape 区间最敏感，应该作为线上预编译/预 capture 的优先候选。

## 11. DBO 对首个抖动的影响怎么评估

DBO 很可能不是首个抖动的主因，但它可能影响：

- 首次是否进入 ubatch / DBO 路径；
- 某个 shape 首次触发时的实际执行拓扑；
- graph cache 命中后是 replay 还是 fallback。

因此可以把 DBO 的影响当成一个独立对照维度：

```text
固定同一套 compile range / capture sizes / sequence
  ├─ DBO=0
  └─ DBO=1
```

如果 DBO on/off 的首回合延迟差异很小，而 compile-range / ACLGraph 的差异很明显，
就可以把首个抖动基本归因给编译或 capture，而不是 DBO 本体。

推荐判读顺序：

1. 先看 `DBO=0` 是否已经存在明显首回合抖动；
2. 再看 `DBO=1` 是否放大抖动或改变 replay/capture 命中；
3. 最后比较 warmup 前后是否把抖动消掉。
