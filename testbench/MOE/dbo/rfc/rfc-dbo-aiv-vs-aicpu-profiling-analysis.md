# RFC: DBO 场景下 AIV vs AI_CPU 优劣分析与 Core 分配实验

## 状态

- 状态：实验设计阶段，待执行
- 日期：2026-07-07
- Profiling 数据：2026-07-06 采集，DBO + FC1 + compile fullgraph + DeepSeek-V2-Lite

## 摘要

在 DBO + FC1 + compile fullgraph 下对比 `HCCL_OP_EXPANSION_MODE=AI_CPU` vs `AIV`：

| 指标 | AI_CPU | AIV |
|---|---|---|
| TTFT (prefill4k) | **~4900ms** | ~5800ms |
| Compute per call | **164ms** | 196ms (+19%) |
| Free/idle per call | 38ms | **22ms** |
| Overlap rate | 61.1% | **73.3%** |
| Kernel avg wait time | 40-126μs | **3-17μs** |

本 RFC 从硬件架构角度分析两种模式在 DBO 下的优劣根因，并设计最简实验：**AIV + core 分区能否隔离通信、加速 compute？**

---

## 1. 硬件背景

### 1.1 Ascend NPU 计算单元

```
┌─────────────────────────────────────────────┐
│ Ascend 910B3 NPU Chip                        │
│                                              │
│  AI Cube cores (AIC)  ← 矩阵乘法专用         │
│    数量: torch.npu.get_device_limit()["cube_core_num"]  │
│    代码 fallback: 24                          │
│                                              │
│  AI Vector cores (AIV) ← 向量运算 + 激活等    │
│    数量: torch.npu.get_device_limit()["vector_core_num"] │
│                                              │
│  AI CPU cores          ← 独立通用 CPU 核      │
│    数量: 硬件固定，不在 get_device_limit 中    │
│    HCCL_OP_EXPANSION_MODE=AI_CPU 时使用       │
└─────────────────────────────────────────────┘
```

### 1.2 MIX_AIC Pipeline 架构

MIX_AIC kernel（Attention、MoE routing、GroupedMatmul）使用 **Cube + Vector 交替流水线**：

```
[Vector: 数据格式化] → [Cube: 矩阵乘法] → [Vector: 后处理/激活] → [Cube: ...] → ...
```

Cube 和 Vector **不是独立并行的**，而是一个紧耦合流水线。**Vector 段被阻塞 → Cube 段也 stall。**

---

## 2. AIV 模式分析

### 2.1 AIV 通信的执行路径

```
HCCL_OP_EXPANSION_MODE=AIV

NPU Stream:
  [Compute kernel] → [HCCL AllGather (跑在 AIV core 上)] → [Compute kernel] → ...
                       ↑
                      通信和计算共享同一批 Vector 核
                      在同一 NPU 调度域内
```

### 2.2 AIV 的优势：低 Sync 开销

DBO 的乒乓同步全部发生在 NPU 调度域内：

```
ubatch[0] stream:  [Compute]──record npu.Event──→ yield ──→ wait npu.Event ──→ [Compute]
                        ↓  ~10μs                        ~10μs  ↑
ubatch[1] stream:  wait npu.Event ──→ [Comm AIV] ──→ record npu.Event ──→ yield
                         ↑                                    ↓
                    同域 GPU event，延迟 ~10μs
```

**每次 DBO 切换只需要 ~10μs 的 GPU event 同步。** 这就是 AIV 的 kernel wait time 低（3-17μs）、Free/idle 少（8.6%）的原因。

### 2.3 AIV 的劣势：Compute Pipeline Stall

AIV 通信跑在 Vector 核上 → 与 MIX_AIC kernel 的 Vector 段**直接冲突**：

```
AIV 模式下 MoeInitRoutingCustom 的时间线:

  Vector: [HCCL AllGather...] [token sort] [等待...] [HCCL ReduceScatter...] [permute]
  Cube:   [等待 Vector...]    [score matmul] [等待 Vector...] [等待...]
           ↑                                                      ↑
          Vector 被通信占用                                    又一次被占用
          → Cube 空闲等待                                       → Cube 空闲等待
          → kernel 总耗时 916μs                                 → AI_CPU 只需 285μs (−69%)
```

各 kernel 受 AIV 通信影响的程度（与 AI_CPU 对比）：

| Kernel | Core Type | AIV | AI_CPU | 慢多少 | 根因 |
|---|---|---|---|---|---|
| MoeInitRoutingCustom | MIX_AIC | 916μs | 285μs | **+221%** | Vector(token sort/permute) + Cube(score) 频繁交替 |
| MatMulV3 (MIX_AIC) | MIX_AIC | 2016μs | 1014μs | **+99%** | 混合精度 matmul，Vector 做量化/反量化 |
| MatMulV3 | AI_CORE | 594μs | 325μs | **+83%** | 虽标记 AI_CORE，数据预处理需要 Vector 格式化 |
| MatMulV2 | AI_CORE | 246μs | 174μs | +41% | 同上，矩阵小所以预处理占比低 |
| FusedInferAttentionScore | MIX_AIC | 1317μs | 1098μs | +20% | QK^T(Cube) + softmax/post(Vector) |
| GroupedMatmul | MIX_AIC | 1157μs | 1075μs | +8% | Cube 主导，Vector 段短 |
| SwiGlu | AI_VECTOR_CORE | 240μs | 250μs | −4% | 纯 Vector，同等竞争，无额外 stall |
| AddRmsNormBias | AI_VECTOR_CORE | 188μs | 187μs | −0% | 纯 Vector，同等竞争，无额外 stall |

**规律：Vector 依赖越重、Cube+Vector 交替越频繁的 kernel，被拖慢越严重。纯 Vector kernel 不受影响。**

### 2.4 AIV 的 Overlap 质量："微重叠"尖角

在 MindStudio trace 中，AIV 通信呈"尖角"状：

```
AIV — 通信 kernel 分布在两个 stream:
  Stream N/A (主):  5319 个 comm kernel, 5099ms, avg 959μs
  Stream 39 (DBO):  4833 个 comm kernel, 5055ms, avg 1046μs
  → 两个 stream 各 ~5000 个 ~1ms 通信碎片

Timeline:
Stream N/A: ═══[Compute 5ms]═══ ─[1ms Comm]─ ═══[Compute 3ms]═══ ─[1ms Comm]─ ═══
Stream 39:  ─[1ms Comm]─ ═══[Compute 5ms]═══ ─[1ms Comm]─ ═══[Compute 3ms]═══
                 ↑                           ↑
              "尖角" — 1ms 通信尖峰插入对面 compute
```

73.3% overlap 率是由 ~10k 个 1ms 碎片拼成的**微重叠**。每个碎片都是一次 stream 切换 + resource 争抢。

---

## 3. AI_CPU 模式分析

### 3.1 AI_CPU 通信的执行路径

```
HCCL_OP_EXPANSION_MODE=AI_CPU

NPU Stream:                                    AI_CPU (独立硬件):
  [Compute kernel]                              [空闲]
       ↓ launch AI_CPU task                         ↓ 收到任务
  [继续...]           ←── 跨域提交 ──→         [allgatherAicpuKernel]
       ↓ wait AI_CPU done                          ↓ 完成
  [等待...]           ←── 跨域同步 ──→         [发完成信号]
       ↓ 收到完成信号
  [Compute kernel]
```

### 3.2 AI_CPU 的优势：Compute 全速

通信物理隔离到 AI_CPU → **NPU 全部 Cube + Vector 核只做计算**：

```
AI_CPU 模式下 MoeInitRoutingCustom 的时间线:

  Vector: [token sort] [permute] [后处理]     ← 全部给计算，无通信争抢
  Cube:   [score matmul]                     ← 无等待
          → kernel 总耗时 285μs (vs AIV 的 916μs)
```

这就是为什么 AI_CPU 的 TTFT 比 AIV 快 15.5% — prefill 路径上的 compute kernel 全部全速运行。

### 3.3 AI_CPU 的劣势 (1)：通信任务调度延迟

`allgatherAicpuKernel` 本身的执行很快（avg 103μs），但**调度延迟**高：

```
allgatherAicpuKernel wait time 分布:
  P50:     0.0 μs    (一半立即执行)
  P90:   264.8 μs
  P95:   950.9 μs    ← 长尾开始
  P99:  1439.9 μs
  Max:  4170.7 μs
  Avg:   126.1 μs    ← 平均调度延迟 > 平均执行时间!
  >1ms:  169 次 (4.6%)
```

**调度延迟（126μs）比执行时间（103μs）还长！** 根因是跨域任务提交路径：

```
NPU kernel 启动:   ~5-10μs   (同域，硬件调度器直接发射)
AI_CPU 任务启动:   ~126μs avg (NPU → driver → AI_CPU scheduler → AI_CPU core)
                   P95=951μs (AI_CPU scheduler 队列深度/优先级等因素)
```

### 3.4 AI_CPU 的劣势 (2)：DBO Sync Cascade 放大

AI_CPU 的跨域延迟通过 DBO 乒乓同步被**级联放大**：

```
正常 DBO (AIV, 同域):
  ubatch[0]: [Compute 5ms] → event(~10μs) → [Comm AIV 1ms] → event(~10μs) → ...
  ubatch[1]:        [Comm AIV 1ms] → event(~10μs) → [Compute 5ms] → event(~10μs)
                        ↑                                        ↑
                   干净的大块 overlap，切换代价 ~10μs

AI_CPU DBO (跨域):
  ubatch[0]: [Compute] → launch AI_CPU(~126μs) → [等待 AI_CPU 完成...] → event → ...
  ubatch[1]:        [等待 event...] → launch AI_CPU(~126μs) → [等待...] → event → ...
                        ↑                    ↑                   ↑
                   每次跨域都有延迟    AI_CPU 调度抖动    cascade 放大
```

**Cascade 传导路径**：

1. AI_CPU allgather 调度延迟 ~126μs → ubatch[1] communication block 延迟
2. ubatch[0] compute 等 ubatch[1] comm 完成 → compute kernel wait time ↑
3. ubatch[0] compute 延迟完成 → ubatch[1] 下个 comm 等更久
4. **正反馈** → 两边 wait time 同时被推高

这就是为什么 AI_CPU 下**所有 kernel 的 wait time 都高了 5-14x**，而不只是通信 kernel：

| Kernel | AIV avg wait | AI_CPU avg wait | 倍数 |
|---|---|---|---|
| FusedInferAttentionScore | 6.8μs | **62.4μs** | 9.2x |
| MatMulV2 | 3.5μs | **41.2μs** | 11.8x |
| SwiGlu | 3.1μs | **44.4μs** | 14.3x |
| MoeInitRoutingCustom | 16.0μs | **97.9μs** | 6.1x |
| allgatherAicpuKernel | — | **126.1μs** | N/A |

---

## 4. 优劣总结

| 维度 | AIV | AI_CPU |
|---|---|---|
| **通信执行硬件** | NPU Vector 核 | 独立 AI CPU 核 |
| **Compute kernel 速度** | ❌ 慢 (与通信争 Vector，pipeline stall) | ✅ 快 (NPU 全部资源给计算) |
| **通信执行延迟** | ✅ 低 (同域调度) | ❌ 高 (跨域调度 avg 126μs, P95 951μs) |
| **DBO sync 开销** | ✅ 低 (~10μs/次 GPU event) | ❌ 高 (跨域 event + cascade，wait time 5-14x) |
| **Overlap 率** | ✅ 73.3% (高) | ❌ 61.1% (较低) |
| **Overlap 质量** | ❌ 微重叠 (~10k 碎片，"尖角") | ✅ 宏重叠 (大块覆盖) |
| **Free/idle** | ✅ 8.6% | ❌ 15.2% |
| **TTFT** | ❌ ~5800ms | ✅ ~4900ms |
| **对 prefill 的影响** | compute 被拖慢 → TTFT 差 | compute 全速 → TTFT 好 |
| **对 decode 的影响** | sync 开销低 → decode 效率可能更好 | sync 开销高 → decode 可能被拖累 |
| **FC2 兼容性** | ✅ 兼容 | ❌ 已知 CANN 9.0.0 crash (AlltoAll) |

**本质 trade-off**：

```
AIV    = "NPU 上什么都做，但什么都互相拖累"
         通信拖慢计算 (pipeline stall)，但同步开销低

AI_CPU = "计算和通信物理隔离，但跨域沟通代价大"
         计算全速，但每次和 AI_CPU 沟通都要付 ~126μs 的"跨域税"
         这个税在 DBO 乒乓同步中被 cascade 放大
```

---

## 5. Core 分配实验

### 5.1 为什么只调 AIV 有意义

当前 AI_CPU 的短板是**跨域调度延迟 + DBO cascade**，这是 CANN/HCCL 层面的问题，无法通过 Core 分区解决。而且 AI_CPU 通信不占用 NPU Vector 核，设置 `COMM_AIV_NUM` 只会白白浪费核。

AIV 的短板是**通信和计算争 Vector 核**，这恰好是 Core 分区可以解决的。思路：

```
AIV + Core 分区 (COMM_AIV_NUM=16):

  时刻 T1:
    Stream A (comp): 独占 total−16 个 Vector 核 → compute 全速，无通信干扰
    Stream B (comm): 只接触 16 个 Vector 核 → AIV 通信被隔离在这 16 个核上

  时刻 T2 (swap):
    Stream A (comm): 只接触 16 个 Vector 核
    Stream B (comp): 独占 total−16 个 Vector 核

  → 通信和计算在 Vector 核上物理隔离
  → 同域调度优势保留（sync 开销仍低）
  → 代价: 通信只有 16 核 → 可能变慢
  → 净效果 = compute 加速 − 通信变慢
```

### 5.2 代码路径

```python
# vllm_ascend/envs.py:113-119
"VLLM_ASCEND_DBO_COMM_AIV_NUM": lambda: int(os.getenv("VLLM_ASCEND_DBO_COMM_AIV_NUM", -1)),
# 默认 -1 = 不分区

# vllm_ascend/worker/ubatching.py:78-84
self.comm_vector_core = envs.VLLM_ASCEND_DBO_COMM_AIV_NUM
self.comp_vector_core = props["vector_core_num"] - self.comm_vector_core

# vllm_ascend/worker/ubatching.py:149-165 — 每个 DBO hook 动态切换
def record_current_stream(self, event):      # 进入通信阶段
    torch.npu.set_stream_limit(stream, vector_num=comm_vec)   # 只用 16 核

def wait_current_stream_and_yield(self, ...): # 回到计算阶段
    torch.npu.set_stream_limit(stream, vector_num=comp_vec)   # 用 total−16 核
```

### 5.3 实验矩阵（3 组，需新跑 1 组）

| # | HCCL_MODE | COMM_AIC | COMM_AIV | 状态 | 代号 |
|---|---|---|---|---|---|
| 1 | AI_CPU | -1 | -1 | **已有** | baseline 上界 |
| 2 | AIV | -1 | -1 | **已有** | baseline 下界 |
| 3 | AIV | -1 | **16** | **待跑** | 核心实验 |

固定配置：DBO=ON, FC1=ON, FC2=OFF, compile=fullgraph, DeepSeek-V2-Lite, TP=2, prefill4k。

### 5.4 判断标准

| `aiv_vec16` TTFT | 结论 |
|---|---|
| **≤ 4900ms** | AIV + core 分区追平/超过 AI_CPU：通信隔离有效，且保留了 AIV 的低 sync 优势 |
| **4900~5800ms** | 有改善但不够：16 核太少导致通信显著变慢，需扫更大值 (20, 24, 28...) |
| **> 5800ms** | 更差：16 核对 AIV 通信是瓶颈，反向验证需要更多核 |
| **≈ 5800ms** | 无效果：`set_stream_limit` 可能在 compile fullgraph 下被优化掉 |

### 5.5 环境变量

```bash
# 启动前确认 core 总数
python3 -c "import torch; print(torch.npu.get_device_limit(0))"

export VLLM_ASCEND_ENABLE_DBO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0

# Group 3 (唯一新实验)
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_DBO_COMM_AIC_NUM=-1
export VLLM_ASCEND_DBO_COMM_AIV_NUM=16
```

---

## 6. 附录：关键代码路径

| 文件 | 行号 | 内容 |
|---|---|---|
| `vllm_ascend/envs.py` | 113-119 | 参数定义 |
| `vllm_ascend/worker/ubatching.py` | 78-84 | Core 获取 + 分区计算 |
| `vllm_ascend/worker/ubatching.py` | 149-165 | `set_stream_limit` 动态切换 |
| `vllm_ascend/worker/npu_ubatch_wrapper.py` | 52-90 | `NPUCoreControlContextManager` |
| `vllm_ascend/platform.py` | 316-338 | `num_compute_units()` |
