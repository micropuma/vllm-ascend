# FlashComm 有效性 & DBO Overlap 数学分析

> 基于 Profiling 数据：DBO + FC1 + AIV + compile fullgraph + DeepSeek-V2-Lite, TP=2, EP=2

## 一、FlashComm 为什么有效

### 1.1 传统 TP 的通信问题

传统 Megatron-style TP 下，每个 Transformer 层的通信模式：

```
QKV column-parallel → [无通信, heads 本地计算]
Attention (per-head local)
o_proj row-parallel  → AllReduce [T, H]
RMSNorm              → 每个 rank 持有全部 token，per-token 操作在 TP 间重复计算
```

o_proj 后的 **AllReduce** 是主要通信开销。以 TP=2, Ring 算法为例：

```
AllReduce 通信量 (per rank):  2 × (TP-1)/TP × S = S     (S = T × H_out)
AllReduce 时间:               T_ar = S / B               (B = HCCS 带宽)
```

从 profiling 数据（no-dbo, AI_CPU 模式，rank 0, `communication_matrix.json` → triage 汇总），HCCS 实际有效带宽：

| 通信类型 | HCCS 带宽 | 数据来源 |
|---|---|---|
| AllReduce | **19.0 GB/s** | 53.5 GB / 2.82s, rank 0↔1 |
| AllGather | **19.4 GB/s** | 71.7 MB / 3.70ms, rank 0↔1 |

> 注意：这是整个 profiling 窗口的有效带宽（含 HCCL 调度开销），非峰值硬件带宽。

### 1.2 FlashComm1 做了什么

核心改造：**把 o_proj 后的 AllReduce 拆成 AllGather（前置） + ReduceScatter（后置）**。

```
传统 TP:
  [全量 token] → QKV(无 AG) → Attn → o_proj → AllReduce → [全量 token]
                                                      ↑
                                           per-token 操作在所有 rank 上重复

FlashComm1:
  [T/TP token] → AllGather → QKV → Attn → o_proj → ReduceScatter → [T/TP token]
       ↑                                                            ↑
  每 rank 只持有部分 token                                    每 rank 只持有部分 token
  → RMSNorm/quant 等 per-token ops 只需处理 T/TP 个 token 而非 T 个
```

**通信量变化**：

| 操作 | 传统 TP | FlashComm1 | 变化 |
|---|---|---|---|
| 前置 | — | AllGather: (TP-1)/TP × T × H | +新增 |
| 后置 | AllReduce: 2×(TP-1)/TP × T × H | ReduceScatter: (TP-1)/TP × T × H | **−50%** |
| **总通信量** | 2×(TP-1)/TP × T × H | 2×(TP-1)/TP × T × H | 相同 |

通信量本身不减少。**FlashComm1 的收益不来自通信量减少，而来自：**

1. **消除冗余计算**：RMSNorm、per-token dynamic quant、压缩映射等操作不再在 TP 间重复执行。TP=2 时，这些操作的计算量减半。
2. **更好的流水线**：AllGather → QKV → Attn → o_proj → ReduceScatter 形成连续流水线，AllReduce 需要等所有 rank 的 partial results 再 sum，ReduceScatter 可以边算边散。

### 1.3 Profiling 验证

从 profiling 数据（AIV, DBO+FC1），每层通信分解：

```
Per layer communication (from communication.json):
  AllGather:     avg 803 μs × ~2.4 ops/layer  ≈ 1930 μs/layer  (MLA AG + EP AG)
  ReduceScatter: avg 1266 μs × ~1.8 ops/layer ≈ 2280 μs/layer  (o_proj RS + EP RS)
  Total comm per layer:                        ≈ 4210 μs/layer
```

对比：如果没有 FC1，o_proj 后的 AllReduce（等于 AG + RS 的时间）≈ 803 + 1266 = 2069 μs，但 FC1 还需要前置 AllGather（额外 803 μs），同时节省了每层 ~1000+ μs 的冗余 per-token 计算（RMSNorm、quant 等每个 token 都要做）。

**Benchmark 数据支持**（prefill4k, DeepSeek-V2-Lite）：

| 配置 | TTFT | QPS |
|---|---|---|
| DBO only, AIV | 5959ms | 6.40 |
| DBO + FC1, AIV | 5778ms (−3.0%) | 6.56 (+2.5%) |
| DBO + FC1, AI_CPU | 5027ms (−15.6%) | 7.61 (+18.9%) |

### 1.4 与 MatMul+ReduceScatter Fused Op 的关联

参考 [AscendC 最佳实践：MatMul + ReduceScatter Custom Op](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/22_matmul_reduce_scatter_custom)：

```
y = ReduceScatter(x1 × x2)

传统做法:
  Step 1: MatMul → 完整 [M, N] 输出 (每个 rank)
  Step 2: ReduceScatter → 对 M 维 reduce 后 scatter 到各 rank → [M/rankDim, N]

Fused 做法:
  MatMul tile 计算 → 本 tile 的 partial 结果直接进入 ReduceScatter
  → 不物化完整输出 → 节省 [M, N] 大小的中间 buffer
  → Cube(Vector) 做 MatMul + Vector 做 ReduceScatter 在硬件上流水线并行
```

FlashComm1 在概念上做了类似的事情：**把 o_proj MatMul 和 ReduceScatter 紧密耦合**，消除了全量 AllReduce 的中间 buffer。区别在于 FlashComm1 在 PyTorch 层做（custom op 拆分 + 通信组重排），而 AscendC 例子在 C++ kernel 层做。

---

## 二、DBO Overlap 数学推导

### 2.1 问题形式化

对 DeepSeek-V2-Lite 的第 i 层，定义以下操作序列（per ubatch）。所有数值来自 `flashcomm1_aiv` profiling 的 `op_statistic.csv` 和 `communication.json`，取 per-op 平均值，同一类别多 kernel 的做求和：

```
符号    操作                     通信/计算    时间(μs)*   数据来源
──────────────────────────────────────────────────────────────────
C₁      MLA AllGather             通信          803       comm.json allgather avg
K₁      MLA Attention 计算        计算         1317       FusedInferAttentionScore avg
C₂      o_proj ReduceScatter      通信         1266       comm.json "other" avg
K₂      RMSNorm + MoE Routing     计算         1317       AddRmsNormBias+MoeGatingTopK+MoeInitRoutingCustom
C₃      EP AllGather              通信          803       comm.json allgather avg
K₃      Expert 计算 (GM+Swiglu)   计算         1397       GroupedMatmul+SwiGlu
C₄      EP ReduceScatter          通信         1266       comm.json "other" avg
──────────────────────────────────────────────────────────────────
T_comm  Σ Cᵢ                                   4138
T_comp  Σ Kᵢ                                   4031
T_layer = T_comm + T_comp                      8169
```

> \* 注意：
> - C₁/C₃ 均使用 comm.json 的 allgather 总平均 (0.803ms)，但实际 MLA AG（tensor 较小）和 EP AG（tensor 较大）的单次时间不同。受限于 Level1 profiling 不区分通信调用来源，此处取总平均近似。
> - C₂/C₄ 使用 comm.json 的 "other" 类别 avg (1.266ms)，该类别包含 ReduceScatter 和少量其他通信。
> - K₂ = AddRmsNormBias(187.5) + MoeGatingTopK(212.8) + MoeInitRoutingCustom(916.1) = 1316.4μs（来自 AIV op_statistic.csv per-op avg）
> - K₃ = GroupedMatmul(1157.3) + SwiGlu(239.5) = 1396.8μs（同上）

### 2.2 DBO 乒乓 Overlap 的时序模型

DBO 将 batch 对半切为 ubatch[0] 和 ubatch[1]，两条 NPU stream 交替推进。

DeepSeek A2 overlap 模板定义的同步事件：

```
Event         record 位置              wait 位置
─────────────────────────────────────────────────
ATTN_PRE      C₄ 之后 (ubatch[0])      C₁ 之前 (ubatch[1])
ATTN_POST     C₂ 之后 (ubatch[0])      C₃ 之前 (ubatch[1])
```

**时序展开**（时序图）：

```
时间 →

ubatch[0] stream:
  [ C₁ ] [   K₁   ] [ C₂ ] ──record ATTN_POST──→ [ K₂ ] [ C₃ ] [   K₃   ] [ C₄ ] ──record ATTN_PRE──→
                                                         ↑                              ↓
ubatch[1] stream:                                          │                              │
  ──wait ATTN_PRE──→ [ C₁ ] [   K₁   ] [ C₂ ] ──record ATTN_POST──→ [ K₂ ] [ C₃ ] [   K₃   ] [ C₄ ] ──
       ↑
       └── ubatch[0]'s C₄ (EP RS) + ubatch[1]'s C₁ (MLA AG) 与
           ubatch[1]'s K₁+K₂ (Attn + Norm/Route) 重叠
```

### 2.3 重叠窗口分析

两个 ubatch 的执行被同步事件划分为 3 个重叠窗口：

**Window 1**: ubatch[0] `{C₃, K₃, C₄}` ↔ ubatch[1] `{C₁, K₁, C₂}`

```
ubatch[0]: [C₃ (803)] [K₃ (1397)] [C₄ (1266)]  → record ATTN_PRE
ubatch[1]:                    [C₁ (803)] [K₁ (1317)] [C₂ (1266)]  → record ATTN_POST
                              ↑ ubatch[1] 等 ATTN_PRE 后启动
```

ubatch[0] 通信块 `C₃ + C₄ = 803 + 1266 = 2069 μs` 与 ubatch[1] 计算块 `K₁ = 1317 μs` 的重叠度：
- 重叠时间: `min(2069, 1317) = 1317 μs`
- 暴露通信: `max(0, 2069 - 1317) = 752 μs`

**Window 2**: ubatch[0] `{C₁(next_layer), K₁(next_layer)}` ↔ ubatch[1] `{K₂, C₃, K₃}`

```
ubatch[0]: [C₁ (803)] [K₁ (1317)] [C₂ (1266)]
ubatch[1]:     [K₂ (1317)] [C₃ (803)] [K₃ (1397)]
```

ubatch[1] 计算块 `K₂ + K₃ = 1317 + 1397 = 2714 μs` 与 ubatch[0] 通信块 `C₁ + C₂ = 803 + 1266 = 2069 μs` 的重叠度：
- 重叠时间: `min(2714, 2069) = 2069 μs`
- 暴露计算: `max(0, 2714 - 2069) = 645 μs`

**Window 3**: ubatch[0] `{K₂, C₃, K₃}` ↔ ubatch[1] `{C₂, K₂, next_C₁}`

类似分析。

### 2.4 单层 DBO 加速比公式

定义：

$$T_{comm} = \sum_{j} C_j \quad T_{comp} = \sum_{j} K_j \quad T_{layer} = T_{comm} + T_{comp}$$

DBO 通过乒乓同步将 T_comm 和 T_comp 分到两条 stream 上并行：

$$T_{layer}^{dbo} = \max(T_{comm}, T_{comp}) + T_{sync}$$

其中 T_sync 是 GPU event record/wait + CPU thread yield 的总开销。

**加速比**：

$$S = \frac{T_{layer}}{T_{layer}^{dbo}} = \frac{T_{comm} + T_{comp}}{\max(T_{comm}, T_{comp}) + T_{sync}}$$

代入 profiling 数据（AIV, per layer per ubatch）：

$$S = \frac{4138 + 4031}{\max(4138, 4031) + T_{sync}} = \frac{8169}{4138 + T_{sync}}$$

当 T_sync → 0 时，S → 8169/4138 ≈ 1.97×。实际 overlap 率为 73.3%，反推：

$$1 - \frac{T_{sync} + T_{imbalance}}{T_{comm}} = 0.733$$

$$T_{sync} + T_{imbalance} = 0.267 \times 4138 \approx 1105 \mu s$$

即每层约 1.1ms 的同步+不均衡开销不能被 overlap 覆盖。

### 2.5 Overlap 效率的约束条件

**约束 1: 通信-计算平衡**

每个 overlap window 内，通信时间和计算时间必须接近：

$$\min(T_{comm\_window}, T_{comp\_window}) / \max(T_{comm\_window}, T_{comp\_window}) \to 1$$

如果 T_comm >> T_comp，则通信暴露；如果 T_comp >> T_comm，则计算空闲。

**约束 2: Hook 可用性**

DBO hook 只能包裹通信算子。两个通信之间如果没有通信操作，无论中间计算多长，都无法插入 event：

```
[C₂: o_proj RS] → [纯计算: RMSNorm + MoE Gate + token routing] → [C₃: MoE Prep AG]
                  ↑ 没有通信算子 = 没有 hook = 无法插入 event
```

因此 C₂ 和 C₃ 在实际模板中被一对 ATTN_POST event 包裹为一个整体。这不是因为中间计算短（实测 gap ~2.9ms），而是因为中间没有可用的 hook 点。

**约束 3: 前向上下文一致性**

第一个 layer 的 `dbo_first_layer_sync` 需要特殊处理 —— ubatch[1] 的第一个 C₁ 没有 ubatch[0] 的前序 C₄ 可重叠，必须等 ubatch[0] 先完成其 C₁ 并 record event。

### 2.6 Overlap 界限的理论推导

参考 [MatMul+ReduceScatter Fused Op](https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/22_matmul_reduce_scatter_custom) 的分析方法，DBO overlap 的**上界**由以下因素决定：

**上界 1: 通信带宽上界**

对 ReduceScatter（TP=2, Ring），通信时间下界：

$$T_{rs}^{min} = \frac{S_{data}}{B_{HCCS}} \times \frac{TP-1}{TP}$$

其中 B_HCCS 是实测 HCCS 带宽（**19.0 GB/s**, no-dbo triage `communication_matrix.json` → HCCS allreduce 汇总）。

**上界 2: 计算吞吐上界**

对 MatMul（M=token_count, K=hidden, N=intermediate），计算时间下界：

$$T_{matmul}^{min} = \frac{2 \times M \times K \times N}{FLOPS_{AIC}}$$

其中 FLOPS_AIC 是 AI Cube 的峰值算力。对于 MIX_AIC kernel，实际时间受 Cube+Vector 流水线效率影响，需加 stall 系数 α ≥ 1：

$$T_{matmul}^{actual} = \alpha \cdot T_{matmul}^{min}$$

从 profiling 数据，AIV 的 α_AIV ≈ 1.8-3.2（受通信争 Vector 核影响），AI_CPU 的 α_AI_CPU ≈ 1.0-1.2。

**上界 3: Overlap 效率上界**

DBO 能达到的最大 overlap 率：

$$\eta_{max} = \frac{\min(T_{comm}, T_{comp})}{T_{comm}}$$

当 T_comm ≤ T_comp 时，理论重叠率 = 100%（通信完全隐藏在计算后）。当 T_comm > T_comp 时，最多重叠 T_comp/T_comm。

对于我们的 profiling 数据，T_comm (4138) > T_comp (4031)，理论最大 overlap 率:

$$\eta_{max} = \frac{4031}{4138} = 97.4\%$$

实际 overlap 率 73.3%，差距来自 T_sync（~1105μs/layer）——即 DBO 的 GPU event + thread yield 开销。

---

## 三、改进方向

### 3.1 减少同步开销

当前每层 ~1.1ms 的不可 overlap 开销估算（未逐项 profiling 验证）：
- GPU event record/wait: ~10μs/次（同域 NPU event，来源：kernel_details wait time P50≈0 反推）
- CPU thread yield (threading.Event + GIL): 数量级估计 10-50μs/次
- 实际 T_sync ≈ 1105μs 来自公式反推（η_max − η_actual = T_sync / T_comm），非直接测量
- 长尾 wait（AI_CPU 模式下的跨域调度延迟）→ 改用 AIV 模式可显著缓解

### 3.2 改善通信-计算平衡

当前 T_comm (4138μs) > T_comp (4031μs)，差距 ~2.6%。优化方向：
- 增大 batch size → T_comp 增长（O(batch_size)）速度快于 T_comm（O(batch_size/TP)），平衡改善
- 使用 FC2 减少 o_proj 通信 → T_comm 降低

### 3.3 MatMul+ReduceScatter Fusion 的借鉴

AscendC 的 fused op 展示了如何将通信与计算在硬件层面流水线化。对 DBO 的启示：
- 如果能将 o_proj MatMul 和 ReduceScatter 融合为单个 AscendC kernel，可消除一次 DBO 同步点
- 融合后通信不再作为独立 block 出现，DBO overlap 策略需相应调整

---

## 四、冷启动抖动

> 冷启动抖动实验数据见 **[`benchmark-doc.md`](benchmark-doc.md) Section 2**（权威源）。
> 核心结论：首请求 65-70x 延迟来自 ACLGraph capture，非 recompile；warmup 完全消除；DBO 对抖动无影响。

---

## 参考文献

- AscendC MatMul+ReduceScatter: https://gitee.com/ascend/samples/tree/master/operator/ascendc/4_best_practices/22_matmul_reduce_scatter_custom
- [`benchmark-doc.md`](benchmark-doc.md) — Benchmark 数据 + 冷启动抖动（权威源）
- [`profile-deep-analysis-4way.md`](profile-deep-analysis-4way.md) — AI_CPU vs AIV Profiling 深度分析（权威源）
- `testbench/MOE/dbo/docs/flashcomm/flashcomm.md` — FlashComm1/2 原理
- `vllm_ascend/dbo/overlap_templates/deepseek.py` — DeepSeek overlap 模板
