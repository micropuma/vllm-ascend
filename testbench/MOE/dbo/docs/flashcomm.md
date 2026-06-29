# FlashComm算子库原理

这份文档把你给的 FlashComm 论文内容，逐项对照到 `vllm-ascend` 里的实现。

目标只有一个：

- 说明 FlashComm1 / FlashComm2 在代码里到底落在哪些层
- 说明它们为什么能减少通信量和冗余计算
- 说明它们为什么还能和 DBO 叠加

---

## TLDR  

> 传统通信并行分为：TP + DP + EP + SP。而华为通信算子库分为：Flashcomm1 + Flashcomm2 + Flashcomm3。并行算法 和 华为通信优化是正交的，排列组合分析对于模型推理性能影响比较困难，也难以理解。 此部分以TP 和 MOE模型 为例，解析flashcomm通信库的作用。

### TP并行 + flashcomm通信优化  
#### 传统TP做了什么？

   ```shell
   输入 X：replicated [T,H]
            │
            ▼
   QKV column-parallel
            │
      ┌─────┴─────┐
      │           │
   rank 0         rank 1
   heads 0~15     heads 16~31
      │           │
   local attn     local attn
      │           │
   O0 [T,H/2]    O1 [T,H/2]
      │           │
   o_proj row-parallel
      │           │
   Y0 [T,H]      Y1 [T,H]
      └─────┬─────┘
            │
      AllReduce / RS
            │
            ▼
   完整 Attention output
   ```

   理解如下规律：  

   * 每个TP rank拿到完整的输入token。
   * QKV projection对Wqkv做列划分，因为该计算只用all gather语义（concat即可，实际不用all gather，因为后续attn可以沿用）。
   * attn实际计算是多头的，之前的**QKV project切分不会破坏多头粒度，所以省去一次 all gather**。
   * O projection需要对所有头做统一的矩阵乘。所以使用行划分。列划分 乘以 行划分，需要all reduce。

### flashcommv1做了什么？  

* 分析传统TP的劣势：

   > QKV projection、按 head 切分的 Attention、o_proj 本地 GEMM 都基本不变；它把链路入口从“每卡持有全部 token”改成“每卡只持有部分 token”，并把 o_proj 后的 TP AllReduce 改成 TP ReduceScatter。

* flashcommv1做了什么：

   > * 在模型forward中，有很多诸如rmsnorm或是量化操作，大多是per token操作。 对于传统TP，在all reduce后每个rank持有完整token，引入大量重复运算。   
   > * flashcomm1将一次完整all reduce 拆分成 all gather + reduce scatter。通过分析model forward链路，发现是qkv projection后，需要全局token（attn注意力模块），所以all gather插入qkv projection后。  

* 优势是：      
   > 在传统 Megatron-style Tensor Parallel 中，Transformer 子层边界处的 hidden states 通常在同一 TP group 内保持 replicated：每个 TP rank 都持有其所属 DP replica 当前 batch 的全部 token，以及每个 token 的完整 hidden vector。因此，位于这些边界上的 RMSNorm、per-token dynamic quant、部分压缩映射等 token-local 操作，可能在所有 TP rank 上重复执行。

   > FlashComm1 将 row-parallel 层后的 AllReduce 改为沿 token 维的 ReduceScatter，使每个 TP rank 只保留部分 token、但仍持有这些 token 的完整 hidden vector。这样，RMSNorm、per-token quant 等操作可以在本地 token shard 上执行；直到下一次 column-parallel 层真正需要全部 token 时，再执行 AllGather。

如下图是完整的链路：  

```shell
X：sequence-sharded
每个 rank 只有 T/P 个 token
        │
        ▼
Local RMSNorm
只处理本地 token
        │
        ▼
TP AllGather
恢复当前 TP group 的全部 token
        │
        ▼
QKV column parallel
        │
        ▼
Local-head Attention
        │
        ▼
o_proj row parallel
每个 rank 得到 partial output [T,H]
        │
        ▼
TP ReduceScatter
先对 TP partial output 求和
再沿 token 维把结果切回各 rank
        │
        ▼
Y：sequence-sharded [T/P,H]
        │
        ▼
Local Residual Add
```

#### 对于MOE模型有什么作用？  

MoE 与 Attention 的总体思想相同：

> 先让每个 rank 只保留部分 token，在本地完成 Norm、Router、Quant 等逐 token 操作；只有进入专家计算时，才执行 EP 范围的 token 汇聚。

不同之处是：
   * Attention 使用 TP AllGather；
   * MoE 使用 EP AllGather；
   * MoE 结束后使用 EP ReduceScatter。

这里重点讨论一下 TP + DP + EP的all gather问题。 

### flashcommv2做了什么？

> 想要解决什么问题？

回顾 FlashComm1 的链路：Attention 输出经过 o_proj（row-parallel）后，执行 TP **ReduceScatter** 沿 token 维切分结果，使每个 rank 只保留 `T/TP` 个 token 的完整 hidden state。这种通信模式有其优劣：

* 优势：下游的 RMSNorm、per-token quant 等操作只需在本地 token shard 上执行，避免 TP 间冗余计算。
* 劣势：o_proj 的 ReduceScatter 通信量 = `T * H_out`（每个 rank 都产出完整 [T, H_out] 的 partial 结果，RS 阶段需要先 sum 再 scatter）。

此外，在 DP > 1 的场景下，o_proj 之后的 token 仍按 DP 维度分布，后续 MoE 需要先 DP AllGather 再 EP 分发，通信链路长。

> FlashCommv2 的核心思路

一句话概括：在 ODP group（output-data-parallel，跨 DP 维度）内做 AllToAll，把 Attention 的 **head 维切分** 转换为 **sequence 维切分**，从而将 o_proj 从"TP 计算 + 大通信"转为"DP 计算 + 小通信"，并通过扩展 weight 输入维度消除 ODP 维度对应的输出规约。

实现上，FlashComm2 将全局 TP 拆成两级通信组（以 TP=8, otp=4 为例）：

```
全局 TP=8 拆分为：OTP size=4, ODP size=2

OTP (Output Tensor Parallel) = interleaved grouping:
  Group 0: {rank0, rank2, rank4, rank6}
  Group 1: {rank1, rank3, rank5, rank7}

ODP (Output Data Parallel) = contiguous grouping:
  Group 0: {rank0, rank1}
  Group 1: {rank2, rank3}
  Group 2: {rank4, rank5}
  Group 3: {rank6, rank7}
```

关键性质：OTP 和 ODP 是正交的（crossed），同一个 OTP group 内的 rank 分布在不同的 ODP group 中。

> 完整数据流（对照 linear_op.py:315-410）

```shell
输入 X: [T, H_h/TP]
每个 rank 已有 sequence-parallel 布局（FC1 的产物）
         │
         ▼
  [1] Pad 到 TP 对齐: [T+pad, H_h/TP]
         │
         ▼
  [2] Batch 重排 (group_indices)
  将 token 按 OTP group 归类，确保后续 AllToAll 后
  同 OTP group 的 token 在一起
         │
         ▼
  [3] ODP AllToAll (跨 odp_size 个 rank)
  在 ODP group 内交换 token → 汇总 head 维度
  结果: [T/odp, odp * H_h/TP]
         │
         ▼
  [4] o_proj MatMul
  Weight: [odp * H_h/TP, H_out/otp]
  输出: [T/odp, H_out/otp]
         │                              ← 这里输出的 H_out/otp
         ▼                                只是部分 hidden dim
  [5] OTP ReduceScatter (dim=0)
  在 OTP group 内沿 token 维归约
  结果: [T/odp/otp, H_out/otp]
         │
         ▼
  [6] 可选 TP AllGather
  ─ 如果 FlashComm1 已启用 → 跳过
  ─ 如果 FlashComm1 未启用 → 执行全局 AllGather
         │
         ▼
  输出 Y: [T/TP, H_out/otp]
```

与代码的对应关系（`vllm_ascend/ops/linear_op.py`）：

| 步骤 | 代码行 | 说明 |
|------|--------|------|
| Padding | L327-329 | `F.pad(input_parallel, (0, 0, 0, num_padding_tokens))` |
| 重排 | L341 | `chunked = chunked[self.group_indices]` |
| ODP AllToAll | L355 | `dist.all_to_all_single(recv_buf, send_buf, group=self.odp_group.device_group)` |
| MatMul | L375 | `self.quant_method.apply(self.layer, input_parallel, bias=bias_)` |
| OTP ReduceScatter | L379 | `self.comm_group.reduce_scatter(output_parallel, dim=0)` |
| TP AllGather | L383 | `get_tp_group().all_gather(output, 0)` — 仅 FC1 未启用时执行 |

> FlashComm2 vs 标准 TP vs 普通 OTP 的通信对比

以 TP=8, otp=4, odp=2 为例，设单 rank o_proj 输出张量大小 = `T * H_out/8`：

| 方案 | 通信操作 | 通信组大小 | 特点 |
|------|----------|-----------|------|
| 标准 TP | AllReduce | 8-rank | 一次全局 AllReduce（内部 = RS + AG） |
| 普通 OTP (OProjRowParallelOp) | AllToAll + ReduceScatter | 均在 OTP group 内 | OTP 组内通信，不涉及 ODP |
| **FlashComm2** | **ODP AllToAll + OTP RS + (可选 TP AG)** | ODP=2, OTP=4 | 两级小组合通信 |

> FlashComm2 的 4 个核心优势

**优势 1 (最关键)：配合 FlashComm1 消除全局 AllGather**

当 FlashComm1 启用时，`linear_op.py:401-405` 中 **TP AllGather 被直接跳过**。输出保持 `[T/TP, H]` 的 sequence-parallel 布局向下传递。

```python
# linear_op.py:401-405
if not _EXTRA_CTX.flash_comm_v1_enabled:
    output = get_tp_group().all_gather(output, 0)  # ← FC1 启用时跳过！
```

这意味着 FlashComm2 的实际通信只有 ODP AllToAll + OTP ReduceScatter，两个操作都在 **更小的子组** 内完成。相比标准 TP 需要一次全局 AllReduce（8 个 rank 参与），FlashComm2 的两级通信参与 rank 数分别为 2 和 4，对 HCCL/NIC 的争抢小得多，通信拓扑更优。

**优势 2：W8A8 量化让 AllToAll 通信量减半**

当启用 W8A8 量化时（`w8a8_static.py:90-108`），ODP AllToAll 传输的是 int8 量化后的数据，而不是 fp16：

```python
# w8a8_static.py:90-108
quant_x = torch.ops.vllm.quantize(x, ...)  # fp16 → int8, 大小减半
x = comm_fn(comm_input)                     # AllToAll 传的是 int8
```

相比标准 TP 的 AllReduce 无法做这种量化+通信融合（AllReduce 需要 float32 做归约），FlashComm2 的 AllToAll 是 pure data movement，天然可以传量化数据，**直接节省 50% 的 ODP 通信带宽**。

**优势 3：小组合通信拓扑更优**

标准 TP AllReduce 横跨所有 8 个 rank，NCCL ring 跳数长、带宽利用不充分。FlashComm2 拆成：
- ODP AllToAll（2 个 rank）：ring 只有 2 跳
- OTP ReduceScatter（4 个 rank）：ring 只有 4 跳

小组合通信的总开销（启动延迟 × 跳数）显著低于一次大组 AllReduce。这在 Ascend HCCL 上尤其明显。

**优势 4：支持 Weight O-Shard 显存优化**

FlashComm2 配合 `layer_sharding=["o_proj"]` 时（见 `flashcomm2_oshard_manager.py`），o_proj 权重可以在 rank 间分片存储、按层异步 broadcast 加载，减少单卡显存占用。每张卡只需常驻 `1/otp` 的 o_proj 权重。

> 代价："以存换传"的本质

FlashComm2 的 weight 输入维度从 `H_h/TP` 扩展为 `odp * H_h/TP`，即扩大了 odp 倍。但注意：
- 全局 weight 总量不变：`(odp * H_h/TP) * (H_out/otp) = H_h * H_out / TP`（与传统 TP 相同）
- 只是同一 rank 需要加载的 weight 列数增多（输入维扩大），输出维减少（H_out/otp vs H_out/TP）
- 真正的额外存储被 O-Shard 机制分担

> 总结

| 维度 | 标准 TP | FlashComm2 |
|------|---------|------------|
| 通信模式 | AllReduce (8-rank) | ODP AllToAll (2-rank) + OTP RS (4-rank) |
| FC1 集成 | 与 FC1 无耦合 | FC1 启用时跳过 TP AllGather |
| 量化融合 | 不支持 | W8A8 让 AllToAll 传 int8 |
| 通信拓扑 | 大组 (8-rank ring) | 小组 (2-rank/4-rank ring) |
| Weight 存储 | H_h/TP × H_out/TP | odp×H_h/TP × H_out/otp (O-Shard 可分担) |
| 本质 | TP 计算 + TP 通信 | DP 计算 + 小组合通信 |

#### O Shard优化  

> TLDR: FlashComm2 OShared 是一套 O-Proj 权重的 layer-sharding 和 runtime prefetch/broadcast 机制。它不减少 ODP AllToAll，也不改变 FlashComm2 的矩阵乘数学语义；它只是用额外的权重通信和更复杂的 stream/event 管理，换取更低的单卡权重常驻显存。  

```shell
FlashComm2 OShared / O-Shard

1. OShared 关闭：所有设备常驻全部 O-Proj 权重

Rank 0                           Rank 1
┌─────────────────────┐        ┌─────────────────────┐
│ W_o Layer 0         │        │ W_o Layer 0         │
│ W_o Layer 1         │        │ W_o Layer 1         │
│ W_o Layer 2         │        │ W_o Layer 2         │
│ ...                 │        │ ...                 │
└─────────────────────┘        └─────────────────────┘
          │                              │
          └──── 每张卡都保存所有层权重 ────┘

优点：计算时无需搬运权重
缺点：单卡显存占用较大


2. OShared 开启：权重分层存储，运行时广播

Rank 0                           Rank 1
┌─────────────────────┐        ┌─────────────────────┐
│ W_o Layer 0         │        │ W_o Layer 1         │
│ W_o Layer 2         │        │ W_o Layer 3         │
└─────────────────────┘        └─────────────────────┘

执行 Layer 0：

Rank 0                                            Rank 1
W_o Layer 0 ─────── weight broadcast ──────────> 临时权重 Buffer
      │                                                   │
      └─────────── 两张卡分别计算不同 token ───────────────┘

执行 Layer 1：

Rank 0                                            Rank 1
临时权重 Buffer <──── weight broadcast ───────── W_o Layer 1
      │                                                   │
      └─────────── 两张卡分别计算不同 token ───────────────┘


3. FlashComm2 完整计算路径

Attention 输出
    │
    ▼
ODP AllToAll
将「所有 token × 部分 feature」
转换为「部分 token × 完整/更多 feature」
    │
    ▼
等待 O-Proj 权重就绪
    │
    ├── OShared 关闭：读取本卡常驻权重
    │
    └── OShared 开启：读取预取/广播得到的临时权重
    │
    ▼
O-Proj MatMul
    │
    ▼
OTP ReduceScatter
    │
    ▼
可选 TP AllGather


核心关系：

FlashComm2
  └── 优化激活通信：AllToAll 前置到 O-Proj MatMul 之前

OShared / O-Shard
  └── 优化权重存储：分散常驻 + 异步广播 + prefetch + 临时复用

注意：
OShared 关闭后，ODP AllToAll 仍然存在。
因此 OShared 不是 FlashComm2 AllToAll 的开关。
```



##  TP + DP + EP 场景下的通信融合

> 有了前面的基础，接下来来看真实的场景：TP + DP + EP下的通信算子融合。

假设：

```text
TP = 2
DP = 2
EP = TP × DP = 4
```

四个 rank 可以表示成：

```text
                 TP 方向
              t0        t1
          ┌────────┬────────┐
DP d0     │   r0   │   r1   │
          ├────────┼────────┤
DP d1     │   r2   │   r3   │
          └────────┴────────┘
```

对应的通信组为：

```text
TP group：
  {r0, r1}
  {r2, r3}

DP group：
  {r0, r2}
  {r1, r3}

EP group：
  {r0, r1, r2, r3}
```

其中，EP group 覆盖了完整的 `TP × DP` rank 网格。

---

### 1. FlashCommV1 关闭时

假设：

```text
DP d0 处理 batch A
DP d1 处理 batch B
```

Attention 的 `o_proj` 后执行 TP AllReduce，因此：

```text
r0：A
r1：A

r2：B
r3：B
```

同一 TP group 内的 hidden states 是重复的。

进入 MoE 前，为了让各个专家 rank 看到所有 DP batch，需要执行 DP AllGather：

```text
TP AllReduce
        │
        ▼
r0=A，r1=A，r2=B，r3=B
        │
        ▼
DP AllGather
        │
        ▼
所有 rank 都得到 [A,B]
```

MoE 计算完成后，通常先执行 DP ReduceScatter，再执行 TP AllReduce：

```text
Local Experts
        │
        ▼
DP ReduceScatter
将结果返回不同 DP batch
        │
        ▼
TP AllReduce
合并 TP rank 上的专家贡献
        │
        ▼
r0=A，r1=A，r2=B，r3=B
```

因此传统路径可以简化为：

```text
Attention
→ TP AllReduce
→ DP AllGather
→ MoE
→ DP ReduceScatter
→ TP AllReduce
```

---

### 2. FlashCommV1 开启时

FlashCommV1 将 Attention 后的 TP AllReduce 改为 TP ReduceScatter。

假设 batch A 和 batch B 分别被切成两个 token shard：

```text
A = [A0, A1]
B = [B0, B1]
```

TP ReduceScatter 后：

```text
r0：A0
r1：A1

r2：B0
r3：B1
```

此时四个 rank 持有的是互不重复的 token shard：

```text
A0 + A1 + B0 + B1 = 全部 token
```

因此，不再需要先执行 TP AllGather、再执行 DP AllGather，而是可以直接执行一次 EP AllGather：

```text
r0=A0
r1=A1
r2=B0
r3=B1
        │
        ▼
EP AllGather
        │
        ▼
所有 rank 都得到 [A0,A1,B0,B1]
```

因此：

```text
TP AllGather + DP AllGather
```

可以融合为：

```text
EP AllGather
```

即：

```text
TP AG → DP AG
        ↓
      EP AG
```

---

### 3. MoE 结束后的 EP ReduceScatter

EP AllGather 后，每个 rank 都看到所有 token，但只计算本地专家。

设不同 rank 的专家输出分别为：

```text
Z0、Z1、Z2、Z3
```

完整 MoE 输出为：

```text
Y = Z0 + Z1 + Z2 + Z3
```

传统路径需要：

```text
DP ReduceScatter
→ TP ReduceScatter
```

先沿 DP 维度归约，再沿 TP 维度归约和切分。

由于 EP group 恰好覆盖完整的 `TP × DP` rank 网格，这两级通信可以合并成一次 EP ReduceScatter：

```text
Z0、Z1、Z2、Z3
        │
        ▼
EP ReduceScatter
        │
        ├── r0 得到 Y[A0]
        ├── r1 得到 Y[A1]
        ├── r2 得到 Y[B0]
        └── r3 得到 Y[B1]
```

因此：

```text
DP ReduceScatter + TP ReduceScatter
```

可以融合为：

```text
EP ReduceScatter
```

即：

```text
DP RS → TP RS
        ↓
      EP RS
```

---

### 4. FlashCommV1 下的完整 MoE 链路

最终，FlashCommV1 下的 MoE 链路可以表示为：

```text
Attention
        │
        ▼
TP ReduceScatter
每个 rank 得到唯一的 token shard
        │
        ▼
Local RMSNorm / Router / Quant
        │
        ▼
EP AllGather
收集 TP × DP 范围内的全部 token
        │
        ▼
Local Experts
每个 rank 只计算本地专家
        │
        ▼
EP ReduceScatter
汇总专家贡献并返回 token owner
        │
        ▼
Sequence-Sharded Hidden States
```

与传统路径相比：

```text
传统路径：

TP AllReduce
→ DP AllGather
→ MoE
→ DP ReduceScatter
→ TP AllReduce
```

FlashCommV1 路径：

```text
TP ReduceScatter
→ EP AllGather
→ MoE
→ EP ReduceScatter
```

---

### 5. 为什么这种融合成立

这种通信融合需要满足以下条件：

1. `EP group` 覆盖完整的 `TP × DP` rank 集合；
2. TP ReduceScatter 后，各 rank 持有的 token shard互不重复；
3. 所有 token shard 合起来能够覆盖完整 token 集合；
4. EP AllGather 和 EP ReduceScatter 使用一致的 rank 顺序；
5. hidden states、router logits、quant scale 等相关 tensor 使用相同的 token 排列。

其本质是将二维分层通信：

```text
先沿 TP 方向通信
再沿 DP 方向通信
```

展平为一次覆盖整个 `TP × DP` rank 网格的 EP collective：

```text
TP AG + DP AG → EP AG

DP RS + TP RS → EP RS
```

因此，FlashCommV1 在 MoE 场景中的核心作用是：

> 让 Attention 输出的 TP token shard 直接衔接 MoE 的 EP 通信，避免先恢复 TP replicated 状态，再重新进行 DP/EP 数据分发。
 

