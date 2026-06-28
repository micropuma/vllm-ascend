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

传统TP通信，在O projection阶段会执行 `reduce scatter`通信。该通信是：（1）X是完整token维度，部分hidden dim维度（H/TP_SIZE）；（2）weight是（H/TP_SIZE）x H 维度。这种通信模式有其优劣：  

* 优势：weight每卡只用存储 H/TP_SIZE X H，大大减少了存储量  
* 劣势：最后做的通信，通信量很大：H X H   

> FlashCommv2 怎么做的？  

首先先用一句话概括：对于attention的输出做 AllToAll 把 head 维切分转换成 sequence 维切分，使得后续O projection 从 TP 计算转成 DP 计算。其逻辑链条如下：  

* 先对 Attention 输出做 all to all        
   * 原来每张卡持有：所有 token × 部分 head   
   * 现在每张卡持有：部分 token × 所有 head       
具体过程如下：  
        ```shell
        1. 每张卡把自己的 T 个 token 切成 P 份；
        2. 第 j 份 token 发给第 j 张卡；
        3. 每张卡收到同一批 token 在所有 TP rank 上的 head 分片；
        4. 沿 head 维拼接。
        ```
* 优势：通信迁移，并且只用传输 (H / TP) x H。  
* 劣势：weight矩阵给是完整的。  

一个总结对比图：  

```shell
传统：
[T, 2048]
   ↓ 局部 o_proj
[T, 7168]
   ↓ ReduceScatter
[T/8, 7168]

FlashComm2：
[T, 2048]
   ↓ AllToAll
[T/8, 16384]
   ↓ 完整 o_proj
[T/8, 7168]
```

> 多节点情况下怎么弄？  

对于大模型而言，flashcomm2会显著增加存储量。所以如果集群很大，可以按照如下配置方案：  

将跨节点的 All2All替换成节点内All2All和一次矩阵乘后的跨节点ReduceScatter操作。该方法将Output 投影矩阵计算由TP $N$转化为DP8+TP $N/8$，而非节点间All2All的技术的DP $N$，其中$N$为节点数。此方法相比于节点间All2All的技术，显著减少了单卡所需载入的权重，且以节点间两卡ReduceScatter的操作替换了原跨节点All2All操作。

> 总结：（1）以存换传（2）TP计算 转成 DP计算

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
 

