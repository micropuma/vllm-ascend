# TP/DP、FlashComm1/2 与 DBO 的统一 Shape Contract

## 1. 目的与适用范围

本文基于当前源码和
[`rfc-dbo-cold-start-problem.md`](../rfc/rfc-dbo-cold-start-problem.md)
梳理 token 维度的统一 shape contract，重点回答三个问题：

1. TP/DP本身要求哪些pad、gather、scatter与unpad；
2. FlashComm1/FlashComm2如何改变token维的分布与生命周期；
3. DBO把一个batch拆成两个ubatch后，为什么必须同时保存graph/padded与logical
   两套长度，以及它如何在不改变数学shape的前提下改变通信时序。

本文以当前源码（包含 `4fc82d04` shape修复）为主要实现依据。RFC保留了
2026-07-01的故障发现过程，其中若干“当前仍失败”结论已被后续提交修正，因此本文
将RFC作为历史证据，而不是直接把其中每个中间判断视为当前实现。
[VERIFY: testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md:1](../../../../testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md#L1)

本文聚焦：

- token维，hidden维只在FlashComm2布局变换时展开；
- TP、DP/EP、FlashComm1、FlashComm2、DBO；
- eager、torch.compile fake shape与ACL Graph buffer之间的一致性。

PCP、DCP、PP、speculative decoding只在与主contract直接相交处提及，不作为本文
完整覆盖范围。

---

## 2. 为什么必须建立多套token长度

### 2.1 一个 `num_tokens` 不足以描述当前系统

当前执行路径至少同时存在以下长度：

| 符号 | 含义 | 典型所有者 |
|---|---|---|
| `L_d` | DP rank `d` 的业务逻辑token数 | scheduler / attention metadata |
| `S_d` | scheduler/graph为rank `d` 准备的tensor row数 | `BatchDescriptor` / model input |
| `C_d` | DP collective metadata声明的rank `d` 有效长度 | `num_tokens_across_dp_cpu` |
| `M` | DP组内需要对齐的最大token数 | `max_tokens_across_dp` |
| `A` | 为TP/EP通信对齐后的每rank长度 | `padded_length` |
| `q` | 某次TP通信临时引入的padding | `pad_size`或tensor shape公式 |
| `R` | TP sequence-local shard长度 | ReduceScatter输出 |
| `L_(d,u)` | DP rank `d` 上ubatch `u` 的logical token数 | per-ubatch metadata |
| `S_(d,u)` | DP rank `d` 上ubatch `u` 的padded slice长度 | `UBatchSlices` |

当前model runner把graph/padded ubatch slices与attention/logical slices同时传入
forward context，说明两套长度是显式contract，而不是实现细节。
[VERIFY: vllm_ascend/worker/model_runner_v1.py:2221](../../../../vllm_ascend/worker/model_runner_v1.py#L2221)
[VERIFY: vllm_ascend/worker/model_runner_v1.py:2234](../../../../vllm_ascend/worker/model_runner_v1.py#L2234)

`C_d` 必须单列，因为它不保证永远等于业务 `L_d`。例如dummy/capture路径在
dispatcher增加graph padding后，会把 `num_tokens_across_dp` 更新为
`num_tokens_padded`；此时DP collective看到的是graph contract，而不是原始业务
长度。[VERIFY: vllm_ascend/worker/model_runner_v1.py:3403](../../../../vllm_ascend/worker/model_runner_v1.py#L3403)

### 2.2 三类padding必须由引入者闭环

本文把padding分为三类：

1. **scheduler/graph padding**
   - 为batch descriptor、capture size、DP协调或固定buffer服务；
   - 生命周期可能贯穿整个model forward；
   - 对应 `S_d - L_d` 或 `S_(d,u) - L_(d,u)`。
2. **DP collective padding**
   - 为不同DP rank获得相同collective输入shape服务；
   - 每rank先pad到 `M` 或 `A`，gather/scatter后恢复本rank长度；
   - 由DP/EP prepare-finalize闭环。
3. **TP/FlashComm padding**
   - 为token维能被TP/OTP group整除服务；
   - 在某次AllGather/ReduceScatter通信边界内引入和删除；
   - 由FlashComm通信层闭环。

RFC中最早的失败正是把scheduler padding当成FlashComm padding：用已包含
scheduler pad的slice长度计算FC1 `pad_size`，使FlashComm认为无需unpad，而
attention仍按logical token工作，最终在AddRmsNormBias汇合处出现首维不一致。
[VERIFY: testbench/MOE/dbo/rfc/rfc-dbo-flashcomm1-compile-shape-contract.md:17](../../../../testbench/MOE/dbo/rfc/rfc-dbo-flashcomm1-compile-shape-contract.md#L17)
[VERIFY: testbench/MOE/dbo/rfc/rfc-dbo-flashcomm1-compile-shape-contract.md:92](../../../../testbench/MOE/dbo/rfc/rfc-dbo-flashcomm1-compile-shape-contract.md#L92)

统一原则是：

> 每一层只能删除自己引入的padding；跨层共享长度时必须标明它是logical、
> graph-padded、DP-padded还是TP-local，不能使用无语义的 `num_tokens` 猜测。

---

## 3. 基础TP contract

### 3.1 普通TP不改变token分布

传统TP主要切hidden/weight维：

```text
输入 X: [N, H]

ColumnParallel:
  每rank计算 [N, H_out / P]
  token维仍为 N

RowParallel:
  每rank计算partial [N, H_out]
  TP AllReduce
  每rank输出 [N, H_out]
```

这里 `P` 是TP size，所有TP rank都保有相同的 `N` 个token。普通
`SequenceRowParallelOp` 在FlashComm1关闭时执行matmul后TP AllReduce，输出首维
保持不变。[VERIFY: vllm_ascend/ops/linear_op.py:567](../../../../vllm_ascend/ops/linear_op.py#L567)

因此传统TP的稳定层间接口是：

```text
FULL(N) := [N, H]
```

### 3.2 Sequence Parallel把token维也纳入TP

FlashComm1把row-parallel末尾的AllReduce替换成token维ReduceScatter：

```text
FULL(N)
  → pad到 Np = ceil(N/P) * P
  → TP ReduceScatter(dim=0)
  → LOCAL(R), R = Np/P = ceil(N/P)
```

当前row-parallel实现直接从 `x.shape[0]` 计算：

```python
pad_size = (-x.shape[0]) % world_size
```

随后pad并执行ReduceScatter，因此：

```text
q_tp(N, P) = (-N) mod P
N_p        = N + q_tp
           = ceil(N / P) * P
R          = N_p / P
           = ceil(N / P)
```

[VERIFY: vllm_ascend/ops/linear_op.py:579](../../../../vllm_ascend/ops/linear_op.py#L579)
[VERIFY: vllm_ascend/ops/linear_op.py:637](../../../../vllm_ascend/ops/linear_op.py#L637)

下一次column-parallel计算需要恢复full token：

```text
LOCAL(R)
  → TP AllGather(dim=0)
  → [P*R, H] = [Np, H]
  → 删除本次TP pad qtp
  → FULL(N)
```

非DBO路径通过 `maybe_all_gather_and_maybe_unpad` 完成这一步；runtime先
AllGather，再按 `_EXTRA_CTX.pad_size` 删除尾部。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:89](../../../../vllm_ascend/ops/register_custom_ops.py#L89)
[VERIFY: vllm_ascend/ops/register_custom_ops.py:98](../../../../vllm_ascend/ops/register_custom_ops.py#L98)

### 3.3 FakeImpl必须表达同一公式

runtime是pad后ReduceScatter，因此FakeImpl不能使用floor division。当前helper为：

```python
return (num_tokens + tp_size - 1) // tp_size
```

即 `ceil(N / P)`。这条公式允许 `num_tokens` 为 `torch.SymInt`，所以
compiled graph的token维可以保持symbolic。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:216](../../../../vllm_ascend/ops/register_custom_ops.py#L216)
[VERIFY: vllm_ascend/ops/register_custom_ops.py:220](../../../../vllm_ascend/ops/register_custom_ops.py#L220)

shape contract要求：

```text
fake_output.shape[0]
== runtime_output.shape[0]
== ceil(input.shape[0] / TP)
```

否则错误可能在compile阶段表现为FakeTensor shape错误，也可能延迟到runtime fused
op或graph replay才暴露。
[VERIFY: testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md:80](../../../../testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md#L80)

---

## 4. 基础DP contract

### 4.1 Dense DP与MoE DP的区别

普通dense DP中，每个DP rank独立处理自己的token：

```text
rank d: [L_d, H] → local model → [L_d, H]
```

不同rank的 `L_d` 可以不同，只要没有要求等shape的跨DP collective。

MoE expert parallel或shared-expert路径会跨DP/EP组交换token，因此collective前必须
对齐各rank首维。令当前进入该MoE边界的rank-local Tensor长度为 `N_d`。
当前 `_prepare_with_dp_group()` 读取 `max_tokens_across_dp`，保存本rank
`hidden_states.shape[0]`，pad到最大长度后AllGather。
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:471](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L471)
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:489](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L489)

令：

```text
M = max(N_0, N_1, ..., N_(D-1))
```

则：

```text
rank d local:       [N_d, H]
pad:                [M, H]
DP AllGather:       [D*M, H]
MoE compute:        [D*M, H]
DP ReduceScatter:   [M, H]
slice [:N_d]:       [N_d, H]
```

finalize执行ReduceScatter后按之前保存的 `self.num_tokens` 裁回本rank长度。
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:605](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L605)
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:615](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L615)

### 4.2 DP padding不是FlashComm padding

DP padding补齐的是不同rank间的不平衡：

```text
q_dp,d = M - N_d
```

TP padding补齐的是某一tensor无法被TP size整除：

```text
q_tp(N, P) = (-N) mod P
```

它们的触发条件和删除位置不同：

```text
DP pad:
  rank-local N_d → M
  在DP gather/scatter闭环后按N_d删除

TP pad:
  当前tensor N → ceil(N/P)*P
  在TP RS/AG闭环后按qtp删除
```

不能使用一个全局 `pad_size` 同时表达 `q_dp,d` 和 `q_tp`。

### 4.3 FC1 + DP/EP的对齐长度

FC1与DP metadata同时存在时，当前forward context以DP metadata中的
`C_d` 计算：

```text
M_C = max(C_0, C_1, ..., C_(D-1))
A   = ceil(M_C / P) * P
```

并设置：

```text
padded_length = A
pad_size = A - current_rank_num_tokens
```

[VERIFY: vllm_ascend/ascend_forward_context.py:180](../../../../vllm_ascend/ascend_forward_context.py#L180)
[VERIFY: vllm_ascend/ascend_forward_context.py:185](../../../../vllm_ascend/ascend_forward_context.py#L185)

这里的 `pad_size` 是复合差值：

```text
A - C_d = (M_C - C_d) + (ceil(M_C / P) * P - M_C)
```

它同时包含“追平DP最大值”和“让最大值对齐TP”的两段尾部padding。因此在DP/EP
路径中不能把它简单解释成纯TP `(-C_d) mod P`。

当前EP custom-op runtime会把gather后的tensor reshape为：

```text
[D, padded_length, ...]
```

然后对每个rank仅拷贝 `num_tokens_across_dp_cpu[d]` 行，拼成DP metadata定义的
compact global token序列。[VERIFY: vllm_ascend/ops/register_custom_ops.py:105](../../../../vllm_ascend/ops/register_custom_ops.py#L105)
[VERIFY: vllm_ascend/ops/register_custom_ops.py:111](../../../../vllm_ascend/ops/register_custom_ops.py#L111)

因此DP+FC1稳定接口有两种：

```text
collective layout: [D*A, H]     # 每rank固定slot
compact global:    [sum(C_d), H] # 删除每rank slot尾部padding后紧凑拼接
```

两者不能只通过删除整个tensor末尾的总padding互换，因为rank0、rank1等中间rank的
padding位于global tensor内部。

---

## 5. FlashComm1 contract

### 5.1 FC1的核心状态机

FlashComm1在层间维护两种合法状态：

```text
FULL(N)  = 每TP rank持有全部N个token
LOCAL(R) = 每TP rank持有ceil(N/P)个sequence shard
```

转换规则：

```text
RowParallel / O-Proj / MoE finalize:
  FULL(N) --pad + ReduceScatter--> LOCAL(ceil(N/P))

ColumnParallel / MLA preprocess / MoE prepare:
  LOCAL(ceil(N/P)) --AllGather + unpad--> FULL(N)
```

核心不变量是：

> 任何消费FULL接口的算子都不能收到LOCAL；任何把LOCAL恢复为FULL的路径都必须
> 删除自己对应的TP communication padding。

### 5.2 Residual必须与当前local shard同长

FC1后，主分支可能是 `LOCAL(R)`，而residual仍可能来自full/padded buffer。如果
AddRmsNormBias接收：

```text
x:        [R, H]
residual: [N, H] 或 [floor(N/P), H]
```

就会发生RFC中的tiling失败。
[VERIFY: testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md:177](../../../../testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md#L177)

当前 `_slice_tp_local_residual()` 不再使用全局 `pad_size` 后直接
`torch.chunk`，而是以消费者 `x.size(0)` 作为local contract：

```text
local_num_tokens  = x.size(0) = R
required_global   = R * P
rank d slice      = [d*R : (d+1)*R]
```

若residual不足 `R * P`，只补到该长度。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:65](../../../../vllm_ascend/ops/register_custom_ops.py#L65)

这个方向的意义是：residual适配当前真实consumer shape，而不是从可能混合
scheduler/DP/TP语义的context字段反推。

### 5.3 MLA的graph buffer与valid prefix

当前MLA wrapper按输入Tensor shape建立output buffer，因此compiled graph可以从
symbolic input传播shape；VL first layer是显式静态特例。
[VERIFY: vllm_ascend/ops/mla.py:155](../../../../vllm_ascend/ops/mla.py#L155)
[VERIFY: vllm_ascend/ops/mla.py:174](../../../../vllm_ascend/ops/mla.py#L174)

MLA runtime内部存在两个不同概念：

- `output`：compiled/graph拥有的buffer；
- `o_proj_output`：O-Proj与FC1通信真实产生的结果。

当前实现保留graph buffer shape，先清零，再只写
`o_proj_output.shape[0]` 行：

```python
output.zero_()
output[:o_proj_output.shape[0]] = o_proj_output
```

[VERIFY: vllm_ascend/attention/mla_v1.py:1809](../../../../vllm_ascend/attention/mla_v1.py#L1809)

对应contract是：

```text
0 <= valid_rows = o_proj_output.shape[0] <= output.shape[0]
output[:valid_rows]       = valid data
output[valid_rows:]       = zero padding
returned tensor shape     = graph-owned output shape
```

这与历史RFC中“直接把output缩成logical shape”的尝试不同。当前策略优先保持
compiled/ACL Graph buffer地址与shape稳定，把有效长度通过前缀语义表达。

### 5.4 MoE prepare/finalize

A2 AllGather模式下，DBO + FC1的MoE prepare显式执行：

```text
hook(record)
TP或EP AllGather
hook(wait/yield)
unpad
MoE compute
```

无DP metadata时，unpad目标是当前 `forward_context.num_tokens`；有DP metadata
时，目标是 `sum(C_d)`，并提供每rank `padded_length=A` 以删除分散在各slot
尾部的padding。
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:390](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L390)
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:405](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L405)
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:419](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L419)

finalize反向建立collective layout：

```text
无DP:
  FULL(N)
    → prepare到 N + qtp
    → TP ReduceScatter
    → LOCAL(ceil(N/P))

有DP:
  compact [sum(C_d), H]
    → pack为 [D*A, H]
    → EP ReduceScatter
    → rank-local aligned shard
```

对应 `prepared_length` 分别为：

```text
N + pad_size
D * padded_length
```

[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:570](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L570)
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:575](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L575)
[VERIFY: vllm_ascend/ops/fused_moe/prepare_finalize.py:586](../../../../vllm_ascend/ops/fused_moe/prepare_finalize.py#L586)

---

## 6. FlashComm2 contract

### 6.1 FC2改变的不只是collective名字

FlashComm2 O-Proj把传统row-parallel链路重排为：

```text
input
  → token pad
  → ODP AllToAll
  → matmul
  → OTP ReduceScatter
  → optional TP AllGather + unpad
```

当前实现先使用 `_EXTRA_CTX.pad_size` pad输入，然后要求padded batch能被
`chunk_num`整除；代码中 `chunk_num` 等于TP size。
[VERIFY: vllm_ascend/ops/linear_op.py:314](../../../../vllm_ascend/ops/linear_op.py#L314)
[VERIFY: vllm_ascend/ops/linear_op.py:325](../../../../vllm_ascend/ops/linear_op.py#L325)
[VERIFY: vllm_ascend/ops/linear_op.py:336](../../../../vllm_ascend/ops/linear_op.py#L336)

设：

```text
P = total TP size
O = ODP group size
Q = OTP group size
P = O * Q
Np = ceil(N/P) * P
```

ODP AllToAll后的布局是：

```text
rows:   Np / O
hidden: O * H_partition
```

实现将recv buffer重排为 `[chunk_size, -1]`，其中
`chunk_size = Np / O`。[VERIFY: vllm_ascend/ops/linear_op.py:357](../../../../vllm_ascend/ops/linear_op.py#L357)
[VERIFY: vllm_ascend/ops/linear_op.py:371](../../../../vllm_ascend/ops/linear_op.py#L371)

随后OTP ReduceScatter再把token rows除以 `Q`：

```text
(N_p / O) / Q = N_p / (O * Q) = N_p / P
```

所以FC2 O-Proj的sequence-local输出仍满足：

```text
LOCAL(R), R = Np/P = ceil(N/P)
```

当前代码在 `tp_size > 1` 时对matmul输出执行OTP ReduceScatter。
[VERIFY: vllm_ascend/ops/linear_op.py:399](../../../../vllm_ascend/ops/linear_op.py#L399)
[VERIFY: vllm_ascend/ops/linear_op.py:401](../../../../vllm_ascend/ops/linear_op.py#L401)

### 6.2 FC2与FC1组合后的两个出口

FC2 O-Proj有两个稳定出口：

#### FC2开启、FC1关闭

```text
OTP local output [Np/P, H]
  → TP AllGather
  → [Np, H]
  → 删除qtp
  → FULL(N)
```

当前实现只在FC1关闭时执行TP AllGather并删除
`num_padding_tokens`。[VERIFY: vllm_ascend/ops/linear_op.py:407](../../../../vllm_ascend/ops/linear_op.py#L407)

#### FC2开启、FC1开启

```text
OTP local output [Np/P, H]
  → 不做TP AllGather
  → 保持LOCAL(R)
  → 下一次FC1 column/MLA/MoE prepare恢复FULL
```

这避免FC2出口立刻AllGather、下一层又ReduceScatter的冗余通信。其contract与
FC1定义的 `LOCAL(ceil(N/P))` 完全对齐，因此FC2不应创造第三种层间token状态。

### 6.3 FC2 + DBO hook覆盖范围

当前FC2把row hook放在ODP AllToAll之前，并在：

- ODP AllToAll；
- matmul；
- OTP ReduceScatter；
- 可选TP AllGather/unpad；

全部结束后调用结束hook。这把整段视为一个DBO overlap phase。
[VERIFY: vllm_ascend/ops/linear_op.py:382](../../../../vllm_ascend/ops/linear_op.py#L382)
[VERIFY: vllm_ascend/ops/linear_op.py:385](../../../../vllm_ascend/ops/linear_op.py#L385)
[VERIFY: vllm_ascend/ops/linear_op.py:413](../../../../vllm_ascend/ops/linear_op.py#L413)

shape上要求hook为identity：

```text
hook(record, x).shape == x.shape
hook(wait, y).shape   == y.shape
```

当前DBO hook FakeImpl直接返回输入，因此不会自行制造或删除padding。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:61](../../../../vllm_ascend/ops/register_custom_ops.py#L61)

---

## 7. DBO contract

### 7.1 DBO只改变调度，不应改变数学shape

DBO把一个scheduler batch拆成多个ubatch，在不同Python线程和NPU stream上交替
提交compute与communication。它的目标是重叠，不是改变模型数学结果。

因此：

```text
model(B)
==
concat(model(B0), model(B1))
```

其中等号指删除所有scheduler/graph/communication padding后的logical output一致。

DBO hook只负责：

- record compute-done event；
- 建立stream wait；
- CPU yield给另一个ubatch；
- 保证collective提交顺序。

hook custom op的fake shape为identity，证明它不拥有tensor shape变换职责。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:34](../../../../vllm_ascend/ops/register_custom_ops.py#L34)
[VERIFY: vllm_ascend/ops/register_custom_ops.py:61](../../../../vllm_ascend/ops/register_custom_ops.py#L61)

### 7.2 每个ubatch必须拥有独立长度

令ubatch `u`：

```text
L_u = logical token count
S_u = scheduler/graph slice rows
```

通常：

```text
0 <= L_u <= S_u
```

当前outer context同时保存：

```text
ubatch_slices         = padded/graph slices
ubatch_slices_logical = attention/logical slices
```

[VERIFY: vllm_ascend/worker/model_runner_v1.py:2233](../../../../vllm_ascend/worker/model_runner_v1.py#L2233)

创建per-ubatch context时：

- `slice_num_tokens` 来自padded slice；
- FULL graph优先从attention metadata取actual tokens；
- `num_tokens_logical` 单独从logical slices取得；
- FC1/FC2 pad按该ubatch context重新计算。

[VERIFY: vllm_ascend/ascend_forward_context.py:275](../../../../vllm_ascend/ascend_forward_context.py#L275)
[VERIFY: vllm_ascend/ascend_forward_context.py:276](../../../../vllm_ascend/ascend_forward_context.py#L276)
[VERIFY: vllm_ascend/ascend_forward_context.py:286](../../../../vllm_ascend/ascend_forward_context.py#L286)
[VERIFY: vllm_ascend/ascend_forward_context.py:291](../../../../vllm_ascend/ascend_forward_context.py#L291)

这里存在两个相近但不能随意互换的字段：

```text
ctx.num_tokens
  当前模型/通信路径使用的ubatch长度口径；
  FULL graph下可能来自attention actual tokens，其他模式来自slice。

ctx.num_tokens_logical
  最终输出恢复业务长度使用的明确logical口径。
```

### 7.3 DBO输出合并必须逐ubatch unpad

FC1使每个ubatch最终可能仍是TP-local output。当前wrapper在最后一个PP rank：

1. 对每个ubatch分别TP AllGather；
2. 计算 `num_padded - num_logical`；
3. 分别删除各自尾部padding；
4. 最后concat。

[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:259](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L259)
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:265](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L265)
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:271](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L271)

不能先concat再统一删除总padding。若：

```text
ubatch0 = [valid0, pad0]
ubatch1 = [valid1, pad1]
```

先concat得到：

```text
[valid0, pad0, valid1, pad1]
```

只从末尾删除 `pad0+pad1` 会保留位于中间的 `pad0`，并误删
`valid1` 的尾部。

### 7.4 DBO capture与runtime必须共享contract

dummy/capture path也创建padded与logical slices。FULL graph时logical slices使用
padded slices，其他模式使用普通ubatch slices，随后将它们写入outer context。
[VERIFY: vllm_ascend/worker/model_runner_v1.py:3403](../../../../vllm_ascend/worker/model_runner_v1.py#L3403)
[VERIFY: vllm_ascend/worker/model_runner_v1.py:3409](../../../../vllm_ascend/worker/model_runner_v1.py#L3409)
[VERIFY: vllm_ascend/worker/model_runner_v1.py:3542](../../../../vllm_ascend/worker/model_runner_v1.py#L3542)
[VERIFY: vllm_ascend/worker/model_runner_v1.py:3554](../../../../vllm_ascend/worker/model_runner_v1.py#L3554)

这意味着冷启动shape contract不仅要满足真实请求，还要满足：

```text
profile dummy
compile FakeTensor
piecewise range compile
ACL Graph capture
ACL Graph replay
runtime ubatch
```

任何一个阶段对 `num_tokens` 的logical/padded解释不同，都可能出现RFC记录的
“编译成功但warmup设备算子shape mismatch”。
[VERIFY: testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md:159](../../../../testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md#L159)

---

## 8. 三层机制如何组合

### 8.1 无FC1/FC2、无DBO

```text
rank d input: FULL(L_d)
TP row:       AllReduce，仍FULL(L_d)
MoE DP:
  pad L_d→M
  DP AG → [D*M]
  MoE
  DP RS → [M]
  slice → FULL(L_d)
```

稳定接口始终是FULL。

### 8.2 FC1开启

```text
FULL(N)
  ├─ row/O-Proj/MoE finalize
  │    pad N→Np
  │    TP RS
  │    → LOCAL(Np/P)
  │
  └─ column/MLA/MoE prepare
       TP AG
       Np→unpad N
       → FULL(N)
```

FC1引入FULL/LOCAL二态，所有层边界必须标注当前状态。

### 8.3 FC2开启、FC1关闭

```text
FULL(N)
  → pad N→Np
  → ODP A2A: rows / O, hidden * O
  → matmul
  → OTP RS: rows / Q
  → TP AG
  → unpad Np→N
  → FULL(N)
```

FC2内部改变layout，出口恢复FULL。

### 8.4 FC1 + FC2

```text
FULL(N)
  → FC2 ODP A2A + OTP RS
  → LOCAL(Np/P)
  → 下一column/MLA/MoE prepare执行FC1 AG+unpad
  → FULL(N)
```

FC2出口直接复用FC1 LOCAL contract。

### 8.5 FC1/FC2 + DP

对于DP rank `d`：

```text
logical local:       L_d
DP metadata max:     M_C = max(C_d)
TP-aligned slot:     A = ceil(M_C/P)*P
collective layout:   [D*A, H]
compact MoE input:   [sum(C_d), H]
TP local shard:      [A/P, H] 或与当前N对应的ceil(N/P)
```

关键是区分：

- `D*A`：通信固定slot布局；
- `sum(C_d)`：删除每rank slot内部padding后的紧凑global token；
- `A/P`：每TP/EP rank的local aligned shard。

### 8.6 再叠加DBO

所有上述变量增加ubatch下标：

```text
L_d      → L_(d,u)
S_d      → S_(d,u)
C_d      → C_(d,u)
M_C      → M_(C,u)
A        → A_u
qtp      → q_(tp,u)
R        → R_u
```

DBO的正确性条件是对每个 `u` 独立满足原contract：

```text
对每个 ubatch u：

R_u = ceil(N_u / P)
A_u = ceil(max_d(C_(d,u)) / P) * P
```

然后：

```text
logical_output(B)
    = concat_u(unpad_u(output(B_u)))
```

DBO不能用outer batch的 `q`、`M`、`A` 替代per-ubatch值；也不能让两个
ubatch共享可变forward context。

---

## 9. 必须满足的Shape不变量

### 9.1 算子局部不变量

1. Elementwise/fused residual：

```text
x.shape[0] == residual.shape[0]
```

2. TP ReduceScatter：

```text
input.shape[0] % TP == 0
output.shape[0] == input.shape[0] / TP
```

3. FC1 AllGather + unpad：

```text
gathered.shape[0] == local.shape[0] * TP
unpadded.shape[0] == target logical/full length
```

4. FlashComm2：

```text
padded_rows % total_TP == 0
ODP_rows == padded_rows / ODP
OTP_output_rows == padded_rows / total_TP
```

5. graph-owned output：

```text
valid_rows <= buffer_rows
invalid suffix is initialized
```

### 9.2 跨层不变量

1. FC1 row出口一定是LOCAL；
2. FC1 column/MLA/MoE prepare入口若收到LOCAL，必须恢复FULL；
3. FC1关闭时，layer间接口必须保持FULL；
4. FC2 + FC1开启时，FC2出口是LOCAL；
5. FC2 + FC1关闭时，FC2出口必须恢复FULL；
6. DP gather前每rankslot等长，DP finalize后恢复本rank长度；
7. DBO concat前必须逐ubatch恢复logical output。

### 9.3 Compile contract

每个custom op必须满足：

```text
FakeImpl output shape
== runtime impl output shape
== graph replay visible shape
```

FakeImpl只能依赖：

- 输入Tensor shape；
- 显式shape参数；
- compile-stable配置。

不能依赖per-request forward context、DP metadata、stream或ubatch runtime状态。
[VERIFY: testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md:42](../../../../testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md#L42)
[VERIFY: testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md:109](../../../../testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md#L109)

---

## 10. 当前源码审计结论与风险

### 10.1 已经闭合的关键点

1. ReduceScatter fake shape使用ceil division。
   [VERIFY: vllm_ascend/ops/register_custom_ops.py:216](../../../../vllm_ascend/ops/register_custom_ops.py#L216)
2. FC1 row padding从当前Tensor symbolic shape计算。
   [VERIFY: vllm_ascend/ops/linear_op.py:579](../../../../vllm_ascend/ops/linear_op.py#L579)
3. residual local slice由consumer `x.size(0)`决定。
   [VERIFY: vllm_ascend/ops/register_custom_ops.py:65](../../../../vllm_ascend/ops/register_custom_ops.py#L65)
4. DBO outer context区分padded与logical ubatch slices。
   [VERIFY: vllm_ascend/worker/model_runner_v1.py:2233](../../../../vllm_ascend/worker/model_runner_v1.py#L2233)
5. DBO最终逐ubatch gather、unpad、concat。
   [VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:259](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L259)
6. MLA保留graph buffer并清零无效suffix。
   [VERIFY: vllm_ascend/attention/mla_v1.py:1809](../../../../vllm_ascend/attention/mla_v1.py#L1809)
7. FC2 row hook覆盖完整ODP/OTP通信段。
   [VERIFY: vllm_ascend/ops/linear_op.py:382](../../../../vllm_ascend/ops/linear_op.py#L382)

### 10.2 仍需要重点review的边界

#### 风险一：`ctx.num_tokens` 的模式相关语义

per-ubatch context在FULL graph时优先使用attention actual tokens，非FULL时使用
slice tokens；最终输出另用 `num_tokens_logical`。这是当前明确实现，但调用方若
把 `ctx.num_tokens` 无条件解释为logical，会在不同graph mode下得到不同语义。
[VERIFY: vllm_ascend/ascend_forward_context.py:275](../../../../vllm_ascend/ascend_forward_context.py#L275)
[VERIFY: vllm_ascend/ascend_forward_context.py:291](../../../../vllm_ascend/ascend_forward_context.py#L291)

#### 风险二：全局 `_EXTRA_CTX.pad_size`

当前部分runtime op仍读取 `_EXTRA_CTX.pad_size` 做FC1/FC2 pad/unpad。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:102](../../../../vllm_ascend/ops/register_custom_ops.py#L102)
[VERIFY: vllm_ascend/ops/register_custom_ops.py:170](../../../../vllm_ascend/ops/register_custom_ops.py#L170)
[VERIFY: vllm_ascend/ops/linear_op.py:325](../../../../vllm_ascend/ops/linear_op.py#L325)

它在runtime custom-op内部可以代表当前context，但不能被FakeImpl当作动态shape
来源；同时DP路径下它可能表示 `A - C_d`，不是纯 `q_tp`。调用点必须明确自己
需要的是哪一种padding。

#### 风险三：MLA valid prefix缺少显式上界断言

当前写法依赖：

```text
o_proj_output.shape[0] <= output.shape[0]
```

但代码没有在赋值前显式断言。若producer返回比graph buffer更长的结果，错误会在
slice assignment处暴露，诊断信息仍不够直接。
[VERIFY: vllm_ascend/attention/mla_v1.py:1809](../../../../vllm_ascend/attention/mla_v1.py#L1809)

#### 风险四：DBO graph capture与普通runtime合并路径不同

DBO graph capture内部直接concat两个ubatch结果；普通runtime在FC1下会先逐ubatch
TP AllGather与unpad再concat。二者可能有意依赖外层graph buffer contract，但必须
通过capture/replay端到端测试证明结果shape一致。
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:200](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L200)
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:206](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L206)
[VERIFY: vllm_ascend/worker/npu_ubatch_wrapper.py:259](../../../../vllm_ascend/worker/npu_ubatch_wrapper.py#L259)

#### 风险五：DP metadata一致性

`_get_actual_num_tokens()` 对metadata dict返回第一个具有
`num_actual_tokens` 的值，没有验证多个attention group是否一致。
[VERIFY: vllm_ascend/ascend_forward_context.py:212](../../../../vllm_ascend/ascend_forward_context.py#L212)
[VERIFY: vllm_ascend/ascend_forward_context.py:223](../../../../vllm_ascend/ascend_forward_context.py#L223)

若不同group报告不同logical长度，当前contract没有显式失败点。

---

## 11. 推荐测试矩阵

### 11.1 纯shape单测

最少覆盖：

| TP | DP | logical tokens/rank | 目的 |
|---:|---:|---|---|
| 1 | 1 | 1, 2, 3 | identity基线 |
| 2 | 1 | 1, 2, 3, 2049 | odd/even FC1 pad |
| 4 | 1 | 1, 3, 4, 5 | `ceil(N/TP)` |
| 2 | 2 | `[3, 5]` | DP max + TP align |
| 4 | 2 | `[5, 8]` | DP slot内部padding |
| 2 | 4 | `[0, 1, 3, 8]` | 空rank与不均衡DP |

每组断言：

```text
RS rows == ceil(N/TP)
AG+unpad rows == N
DP compact rows == sum(N_d)
DP RS+slice rows on rank d == N_d
residual rows == activation rows
```

### 11.2 功能组合矩阵

```text
FC1 ∈ {0,1}
FC2 ∈ {0,1}
DBO ∈ {0,1}
DP  ∈ {1,2}
TP  ∈ {2,4}
mode ∈ {eager, compile, PIECEWISE graph, FULL graph}
```

高优先级组合：

1. FC1=1、FC2=0、DBO=1、odd ubatch；
2. FC1=0、FC2=1、DBO=1；
3. FC1=1、FC2=1、DBO=1；
4. DP=2且两个rank token不相等；
5. cold cache profile/compile/capture；
6. capture size明显大于logical token，例如16或4096对8。

### 11.3 每层debug记录

出现shape问题时，不要只打印一个 `num_tokens`，至少记录：

```text
dp_rank
ubatch_id
cudagraph_runtime_mode
batch_descriptor.num_tokens
attn_metadata.num_actual_tokens
ctx.num_tokens
ctx.num_tokens_logical
ctx.max_tokens_across_dp
ctx.padded_length
ctx.pad_size
input.shape[0]
residual.shape[0]
collective output.shape[0]
graph buffer.shape[0]
```

这些字段足以判断断裂发生在：

- scheduler/graph → ubatch；
- DP pack/unpack；
- TP pad/RS/AG/unpad；
- FC2 ODP/OTP布局；
- DBO output merge；
- fake/runtime/graph buffer。

---

## 12. 最终统一定义

整个系统只应暴露两种稳定层间token状态：

```text
FULL(N):
  当前并行域需要的完整、紧凑token序列。

LOCAL(R):
  FC1/FC2产生的TP sequence shard，
  R = ceil(N/TP)。
```

DP、graph和communication可以在内部使用固定slot或padded buffer，但必须明确：

```text
logical length  != buffer capacity
DP slot length  != compact global length
TP local length != full token length
```

三类优化的职责边界为：

```text
TP/DP:
  定义数据在哪些rank复制、切分与聚合。

FlashComm1/2:
  重排collective并在FULL与LOCAL之间转换；
  对自己引入的通信padding负责。

DBO:
  对每个ubatch复制同一shape contract；
  只改变compute/communication提交时序；
  concat前逐ubatch恢复logical output。
```

最终正确性条件可以写成：

```text
Output_logical
    = Unpad_scheduler_or_graph(
        Unpad_DP(
          Unpad_TP(
            Execute_TP_DP_FC1_FC2_DBO(X)
          )
        )
      )
```

其中每个 `Unpad` 只能删除对应层自己引入的padding。只要某一层使用了另一层的
长度或pad值，contract就不再可组合，错误最终会在elementwise融合、collective
整除检查、custom-op fake shape或graph buffer写回处暴露。
