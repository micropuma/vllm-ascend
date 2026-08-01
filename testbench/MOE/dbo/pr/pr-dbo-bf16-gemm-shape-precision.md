# [DBO][Precision] Ascend BF16 GEMM shape-dependent precision drift

## 状态

- 状态：根因已定位，底层数值稳定路径待确认
- 日期：2026-07-30
- 适用范围：DeepSeek-V2-Lite-Chat、Ascend 910B3、TP=2、EP、DBO
- 影响模式：eager prefill；问题与 FlashComm1/event hook 无关

## 摘要

DBO 将一个长 prefill 请求拆成两个 microbatch。对于同一份输入和同一份权重，
Ascend BF16 GEMM 的输出会随 GEMM 的 M 维度和分块方式变化：

```text
一次计算：M = 2050
DBO 计算：M = 1025 + 1025
```

因此 DBO 与非 DBO 的前向结果在第 0 层 q projection 就已经不同，后续层将该
差异逐步放大，最终造成输出 token logprob 漂移。当前证据支持的本质结论是：
**Ascend BF16 GEMM 当前不是 M-shape invariant 的；DBO 的切分暴露了该数值差异。**

这不是 DBO event 依赖、attention metadata、KV cache 或 `npu_attention_update`
导致的同步/状态错误。

## 模型路径与矩阵尺寸

DeepSeek-V2-Lite 配置为：

```text
hidden_size       = 2048
num_attention_heads = 16
qk_head_dim       = 128 + 64 = 192
q projection      = [M, 2048] x [2048, 3072]
```

上游 `DeepseekV2Attention` 在 `deepseek_v2.py:464-469` 创建
`ColumnParallelLinear(hidden_size, num_heads * qk_head_dim)`。Ascend
`AscendUnquantizedLinearMethod.apply` 在 `vllm_ascend/ops/linear.py:89-95`
调用 `torch.ops.vllm.unquantized_gemm`，其实现是
`torch.nn.functional.linear`，最终落到 Ascend GEMM。

底层定位可以继续沿 PyTorch dispatch 查：当前环境的 dispatch dump 显示
`aten::mm`、`aten::matmul` 和 `aten::addmm` 的 NPU kernel 注册点为
`torch_npu/csrc/aten/RegisterNPU.cpp:33542`。它再调用 CANN ACLNN 的
`aclnnMatmul`（或权重为 FRACTAL_NZ 时的 `aclnnMatmulWeightNz`）；接口定义可见
`/usr/local/Ascend/cann-9.0.0/include/aclnnop/aclnn_matmul.h:25-60`。CANN 安装
包还提供可读的 AscendC tiling 实现：

```text
/usr/local/Ascend/cann-9.0.0/aarch64-linux/asc/impl/adv_api/
  tiling/matmul/matmul_tiling_algorithm.h
  detail/matmul/param/matmul_cross_core_sync.h
```

`MatmulRunParas` 显式保存 `oriShapeM`，tiling 算法基于 M/N/K 选择
`SPLIT_MN` 或 `SPLIT_SMALL_MN`、`coreUse`、single-core M 和 base M/N/K；其中
small-MN 分支条件为 `M*N < 128*256*corenum*0.8`。这提供了直接源码证据：M 是
kernel tiling 与跨核策略的输入，完整 M 和 DBO split M 没有数值等价保证。

仓库侧能直接控制的是调用入口和权重布局（`maybe_trans_nz`，
`vllm_ascend/utils.py:210-247`），不能修改 CANN 内部的 tiling、split-K 或累加
顺序。注意默认 `weight_nz_mode=1` 只转换量化权重；非量化 BF16 权重仅在
`weight_nz_mode=2` 时转 NZ（`vllm_ascend/ascend_config.py:218-225`）。所以未
显式配置 mode 2 的当前 DeepSeek BF16 q_proj 应走普通 `aclnnMatmul`，而不是
`aclnnMatmulWeightNz`。

## 端到端证据

### DBO 确实执行

在以下日志中，两个 TP worker 都记录了 DBO ubatch：

`testbench/MOE/dbo/demos/precision/results/`
`single-long-eager-preprocess-probe-20260730T115000Z/dbo_server.log`

```text
Worker_TP0_EP0 ... BatchDescriptor(num_tokens=2050) ... should_ubatch: True
Worker_TP1_EP1 ... BatchDescriptor(num_tokens=2050) ... should_ubatch: True
```

### 首个差异出现在 q projection

在同一输入校验和、同一权重下，重新计算完整 M 和两个 M=1025 分块，并逐元素
比较 q projection 输出：

`testbench/MOE/dbo/demos/precision/results/`
`single-long-eager-preprocess-probe-20260730T120000Z/baseline_server.log`

```text
rank=0: maxabs=0.015625,       meanabs=1.74606029e-07, nonzero=261
rank=1: maxabs=0.00048828125,  meanabs=1.58006819e-09, nonzero=213
```

同一 probe 中，后续 `kv_b_proj` 输出没有对应的首发差异，说明差异不是输入
或 KV 更新先发生改变。

### 输出层面

`single-long-20260730T080000Z/comparison.json` 报告：

```text
generated token: 相同
token 0 logprob delta: 0.035461246967315674
allowed tolerance:     0.001
passed:                false
```

此前更长的生成测试中，累计 logprob 偏差达到 0.22454。

## 独立 NPU GEMM 复现

停止 vLLM 服务后，在同一台 Ascend 910B3 上使用 BF16 随机输入运行：

```python
K, N = 2048, 3072
x = torch.randn((2050, K), device="npu", dtype=torch.bfloat16)
w = torch.randn((N, K), device="npu", dtype=torch.bfloat16)

y_full = torch.mm(x, w.t())
y_split = torch.cat([
    torch.mm(x[:1025], w.t()),
    torch.mm(x[1025:], w.t()),
])
```

逐元素比较 `y_full` 与 `y_split` 得到非零差异。一次复现结果为：

```text
max abs diff = 0.5
mean abs diff = 5.945300017629052e-06
nonzero       = 829
```

`torch.nn.functional.linear` 和显式 `torch.mm` 均可复现。使用单个
`M=2050` GEMM 时结果与自身比较为零；只有改变 M 分块后出现差异。将分块改成
`1024+1026`、`1026+1024` 或四段切分仍能复现，说明触发因素是 GEMM 的形状/调度，
不是特定 event 顺序。

该独立实验使用连续 BF16 权重，数值幅度不应直接与模型权重的 probe 数值比较；
它用于证明 Ascend BF16 GEMM 的形状相关性。模型 probe 则证明该性质确实落在
实际 q projection 路径上。

### 与 FP32 reference 的误差分解

使用同一组 BF16 输入和权重，先在 CPU 上将 BF16 操作数转换为 FP32，再计算
FP32 reference。NPU 输出保持 BF16，统计结果如下：

| 计算方式 | abs mean | abs P99 | abs max | relative mean | relative P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M=2050 | 0.0508723 | 0.237534 | 0.499985 | 0.00140956 | 0.00346522 |
| 1025+1025 | 0.0508723 | 0.237534 | 0.499985 | 0.00140958 | 0.00346528 |

完整 GEMM 与切分 GEMM 之间的差异为：

```text
abs mean = 4.91572e-06
abs max  = 0.5
nonzero  = 749 / 6,297,600
P99      = 0
```

这组数据不能单独定义“正常 BF16”的通用阈值，因为它使用随机输入，且 NPU
reference 仍是具体 kernel 的实现结果；但它清楚地区分了两件事：单次 BF16
GEMM 相对 FP32 的误差是普遍存在的，而 M 形状改变引起的额外误差非常稀疏，
主要表现为少量元素的舍入路径变化，而不是整个矩阵系统性偏移。

## 建议补充的模型级统计

独立 GEMM 不能说明误差在真实模型中如何传播。应在同一请求、同一 TP rank、同一
组 BF16 输入下保存完整 M 和切分 M 的中间张量，并为每层计算：

```text
abs_max      = max(abs(a - b))
abs_mean     = mean(abs(a - b))
relative_max = max(abs(a - b) / max(abs(a), eps))
relative_mean
cosine       = cosine_similarity(a.flatten(), b.flatten())
nonzero      = count(a != b)
```

建议至少保存以下边界：`q_proj`、`kv_b_proj`、attention 输出、MLP 输出、layer
输出和最终 logits。把每层的 `abs_max`、`abs_mean`、`relative_mean`、`1-cosine`
画成 layer 曲线，即可区分“首层稀疏差异被传播”与“某个后续算子重新引入大误差”。

当前已有 layer probe 可以提供 sum/maxabs/尾段校验和；要完成上述统计，probe
需要在固定请求中保存相同 token 区间的张量摘要，或保存压缩后的 FP32/BF16
中间张量。不得只比较最终 token，因为 token 相同并不代表 logits/logprob 相同。

## 已排除项

- DBO event hook 和 stream dependency
- MLA metadata/cache context
- `npu_attention_update`
- 输入尾段校验和不一致
- q projection 之前的 layer-0 输入差异
- FlashComm1 开关作为必要条件

这些修改或排查不能消除 q projection 的首发差异，因为差异发生在 GEMM 本身。

## `VLLM_BATCH_INVARIANT` 结果

尝试设置 `VLLM_BATCH_INVARIANT=1` 做 A/B 时，baseline 在首个请求即因
`x must be a 2D tensor` 退出，DBO 没有完成有效请求。因此该运行不能证明
batch-invariant 路径已经修复此问题。

现有 Ascend `AscendUnquantizedLinearMethod.apply` 也没有直接复用上游
`linear_batch_invariant` 实现；即使补上该路径，也必须先验证其 NPU kernel 对
实际 BF16/NZ 权重和所有 DBO shapes 的正确性。

仓库中已有 counterpart：`vllm_ascend/batch_invariant.py:90-123` 会将
`aten::mm/matmul/linear` 重定向到 AscendC `batch_invariant_ops` 或 Triton
实现。AscendC 路径优先注册 `npu_mm_batch_invariant` 和
`npu_matmul_batch_invariant`，并在 `override_envs_for_invariance()` 中关闭 NZ
权重（`weight_nz_mode=0`）和 matmul-allreduce；Triton fallback 位于
`vllm_ascend/ops/triton/batch_invariant/matmul.py`，使用固定的
`BLOCK_M/BLOCK_N/BLOCK_K` 与 persistent 1-D grid。当前环境没有安装
`batch_invariant_ops` 扩展，因此不能把该路径当作已验证的 q_proj 修复；而且
关闭 NZ 后的权重路径与生产 DBO 仍不是同一个 A/B。

新增的最小复现脚本为：

```text
testbench/MOE/dbo/demos/precision/quick/test_npu_gemm_m_shape.py
```

它固定比较完整 `M=2050` 与 `1025+1025` 的 BF16 `F.linear`，输出
`max_abs`、`mean_abs` 和 nonzero。当前机器上该独立进程在首次 NPU GEMM 调用
处异常退出，未生成新的统计；历史同机 probe 的非零结果仍是当前数值结论的
依据。要验证 batch-invariant counterpart，必须先安装对应的 AscendC 扩展，
再用同一脚本同时测试普通 `F.linear`、AscendC `npu_mm_batch_invariant` 和
真实 NZ 权重形态。

## 影响

- 严格 token-logprob 对齐场景：当前 DBO 不满足基线一致性要求。
- greedy/temperature=0 场景：短输出可能 token 相同，但 logprob 已经漂移；长生成
  存在后续 token 分叉风险。
- 性能收益与精度要求冲突：DBO 的 overlap 依赖切分，而切分会改变 GEMM 数值路径。

## 建议方案

### P0：严格对齐模式

在底层数值稳定方案可用前，禁止将同一长请求按 token 切入 DBO；可保留 DBO
用于多请求场景，但单请求长 prefill 应走完整 M 的 baseline 路径。

这是当前唯一不依赖算子改动、可以立即保证基线一致性的方案。代价是该类请求
失去 DBO 的单请求 prefill overlap；服务端应将其作为 precision mode 或按请求
长度启用的保护策略，而不是默默接受 logprob 漂移。

### P0 过渡方案：固定 M 的 padding

如果业务必须保留 DBO 调度，可以让两个 microbatch 都以 baseline 的 M 形状调用
GEMM：不足的 row 用零填充，计算后只取真实 token 的输出。这样完整 M 和每个
microbatch 使用同一 kernel shape，理论上可以消除当前 shape-dependent 差异。

该方案必须单独验证 padding row 不参与 attention、MoE routing、collective 和
KV cache 更新；同时它会增加无效 GEMM 计算，可能抵消 DBO 的性能收益，因此只
适合作为实验性兼容模式，不应直接作为默认优化。

### P1：底层修复

由 Ascend 算子/运行时提供以下任一保证，并用实际 q projection 尺寸验证：

1. M-shape invariant 的 BF16 GEMM 调度/累加路径；或
2. 更高精度累加（例如 FP32 accumulation）且结果不依赖 M 分块；或
3. 对 DBO microbatch 使用可证明与完整 M 等价的固定 tile/persistent kernel。

这是推荐的长期方案。vLLM-Ascend 侧应向 Ascend 算子团队提交最小复现（矩阵
`[2050,2048] x [2048,3072]`、BF16、完整与 `1025+1025`），要求明确：累加精度、
split-K/tiling 选择、NZ padding 规则，以及是否存在 batch-invariant kernel 开关。
在拿到底层保证前，不应仅通过 Python 侧 event 或张量重排宣称已修复。

验证标准应至少包括 `M=2050` 与 `1025+1025` 的逐元素结果、模型 1 层至最终
logprob 对齐，以及 TP=2 两个 rank 的一致性。

### 不建议的修复方向

- 继续调整 event 依赖；
- 仅替换 attention update；
- 仅依赖 `VLLM_BATCH_INVARIANT=1` 而不验证 Ascend linear/NZ 权重路径。
- 仅提高最终 logits 或 logprob 的计算精度；这无法恢复已经在 q projection
  发生的差异。

## DP+EP 逐层验证补充（2026-07-31）

使用同一套 layer-boundary precision tracer，在 `TP=1, DP=2, EP=2` 上运行
16 个并发、2048-token 的确定性请求波次。baseline 与 DBO 请求均成功，两个
DP worker 均出现 `should_ubatch: True`。

权威产物：

```text
/data/tmp/dbo-dp-layer-trace-20260731T031000Z/divergence.json
```

结果显示首个差异仍是：

```text
DeepseekV2ForCausalLM.model.layers.0.self_attn.q_proj (rank 0)
baseline:    [3274, 3072], aggregate mean=-0.024164944887161255, abs_max=21.75
DBO stitched:[3274, 3072], aggregate mean=-0.02190499217367978, abs_max=21.75
```

这些 mean/abs_max 是对 `3274*3072` 个 BF16 元素的聚合摘要，不是逐元素误差。
聚合 mean 差为 `0.0022599527134814744`（相对 baseline 约 9.35%），两侧
abs_max 都是 `21.75`。它们不能被解释成 `mean(abs(baseline - DBO))` 或
`max(abs(baseline - DBO))`；当前 serving trace 没有把 module record 绑定到
request id，baseline 与 DBO 可能把不同请求打包进同一 shape。因此 DP+EP trace
用于确定“首个候选边界”，不单独用于宣称逐元素误差幅度。

TP probe 的量化证据更强：它在相同输入尾段和相同权重上做 full/split 重算，
直接统计 elementwise `maxabs`、`meanabs` 和 nonzero。要得到 DP+EP 的严格
幅度，还需要固定 request identity 或保存并重放完全相同的 q_proj 输入。

这与 TP=2 的 layer-0 `q_proj` probe 完全一致，说明 DP+EP 没有引入新的
首发算子；DP 场景只是通过 DBO microbatch split 触发同一个 Ascend BF16
GEMM 的 M-shape-dependent rounding path。当前 DBO batch 的 DP token vector
是等长的（例如 `[3274,3274]`），因此 `make_ubatch_dp_metadata` 的不等长
peer 修复不是本次漂移来源，但该修复仍是 DP+EP uneven batch 的必要通信正确性
修复。

后续修改方向保持分层：短期严格精度模式绕过长请求 DBO，或使用固定 M padding
并在 attention、MoE routing、collective、KV cache 前严格去除 padding；长期由
Ascend linear/GEMM 提供 M-shape invariant 的 BF16 累加/tiling 路径，使用真实
`[M,2048] x [2048,3072]` q projection 做 full-vs-split 验证。不要通过 event
顺序、attention update 或最终 logits 调整来掩盖已经在 q_proj 发生的差异。

## 验收标准

- 两个 TP rank 的 q projection：`maxabs == 0` 或由底层明确给出可接受误差上限；
- 端到端 sampled-token logprob delta `<= 1e-3`；
- 生成 token、token 数和请求完成率与 baseline 一致；
- DBO 触发证据仍存在，且没有引入新的 collective/event 错误；
- 至少重复三次，覆盖 eager 和实际生产使用的 compile/graph 配置。
