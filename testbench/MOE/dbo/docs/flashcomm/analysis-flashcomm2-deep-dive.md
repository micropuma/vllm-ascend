# FlashComm2 源码深度分析

## 0. 概述

FlashComm2 是针对 **Attention O-Proj 线性层** 的分布式通信优化方案。核心思路是用一个两级通信组（ODP all-to-all + OTP reduce-scatter）替代标准 TP 的 OProj 通信，减少跨 rank 数据传输量。

**开启方式**：环境变量 `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=N` (N > 0, 表示 OTP group size)。

> 本文所有文件路径基于仓库根目录 `/home/douliyang/large/mlsys/tmp/vllm-ascend/`。

---

## 1. 配置入口 → 校验 → 存储

### 1.1 环境变量定义

> [`vllm_ascend/envs.py:76-80`](../../../../vllm_ascend/envs.py#L76)

```python
"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
```

### 1.2 读入 AscendConfig

> [`vllm_ascend/ascend_config.py:162-167`](../../../../vllm_ascend/ascend_config.py#L162)

```python
self.enable_flashcomm2_parallel_size = self._get_config_value(
    additional_config,
    "enable_flashcomm2_parallel_size",
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE",
    ascend_envs.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE,
)
```

校验入口 [`vllm_ascend/ascend_config.py:210-212`](../../../../vllm_ascend/ascend_config.py#L210)：
```python
self.flashcomm2_oproj_tensor_parallel_size = get_flashcomm2_config_and_validate(self, vllm_config)
```

### 1.3 校验函数

> [`vllm_ascend/utils.py:1203-1250`](../../../../vllm_ascend/utils.py#L1203)

`get_flashcomm2_config_and_validate()` 做如下校验：

| 行号 | 检查项 | 条件 | 失败行为 |
|------|--------|------|----------|
| [L1207-1208](../../../../vllm_ascend/utils.py#L1207) | 是否启用 | `enable_flashcomm2_parallel_size <= 0` | 返回 0 (禁用) |
| [L1213-1220](../../../../vllm_ascend/utils.py#L1213) | layer_sharding 兼容 | 只允许 `["o_proj"]` | 抛 ValueError |
| [L1221-1224](../../../../vllm_ascend/utils.py#L1221) | FlashComm1 建议 | FC1 未开启 | 警告（推荐同时开启） |
| [L1225-1228](../../../../vllm_ascend/utils.py#L1225) | fine-grained OTP 互斥 | `oproj_tensor_parallel_size > 0` | 抛 AssertionError |
| [L1229-1233](../../../../vllm_ascend/utils.py#L1229) | TP 大小检查 | `global_tp_size <= flashcomm2_otp_size` | 抛 AssertionError |
| [L1234-1238](../../../../vllm_ascend/utils.py#L1234) | TP 可整除检查 | `global_tp_size % flashcomm2_otp_size != 0` | 抛 AssertionError |
| [L1239-1243](../../../../vllm_ascend/utils.py#L1239) | P/D 场景 | `kv_transfer_config=None` | 警告（推荐 P-scenario） |
| [L1244-1248](../../../../vllm_ascend/utils.py#L1244) | D 节点禁止 | `is_kv_consumer=True` | 抛 AssertionError |

### 1.4 本地判断函数

> [`vllm_ascend/utils.py:1191-1193`](../../../../vllm_ascend/utils.py#L1191)

```python
def flashcomm2_enable() -> bool:
    config_val = get_ascend_config().enable_flashcomm2_parallel_size
    return config_val > 0
```

---

## 2. 通信组初始化

### 2.1 模块级变量

> [`vllm_ascend/distributed/parallel_state.py:18-19`](../../../../vllm_ascend/distributed/parallel_state.py#L18)

```python
_FLASHCOMM2_OTP: GroupCoordinator | None = None   # Output Tensor Parallel
_FLASHCOMM2_ODP: GroupCoordinator | None = None   # Output Data Parallel
```

### 2.2 组初始化函数

> [`vllm_ascend/distributed/parallel_state.py:152-189`](../../../../vllm_ascend/distributed/parallel_state.py#L152)

以 **TP=4, flashcomm2_otp_size=2** 为例的建组过程：

```
Step 1: 计算 OTP 组数
  num_fc2_oproj_tensor_parallel_groups = global_tp_size // flashcomm2_otp_size
  = 4 // 2 = 2

Step 2: 默认值 ([L159-160](../../../../vllm_ascend/distributed/parallel_state.py#L159))
  _FLASHCOMM2_OTP = None
  _FLASHCOMM2_ODP = get_tp_group()  # 默认 = 全局 TP group

Step 3: 当 flashcomm2_otp_size > 1 时 ([L162-189](../../../../vllm_ascend/distributed/parallel_state.py#L162))
```

**OTP 组（interleaved mapping）** — [`L173-182`](../../../../vllm_ascend/distributed/parallel_state.py#L173)：
```
遍历 num_otp_groups=2, otp_size=2:
  group 0: tp_local_rank = 0 + 0*2 = 0  → global_rank 0
            tp_local_rank = 0 + 1*2 = 2  → global_rank 2
  group 1: tp_local_rank = 1 + 0*2 = 1  → global_rank 1
            tp_local_rank = 1 + 1*2 = 3  → global_rank 3

结果: OTP Group 0 = {rank0, rank2}, OTP Group 1 = {rank1, rank3}
```

**ODP 组（consecutive mapping）** — [`L181`](../../../../vllm_ascend/distributed/parallel_state.py#L181)：
```
odp_group_index = odp_base_index + j:
  j=0: odp_group_ranks[0] = [rank0, rank1]
  j=1: odp_group_ranks[1] = [rank2, rank3]

结果: ODP Group 0 = {rank0, rank1}, ODP Group 1 = {rank2, rank3}
```

**关键性质**：OTP 和 ODP 的 rank 分组是正交的（crossed）。OTP = stride grouping（每隔 gap 取一个），ODP = contiguous grouping（连续的）。

### 2.3 访问器函数

> [`vllm_ascend/distributed/parallel_state.py:258-264`](../../../../vllm_ascend/distributed/parallel_state.py#L258)

```python
def get_flashcomm2_otp_group() -> GroupCoordinator:
    return _FLASHCOMM2_OTP

def get_flashcomm2_odp_group() -> GroupCoordinator:
    assert _FLASHCOMM2_ODP is not None
    return _FLASHCOMM2_ODP
```

---

## 3. FlashComm2 OProj 操作符

### 3.1 Op 选择逻辑

> [`vllm_ascend/ops/linear_op.py:694-730`](../../../../vllm_ascend/ops/linear_op.py#L694)

```python
def _get_row_parallel_op(prefix, layer):
    if "o_proj" in prefix and oproj_tp_enable():
        return OProjRowParallelOp(layer)       # [A] 普通 OProj (有 DBO hook)
    ...
    if flashcomm2_enable():
        if "o_proj" in prefix or "out_proj" in prefix:
            return Flashcomm2OProjRowParallelOp(layer)  # [B] FlashComm2 (无 DBO hook)
    if enable_sp():
        ...
        return SequenceRowParallelOp(layer)    # [C] SP OProj
```

优先级：[A] > [B] > [C]。当 `flashcomm2_enable()` 且 `oproj_tp_enable()` 为 False 时，选 [B]。

### 3.2 Flashcomm2OProjRowParallelOp 类定义

> [`vllm_ascend/ops/linear_op.py:289-399`](../../../../vllm_ascend/ops/linear_op.py#L289)

#### 3.2.1 构造函数 — [`L289-297`](../../../../vllm_ascend/ops/linear_op.py#L289)

```python
class Flashcomm2OProjRowParallelOp(CustomRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.odp_group = get_flashcomm2_odp_group()     # ODP 通信组
        self.odp_size = self.odp_group.world_size        # ODP size
        self.otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
        self.reorgnized_batch_ids = get_flashcomm2_reorgnized_batch_ids(
            get_tp_group().world_size
        )
        self.group_indices = torch.tensor(self.reorgnized_batch_ids).npu()
```

**`reorgnized_batch_ids` 生成逻辑** — [`vllm_ascend/utils.py:1253-1269`](../../../../vllm_ascend/utils.py#L1253)：

```python
def get_flashcomm2_reorgnized_batch_ids(global_tp_size):
    flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
    num_oproj_tensor_parallel_groups = global_tp_size // flashcomm2_otp_size

    reorgnized_batch_ids = []
    for i in range(num_oproj_tensor_parallel_groups):
        ranks = []
        for j in range(flashcomm2_otp_size):
            rank_idx = i + j * num_oproj_tensor_parallel_groups
            ranks.append(rank_idx)
        reorgnized_batch_ids.append(ranks)
    return reorgnized_batch_ids
```

TP=4, otp_size=2：`[[0, 2], [1, 3]]` → `group_indices = tensor([0, 2, 1, 3])`

#### 3.2.2 `tp_rank` / `tp_size` 属性 — [`L303-313`](../../../../vllm_ascend/ops/linear_op.py#L303)

当 `flashcomm2_otp_size == 1` 时，返回 `tp_rank=0, tp_size=1`；否则使用 OTP 组的 rank/size。

#### 3.2.3 核心 forward: `apply_impl()` — [`L315-392`](../../../../vllm_ascend/ops/linear_op.py#L315)

完整数据流逐步解读：

```python
def apply_impl(self, input_: torch.Tensor):
    # ===================================================================
    # Step 1: 获取输入 — [L324](../../../../vllm_ascend/ops/linear_op.py#L324)
    # ===================================================================
    input_parallel = self.get_input_parallel(input_)
    # 若 input_is_parallel=True → 直接返回 input_
    # 否则 → split_tensor_along_last_dim(input_, tp_size)
    # 对于 FlashComm2，input_is_parallel 通常为 True
    # shape: [batch_tokens, headnum * headdim / TP]

    # ===================================================================
    # Step 2: Token Padding — [L327-329](../../../../vllm_ascend/ops/linear_op.py#L327)
    # ===================================================================
    num_padding_tokens = _EXTRA_CTX.pad_size
    if num_padding_tokens > 0:
        input_parallel = nn.functional.pad(
            input_parallel, (0, 0, 0, num_padding_tokens)
        )
    # shape: [batch_tokens + pad, headnum * headdim / TP]
    # pad_size 由 ascend_forward_context.py:L148 决定：
    #   pad_size = (TP - num_tokens % TP) % TP
    # DP 场景: pad 到 max_tokens_across_dp 对齐 TP

    # ===================================================================
    # Step 3: Batch 重排 + ODP All-to-All — [L331-357](../../../../vllm_ascend/ops/linear_op.py#L331)
    # ===================================================================
    def otp_maybe_quant_comm(x):
        # chunk_num = num_otp_groups * otp_size = 2 * 2 = 4 (for TP=4)
        chunk_num = len(self.reorgnized_batch_ids) * len(
            self.reorgnized_batch_ids[0]
        )

        batch_size = x.size(0)                    # padded_bs
        assert batch_size % chunk_num == 0
        batch_size_per_chunk = batch_size // chunk_num  # padded_bs / 4

        # 将 x 切成 chunk_num 份
        chunked = x.view(chunk_num, batch_size_per_chunk, x.shape[1])
        # [4, bs/4, dim]

        if self.otp_size != 1:
            chunked = chunked[self.group_indices]  # ← batch 重排！[L341](../../../../vllm_ascend/ops/linear_op.py#L341)
            # group_indices=[0,2,1,3] → 原本 [c0,c1,c2,c3] 变为 [c0,c2,c1,c3]
        send_buf = chunked.flatten(1, 2)
        # [4, bs/4 * dim]

        # All-to-All 参数
        all2all_tp_size = self.odp_size           # ODP group size
        local_intermediate_size = x.size(1)        # dim = headnum*headdim/TP
        chunk_size = x.size(0) // all2all_tp_size   # padded_bs / odp_size
        total_intermediate_size = local_intermediate_size * all2all_tp_size

        recv_buf = torch.empty(
            total_intermediate_size * chunk_size,
            dtype=x.dtype, device=x.device
        )

        # All-to-all across ODP group — [L355](../../../../vllm_ascend/ops/linear_op.py#L355)
        dist.all_to_all_single(recv_buf, send_buf,
                               group=self.odp_group.device_group)

        return (recv_buf
                .view(all2all_tp_size, chunk_size, -1)
                # [odp_size, padded_bs/odp_size, dim]
                .transpose(0, 1)
                # [padded_bs/odp_size, odp_size, dim]
                .reshape(chunk_size, -1))
                # [padded_bs/odp_size, odp_size * dim]
                # = [padded_bs/odp_size, odp_size * headnum*headdim/TP]

    # ===================================================================
    # Step 4: W8A8 融合路径 — [L359-367](../../../../vllm_ascend/ops/linear_op.py#L359)
    # ===================================================================
    self.layer._quant_comm_config["communication_fn"] = otp_maybe_quant_comm
    # 将通信函数注入 layer，供 W8A8 quant_method 延迟调用

    from vllm_ascend.quantization.methods.w8a8_static import (
        AscendW8A8LinearMethod
    )
    if not isinstance(actual_quant_method, AscendW8A8LinearMethod):
        # 非 W8A8 路径：立即执行通信 — [L367](../../../../vllm_ascend/ops/linear_op.py#L367)
        input_parallel = otp_maybe_quant_comm(input_parallel)

    # ===================================================================
    # Step 5: 矩阵乘法 — [L370-375](../../../../vllm_ascend/ops/linear_op.py#L370)
    # ===================================================================
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output_parallel = self.quant_method.apply(
        self.layer, input_parallel, bias=bias_
    )
    # input:  [padded_bs/odp_size, odp_size * headnum*headdim/TP]
    # output: [padded_bs/odp_size, hidden_state]

    # ===================================================================
    # Step 6: OTP Reduce-Scatter — [L377-381](../../../../vllm_ascend/ops/linear_op.py#L377)
    # ===================================================================
    if self.tp_size > 1:
        output = self.comm_group.reduce_scatter(output_parallel, dim=0)
        # self.comm_group = get_flashcomm2_otp_group()  — [L301](../../../../vllm_ascend/ops/linear_op.py#L301)
        # OTP Group {0,2} 和 {1,3} 内部做 reduce-scatter
    else:
        output = output_parallel

    # ===================================================================
    # Step 7: TP All-Gather + De-pad — [L383-387](../../../../vllm_ascend/ops/linear_op.py#L383)
    # ===================================================================
    if not _EXTRA_CTX.flash_comm_v1_enabled:
        # FlashComm1 未启用时需要手动全局 TP all-gather
        output = get_tp_group().all_gather(output, 0)
        if num_padding_tokens > 0:
            output = output[:-num_padding_tokens]   # 去除 padding

    # ===================================================================
    # Step 8: 返回
    # ===================================================================
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

### 3.3 与普通 OProj 的代码级对比

**普通 `OProjRowParallelOp.apply_impl()`** — [`vllm_ascend/ops/linear_op.py:240-281`](../../../../vllm_ascend/ops/linear_op.py#L240)

```
通信模式:
  input → all_to_all (get_otp_group) → linear → reduce_scatter (get_otp_group)

DBO hook:
  [L264](../../../../vllm_ascend/ops/linear_op.py#L264): _dbo_call_linear_row_hook(is_record=True)   ← all-to-all 前
  [L276](../../../../vllm_ascend/ops/linear_op.py#L276): _dbo_call_linear_row_hook(is_record=False)  ← reduce-scatter 后
```

**`Flashcomm2OProjRowParallelOp.apply_impl()`** — [`vllm_ascend/ops/linear_op.py:315-392`](../../../../vllm_ascend/ops/linear_op.py#L315)

```
通信模式:
  input → ODP all-to-all → linear → OTP reduce-scatter → (optional TP all-gather)

DBO hook:
  无 ← 完全缺失！
```

---

## 4. `_EXTRA_CTX` 与 padding_size 计算

### 4.1 _EXTRA_CTX 代理定义

> [`vllm_ascend/ascend_forward_context.py:481-540`](../../../../vllm_ascend/ascend_forward_context.py#L481)

```python
class _ExtraForwardContextProxy:
    extra_attrs = (
        ...
        "flash_comm_v1_enabled",    # [L490](../../../../vllm_ascend/ascend_forward_context.py#L490)
        "flashcomm_v2_enabled",    # [L491](../../../../vllm_ascend/ascend_forward_context.py#L491)
        "pad_size",                # [L492](../../../../vllm_ascend/ascend_forward_context.py#L492)
        ...
    )

_EXTRA_CTX = _ExtraForwardContextProxy()   # [L540](../../../../vllm_ascend/ascend_forward_context.py#L540)
```

这是对 `forward_context` 的代理对象，统一 v1/v2 model runner 的访问接口。
详见 `__getattr__` 在 [`L521-528`](../../../../vllm_ascend/ascend_forward_context.py#L521) — v1 从 `ctx` 属性读，v2 从 `ctx.additional_kwargs` 读。

### 4.2 FlashComm2 开关与 pad_size 设置

> [`vllm_ascend/ascend_forward_context.py:144-148`](../../../../vllm_ascend/ascend_forward_context.py#L144)

```python
# L144 — flashcomm_v2_enabled 条件：fc2 开启 + TP>1 + num_tokens 非空
forward_context.flashcomm_v2_enabled = (
    flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None
)

# L147-148 — FC1 和 FC2 共享 pad_size
if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
    pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
```

DP 场景的额外处理 — [`vllm_ascend/ascend_forward_context.py:182-186`](../../../../vllm_ascend/ascend_forward_context.py#L182)：
```python
if dp_world_size > 1 and forward_context.dp_metadata is not None:
    max_tokens_across_dp = dp_meta.num_tokens_across_dp_cpu.max().item()
    padded_length = ((max_tokens_across_dp + tp_world_size - 1)
                     // tp_world_size * tp_world_size)
    pad_size = padded_length - num_tokens      # pad 到全 DP 最大 token 数
```

`platform.py` 中也有相同逻辑 — [`vllm_ascend/platform.py:1018-1023`](../../../../vllm_ascend/platform.py#L1018)。

---

## 5. Batch ID 重排机制详解

### 5.1 映射表生成

> [`vllm_ascend/utils.py:1253-1269`](../../../../vllm_ascend/utils.py#L1253)

```python
def get_flashcomm2_reorgnized_batch_ids(global_tp_size):
    flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
    num_oproj_tensor_parallel_groups = global_tp_size // flashcomm2_otp_size
    # TP=4, otp=2 → groups=2
    # i=0: ranks=[0+0*2=0, 0+1*2=2] = [0, 2]
    # i=1: ranks=[1+0*2=1, 1+1*2=3] = [1, 3]
    # → [[0, 2], [1, 3]]
```

### 5.2 重排的数据含义

以 4 tokens, TP=4, otp=2 为例：

```
原始 token 分配（假设每个 token 属于不同 batch_id）:
  token_0 → batch_id=0
  token_1 → batch_id=1
  token_2 → batch_id=2
  token_3 → batch_id=3

chunked = input.view(4, 1, dim)  # 4 个 chunk，每个 1 token
chunked = [t0, t1, t2, t3]       # chunk 按 batch_id 顺序排列

group_indices = [0, 2, 1, 3]
chunked = [t0, t2, t1, t3]       # 重排后

ODP all-to-all (rank0 ↔ rank1):
  rank0 sends {t1, t3}, receives rank1's {t1', t3'}
  rank1 sends {t0, t2}, receives rank0's {t0', t2'}
```

**效果**：同 OTP group（rank0, rank2）的 batch 被分到 ODP all-to-all 的同一侧，确保后续 OTP reduce-scatter 时数据在正确的 rank 上。

---

## 6. W8A8 量化融合路径

### 6.1 FlashComm2 注入通信函数

> [`vllm_ascend/ops/linear_op.py:360-361`](../../../../vllm_ascend/ops/linear_op.py#L360)

```python
self.layer._quant_comm_config["communication_fn"] = otp_maybe_quant_comm
```

### 6.2 W8A8 量化方法中的融合执行

> [`vllm_ascend/quantization/methods/w8a8_static.py:90-108`](../../../../vllm_ascend/quantization/methods/w8a8_static.py#L90)

```python
comm_fn = quant_comm_config.get("communication_fn")
enable_flashcomm2_quant_comm = (
    comm_fn is not None and
    ("o_proj" in layer.prefix or "out_proj" in layer.prefix)
)
if enable_flashcomm2_quant_comm:
    # 先量化 → 再通信（节省一次显存读写 + 通信量减半）
    quant_x = torch.ops.vllm.quantize(
        x, layer.aclnn_input_scale,
        layer.aclnn_input_scale_reciprocal, layer.aclnn_input_offset
    )
    comm_input = quant_x.view(x.size(0), -1)
    x = comm_fn(comm_input)  # 调用 otp_maybe_quant_comm
```

**优化效果**：quantize 输出是 int8，大小只有 fp16 的一半，all-to-all 通信量减半。

### 6.3 tp_rank 的正确获取

> [`vllm_ascend/quantization/method_adapters.py:151-155`](../../../../vllm_ascend/quantization/method_adapters.py#L151)

```python
elif ("o_proj" in layer.prefix or "out_proj" in layer.prefix) \
        and flashcomm2_enable():
    if get_ascend_config().flashcomm2_oproj_tensor_parallel_size == 1:
        tp_rank = 0
    else:
        tp_rank = get_flashcomm2_otp_group().rank_in_group
```

---

## 7. O-Shard Weight 分片（可选增强）

### 7.1 O-Shard 原理概述

O-Shard 是 FlashComm2 配套的显存优化方案，核心思路是**将 Attention O-Proj weight 分散到多张卡存储，通过异步 broadcast 按需加载**。

**启用条件**（两个同时满足）：
- `flashcomm2_enable()` = True → `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0`
- `o_shard_enable()` = True → `layer_sharding` 包含 `"o_proj"` — [`vllm_ascend/utils.py:1196-1200`](../../../../vllm_ascend/utils.py#L1196)

```python
def o_shard_enable() -> bool:
    layer_sharding = get_ascend_config().layer_sharding
    if layer_sharding is None:
        return False
    return "o_proj" in layer_sharding
```

### 7.2 Weight 分片存储策略

> [`vllm_ascend/ops/layer_shard_linear.py:18-56`](../../../../vllm_ascend/ops/layer_shard_linear.py#L18)

**关键数据结构**：

```python
@dataclass
class SeriesMetadata:
    """Weight shard series 的元数据"""
    group: GroupCoordinator              # shard_weight_group（即 TP group）
    start_layer: int                     # 系列的起始层索引
    end_layer: int                       # 系列的结束层索引
    num_layers: int                      # 该系列包含的总层数
    prefetch_step: int                   # 预取步数（通常=1）
    dummy_weight: torch.Tensor           # 临时占位符 weight
    layers: list[LayerMetadata]          # 所有注册的层对象
    shard_windows: list[ShardWindowMetadata]  # 环形缓冲（大小 = prefetch_step + 1）
    window_offset: int                   # 下一个待使用的 window 索引

    def is_source(self, layer_idx) -> bool:
        # 该层的 weight 是否存储在当前 rank 上
        return layer_idx % self.group.world_size == self.group.rank_in_group
```

**分片规则**：层 `i` 的 weight 存储在 rank `(i % TP_size)` 上。

例如 TP=4、o_proj 的 4 层分布：
```
layer 0 → rank 0（owner）    layer 1 → rank 1（owner）
layer 2 → rank 2（owner）    layer 3 → rank 3（owner）
```

每个 rank 只存储自己负责的 layer weight，其他层的 weight 通过按需 broadcast 获取。

### 7.3 环形缓冲与 Prefetch 机制

> [`vllm_ascend/ops/layer_shard_linear.py:57-114`](../../../../vllm_ascend/ops/layer_shard_linear.py#L57)

**初始化过程** — `post_process_after_loading()`：

```
Step 1: 排序所有注册的层
        layers.sort(key=lambda x: x.layer_idx)

Step 2: 对每一层执行 broadcast
        for layer_idx in range(start_layer, end_layer):
            is_source = (layer_idx % group_size == rank_in_group)
            if is_source:
                # 该层的 owner rank：直接使用本地 weight
            else:
                # 其他 rank：创建空 tensor 接收 broadcast
                layer.weight.set_(torch.empty_like(dummy_weight))

            dist.broadcast(layer.weight, src=source_rank, group=shard_group)
            layer.quant_method.process_weights_after_loading()

Step 3: 建立 shard_windows（环形缓冲）
        缓冲大小 = prefetch_step + 1（通常=2）

        前 prefetch_step 层（0 到 prefetch_step-1）:
            ├─ 创建 window，clone weight
            └─ 非源 rank：让 layer.weight 指向该 window

        第 prefetch_step 层:
            ├─ 创建 empty window（用于异步加载下一层）
            └─ 非源 rank：dispose 原 weight（释放显存）

Step 4: dispose dummy_weight
```

**Prefetch 流程** — `reach_layer(layer_idx)` 在 forward 中被调用：

```python
def reach_layer(self, layer_idx: int):
    # 计算待预取的下一层
    next_layer_idx = (layer_idx - self.start_layer + self.prefetch_step) % self.num_layers + self.start_layer
    next_window = self.shard_windows[self.window_offset]

    # 源 rank 拷贝 weight 到 next_window
    if self.is_source(next_layer_idx):
        next_window.weight.copy_(self.layers[next_layer_idx - self.start_layer].weight)

    # 异步 broadcast
    work = dist.broadcast(
        next_window.weight,
        src=self.group.ranks[next_layer_idx % self.group.world_size],
        group=self.group.device_group,
        async_op=True  # 异步执行
    )

    next_window.work = work
    next_window.data_layer_idx = next_layer_idx
    self.window_offset = (self.window_offset + 1) % len(self.shard_windows)
```

**等待 Weight 就绪** — `wait_weight(layer_idx)` 在 layer.forward 前被调用：

```python
def wait_weight(self, layer_idx: int):
    window_idx = (layer_idx - self.start_layer) % len(self.shard_windows)
    window = self.shard_windows[window_idx]

    # 等待异步 broadcast 完成
    if window.work is not None:
        window.work.wait()
        window.work = None
```

**时间线示例**（prefetch_step=1，TP=4）：

```
t=0: forward(layer_0)
     ├─ wait_weight(layer_0) → window[0] 已就绪
     ├─ layer_0.forward()
     └─ reach_layer(layer_0)  [在 QKV forward 时调用]
        └─ 异步加载 layer_1 weight 到 window[1]

t=1: forward(layer_1)
     ├─ wait_weight(layer_1) → 等待 window[1] broadcast 完成
     ├─ layer_1.forward()
     └─ reach_layer(layer_1)
        └─ 异步加载 layer_2 weight 到 window[0]（轮转覆盖）

t=2: forward(layer_2)
     ├─ wait_weight(layer_2) → 等待 window[0] broadcast 完成
     └─ ...
```

### 7.4 集成架构

#### 7.4.1 Flashcomm2OShardManager（顶层 Wrapper）

> [`vllm_ascend/ops/flashcomm2_oshard_manager.py:15-101`](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L15)

```python
class Flashcomm2OShardManager:
    def __init__(self):
        self._shard_layers: dict[int, Any] = {}  # layer_idx → layer object

    def flashcomm2_oshard_enable(self):
        return flashcomm2_enable() and o_shard_enable()

    def register_layer(self, layer: Any, prefetch_step: int = 1):
        """在模型初始化时被 Flashcomm2OProjRowParallelOp.update_attrs() 调用"""
        if is_hidden_layer(layer):
            layer_idx = extract_layer_index(layer.prefix)
            self._shard_layers[layer_idx] = layer

            register_layer_to_shard_weight_series(
                series_name="o_proj",
                group=get_shard_weight_group(),
                layer=layer,
                prefetch_step=prefetch_step
            )

    def trigger_broadcast_for_layer(self, layer_prefix: str):
        """在 forward 时被 Flashcomm2OshardQKVParallelOp.apply_impl() 调用"""
        layer_idx = extract_layer_index(layer_prefix)
        target_layer = self.get_layer(layer_idx)

        if target_layer and is_hidden_layer(target_layer):
            reach_layer_for_shard_weight_series(target_layer)

    def post_process_after_loading(self):
        """在权重加载完成后被 AttentionV1.process_weights_after_loading() 调用"""
        if self._shard_layers:
            any_layer = next(iter(self._shard_layers.values()))
            post_process_after_loading_for_shard_weight_series(any_layer)

# 全局单例
flashcomm2_oshard_manager = Flashcomm2OShardManager()
```

#### 7.4.2 集成入口 1：模型初始化

> [`vllm_ascend/ops/linear_op.py:422-427`](../../../../vllm_ascend/ops/linear_op.py#L422)

```python
class Flashcomm2OProjRowParallelOp(CustomRowParallelOp):
    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.input_size_per_partition = self.layer.input_size_per_partition

        # 只有 FlashComm2 O-Proj Op 在初始化时注册层到 O-Shard manager
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.register_layer(self.layer, prefetch_step=1)
```

**关键点**：
- 只有 `Flashcomm2OProjRowParallelOp` 会注册（普通 `OProjRowParallelOp` 不会）
- 在 `CustomRowParallelOp.update_attrs()` 被模型初始化调用时触发
- 每个 O-Proj layer 被注册到底层的 `layer_shard_linear` 的全局 series dict

#### 7.4.3 集成入口 2：权重加载完成

> 调用链：`AttentionV1.process_weights_after_loading()` → `flashcomm2_oshard_manager.post_process_after_loading()` → `post_process_after_loading_for_shard_weight_series()`

初始化 shard_windows、执行所有 broadcast、建立 forward wrapper。

#### 7.4.4 集成入口 3：前向传播（Prefetch 触发）

> [`vllm_ascend/ops/linear_op.py:520-525`](../../../../vllm_ascend/ops/linear_op.py#L520)

在 **Flashcomm2OshardQKVParallelOp.apply_impl()** 中触发：

```python
class Flashcomm2OshardQKVParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_):
        # ... QKV 计算 ...

        # 在 matmul 前触发异步加载下一层（O-Proj）的 weight
        flashcomm2_oshard_manager.trigger_broadcast_for_layer(self.layer.prefix)

        output_parallel = self.quant_method.apply(self.layer, input_, bias)
        # ...
```

**为什么在 QKV 这里触发**？
- QKV forward 完成 → Attention 计算
- Attention 完成后立即需要 O-Proj
- 在 QKV 执行时异步加载 O-Proj weight，**overlap communication 和计算**

#### 7.4.5 集成入口 4：Layer Forward 包装

> [`vllm_ascend/ops/layer_shard_linear.py:162-168`](../../../../vllm_ascend/ops/layer_shard_linear.py#L162)

```python
def _create_forward_wrapper(forward: Callable, series: SeriesMetadata, layer_idx: int) -> Callable:
    def wrapped_forward(*args, **kwargs):
        # 在 forward 前等待 weight 加载完成
        series.wait_weight(layer_idx)
        return forward(*args, **kwargs)

    return wrapped_forward

# 在 register_layer_to_shard_weight_series 中被使用
layer.forward = _create_forward_wrapper(layer.forward, series, layer_idx)
```

每个注册的 O-Proj layer 的 forward 都被包装，确保 weight 已就位。

### 7.5 完整调用链与时间轴

```
┌─ 配置阶段 ──────────────────────────────────────────────┐
│  env: VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=N            │
│  env: VLLM_ASCEND_LAYER_SHARDING=[o_proj]               │
│  → o_shard_enable() = True                              │
└──────────────────────────────────────────────────────────┘
         ↓
┌─ 模型初始化阶段 ────────────────────────────────────────┐
│  for each layer in model.layers:                         │
│    if "o_proj" in layer.prefix:                          │
│      Flashcomm2OProjRowParallelOp(layer).update_attrs()  │
│        → flashcomm2_oshard_manager.register_layer()      │
│          → register_layer_to_shard_weight_series()       │
│            └─ 记录 layer metadata                        │
│            └─ 非源 rank dispose weight                   │
│            └─ 包装 layer.forward() with wait_weight()    │
└──────────────────────────────────────────────────────────┘
         ↓
┌─ 权重加载完成后 ────────────────────────────────────────┐
│  AttentionV1.process_weights_after_loading()             │
│    → flashcomm2_oshard_manager.post_process_after_loading()
│      → post_process_after_loading_for_shard_weight_series()
│        ├─ Broadcast all O-Proj weights                  │
│        ├─ 初始化 shard_windows（prefetch_step+1 个）    │
│        └─ forward wrapper 已就位                        │
└──────────────────────────────────────────────────────────┘
         ↓
┌─ 前向传播阶段（迭代 per token） ────────────────────────┐
│  for each layer_idx in range(num_layers):                │
│                                                           │
│    t=k: Flashcomm2OshardQKVParallelOp[k].apply_impl()    │
│        ├─ Attention QKV + matmul                         │
│        └─ trigger_broadcast_for_layer(prefix[k])         │
│          → reach_layer(o_proj[k+1])                      │
│            └─ 异步加载 o_proj[k+1] weight 到 window     │
│                                                           │
│    t=k+1: Flashcomm2OProjRowParallelOp[k].apply_impl()   │
│        ├─ wrapped_forward() 自动调用：                   │
│        │  └─ wait_weight(k)                              │
│        │    └─ 等待 t=k 的 broadcast 完成               │
│        └─ O-Proj 本地 forward + 通信                    │
└──────────────────────────────────────────────────────────┘
```

### 7.6 与 FlashComm2 通信的协作

O-Shard 本身**不参与 FlashComm2 的 ODP all-to-all 和 OTP reduce-scatter**，只负责 O-Proj weight 的按需加载。

但两者配合工作流程：

```
Layer N Attention:
  QKV forward (Flashcomm2OshardQKVParallelOp)
  ├─ trigger_broadcast_for_layer(qkv[N]) → 异步加载 o_proj[N+1] weight
  ├─ QKV matmul
  └─ Attention 计算

  O-Proj forward (Flashcomm2OProjRowParallelOp)
  ├─ wait_weight(N) → 等待 o_proj[N] weight ready
  ├─ ODP all-to-all（FlashComm2 通信）
  ├─ O-Proj matmul
  └─ OTP reduce-scatter（FlashComm2 通信）

Layer N+1 Attention:
  QKV forward (Flashcomm2OshardQKVParallelOp)
  ├─ trigger_broadcast_for_layer(qkv[N+1]) → 异步加载 o_proj[N+2] weight
  └─ （此时 o_proj[N+1] weight 已就绪）

  O-Proj forward (Flashcomm2OProjRowParallelOp)
  ├─ wait_weight(N+1) → 等待 o_proj[N+1] weight ready（通常已就绪）
  └─ ...
```

**显存优化效果**：
- 无 O-Shard：每张卡存储所有 O-Proj layer 的 weight（显存占用 = layers × weight_size）
- 有 O-Shard：每张卡只存储 `num_layers / TP` 个 O-Proj layer weight + 2 个 window（显存占用 ≈ `(layers / TP) × weight_size`）

---

## 8. DBO 集成分析

### 8.1 普通 OProj 的 DBO Hook 位置

> [`vllm_ascend/ops/linear_op.py:240-281`](../../../../vllm_ascend/ops/linear_op.py#L240)

```python
# OProjRowParallelOp.apply_impl()
def apply_impl(self, input_):
    input_parallel = self.get_input_parallel(input_)
    ...
    forward_context = get_forward_context()
    if forward_context.dbo_enabled:
        _dbo_call_linear_row_hook(forward_context, is_record=True)
        # [L264](../../../../vllm_ascend/ops/linear_op.py#L264) — 在 all-to-all 前 record ATTN_POST

    dist.all_to_all_single(recv_buf, send_buf, group=self.comm_group.device_group)
    input_parallel = recv_buf.view(total_batch_size, chunk_size)
    ...
    output_parallel = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
    output = self.comm_group.reduce_scatter(output_parallel, dim=0)

    if forward_context.dbo_enabled:
        _dbo_call_linear_row_hook(forward_context, is_record=False)
        # [L276](../../../../vllm_ascend/ops/linear_op.py#L276) — 在 reduce-scatter 后 wait ATTN_POST + yield

    return output, output_bias
```

### 8.2 FlashComm2 OProj 缺少 DBO Hook

> [`vllm_ascend/ops/linear_op.py:315-392`](../../../../vllm_ascend/ops/linear_op.py#L315)

整个 `apply_impl` 中**没有 `_dbo_call_linear_row_hook` 的任何调用**。

### 8.3 DBO Overlap 模板对 ATTN_POST 的依赖

Hook wrapper 定义 — [`vllm_ascend/dbo/compile_guard.py:18-20`](../../../../vllm_ascend/dbo/compile_guard.py#L18)：
```python
@torch.compiler.disable()
def _dbo_call_linear_row_hook(forward_context, is_record: bool) -> None:
    forward_context.dbo_template.dbo_linear_row_hook(is_record=is_record)
```

A2 Overlap Template 实现 — [`vllm_ascend/dbo/overlap_templates/deepseek.py:7-37`](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L7)：

| 位置 | hook | 事件操作 | 含义 |
|------|------|----------|------|
| [L19-22](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L19) | `dbo_linear_row_hook(is_record=True)` | **record** ATTN_POST | OProj 通信开始时通知下游 |
| [L24-27](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L24) | `dbo_linear_column_hook(is_record=False)` | **wait** ATTN_POST | MLP down_proj 等待 OProj 通信完成 |
| [L29-31](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L29) | `dbo_moe_prepare_hook(is_record=False)` | **wait** ATTN_POST | MoE prepare 等待 OProj 通信完成 |
| [L34-36](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L34) | `dbo_moe_finalize_hook(is_record=True)` | **record** ATTN_PRE | 通知下一层 MLA preprocess |

### 8.4 断裂的事件链

```
Layer N 的正常 DBO 流程 (A2):
  ┌─ dbo_mla_preprocess_hook: wait ATTN_PRE          ← 等上一层 moe_finalize
  ├─ MLA QKV 计算 + all-gather
  ├─ Attention 计算
  ├─ dbo_linear_row_hook: record ATTN_POST           ← FlashComm2 没做 ← 断裂点！
  ├─ OProj 通信 + Linear
  ├─ dbo_linear_column_hook: wait ATTN_POST          ← 永远等不到！
  ├─ MLP
  ├─ dbo_moe_prepare_hook: wait ATTN_POST            ← 永远等不到！
  ├─ MoE prepare + 计算
  └─ dbo_moe_finalize_hook: record ATTN_PRE          ← 给下一层

结果: column_hook 和 moe_prepare_hook 会阻塞等待 ATTN_POST，
     但 FlashComm2 从未 record 该事件 → ubatch 线程死锁。
```

核心原因：`_get_row_parallel_op` 中 FlashComm2 替换了普通 OProj（[`L714-715`](../../../../vllm_ascend/ops/linear_op.py#L714)），但 FlashComm2 的实现没有复制普通 OProj 的 DBO hook 逻辑。

### 8.5 修复方向

需要在 `Flashcomm2OProjRowParallelOp.apply_impl()` 中围绕通信块添加 hook：

```
位置 1: ODP all-to-all 执行前 → _dbo_call_linear_row_hook(is_record=True)  # record ATTN_POST
位置 2: OTP reduce-scatter 完成后 → _dbo_call_linear_row_hook(is_record=False)  # wait + yield
```

需要考虑 W8A8 延迟通信路径。当 W8A8 启用时，`otp_maybe_quant_comm` 被延迟到 `quant_method.apply()` 内部执行（[`w8a8_static.py:98-108`](../../../../vllm_ascend/quantization/methods/w8a8_static.py#L98)），所以 record hook 不能简单放在 `otp_maybe_quant_comm` 调用处，而应放在实际通信发生的位置之前。

对于 W8A8 路径，整个 matmul+通信 是一个融合操作，record hook 应放在 `quant_method.apply()` 之前或 `otp_maybe_quant_comm` 内部的 `all_to_all_single` 之前。

---

## 8. O-Shard 与 DBO 的交互

### 8.1 O-Shard 中的 Forward Wrapper 不会额外触发 DBO Hook

> [`vllm_ascend/ops/layer_shard_linear.py:162-168`](../../../../vllm_ascend/ops/layer_shard_linear.py#L162)

O-Shard 的 forward wrapper 只负责等待 weight ready：

```python
def _create_forward_wrapper(forward: Callable, series: SeriesMetadata, layer_idx: int) -> Callable:
    def wrapped_forward(*args, **kwargs):
        # 仅等待 weight 加载，不涉及 DBO
        series.wait_weight(layer_idx)
        return forward(*args, **kwargs)

    return wrapped_forward
```

**DBO hook 仍由 `Flashcomm2OProjRowParallelOp.apply_impl()` 负责** — [`L387-388, 414-415`](../../../../vllm_ascend/ops/linear_op.py#L387)

### 8.2 FlashComm2 DBO Hook 覆盖范围

FlashComm2 的 DBO hook 需要覆盖完整的通信块（包括 O-Shard weight 加载）：

```python
# Flashcomm2OProjRowParallelOp.apply_impl()
def apply_impl(self, input_):
    # ... get input ...

    # DBO record: 通知下游可以开始准备 O-Proj 的输入
    if forward_context.dbo_enabled:
        _dbo_call_linear_row_hook(forward_context, is_record=True)  # ← L388

    # ... ODP all-to-all ...
    # ... matmul（此时 O-Shard 的 wrapped_forward 会自动等待 weight） ...
    # ... OTP reduce-scatter ...

    # DBO wait: 确保所有通信完成后再让下游（MLP 等）执行
    if forward_context.dbo_enabled:
        _dbo_call_linear_row_hook(forward_context, is_record=False)  # ← L415
```

**关键点**：O-Shard 的 weight broadcast 发生在 **record 和 wait 之间**，属于 DBO 管理的 overlap 范围内。

---

## 9. O-Shard 数据结构总览

### 9.1 全局状态管理

> [`vllm_ascend/ops/layer_shard_linear.py:157-160`](../../../../vllm_ascend/ops/layer_shard_linear.py#L157)

```python
# 全局字典：记录所有注册的 shard series
_series_dict: dict[str, SeriesMetadata] = {}
# 例：{"o_proj": SeriesMetadata(...)}

# 全局字典：记录每个 layer 的外部元数据（指回 series）
_layer_external_dict: dict[int, LayerExternalMetadata] = {}
# 例：{id(o_proj_layer_0): LayerExternalMetadata(series=series_dict["o_proj"], layer_idx=0), ...}
```

### 9.2 Flashcomm2OShardManager 的本地缓存

```python
# flashcomm2_oshard_manager._shard_layers — vllm_ascend/ops/flashcomm2_oshard_manager.py:L32
{
    0: o_proj_layer_0,  # layer_idx → layer object
    1: o_proj_layer_1,
    2: o_proj_layer_2,
    3: o_proj_layer_3,
}
```

### 9.3 SeriesMetadata 中的 shard_windows（环形缓冲）

> [`vllm_ascend/ops/layer_shard_linear.py:29-36`](../../../../vllm_ascend/ops/layer_shard_linear.py#L29)

以 prefetch_step=1、TP=4 为例（4 层 O-Proj）：

```python
shard_windows = [
    ShardWindowMetadata(weight=tensor[...], data_layer_idx=0, work=None),
    ShardWindowMetadata(weight=tensor[...], data_layer_idx=1, work=None),
]
# 大小 = prefetch_step + 1 = 2

# 时间轴：
# t=0: window_offset=0
#      reach_layer(0) 异步加载 layer_1 到 window[1]
#      window_offset → 1
#
# t=1: window_offset=1
#      reach_layer(1) 异步加载 layer_2 到 window[0]（覆盖）
#      window_offset → 0
#
# t=2: window_offset=0
#      reach_layer(2) 异步加载 layer_3 到 window[1]（覆盖）
#      window_offset → 1
```

### 9.4 LayerMetadata 记录结构

> [`vllm_ascend/ops/layer_shard_linear.py:18-27`](../../../../vllm_ascend/ops/layer_shard_linear.py#L18)

```python
@dataclass
class LayerMetadata:
    layer_idx: int                    # O-Proj 层在模型中的索引
    layer: LinearBase                 # layer 对象本身
    post_method: Callable             # layer.quant_method.process_weights_after_loading
    weight: torch.Tensor              # layer.weight（在非源 rank 被替换为 window tensor）
    window_idx: int                   # 该 layer 当前关联的 window 索引
```

---

## 10. O-Shard 配置与启用

## 10. O-Shard 配置与启用

### 10.1 环境变量配置

> [`vllm_ascend/utils.py:1196-1200`](../../../../vllm_ascend/utils.py#L1196)

```bash
# 同时设置以下两个环境变量启用 O-Shard：

# 1. 启用 FlashComm2
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2  # 或其他 > 0 的值

# 2. 启用 O-Proj layer sharding
export VLLM_ASCEND_LAYER_SHARDING=[o_proj]
```

**校验逻辑**：

```python
# flashcomm2_enable()
def flashcomm2_enable() -> bool:
    return get_ascend_config().enable_flashcomm2_parallel_size > 0

# o_shard_enable()
def o_shard_enable() -> bool:
    layer_sharding = get_ascend_config().layer_sharding
    if layer_sharding is None:
        return False
    return "o_proj" in layer_sharding

# 两者都满足才启用
def flashcomm2_oshard_enable(self):
    return flashcomm2_enable() and o_shard_enable()
```

### 10.2 约束条件检查

> [`vllm_ascend/utils.py:1213-1220`](../../../../vllm_ascend/utils.py#L1213)

O-Shard 只允许 `layer_sharding` 包含 `"o_proj"`，不能包含其他层类型：

```python
if layer_sharding is not None and flashcomm2_enable():
    unsupported_layers = [x for x in layer_sharding if x not in ["o_proj"]]
    if unsupported_layers:
        raise ValueError(
            f"FlashComm2 only supports layer_sharding=['o_proj'], "
            f"got unsupported: {unsupported_layers}"
        )
```

---

## 11. 完整调用链路总览

```
┌────────────────────────────────────────────────────────┐
│ 配置启用（环境变量）                                     │
│  VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=N                │
│  VLLM_ASCEND_LAYER_SHARDING=[o_proj]                   │
└────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────┐
│ 模型初始化阶段                                           │
│  for each layer in model.layers:                       │
│    if "o_proj" in layer.prefix:                        │
│      Op = Flashcomm2OProjRowParallelOp(layer)          │
│        [linear_op.py:L289]                             │
│      Op.update_attrs()  [L422-427]                     │
│        ├─ super().update_attrs()                       │
│        └─ flashcomm2_oshard_manager.register_layer()   │
│          [flashcomm2_oshard_manager.py:L37-59]         │
│            ├─ is_hidden_layer(layer)  [layer_shard_linear.py:L273]
│            ├─ extract_layer_index(layer.prefix)        │
│            └─ register_layer_to_shard_weight_series()  │
│              [layer_shard_linear.py:L205-247]          │
│                ├─ create SeriesMetadata (if new)       │
│                ├─ append LayerMetadata                 │
│                ├─ disable quant_method.process_...()   │
│                ├─ dispose non-source rank weight       │
│                └─ wrap layer.forward()                 │
│                  [L162-168]                             │
└────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────┐
│ 权重加载完成后                                           │
│  AttentionV1.process_weights_after_loading()           │
│    → flashcomm2_oshard_manager.post_process_after_loading()
│      [flashcomm2_oshard_manager.py:L90-98]             │
│      → post_process_after_loading_for_shard_weight_series()
│        [layer_shard_linear.py:L250-252]                │
│        → series.post_process_after_loading()           │
│          [layer_shard_linear.py:L57-114]               │
│            ├─ sort layers by layer_idx                 │
│            ├─ for each layer:                          │
│            │  ├─ broadcast weight                      │
│            │  └─ call quant_method.process_...()       │
│            ├─ build shard_windows（环形缓冲）          │
│            │  size = prefetch_step + 1 = 2              │
│            └─ forward wrapper already attached          │
└────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────┐
│ 前向传播（迭代 per layer）                              │
│                                                        │
│ Layer k Attention:                                     │
│  Flashcomm2OshardQKVParallelOp[k].apply_impl()        │
│    [linear_op.py:L479-504]                            │
│    ├─ QKV forward + matmul                             │
│    └─ flashcomm2_oshard_manager.trigger_broadcast...()│
│      [flashcomm2_oshard_manager.py:L72-88]             │
│      → reach_layer_for_shard_weight_series()           │
│        [layer_shard_linear.py:L255-257]                │
│        → series.reach_layer(layer_idx)                 │
│          [layer_shard_linear.py:L116-136]              │
│          ├─ 计算 next_layer_idx                        │
│          ├─ 异步 broadcast next_layer weight          │
│          │  到 shard_window[next_window_offset]        │
│          └─ window_offset++ (轮转)                     │
│                                                        │
│ Layer k O-Proj:                                        │
│  Flashcomm2OProjRowParallelOp[k].apply_impl()        │
│    [linear_op.py:L315-420]                            │
│    ├─ DBO record hook  [L388]                         │
│    ├─ wrapped_forward() 自动调用：                     │
│    │  → series.wait_weight(layer_idx)                  │
│    │    [layer_shard_linear.py:L138-146]               │
│    │    └─ 等待 shard_window.work broadcast 完成       │
│    ├─ otp_maybe_quant_comm()  [L331-375]              │
│    │  ├─ ODP all-to-all                               │
│    │  └─ reshape output                                │
│    ├─ quant_method.apply()  [L400]                    │
│    │  （W8A8 延迟通信在此）                            │
│    ├─ OTP reduce-scatter  [L404]                      │
│    ├─ TP all-gather (if !FC1)  [L410]                 │
│    ├─ de-pad  [L412]                                   │
│    └─ DBO wait hook  [L415]                           │
└────────────────────────────────────────────────────────┘
```

---

## 12. 文件索引

| 文件 | 关键行号 | 内容 |
|------|----------|------|
| **FlashComm2 配置** | | |
| [vllm_ascend/envs.py](../../../../vllm_ascend/envs.py#L76) | 76-80 | 环境变量定义 |
| [vllm_ascend/ascend_config.py](../../../../vllm_ascend/ascend_config.py#L162) | 162-167, 210-212 | 配置读取 & 校验入口 |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1191) | 1191-1193 | `flashcomm2_enable()` |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1196) | 1196-1200 | `o_shard_enable()` ← **O-Shard 启用检查** |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1203) | 1203-1250 | `get_flashcomm2_config_and_validate()` |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1253) | 1253-1269 | `get_flashcomm2_reorgnized_batch_ids()` |
| **通信组** | | |
| [vllm_ascend/distributed/parallel_state.py](../../../../vllm_ascend/distributed/parallel_state.py#L18) | 18-19 | 全局变量 `_FLASHCOMM2_OTP` / `_FLASHCOMM2_ODP` |
| [vllm_ascend/distributed/parallel_state.py](../../../../vllm_ascend/distributed/parallel_state.py#L152) | 152-189 | ODP/OTP 组初始化 |
| [vllm_ascend/distributed/parallel_state.py](../../../../vllm_ascend/distributed/parallel_state.py#L258) | 258-264 | 访问器 `get_flashcomm2_otp/odp_group()` |
| **FlashComm2 OProj Op** | | |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L289) | 289-427 | `Flashcomm2OProjRowParallelOp` 完整类（包括 O-Shard 集成） |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L694) | 694-730 | `_get_row_parallel_op` 选择逻辑 |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L240) | 240-281 | `OProjRowParallelOp`（有 DBO hook 的参考） |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L479) | 479-504 | `Flashcomm2OshardQKVParallelOp` ← **prefetch 触发点** |
| **O-Shard 核心** | | |
| [vllm_ascend/ops/flashcomm2_oshard_manager.py](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L15) | 15-101 | `Flashcomm2OShardManager` — 顶层 wrapper |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L18) | 18-56 | `SeriesMetadata` 等数据结构 |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L57) | 57-114 | `post_process_after_loading()` — 权重初始化 |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L116) | 116-136 | `reach_layer()` — prefetch 触发 |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L138) | 138-146 | `wait_weight()` — 等待 weight 就绪 |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L162) | 162-168 | `_create_forward_wrapper()` — forward 包装 |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L205) | 205-247 | `register_layer_to_shard_weight_series()` |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L250) | 250-252 | `post_process_after_loading_for_shard_weight_series()` |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L255) | 255-257 | `reach_layer_for_shard_weight_series()` |
| [vllm_ascend/ops/layer_shard_linear.py](../../../../vllm_ascend/ops/layer_shard_linear.py#L273) | 273-276 | `is_hidden_layer()` |
| **Forward Context** | | |
| [vllm_ascend/ascend_forward_context.py](../../../../vllm_ascend/ascend_forward_context.py#L144) | 144-148 | `flashcomm_v2_enabled` + pad_size 设置 |
| [vllm_ascend/ascend_forward_context.py](../../../../vllm_ascend/ascend_forward_context.py#L481) | 481-540 | `_EXTRA_CTX` 代理定义 |
| **量化集成** | | |
| [vllm_ascend/quantization/methods/w8a8_static.py](../../../../vllm_ascend/quantization/methods/w8a8_static.py#L90) | 90-108 | W8A8 量化+通信融合 |
| [vllm_ascend/quantization/method_adapters.py](../../../../vllm_ascend/quantization/method_adapters.py#L151) | 151-155 | FlashComm2 tp_rank 获取 |
| **DBO 集成** | | |
| [vllm_ascend/dbo/compile_guard.py](../../../../vllm_ascend/dbo/compile_guard.py#L18) | 18-20 | `_dbo_call_linear_row_hook` wrapper |
| [vllm_ascend/dbo/overlap_templates/deepseek.py](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L7) | 7-37 | A2 Overlap Template |
