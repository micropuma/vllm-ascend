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

### 7.1 Flashcomm2OShardManager

> [`vllm_ascend/ops/flashcomm2_oshard_manager.py:15-101`](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L15)

当同时启用 `flashcomm2_enable()` 和 `o_shard_enable()`（即 `layer_sharding` 包含 `"o_proj"`）时，O-Proj weight 在 rank 间分片，通过 shard weight group 做异步 broadcast 按需加载。

```python
class Flashcomm2OShardManager:
    # [L34-35](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L34)
    def flashcomm2_oshard_enable(self):
        return flashcomm2_enable() and o_shard_enable()

    # [L37-59](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L37)
    def register_layer(self, layer, prefetch_step=1):
        # 由 Flashcomm2OProjRowParallelOp.update_attrs() 调用 — [L398-399](../../../../vllm_ascend/ops/linear_op.py#L398)
        ...

    # [L72-88](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L72)
    def trigger_broadcast_for_layer(self, layer_prefix):
        # 由 Flashcomm2OshardQKVParallelOp 在 forward 时调用
        ...

    # [L90-98](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L90)
    def post_process_after_loading(self):
        # 由 AttentionV1.process_weights_after_loading() 调用
        ...
```

### 7.2 O-Shard 完整调用链

```
model init:
  Flashcomm2OProjRowParallelOp.update_attrs()       — [L394-399](../../../../vllm_ascend/ops/linear_op.py#L394)
    → flashcomm2_oshard_manager.register_layer()    — [L37](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L37)
      → register_layer_to_shard_weight_series()

forward pass:
  Flashcomm2OshardQKVParallelOp.apply_impl()        — [L479-504](../../../../vllm_ascend/ops/linear_op.py#L479)
    → flashcomm2_oshard_manager.trigger_broadcast_for_layer()  — [L72](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L72)
      → reach_layer_for_shard_weight_series()

post load:
  AttentionV1.process_weights_after_loading()       — attention_v1.py
    → flashcomm2_oshard_manager.post_process_after_loading()   — [L90](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L90)
```

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

## 9. 调用链路总览

```
模型 forward
  └─ _get_row_parallel_op("o_proj", layer)           [linear_op.py:L694]
       ├─ oproj_tp_enable() → OProjRowParallelOp      (有 DBO hook)
       └─ flashcomm2_enable() → Flashcomm2OProjRowParallelOp  (无 DBO hook)
            └─ apply_impl(input_)
                 ├─ get_input_parallel(input_)          [linear_op.py:L324]
                 ├─ pad input                           [linear_op.py:L327]
                 ├─ otp_maybe_quant_comm(x):             [linear_op.py:L331]
                 │    ├─ chunk + reorder (group_indices)  [linear_op.py:L341]
                 │    ├─ all_to_all_single (ODP group)   [linear_op.py:L355]
                 │    └─ reshape output
                 ├─ W8A8 check: inject comm_fn or call  [linear_op.py:L359]
                 ├─ quant_method.apply()                  [linear_op.py:L375]
                 │    └─ (W8A8) → comm_fn inside quant   [w8a8_static.py:L90]
                 ├─ reduce_scatter (OTP group)           [linear_op.py:L379]
                 ├─ all_gather (TP group, if !FC1)       [linear_op.py:L383]
                 └─ de-pad                                [linear_op.py:L387]
```

## 10. 文件索引

| 文件 | 关键行号 | 内容 |
|------|----------|------|
| [vllm_ascend/envs.py](../../../../vllm_ascend/envs.py#L76) | 76-80 | 环境变量定义 |
| [vllm_ascend/ascend_config.py](../../../../vllm_ascend/ascend_config.py#L162) | 162-167, 210-212 | 配置读取 & 校验入口 |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1191) | 1191-1193 | `flashcomm2_enable()` |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1203) | 1203-1250 | `get_flashcomm2_config_and_validate()` |
| [vllm_ascend/utils.py](../../../../vllm_ascend/utils.py#L1253) | 1253-1269 | `get_flashcomm2_reorgnized_batch_ids()` |
| [vllm_ascend/distributed/parallel_state.py](../../../../vllm_ascend/distributed/parallel_state.py#L18) | 18-19 | 全局变量 `_FLASHCOMM2_OTP` / `_FLASHCOMM2_ODP` |
| [vllm_ascend/distributed/parallel_state.py](../../../../vllm_ascend/distributed/parallel_state.py#L152) | 152-189 | ODP/OTP 组初始化 |
| [vllm_ascend/distributed/parallel_state.py](../../../../vllm_ascend/distributed/parallel_state.py#L258) | 258-264 | 访问器 `get_flashcomm2_otp/odp_group()` |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L289) | 289-399 | `Flashcomm2OProjRowParallelOp` 完整类 |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L694) | 694-730 | `_get_row_parallel_op` 选择逻辑 |
| [vllm_ascend/ops/linear_op.py](../../../../vllm_ascend/ops/linear_op.py#L240) | 240-281 | `OProjRowParallelOp`（有 DBO hook 的参考） |
| [vllm_ascend/ascend_forward_context.py](../../../../vllm_ascend/ascend_forward_context.py#L144) | 144-148 | `flashcomm_v2_enabled` + pad_size 设置 |
| [vllm_ascend/ascend_forward_context.py](../../../../vllm_ascend/ascend_forward_context.py#L481) | 481-540 | `_EXTRA_CTX` 代理定义 |
| [vllm_ascend/ascend_forward_context.py](../../../../vllm_ascend/ascend_forward_context.py#L280) | 280-291 | ubatch context 中的 pad_size |
| [vllm_ascend/ops/flashcomm2_oshard_manager.py](../../../../vllm_ascend/ops/flashcomm2_oshard_manager.py#L15) | 15-101 | Flashcomm2OShardManager |
| [vllm_ascend/quantization/methods/w8a8_static.py](../../../../vllm_ascend/quantization/methods/w8a8_static.py#L90) | 90-108 | W8A8 量化+通信融合 |
| [vllm_ascend/quantization/method_adapters.py](../../../../vllm_ascend/quantization/method_adapters.py#L151) | 151-155 | FlashComm2 tp_rank 获取 |
| [vllm_ascend/dbo/overlap_templates/deepseek.py](../../../../vllm_ascend/dbo/overlap_templates/deepseek.py#L7) | 7-37 | A2 Overlap Template |
| [vllm_ascend/dbo/compile_guard.py](../../../../vllm_ascend/dbo/compile_guard.py#L18) | 18-20 | `_dbo_call_linear_row_hook` wrapper |
