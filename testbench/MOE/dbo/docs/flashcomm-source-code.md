# FlashComm 源码解析：vllm-ascend 中的通信优化实现

这份文档是 [FlashComm算子库原理](flashcomm.md) 的源码级对照解析。

目标只有一个：

- 对照 FlashComm1 / FlashComm2 的理论设计，逐层说明在 vllm-ascend 代码中是如何落地的
- 说明关键函数和调用链路，让读者可以按图索骥阅读源码
- 说明 FlashComm 与 DBO 在代码层面如何协同

---

## TLDR

> FlashComm 的源码实现遵循一条清晰的四层架构：**配置层 → 启用判断层 → 自定义算子层 → 消费层**。理解这四层，就能掌握全部代码路径。

```
用户配置 (env / additional_config)
        │
        ▼
ascend_config.py          ← 配置层：读取 enable_flashcomm1 等
        │
        ▼
utils.py: enable_sp()     ← 启用判断层：综合 config + env 决定是否开启
        │
        ▼
ascend_forward_context.py ← 运行时状态层：设置 flash_comm_v1_enabled 等 flag
        │
        ▼
register_custom_ops.py    ← 自定义算子层：maybe_all_gather_and_maybe_unpad
        │                  maybe_pad_and_reduce
        ▼
linear_op.py / prepare_finalize.py  ← 消费层：根据 flag 走不同通信路径
```

---

## 1. 配置层：从用户输入到 AscendConfig

### 1.1 环境变量定义

`vllm_ascend/envs.py:75,80` 定义了两个核心环境变量：

```python
# envs.py:75
"VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),

# envs.py:80
"VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
```

### 1.2 配置读取

`vllm_ascend/ascend_config.py:83-88` 在 `AscendConfig.__init__` 中将环境变量与 `--additional-config` JSON 统一读取：

```python
# ascend_config.py:83-88
self.enable_flashcomm1 = self._get_config_value(
    additional_config, "enable_flashcomm1",
    "VLLM_ASCEND_ENABLE_FLASHCOMM1", ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM1,
)
```

`additional_config` 的优先级高于环境变量，因为 `AscendConfig` 的另一个初始化路径（`AscendConfig.update_from_json`）会首先从 JSON 文件中读取 `enable_flashcomm1` 等字段。

---

## 2. 启用判断层：enable_sp() 与 flashcomm2_enable()

### 2.1 FlashComm1 的启用判断

`vllm_ascend/utils.py:972-998` —— `enable_sp()` 函数：

```python
# utils.py:972-998
def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    # ...
    additional_config = getattr(vllm_config, "additional_config", None)
    if _ENABLE_SP is None or refresh:
        if additional_config is not None and "enable_flashcomm1" in additional_config:
            _ENABLE_SP = bool(additional_config["enable_flashcomm1"])  # additional_config 优先
        else:
            try:
                _ENABLE_SP = get_ascend_config().enable_flashcomm1       # 其次 AscendConfig
            except RuntimeError:
                _ENABLE_SP = envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1  # 兜底环境变量
    return bool(_ENABLE_SP)
```

注意命名：函数叫 `enable_sp()`（sequence parallelism），但内部读取的 key 是 `enable_flashcomm1`。在 vllm-ascend 中，**FlashComm1 = Sequence Parallelism**，二者是同义词。`sp` 指 token 维度的序列并行（区别于 vLLM 原生的 SP）。

### 2.2 FlashComm2 的启用判断

`vllm_ascend/utils.py:1287-1288`：

```python
# utils.py:1287-1288
def flashcomm2_enable() -> bool:
    config_val = get_ascend_config().enable_flashcomm2_parallel_size
    return config_val > 0
```

FlashComm2 需要 `enable_flashcomm2_parallel_size > 0` 才启用。这个值同时也是 O 矩阵的 TP group 大小。

---

## 3. 运行时状态层：ForwardContext 中的 flash_comm_v1_enabled

### 3.1 标志位的设置

`vllm_ascend/ascend_forward_context.py:112-140` 在 `set_ascend_forward_context()` 中决定每个 step 的通信策略：

```python
# ascend_forward_context.py:127-140
is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
if is_context_moe_model:
    flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
    mmrs_fusion = False                              # MoE 模型禁用 mmrs_fusion
elif is_draft_model:
    flash_comm_v1_enabled = False                    # drafter 模型禁用以避免兼容问题
else:
    # 稠密模型：高并发（>1000 token）才开启，经验阈值
    flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000

forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled
forward_context.flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None
```

几个关键判断逻辑：

- **MoE 模型**：`num_tokens is not None` 即开启，不要求 `> 1000`。因为 MoE 场景的通信融合收益与 token 数无关。   
    > 关键是MOE的模型结构天然适配 flashcomm：具有较强的模型结构属性：  
    > * 减少 TP rank 上重复的 RMSNorm、Router、Quant；  
    > * 减少两级 collective 的启动和同步（统一TP,DP,EP三层通信域）；
    > * 避免先恢复 TP replicated，再重新进入 EP 布局；
    > * MoE 输出直接回到 sequence-sharded 状态。
- **稠密模型**：`num_tokens > 1000` 才开启。低并发时 FlashComm1 的通信方式切换开销可能超过收益。
- **Drafter 模型**：直接关闭。FlashComm1 与 MTP 的 dp / graph 机制不兼容。

### 3.2 Padding 计算

```python
# ascend_forward_context.py:142-145
forward_context.pad_size = 0
if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
    pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
    forward_context.pad_size = pad_size
```

FlashComm1 要求在 token 维度做 ReduceScatter / AllGather，因此 token 数必须是 `tp_size` 的整数倍。不足时 pad。

### 3.3 DP 感知的 Padding

```python
# ascend_forward_context.py:174-182
dp_world_size = get_dp_group().world_size
if dp_world_size > 1 and forward_context.dp_metadata is not None:
    dp_meta = forward_context.dp_metadata
    max_tokens_across_dp = dp_meta.num_tokens_across_dp_cpu.max().item()
    if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
        padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
        pad_size = padded_length - num_tokens
        forward_context.padded_length = padded_length
        forward_context.pad_size = pad_size
```

DP > 1 时，各 DP rank 的 token 数可能不同，需要对齐到 `max_tokens_across_dp` 的 `tp_size` 整数倍，以保证 EP AllGather 时各 rank 的 tensor shape 一致。

### 3.4 _EXTRA_CTX：跨模块访问的代理

`vllm_ascend/ascend_forward_context.py:455-514` 定义了 `_ExtraForwardContextProxy` 单例 `_EXTRA_CTX`：

```python
# ascend_forward_context.py:514
_EXTRA_CTX = _ExtraForwardContextProxy()
```

所有的 consumer 代码（`linear_op.py`、`prepare_finalize.py`、`mla.py` 等）都通过 `_EXTRA_CTX.flash_comm_v1_enabled` 来读取当前 step 的通信策略。这个代理兼容 v1/v2 model runner。

---

## 4. 自定义算子层：注册 PyTorch Custom Op

这是 FlashComm 实现最核心的部分。理解这两个 custom op 就理解了 FlashComm1 的代码本质。

### 4.1 `maybe_all_gather_and_maybe_unpad` —— 替代 AllGather

`vllm_ascend/ops/register_custom_ops.py:40-77`：

```python
# register_custom_ops.py:40-77
def _maybe_all_gather_and_maybe_unpad_impl(
    x: torch.Tensor, label: bool, is_ep_comm: bool = False, do_comm: bool = True
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x                                          # 无 forward context → 直通

    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled or (
        enable_sp_by_pass() and is_ep_comm
    )
    if flash_comm_v1_enabled and label:
        dp_metadata = forward_context.dp_metadata
        if dp_metadata is None or not is_ep_comm:
            # 纯 TP 场景：沿 token 维做 AllGather
            if do_comm:
                x = tensor_model_parallel_all_gather(x, 0)
            pad_size = _EXTRA_CTX.pad_size
            if pad_size > 0:
                x = x[:-pad_size]                         # 去掉 padding token
        else:
            # TP + DP 场景：融合为 EP AllGather
            if do_comm:
                x = get_ep_group().all_gather(x, 0)       # 一次通信覆盖 TP×DP 全量
            # DP-aware unpad：各 DP rank 的原始 token 数不同
            num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
            result = torch.empty((num_tokens_across_dp_cpu.sum(), ...), ...)
            dp_size = get_dp_group().world_size
            x = x.view(dp_size, _EXTRA_CTX.padded_length, ...)
            offset = 0
            for idx in range(dp_size):
                num_tokens_dp = num_tokens_across_dp_cpu[idx]
                result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
                offset += num_tokens_dp
            x = result
    return x
```

该 op 在 `register_custom_ops.py:248-254` 注册为 `torch.ops.vllm.maybe_all_gather_and_maybe_unpad`：

```python
direct_register_custom_op(
    op_name="maybe_all_gather_and_maybe_unpad",
    op_func=_maybe_all_gather_and_maybe_unpad_impl,
    fake_impl=_maybe_all_gather_and_maybe_unpad_fake,   # 给 torch.compile 用的形状推导
    ...
)
```

**参数含义**：
- `label`：由调用方决定是否需要 AllGather（例如 QKV projection 需要的列切分场景）
- `is_ep_comm`：是否走 EP group（MoE 场景），否则走 TP group
- `do_comm`：DBO 场景下可以先 padding 再推迟通信（`do_comm=False`）

**核心逻辑**就是 FlashComm1 论文里的关键：
- 如果 `dp_metadata is None`（纯 TP，DP=1）→ 走 `tensor_model_parallel_all_gather`
- 如果 `dp_metadata is not None` 且 `is_ep_comm=True` → 走 `get_ep_group().all_gather`，**一次 EP 通信替代 TP AG + DP AG**

### 4.2 `maybe_pad_and_reduce` —— 替代 AllReduce

`vllm_ascend/ops/register_custom_ops.py:80-122`：

```python
# register_custom_ops.py:80-122
def _maybe_pad_and_reduce_impl(x: torch.Tensor, is_ep_comm: bool = False, do_comm: bool = True):
    try:
        forward_context = get_forward_context()
    except AssertionError:
        if do_comm:
            return tensor_model_parallel_all_reduce(x)   # 无 forward context → 回退 AllReduce
        return x

    flash_comm_v1_enabled = getattr(forward_context, "flash_comm_v1_enabled", False) or (
        enable_sp_by_pass() and is_ep_comm
    )
    if not flash_comm_v1_enabled or (...):
        if do_comm:
            return tensor_model_parallel_all_reduce(x)   # FlashComm1 未开启 → 走传统 AllReduce
        return x

    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None or not is_ep_comm:
        # 纯 TP 场景：pad → TP ReduceScatter
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        if do_comm:
            x = tensor_model_parallel_reduce_scatter(x, 0)
        return x
    else:
        # TP + DP 场景：DP-aware padding → EP ReduceScatter
        dp_size = get_dp_group().world_size
        padded_x = torch.empty((dp_size, _EXTRA_CTX.padded_length, ...), ...)
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            padded_x[idx, :num_tokens_dp] = x[offset : offset + num_tokens_dp]
            offset += num_tokens_dp
        if do_comm:
            res = get_ep_group().reduce_scatter(padded_x.view(-1, ...), 0)
        else:
            res = padded_x.view(-1, ...)
        return res
```

注册为 `torch.ops.vllm.maybe_pad_and_reduce`（`register_custom_ops.py:256-262`）。

**核心逻辑**：
- FlashComm1 未开启 → `tensor_model_parallel_all_reduce`（传统 AllReduce）
- FlashComm1 开启 + 纯 TP → `tensor_model_parallel_reduce_scatter`（token 维切分）
- FlashComm1 开启 + TP+DP → `get_ep_group().reduce_scatter`（EP 融合通信）
- `do_comm=False` → 只 pad 不做通信（给 DBO 用）

### 4.3 Fake Impl：torch.compile 的形状推导

`register_custom_ops.py:125-143`：

```python
# register_custom_ops.py:136-143
def _maybe_pad_and_reduce_fake(x, is_ep_comm=False, do_comm=True):
    if (_EXTRA_CTX.flash_comm_v1_enabled or enable_sp_by_pass()) and do_comm:
        # ReduceScatter 后第一维除以 tp_size
        return torch.empty(
            (x.shape[0] // get_tensor_model_parallel_world_size(), *x.shape[1:]),
            device=x.device, dtype=x.dtype
        )
    return x
```

torch.compile 需要在 trace 时就知道 tensor shape。Fake impl 不执行真实通信，仅推导输出形状。FlashComm1 开启时，`maybe_pad_and_reduce` 的输出第一维会缩小 `tp_size` 倍（ReduceScatter 沿 token 维切分）。

---

## 5. 消费层：各模块如何调用 Custom Op

### 5.1 稠密模型的 Column Parallel Linear（QKV / Gate-Up）

`vllm_ascend/ops/linear_op.py:439-474` —— `SequenceColumnParallelOp.apply_impl`：

```python
# linear_op.py:439-464
class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        need_all_gather = not (extract_layer_index(self.layer.prefix) == 0
                               and is_vl_model() and "attn" in self.prefix)

        forward_context = get_forward_context()
        if forward_context.dbo_enabled:
            # DBO 路径：先做通信，再走 maybe_all_gather_and_maybe_unpad with do_comm=False
            forward_context.dbo_template.dbo_linear_column_hook(is_record=True)
            if get_forward_context().flash_comm_v1_enabled and need_all_gather:
                input_ = tensor_model_parallel_all_gather(input_, 0)
            forward_context.dbo_template.dbo_linear_column_hook(is_record=False)
            input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, do_comm=False, ...)
        else:
            # 非 DBO 路径：直接调用 custom op
            input_ = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(input_, label=need_all_gather)

        output_parallel = self.quant_method.apply(self.layer, input_, bias)
        ...
```

FlashComm1 开启时，此处 `input_` 是 sequence-sharded（只有 `T/tp_size` 个 token），需要 AllGather 恢复到全量 token 才能做 QKV/Gate-Up 的列并行计算。

### 5.2 稠密模型的 Row Parallel Linear（O-Proj / Down-Proj）

`vllm_ascend/ops/linear_op.py:505-612` —— `SequenceRowParallelOp`：

```python
# linear_op.py:529-543
def matmul_and_reduce(self, input_parallel, bias_):
    flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
    mmrs_fusion = _EXTRA_CTX.mmrs_fusion

    if not flash_comm_v1_enabled:
        # 传统路径：matmul → AllReduce
        output_parallel = self.layer.quant_method.apply(self.layer, x, bias=bias_)
        return tensor_model_parallel_all_reduce(output_parallel)

    # FlashComm1 路径：pad → matmul → ReduceScatter
    pad_size = _EXTRA_CTX.pad_size
    if pad_size > 0 and not dsa_cp_attn_out:
        x = F.pad(x, (0, 0, 0, pad_size))
    # ...
```

FlashComm1 开启时，Row Parallel 的输出不再做 AllReduce，而是做 ReduceScatter。结果从 `[T, H]` 变为 `[T/tp_size, H]`——每个 rank 只保留部分 token 的完整 hidden vector。

### 5.3 MoE Prepare 阶段（AllGather 替换）

`vllm_ascend/ops/fused_moe/prepare_finalize.py:390-414` —— `PrepareAndFinalize._prepare_with_tp`：

```python
# prepare_finalize.py:390-414
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)
    if get_forward_context().flash_comm_v1_enabled:
        if get_forward_context().dp_metadata is None:
            # 纯 TP 场景：TP AllGather
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            router_logits = tensor_model_parallel_all_gather(router_logits, 0)
        else:
            # TP + DP 场景：融合为 EP AllGather
            hidden_states = get_ep_group().all_gather(hidden_states, 0)
            router_logits = get_ep_group().all_gather(router_logits, 0)
        forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False)
    # 再走 custom op（此时 do_comm=False，因为 DBO 已完成通信）
    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
        hidden_states, True, True, False
    )
else:
    # 非 DBO：custom op 内部完成通信
    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
        hidden_states, True, True
    )
```

对照 [FlashComm算子库原理](flashcomm.md) 的 2.2 节：

> FlashCommV1 将 Attention 后的 TP AllReduce 改为 TP ReduceScatter

Attention 的 O-Proj 后已经是 token-sharded 状态，进入 MoE 前需要恢复到全量 token。此时：

- **DP=1 时**：走 `tensor_model_parallel_all_gather`（TP AG）——等同于 flashcomm.md 里的 `TP AllGather`
- **DP>1 时**：走 `get_ep_group().all_gather`（EP AG）——等同于 flashcomm.md 里的 `EP AllGather = TP AG + DP AG` 融合

### 5.4 MoE Finalize 阶段（ReduceScatter 替换）

`vllm_ascend/ops/fused_moe/prepare_finalize.py:515-543` —— `_finalize_with_ep_group`：

```python
# prepare_finalize.py:515-543
def _finalize_with_ep_group(self, hidden_states):
    forward_context = get_forward_context()
    if forward_context.dbo_enabled:
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True, do_comm=False)
        forward_context.dbo_template.dbo_moe_finalize_hook(is_record=True)
        if get_forward_context().flash_comm_v1_enabled:
            if get_forward_context().dp_metadata is None:
                hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
            else:
                hidden_states = get_ep_group().reduce_scatter(hidden_states, 0)
        forward_context.dbo_template.dbo_moe_finalize_hook(is_record=False)
    else:
        hidden_states = torch.ops.vllm.maybe_pad_and_reduce(hidden_states, True)
    return hidden_states
```

对照 flashcomm.md 的 3.3 节 —— **DP RS + TP RS → EP RS**。实际代码中：

- DP=1 → `tensor_model_parallel_reduce_scatter`
- DP>1 → `get_ep_group().reduce_scatter`

完全对应理论设计。

### 5.5 `maybe_all_reduce_tensor_model_parallel` —— 跳过冗余 AllReduce

`vllm_ascend/ops/register_custom_ops.py:168-176`：

```python
# register_custom_ops.py:168-176
def _maybe_all_reduce_tensor_model_parallel_impl(final_hidden_states):
    moe_comm_type = _EXTRA_CTX.moe_comm_type
    if (
        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
        or _EXTRA_CTX.flash_comm_v1_enabled
    ):
        return final_hidden_states   # FlashComm1 或 A2A 模式下跳过 AllReduce
    else:
        return tensor_model_parallel_all_reduce(final_hidden_states)
```

FlashComm1 开启时，MoE 输出已在 finalize 阶段通过 `_finalize_with_ep_group` 完成了 ReduceScatter，不需要再在 FusedMoE 层做 AllReduce。

---

## 6. FlashComm2 的实现

### 6.1 O-Shard Manager

`vllm_ascend/ops/flashcomm2_oshard_manager.py:1-101`：

```python
class Flashcomm2OShardManager:
    def flashcomm2_oshard_enable(self):
        return flashcomm2_enable() and o_shard_enable()  # FlashComm2 + o_proj layer_sharding 同时开

    def register_layer(self, layer, prefetch_step=1):
        # 注册 o_proj 层，管理 weight broadcast

    def trigger_broadcast_for_layer(self, layer_prefix):
        # 在 forward 中触发异步 weight broadcast

flashcomm2_oshard_manager = Flashcomm2OShardManager()
```

### 6.2 FlashComm2 O-Proj Row Parallel

`vllm_ascend/ops/linear_op.py:280-398` —— `Flashcomm2OProjRowParallelOp`：

FlashComm2 的核心思路是对 O 矩阵再做一层 TP 切分，通过 all-to-all 通信在 row parallel 的输入上重新分布数据：

```python
# linear_op.py:306-313
def apply_impl(self, input_):
    input_parallel = self.get_input_parallel(input_)
    # padding for all-to-all
    num_padding_tokens = _EXTRA_CTX.pad_size
    if num_padding_tokens > 0:
        input_parallel = nn.functional.pad(input_parallel, (0, 0, 0, num_padding_tokens))

    # all-to-all 在 ODP group 上重分布
    dist.all_to_all_single(recv_buf, send_buf, group=self.odp_group.device_group)
    # ...
    # reduce_scatter 在 OTP group 上汇总
    output = self.comm_group.reduce_scatter(output_parallel, dim=0)

    if not _EXTRA_CTX.flash_comm_v1_enabled:
        # 没开 FC1 时需要 AllGather 恢复全量 token
        output = get_tp_group().all_gather(output, 0)
```

FlashComm2 与 FlashComm1 是正交的优化：
- FlashComm1 优化 token 维度的通信（AllReduce → ReduceScatter + AllGather）
- FlashComm2 对 O 矩阵再做一层切分（OTP + ODP），通过 all-to-all 进一步减少单次通信量

当两者同时开启时（`flash_comm_v1_enabled=True`），O-Proj 输出的 `all_gather` 被跳过（`linear_op.py:380`），因为 FlashComm1 已在上游完成了 token 维度的 ReduceScatter。

---

## 7. 与 DBO 的协同

FlashComm 与 DBO（Dual-Batch Overlap）的协同体现在 custom op 的 `do_comm` 参数上。

### 7.1 协同模式

```text
非 DBO 路径：
  custom_op (do_comm=True) → 在 custom op 内部完成通信

DBO 路径：
  1. custom_op (do_comm=False) → 只做 padding，不通信
  2. dbo_template.dbo_xxx_hook(is_record=True)  → 标记通信区间
  3. 显式调用通信原语 (all_gather / reduce_scatter) → 实际通信
  4. dbo_template.dbo_xxx_hook(is_record=False) → 结束记录
```

### 7.2 MoE Prepare 的 DBO 协同

```python
# prepare_finalize.py:391-406
if forward_context.dbo_enabled:
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=True)  # 开始记录
    if get_forward_context().flash_comm_v1_enabled:
        if get_forward_context().dp_metadata is None:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
        else:
            hidden_states = get_ep_group().all_gather(hidden_states, 0)
    forward_context.dbo_template.dbo_moe_prepare_hook(is_record=False) # 结束记录
    # 此时 do_comm=False，custom op 不再做通信
    hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
        hidden_states, True, True, False
    )
```

DBO 模板通过 hook 记录通信区间，用于生成 overlap 调度——将计算放在一个 batch 上，通信放在另一个 batch 上，实现计算-通信 overlap。

---

## 8. 完整调用链路总结

以 MoE 模型（如 DeepSeek-V3）的一次 forward 为例，FlashComm1 开启时的完整链路：

```text
Attention O-Proj (SequenceRowParallelOp)
        │  matmul_and_reduce()
        │  → flash_comm_v1_enabled=True → ReduceScatter (token 维切分)
        │  输出: [T/tp_size, H]  每个 rank 只有部分 token
        │
        ▼
RMSNorm / Router / Quant (per-token, 本地执行)
        │  此时各 rank 持有互不重复的 token shard
        │
        ▼
MoE Prepare (_prepare_with_tp)
        │  → get_ep_group().all_gather()  一次 EP 通信替代 TP AG + DP AG
        │  输出: [T * DP, H]  全量 token
        │
        ▼
Local Expert 计算
        │
        ▼
MoE Finalize (_finalize_with_ep_group)
        │  → get_ep_group().reduce_scatter()  EP 通信替代 DP RS + TP RS
        │  输出: [T/tp_size, H]  回到 token shard
        │
        ▼
Next Layer ...
```

对比传统路径：

```text
传统路径:
  O-Proj → TP AllReduce → DP AllGather → MoE → DP ReduceScatter → TP AllReduce
  (5 次通信)

FlashComm1 路径:
  O-Proj → TP ReduceScatter → EP AllGather → MoE → EP ReduceScatter
  (3 次通信，且 per-token 计算在本地执行，无冗余)
```

---

## 9. 关键文件索引

| 文件 | 作用 | 关键行号 |
|---|---|---|
| `vllm_ascend/envs.py` | 环境变量定义 | :75, :80 |
| `vllm_ascend/ascend_config.py` | 配置读取 | :83-88 |
| `vllm_ascend/utils.py` | `enable_sp()` / `flashcomm2_enable()` | :972-998, :1287-1288 |
| `vllm_ascend/ascend_forward_context.py` | 运行时 flag 设置 / pad_size / `_EXTRA_CTX` | :127-140, :174-182, :455-514 |
| `vllm_ascend/ops/register_custom_ops.py` | `maybe_all_gather_and_maybe_unpad` / `maybe_pad_and_reduce` | :40-122, :125-143, :168-176 |
| `vllm_ascend/ops/linear_op.py` | `SequenceColumnParallelOp` / `SequenceRowParallelOp` / `Flashcomm2OProjRowParallelOp` | :439-474, :505-612, :280-398 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | MoE 的 AG / RS 替换 | :390-414, :515-543 |
| `vllm_ascend/ops/flashcomm2_oshard_manager.py` | FlashComm2 O-Shard 管理 | :1-101 |
| `vllm_ascend/worker/model_runner_v1.py` | 模型输出后 all_gather 恢复全量 | :2588-2593 |
