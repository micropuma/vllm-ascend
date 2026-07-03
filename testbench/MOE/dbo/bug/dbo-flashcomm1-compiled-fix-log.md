# DBO + FlashComm1 compiled 修复记录

## 约束

- 模型：DeepSeek-V2-Lite-Chat
- 硬件：2 x Ascend 910B3
- 并行：TP=2、EP=2、DeepSeek A2 AllGather/ReduceScatter
- 配置：FlashComm1=1、FlashComm2=0、DBO=1、AI_CPU
- 每一步只改变一个可验证因素；失败时记录最早异常，不用 timeout 推断根因。

## Step 1：移除 runtime eager fallback

### 背景

此前 `model_runner_v1.py` 在 DBO + sequence parallel runtime 强制：

```python
use_dbo_runtime_eager_fallback = enable_dbo and enable_sp(...)
skip_compiled = has_encoder_input or use_dbo_runtime_eager_fallback
```

因此服务虽然完成 torch.compile 和 ACL graph capture，实际 DBO 请求仍走 eager，
无法验证 compiled shape contract。

### 修改

删除 `use_dbo_runtime_eager_fallback`，恢复原有 encoder-only 条件：

```python
skip_compiled = has_encoder_input
```

本步骤不修改 ubatch slicing、padding、custom op、hook 或异常处理。

### 验证方法

1. 启动 FlashComm1 + DBO compiled server。
2. 确认 torch.compile、KV cache 和 ACL graph capture完成，API 返回 200。
3. 运行 4096 输入、1 输出、1 请求、并发 1；实际输入应为 4103 tokens。
4. 检查两个 rank 的第一条 Python/device 异常和 server/test 退出状态。

### 结果

服务启动成功：

- `python -m py_compile`：退出码 0；
- `git diff --check`：退出码 0；
- torch.compile：7.72 s；
- PIECEWISE/FULL ACL graph capture 完成；
- `/v1/models`：HTTP 200。

首个 4103-token warmup 请求进入 compiled runtime 后失败。两个 TP rank、两个
ubatch 均上报：

```text
RuntimeError: The size of tensor a (4096) must match the size of tensor b (1026)
at non-singleton dimension 0
```

调用链：

```text
AscendUBatchWrapper._run_ubatches
→ AOT compiled DeepseekV2 forward
→ piecewise backend
→ AscendCompiler / NPUGraph runnable
→ generated FX forward line 34
→ torch custom op
```

`_check_thread_exceptions` 成功把两个 sub-thread 的异常传回主线程，随后
EngineCore 退出。正式 benchmark 因服务已退出而得到 0 成功、1 失败。

日志：

```text
/data/workspace/logs/deepseek-v2-dbo-fc1-compiled-step1-server.log
/data/workspace/logs/deepseek-v2-dbo-fc1-compiled-step1-test.log
```

### 结论

1. eager workaround 已成功关闭，请求确实进入 AOT/piecewise compiled runtime。
2. 当前第一故障已从旧的 down-proj odd-row RS 断言前移为
   `4096 vs 1026` 的 compiled graph shape mismatch。
3. 这不是 collective 顺序问题，也没有发生静默死锁；下一步应定位 generated
   FX forward line 34 对应的 custom op。
4. 在定位该 op 前不修改 padding 或 collective contract，避免同时改变多个因素。

## Step 2：映射第一处 shape mismatch

### 只读生成图证据

失败 cache：

```text
/root/.cache/vllm/torch_compile_cache/db6e52d5c5/
```

`artifact_compile_range_1_16384_subgraph_1` 的关键序列：

```python
matmul_and_reduce = torch.ops.vllm.matmul_and_reduce(
    npu_swiglu, "model.layers.0.mlp.down_proj"
)
maybe_chunk_residual_1 = torch.ops.vllm.maybe_chunk_residual(
    matmul_and_reduce, getitem_2
)
```

异常中的 generated FX `line 34` 正是 `maybe_chunk_residual`。两侧 shape 为：

```text
matmul_and_reduce: [4096, 2048]
residual getitem_2: [1026, 2048]
```

生成图还显示 layer0 MLP 前的 `maybe_unpad_after_all_gather` 被固化为：

```python
maybe_unpad_after_all_gather(..., 8192)
```

而 layer0 down-proj 后的 `npu_add_rms_norm_bias` 输出被固化为：

```text
[4096, 2048]
```

### Step 2 结论

第一故障不是 `maybe_chunk_residual` 自身算法错误。它只是第一个合流点：

- residual 分支保留当前 ubatch 的 1026 rows；
- MLP/FlashComm1 compiled 分支仍按 capture 常量产生 4096 rows。

下一修复目标应是消除 subgraph 1 中 `num_tokens=8192` 的 Python 常量化，使
`maybe_unpad_after_all_gather` 和 `matmul_and_reduce` 的 row contract 从当前
ubatch symbolic input 推导。不能在 `maybe_chunk_residual` 处裁剪，否则只会
隐藏上游错误。

### Step 2 最小修改

`SequenceColumnParallelOp` 的 DBO 分支已经按 `ubatch_slices_padded` 输入：

```text
local collective rows
→ TP AllGather
→ global collective rows
→ GEMM
→ down-proj ReduceScatter
→ local collective rows
```

因此 AllGather 后不应再按 Python `forward_context.num_tokens` 裁剪。删除：

```python
input_ = torch.ops.vllm.maybe_unpad_after_all_gather(
    input_, forward_context.num_tokens
)
```

非 DBO 分支、MLA preprocess 和 MoE 路径不变。本步骤预期使 layer0 dense MLP
的 row shape 从输入 symbolic shape 传播，不再在生成图中出现
`maybe_unpad_after_all_gather(..., 8192)`。

### Step 2 验证

使用 `cudagraph_capture_sizes=[16]` 缩短与目标无关的 decode FULL graph
启动时间；PIECEWISE compiled runtime 配置不变。

单请求结果：

```text
input tokens: 4103
successful: 1
failed: 0
TTFT: 183.83 ms
```

新 subgraph 1 中已经不存在 `maybe_unpad_after_all_gather(..., 8192)`，并且
MLP output buffer 的 token 维改为 graph input，不再是固定 4096。

## Step 3：per-ubatch collective padding

### 并发暴露的新边界

32 请求、并发 16 的测试中，warmup 2/2 通过；正式测试在一个
`total_num_scheduled_tokens=12318` 的 batch 失败：

```text
DBO split: 6159 + 6159
TP world size: 2
6159 % 2 != 0
```

第一异常：

```text
matmul_and_reduce(layer0.down_proj)
→ tensor_model_parallel_reduce_scatter
→ assert input_tensor.shape[0] % world_size == 0
```

这说明“总 batch 对齐 TP”不足以保证“每个 DBO ubatch 对齐 TP”。

### Step 3 修改

`matmul_and_reduce` 不再使用 Python forward-context 中的
`_EXTRA_CTX.pad_size`，而是从实际 collective input shape 计算：

```python
world_size = self.layer.tp_size
pad_size = (-x.shape[0]) % world_size
```

这与 fake impl 的 ceil division 保持一致：

```python
(num_tokens + tp_size - 1) // tp_size
```

该 padding 属于 ReduceScatter 自身的 collective contract，不依赖 attention
logical token bookkeeping。

### Step 3 验证

服务使用 fresh cache `c4aa6471e3` 完成编译，随后运行：

```text
INPUT_LEN=4096
OUTPUT_LEN=1
NUM_PROMPTS=32
MAX_CONCURRENCY=16
```

结果：

```text
warmup: 2 successful, 0 failed
formal: 32 successful, 0 failed
total input tokens: 131321
total generated tokens: 32
```

相同 workload 在 Step 2 为 17 successful / 15 failed，并触发 odd-row
ReduceScatter assertion。Step 3 后 server 日志中没有 AssertionError、
EngineDeadError 或 device-side error。

日志：

```text
/data/workspace/logs/deepseek-v2-dbo-fc1-compiled-step3-server.log
/data/workspace/logs/deepseek-v2-dbo-fc1-compiled-step3-multi32-test.log
```

### 单元测试

执行：

```text
pytest -q tests/ut/test_ascend_forward_context.py \
  tests/ut/ops/test_register_custom_ops.py
```

结果为 11 passed / 3 failed。通过项包含 odd/even ReduceScatter ceil-division
contract。三个失败均来自已有 MLA 测试错误地 patch
`_ExtraForwardContextProxy._ctx`，在进入被测函数前即抛 AttributeError，与本次
row-linear 修改无关。该测试基础设施问题需要单独修复，不能记录为本次功能通过。
