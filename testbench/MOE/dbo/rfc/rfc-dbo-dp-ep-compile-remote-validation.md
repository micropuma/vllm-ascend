# RFC: DBO + DP/EP + torch.compile 的全局 ubatch 契约

## 状态

- 日期：2026-06-28
- 模型：DeepSeek-V2-Lite-Chat
- 环境：2 × Ascend 910B3
- 配置：TP=1、DP=2、EP=2、DBO、torch.compile/ACL graph
- 状态：根因已定位，修复后 eager 和 compile 实际请求均通过

## 摘要

DP+DBO 的请求挂起不是 attention shape 问题，也不是单纯的 FlashComm 问题。根因是各 DP rank 独立决定是否启用 DBO：有真实 4K prefill 的 rank 执行两个 ubatch，空闲 rank 通过 dummy batch 参与 EP collective，却执行单 batch，导致跨 rank collective 序列不一致并永久等待。

compile/ACL graph 还暴露第二层错误：空闲 rank 的 1 个逻辑 dummy token 被 graph padding 到 4104 后，旧代码用 padded token 数判断 DBO，错误启用 ubatching，随后从 1-token 逻辑请求构造出空 ubatch，编译线程报 `Shape: 0 out of considered ranges`。线程异常没有及时传播，外部仍表现为 HTTP 200 后流不结束。

修复原则是：

```text
local eligibility = check(logical_num_tokens)
global should_ubatch = MIN(local eligibility across DP ranks)
execution shape = DP/graph padded shape
```

DBO eligibility、执行 shape、collective 序列必须分开管理。

## 1. vLLM 需要做的兼容

### 1.1 问题

vLLM-Ascend 为满足上游 DBO backend 校验，会在 A2 上把 `all2all_backend` 标记为 `deepep_low_latency`。实际 A2 路径仍使用 Ascend 的 AllGather/ReduceScatter，这个 backend 值只是兼容上游校验。

上游 `all2all_utils.py` 只在 CUDA-like 平台导入：

```python
DeepEPLLPrepareAndFinalize
```

但 `maybe_roundup_layer_hidden_size()` 仅检查 `use_deepep_ll_kernels`，没有检查平台。NPU 上 backend 字段被兼容性设置为 DeepEP-LL 后，模型加载阶段执行未导入的类：

```text
NameError: name 'DeepEPLLPrepareAndFinalize' is not defined
```

### 1.2 修改

文件：`vllm/model_executor/layers/fused_moe/all2all_utils.py`

```python
if (
    current_platform.is_cuda_alike()
    and moe_parallel_config.use_deepep_ll_kernels
):
    hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
        hidden_size
    )
```

导入条件和使用条件必须一致。这个修改只禁止 NPU 调用 CUDA DeepEP 专属 hidden-size roundup，不改变 CUDA DeepEP，也不改变 Ascend 实际通信。

### 1.3 长期建议

上游不应让 backend 字符串同时承担“通过 DBO 校验”和“选择 CUDA kernel”两种语义。更稳妥的方案是：

- DBO 校验允许 out-of-tree 平台声明自己的通信实现；
- kernel capability 由 platform capability 判断；
- DeepEP 类的导入和所有引用使用同一 capability guard；
- 增加 NPU/out-of-tree + DBO + EP 的模型初始化测试。

## 2. DP+DBO 到底出了什么 bug

### 2.1 单请求下为什么需要 idle rank

DP=2 时请求通常只被一个 DP replica 接收，但 EP group 跨两个 DP rank。active rank 执行 MoE EP collective 时，idle rank 必须通过 `execute_dummy_batch()` 执行匹配 collective，否则 active rank 永远等待。

### 2.2 eager 错误链

```text
DP0 active: logical tokens=4103 -> should_ubatch=True -> 两个 ubatch
DP1 idle:   logical tokens=1    -> should_ubatch=False -> 一个 dummy batch
```

旧 Ascend `_determine_batch_execution_and_padding()` 在每个 rank 本地调用 `check_enable_ubatch()`，没有同步最终布尔值。因此两个 rank 进入不同 forward 结构：

```text
DP0: ubatch0 EP AG/RS -> ubatch1 EP AG/RS
DP1: one-batch EP AG/RS
```

collective 数量、顺序和 shape 不再一致，形成死锁。

### 2.3 compile/graph 错误链

ACL graph 会统一执行 shape：

```text
DP0 active: logical=4103, padded=4104
DP1 idle:   logical=1,    padded=4104
```

旧代码错误地使用 `num_tokens_padded` 判断 eligibility，所以 DP1 把 4104 当成真实 workload，得到 `should_ubatch=True`。但 dummy request 元数据仍只有 1 token，按 2052 切分时产生空 request slice。ubatch 线程进入编译图后报：

```text
AssertionError: Shape: 0 out of considered ranges: [(1, 16384)]
```

随后另一 rank 阻塞在 EP collective。HTTP 层已经发送 200 响应头，因此客户端看到的是流永久不结束，而不是明确 500。

### 2.4 为什么之前只看日志容易误判

- 没有 AICPU/device error；
- NPU AICore 利用率降为 0；
- EngineCore 只报告 60 秒无 shared-memory broadcast block；
- ubatch 子线程异常没有立即终止主请求；
- HTTP 200 只代表响应流建立，不代表 token 生成完成。

## 3. 如何修改

### 3.1 eligibility 使用逻辑 token

文件：`vllm_ascend/worker/model_runner_v1.py`

旧逻辑把 padded token 同时作为 eligibility 和 execution shape：

```python
should_ubatch = check_enable_ubatch(
    num_tokens_padded,
    num_tokens_padded,
    ...
)
```

修改为：

```python
should_ubatch = check_enable_ubatch(
    num_tokens,
    num_tokens_padded,
    ...
)
```

第一个参数是本 rank 逻辑 token 数；第二个参数只用于检查 padding 后是否产生空 ubatch。

### 3.2 在 DP group 上统一 should_ubatch

只有所有 DP rank 都满足条件时才启用 DBO：

```python
ubatch_flag = torch.tensor(
    [int(should_ubatch)], device=device, dtype=torch.int32
)
dist.all_reduce(ubatch_flag, op=dist.ReduceOp.MIN, group=group)
should_ubatch = bool(ubatch_flag.item())
```

MIN 的语义是逻辑 AND。单请求时 idle rank 投 0，所有 rank 统一关闭 DBO；两个 rank 都有大 prefill 时才统一开启。

CPU/NPU group 的选择复用现有 `dp_allreduce_on_npu` 配置，避免引入新的通信域。

### 3.3 dummy run 保留同步结果

旧 dummy 路径明确丢弃 DBO：

```python
# vllm-ascend does not support ubatch now
ubatch_slices, ubatch_slices_padded = None, None
```

修改后保留 `_determine_batch_execution_and_padding()` 返回的 `should_ubatch`，调用 `maybe_create_ubatch_slices()`，并把 slices 传入 `set_ascend_forward_context()`。这样当所有 rank 确实统一启用 DBO 时，dummy/normal forward 的 ubatch 数量和 collective 序列仍保持一致。

### 3.4 后续优化

当前修复增加一次独立 DP MIN all-reduce。正确性优先，后续应把 `should_ubatch` 合并进 `_sync_metadata_across_dp()` 已有 packed tensor，避免额外同步开销。

还应让 ubatch 子线程异常立即传播到主线程，避免再次表现为无诊断信息的请求挂起。

## 4. 验证结果

### eager

```text
单请求: 1/1 success, input=4103, output=16, 2.69 s
双并发: 2/2 success, input=8206, output=32, 2.75 s
exit code: 0
```

### torch.compile + ACL graph

```text
torch.compile cache hit: 2.14 s / 2.38 s
ACL graph capture: 231 s
单请求: 1/1 success, input=4103, output=16, 0.75 s
双并发: 2/2 success, input=8206, output=32, 0.69 s
exit code: 0
```

修复后没有 `Shape: 0`、HCCL、AICPU 或 ubatch thread 异常。

日志：

```text
/data/workspace/logs/deepseek-v2-dbo-server-dp-fix2-eager.log
/data/workspace/logs/deepseek-v2-dbo-test-dp2-fix2-eager.log
/data/workspace/logs/deepseek-v2-dbo-server-dp-final.log
/data/workspace/logs/deepseek-v2-dbo-test-dp2-final-compile.log
```

## 5. FlashComm1 边界

本机只有两卡。`TP=1, DP=2, FlashComm1=on` 会被配置校验拒绝，因为 FlashComm1 要求 TP>1。真正的 DP+TP+FlashComm1 验证至少需要：

```text
TP=2, DP=2, EP=4
```

因此本 RFC 已解决 DP+DBO 的调度/collective bug，但不声称完成四卡 FlashComm1 验证。后续矩阵必须覆盖 eager/compile、odd/even token、单请求/双 rank 并发以及输出与 eager baseline 的逐 token 对比。
