# RFC: DBO 与 FlashComm2 的 AICPU AllToAll 兼容性

## 状态

- 状态：问题已定位，修复方案待实现和 NPU 验证
- 日期：2026-06-27
- 影响范围：DBO + FlashComm2，TP/ODP communicator，多 stream eager/compile

## 摘要

Ascend 910B、TP=2 下同时启用 DBO 和 FlashComm2，会在首个 DBO prefill
step 稳定触发 HCCL AllToAll AICPU 异常，最终导致 EngineCore 退出。

对照实验已经排除 FlashComm1、FlashComm2 O-Shard 和 MLA FIA 为必要条件。
稳定触发条件是：

```text
enable_dbo = true
enable_flashcomm2_parallel_size > 0
```

直接故障源是 `Flashcomm2OProjRowParallelOp` 的 ODP AllToAll。该通信块没有
接入 DBO row hook，两个 ubatch 线程可以在两条 NPU stream 上向同一 communicator
无序提交 collective，破坏 HCCL collective 的跨 rank 配对和 stream 依赖。

## 环境与复现

```text
Device: Ascend 910B
CANN: 9.0.0
vLLM: 0.22.1
Model: DeepSeek-V2-Lite-Chat
TP: 2
all2all backend: deepep_low_latency
HCCL_OP_EXPANSION_MODE: AI_CPU
```

关键配置：

```bash
export VLLM_ASCEND_ENABLE_DBO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
```

测试：

```bash
PORT=8001 \
INPUT_LEN=4096 OUTPUT_LEN=16 \
NUM_PROMPTS=500 MAX_CONCURRENCY=96 \
bash deepseek-v2-dbo-test.sh
```

O-Shard 关闭后仍可复现，因此 O-Shard broadcast 不是必要条件。

## 故障证据

设备侧最早错误：

```text
soName=libccl_kernel.so
funcName=RunAicpuRpcSrvLaunchV2
kernelName=RunAicpuRpcSrvLaunchV2_alltoall
errorCode=0x2a
runtime result=507018
```

Python 侧可能在 FIA、MatMul、Abs、AllGather 或 stream synchronize 处观察到错误。
这是异步执行的结果：后续 API 只是发现 device 已进入 error 状态，不能据此判断
故障发起算子。根因分析必须以 plog 中最早失败的 AICPU task 为准。

对照矩阵：

| DBO | FC1 | FC2 | O-Shard | 结果 |
|---|---:|---:|---:|---|
| 开 | 关 | 关 | 关 | 通过 |
| 开 | 开 | 关 | 关 | 通过 |
| 开 | 关 | 开 | 开 | AICPU AllToAll crash |
| 开 | 关 | 开 | 关 | AICPU AllToAll crash |
| 开 | 开 | 开 | 开 | AICPU AllToAll crash |

## 代码级根因

FlashComm2 attention OProj 路径：

```text
_get_row_parallel_op()
  -> Flashcomm2OProjRowParallelOp.apply_impl()
     -> otp_maybe_quant_comm()
        -> dist.all_to_all_single(..., odp_group)
     -> o_proj matmul
     -> reduce_scatter
     -> optional TP all_gather
```

普通 `OProjRowParallelOp` 会用以下 hook 包住完整通信块：

```python
_dbo_call_linear_row_hook(context, is_record=True)
all_to_all(...)
matmul(...)
reduce_scatter(...)
_dbo_call_linear_row_hook(context, is_record=False)
```

FlashComm2 实现缺少对应 hook。DBO 的两个 ubatch context 交叉使用 compute/comm
stream；没有 hook 时，DBO template 无法建立以下约束：

1. 两个 rank 对同一 collective 的一致提交顺序；
2. 同一 communicator 上前后 collective 的 stream happens-before；
3. OProj AllToAll 与相邻 MLA/MoE 通信块的 event 依赖；
4. ubatch CPU yield 与设备通信完成点之间的关系。

因此根因不是笼统的“HCCL 不支持多线程”，而是 FlashComm2 的 collective
绕过了 DBO 的调度协议。

## 修复方案

### 方案 A：把完整 FC2 OProj 通信块纳入 row hook

推荐方案。hook 边界必须覆盖：

```text
ODP AllToAll
  -> OProj matmul
  -> OTP/TP ReduceScatter
  -> 条件性 TP AllGather
```

不能只包住 `all_to_all_single`，否则后续 collective 仍可能与另一个 ubatch 交错。

伪代码：

```python
context = get_forward_context()
if context.dbo_enabled:
    _dbo_call_linear_row_hook(context, is_record=True)

input_parallel = otp_maybe_quant_comm(input_parallel)
output_parallel = matmul(input_parallel)
output = reduce_scatter(output_parallel)
if not context.flash_comm_v1_enabled:
    output = tp_all_gather(output)

if context.dbo_enabled:
    _dbo_call_linear_row_hook(context, is_record=False)
```

### 方案 B：独立的 FlashComm2 DBO hook/event

如果 row hook 的 event 生命周期无法表达 ODP 和 OTP 两个 communicator，可新增
FC2 专用 event key。只有 profiling 证明普通 row hook 无法正确 overlap 时才采用，
避免扩大模板复杂度。

### 兼容性保护

在方案 A/B 完成验证前，检测到 `DBO + FC2` 应报清晰错误或关闭 FC2。当前
workaround：

```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
```

## 次生问题

`npu_ubatch_wrapper` 必须传播子线程原始异常。当前子线程失败后，主线程可能继续
访问空的 results 并抛出 `IndexError`，掩盖 AICPU 根因。应收集
`result | exception`，join 后优先重新抛出原异常，并校验结果数量。

## 验收标准

1. FC1 off、FC2 on、O-Shard off；
2. FC1 off、FC2 on、O-Shard on；
3. FC1 on、FC2 on；
4. eager 与 torch.compile 分别验证；
5. 4096 输入、500 请求、并发 96，连续三轮；
6. 100% 请求成功；
7. plog 无 `0x2a` 和 `507018`；
8. profiler 证明 collective 顺序一致；
9. 不允许用全局 stream synchronize 消除所有 overlap；
10. 与 DBO off 做输出准确性对比。

## 参考证据

- `../bug/ai_cpu.md`
- `/data/workspace/codex-dbo-analysis-20260627/`

