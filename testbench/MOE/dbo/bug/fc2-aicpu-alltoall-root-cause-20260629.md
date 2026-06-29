# FlashComm2 AI_CPU AllToAll 根因分析

## 结论

DeepSeek-V2 TP=2 场景中的 `507018` 主故障不是由 DBO 引起，也不要求
ACL Graph、FlashComm1、EP 或复杂 tensor layout。问题可以缩减为：

```text
Ascend 910B
+ CANN 9.0.0
+ torch_npu 2.10.0
+ 两 rank HCCL
+ HCCL_OP_EXPANSION_MODE=AI_CPU
+ dist.all_to_all_single()
= RunAicpuRpcSrvLaunchV2_alltoall errorCode=0x2a / 507018
```

同一最小脚本切换为 `HCCL_OP_EXPANSION_MODE=AIV` 后，连续四轮通信和逐元素
结果校验全部通过。

因此当前故障边界位于 CANN/HCCL 的 AI_CPU AllToAll 执行路径。FlashComm2
只是模型中引入该 AllToAll 的调用方。

`Flashcomm2OProjRowParallelOp` 缺少 DBO row hook 仍然是独立的 DBO 集成缺陷，
但不是本次 AI_CPU `507018` 的必要条件，也不能仅靠增加 hook 修复。

## 环境

```text
Date: 2026-06-29
Device: Ascend 910B3, 2 cards
CANN: 9.0.0
torch: 2.10.0+cpu
torch_npu: 2.10.0
vLLM: 0.22.1
vllm-ascend commit: ea5141df718d0855c04a408af87883841ef56f08
Model: DeepSeek-V2-Lite-Chat
```

## 原始误判

原 RFC 将故障归因于：

```text
DBO 两个 ubatch
-> 多 host thread / 多 NPU stream
-> 共用 ODP communicator
-> FC2 缺少 DBO row hook
-> AllToAll 顺序失配
```

这个路径在理论上成立，代码中也确实缺少 hook，但新对照证明 DBO 关闭后仍能
稳定复现完全相同的 AICPU AllToAll 异常。因此它不能解释当前主故障。

## 模型级对照

### 1. 真正关闭 DBO，AI_CPU + FC2 + graph

启动命令没有 `--enable-dbo`，配置日志也没有 `enable_dbo=True`。模型使用
`ACLGraphWrapper`，而不是 `AscendUBatchWrapper`。

结果：

```text
Initial profiling/warmup run completed
Profiling CUDA graph memory
Waiting for pending NCCL work before graph capture
RunAicpuRpcSrvLaunchV2_alltoall
stream_id=11, task_id=367, errorCode=0x2a
runtime result=507018
```

两 rank 在相同 local stream 和 task 上失败，不存在 DBO 双 ubatch stream。

日志：

```text
/data/workspace/logs/tp2-fc2-nodbo-aicpu-20260629.log
/root/ascend/log/debug/plog/plog-27036_20260629111943419.log
/root/ascend/log/debug/plog/plog-27037_20260629111943424.log
```

### 2. 真正关闭 DBO，AIV + FC2

graph-memory profiling 越过 AI_CPU 的固定故障点。随后使用 eager 服务执行真实
请求：

```text
warmup: 16 successful, 0 failed
formal: 2 successful, 0 failed
input: 4096 tokens
output: 16 tokens
```

日志：

```text
/data/workspace/logs/tp2-fc2-nodbo-aiv-20260629.log
/data/workspace/logs/tp2-fc2-nodbo-aiv-eager-20260629.log
/data/workspace/logs/tp2-fc2-nodbo-aiv-eager-test-20260629.log
```

### 3. 真正关闭 DBO，AI_CPU + FC2 + eager

服务能够启动，因为启动阶段没有 graph-memory capture。首批真实请求执行模型后，
服务立即退出：

```text
RunAicpuRpcSrvLaunchV2_alltoall
stream_id=11, task_id=115, errorCode=0x2a
runtime result=507018
```

这证明 ACL Graph 不是必要条件。Graph 只是在启动阶段同步 pending HCCL work，
因而更早暴露异步设备错误。

日志：

```text
/data/workspace/logs/tp2-fc2-nodbo-aicpu-eager-20260629.log
/data/workspace/logs/tp2-fc2-nodbo-aicpu-eager-test-20260629.log
```

### 4. 关闭 FlashComm1

配置：

```text
DBO=off
FC1=off
FC2=on
AI_CPU
eager
```

首批请求仍在两 rank 的 `stream_id=11, task_id=3` 失败：

```text
RunAicpuRpcSrvLaunchV2_alltoall
errorCode=0x2a
```

因此 FlashComm1、MLA sequence-parallel 通信和 post-model AllGather 不是必要条件。

日志：

```text
/data/workspace/logs/tp2-fc2-nodbo-aicpu-eager-fc1off-20260629.log
/data/workspace/logs/tp2-fc2-nodbo-aicpu-eager-fc1off-test-20260629.log
/root/ascend/log/debug/plog/plog-41940_20260629114632066.log
/root/ascend/log/debug/plog/plog-41941_20260629114632076.log
```

## 最小复现

脚本：

```text
testbench/MOE/dbo/bug/reproduce_fc2_alltoall.py
```

该脚本只执行：

1. 两 rank HCCL 初始化；
2. 创建 contiguous BF16 tensor；
3. `dist.all_to_all_single()`；
4. 当前 stream 同步；
5. 逐元素验证接收结果。

不包含 vLLM、模型、FC1、FC2 类、EP、DBO、ACL Graph 或多个 communicator。

### FC2 对应 shape

```bash
HCCL_OP_EXPANSION_MODE=AI_CPU \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  testbench/MOE/dbo/bug/reproduce_fc2_alltoall.py \
  --tokens 4096 --hidden-per-rank 1024 --iterations 4
```

结果：第一次 AllToAll 即失败，两 rank 均为：

```text
stream_id=12
task_id=1
RunAicpuRpcSrvLaunchV2_alltoall
errorCode=0x2a
507018
```

日志：

```text
/data/workspace/logs/minimal-fc2-alltoall-aicpu-20260629.log
```

### AIV 对照

相同命令仅修改：

```bash
HCCL_OP_EXPANSION_MODE=AIV
```

结果：

```text
iteration 0 verified=true
iteration 1 verified=true
iteration 2 verified=true
iteration 3 verified=true
PASS
exit code: 0
```

日志：

```text
/data/workspace/logs/minimal-fc2-alltoall-aiv-20260629.log
```

### 极小消息

```bash
HCCL_OP_EXPANSION_MODE=AI_CPU \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  testbench/MOE/dbo/bug/reproduce_fc2_alltoall.py \
  --tokens 2 --hidden-per-rank 1 --iterations 1
```

结果仍是第一次 AllToAll 的 `stream_id=12, task_id=1, errorCode=0x2a`。

这排除了以下必要条件：

- 大消息；
- FC2 reshape/reorganization；
- 非 contiguous tensor；
- 动态 shape；
- buffer 地址变化；
- AI_CPU cache 的第二次复用；
- 多 communicator；
- EP；
- DBO；
- ACL Graph。

日志：

```text
/data/workspace/logs/minimal-alltoall-aicpu-tiny-20260629.log
```

## 根因边界

### 已证实

1. 该环境的 HCCL AI_CPU AllToAll 基础调用失败。
2. 错误来自设备侧 `libccl_kernel.so` 的
   `RunAicpuRpcSrvLaunchV2_alltoall`。
3. AIV 对相同 group、shape、dtype 和数据内容执行正确。
4. DBO、graph、FC1、消息大小和 tensor layout 都不是必要条件。
5. Python 中 FIA、MatMul、shared expert、event synchronize 或 graph enter
   只是异步观察点。

### 尚不能从应用侧确定

`errorCode=0x2a` 只表明 AICPU task exception，当前 plog 没有更细的 HCCL 内部
错误字段。因此还不能区分：

- CANN 9.0.0 的 AI_CPU AllToAll kernel 缺陷；
- 当前 910B3 型号与该 expansion mode 的兼容性问题；
- HCCL 与驱动/固件版本组合问题；
- ProcessGroupHCCL 对 AI_CPU AllToAll 的下发参数问题。

这些需要携带最小复现和 plog 向 CANN/HCCL 侧继续定位。

## DBO 缺陷的正确定位

远端当前 `Flashcomm2OProjRowParallelOp.apply_impl()` 确实没有
`dbo_linear_row_hook`。在 AI_CPU AllToAll 基础能力恢复后，DBO 场景仍需修复：

```text
row hook record
-> optional quantize
-> ODP AllToAll
-> OProj MatMul
-> optional OTP ReduceScatter
-> optional TP AllGather
-> row hook completion/yield
```

否则两个 ubatch 可能在同一 communicator 上产生不受模板约束的 collective
提交和 stream 依赖。这个风险与当前 AI_CPU 单调用失败相互独立。

## W8A8

W8A8 会把 FC2 `communication_fn` 下沉到 quantization method：

```text
BF16 input
-> INT8 quantize
-> INT8 ODP AllToAll
-> quantized OProj
```

因此：

1. AIV 可减少通信字节数，是当前可验证方向；
2. AI_CPU 连极小 BF16 AllToAll 都失败，当前没有必要先归因于 W8A8 dtype；
3. 后续 DBO hook 必须覆盖 quant method 内真正执行的 AllToAll，不能只覆盖 BF16
   外层分支。

## 建议

### 当前 workaround

优先使用：

```bash
export HCCL_OP_EXPANSION_MODE=AIV
```

或者关闭 FlashComm2：

```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
```

AIV 在 A2 上存在 Vector Core 资源竞争和多 communicator 并发约束，正式采用前需要
继续做压力与性能验证。

### 上报 CANN/HCCL

提交以下材料已经足够复现：

- `reproduce_fc2_alltoall.py`；
- AI_CPU 大 shape 日志；
- AI_CPU tiny shape 日志；
- AIV PASS 日志；
- 两 rank debug plog；
- CANN、torch_npu、驱动和固件版本。

### RFC 修正

原 RFC 应拆成两个独立问题：

1. `HCCL AI_CPU AllToAll` 在当前环境基础调用失败，P0；
2. `Flashcomm2OProjRowParallelOp` 缺少 DBO row hook，P0，但需在基础通信可用后验证。

不应再将 `507018` 直接作为 DBO collective 顺序失配的证据。
