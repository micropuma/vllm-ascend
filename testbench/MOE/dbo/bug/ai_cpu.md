# DBO + FlashComm2 导致 HCCL AllToAll AICPU 异常

## 摘要

在 Ascend 910B、TP=2、DBO 和 FlashComm2 同时启用时，第一个正式 DBO prefill step 会触发 HCCL AICPU 异常，导致 EngineCore 退出及全部请求失败。

远端对照实验确认：

- FlashComm1 不是必要条件；
- FlashComm2 O-Shard 不是必要条件；
- `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1` 与 DBO 同时启用是稳定触发条件；
- 直接故障点是 FlashComm2 OProj 路径中的 `HcclAllToAll`，不是 MLA FIA 内部通信。

根因是 `Flashcomm2OProjRowParallelOp` 发起的 AllToAll 未接入 DBO row hook。两个 ubatch 线程可在两个 NPU stream 上向同一个 FlashComm2 ODP communicator 提交集合通信，DBO 模板无法约束其提交顺序和 stream 依赖，最终使 AICPU HCCL task 异常。

## 环境

| 项目 | 配置 |
|---|---|
| 芯片 | Ascend 910B |
| CANN | 9.0.0 |
| vLLM | 0.22.1 |
| 模型 | DeepSeek-V2-Lite-Chat |
| 并行配置 | TP=2，EP enabled |
| 执行模式 | eager |
| DBO | prefill threshold=1024 |
| MoE backend | `deepep_low_latency` |
| HCCL expansion | `AI_CPU` |

代码版本：

```text
vllm-ascend: 98446bf8f78c11511ceead138a10771c650d6e0b
```

## 最小复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /data/workspace/.venv-dbo/bin/activate

export SOC_VERSION=ascend910b1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1

export HCCL_OP_EXPANSION_MODE=AI_CPU
export VLLM_ASCEND_ENABLE_DBO=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0

cd /data/workspace/vllm-ascend/testbench/MOE/dbo/demos
PORT=8001 bash deepseek-v2-dbo-server.sh 2>&1 | tee server.log
```

另一个终端运行：

```bash
source /data/workspace/.venv-dbo/bin/activate
cd /data/workspace/vllm-ascend/testbench/MOE/dbo/demos

PORT=8001 \
LABEL=dbo_fc2_repro \
INPUT_LEN=4096 \
OUTPUT_LEN=16 \
NUM_PROMPTS=500 \
MAX_CONCURRENCY=96 \
bash deepseek-v2-dbo-test.sh
```

预期现象：首个正式 DBO prefill step 后服务退出，500 个请求全部失败。

## 对照实验

所有实验使用相同模型、TP=2、输入长度 4096 和 `AI_CPU` expansion mode。

| DBO | FlashComm1 | FlashComm2 | O-Shard | 并发/请求数 | 结果 |
|---|---:|---:|---:|---:|---|
| 开 | 关 | 关 | 关 | 16/32 | 32/32 成功 |
| 开 | 开 | 关 | 关 | 16/32 | 32/32 成功 |
| 开 | 开 | 关 | 关 | 96/500 | 500/500 成功 |
| 开 | 关 | 开 | 开 | 96/500 | Engine crash |
| 开 | 关 | 开 | 关 | 96/500 | Engine crash |
| 开 | 开 | 开 | 开 | 96/500 | Engine crash |

该矩阵排除了以下假设：

1. **不是 FlashComm1 MLA AllGather 单独导致。**  
   DBO + FlashComm1 在完整压力下稳定通过。

2. **不是 O-Shard 的异步权重 broadcast 单独导致。**  
   关闭 O-Shard、保留 FlashComm2 后仍然复现。

3. **不是 FIA 算子本身首先失败。**  
   FIA、MatMul、Abs 等算子只是异步检测到已经发生的 device error。

## 故障证据

设备侧首要错误：

```text
Aicpu kernel execute failed,
device_id=0/1,
stream_id=11,
task_id=3,
soName=libccl_kernel.so,
funcName=RunAicpuRpcSrvLaunchV2,
kernelName=RunAicpuRpcSrvLaunchV2_alltoall,
errorCode=0x2a
```

在不同运行中，Python 侧可能显示：

```text
current working operator name is aclnnFusedInferAttentionScoreV3
current working operator name is aclnnMatmul
current working operator name is aclnnAbs
current working operator name is HcclAllGather
```

这些不是稳定的故障发起点。Ascend 默认异步下发，AICPU AllToAll 先失败，后续任意触发 runtime 状态检查的 API 都可能观察到 `507018`。定位时应以 plog 中最早的失败 task 为准。

原始 DBO + FC1 + FC2 运行中，故障前两个 TP AllGather 的 HCCL plog 均记录为成功。因此不能据 Python 栈推断“AllGather 内部转换成 AllToAll 后失败”。

## 代码级根因

FlashComm2 为 attention `o_proj` 选择：

```text
vllm_ascend/ops/linear_op.py
  _get_row_parallel_op()
    -> Flashcomm2OProjRowParallelOp
```

其通信路径为：

```text
Flashcomm2OProjRowParallelOp.apply_impl()
  -> otp_maybe_quant_comm()
     -> dist.all_to_all_single(..., odp_group)
  -> o_proj matmul
  -> optional OTP reduce_scatter
  -> TP all_gather when FlashComm1 is disabled
```

普通 `OProjRowParallelOp` 已显式接入 DBO：

```python
if forward_context.dbo_enabled:
    _dbo_call_linear_row_hook(forward_context, is_record=True)

dist.all_to_all_single(...)
output_parallel = self.quant_method.apply(...)
output = self.comm_group.reduce_scatter(...)

if forward_context.dbo_enabled:
    _dbo_call_linear_row_hook(forward_context, is_record=False)
```

`Flashcomm2OProjRowParallelOp.apply_impl()` 中没有对应的 row hook。

DBO 使用两个 Python 线程和两条互换角色的 NPU stream 执行两个 ubatch。DBO 模板只在已注册的 hook 处记录 event、等待 event 和切换线程。由于 FlashComm2 OProj AllToAll 位于 hook 管理范围之外：

1. ubatch 线程可以独立进入同一个 ODP communicator；
2. AllToAll 的 host 提交顺序未被 DBO 调度约束；
3. AllToAll 与相邻 attention/MoE 通信之间缺少 DBO event 依赖；
4. HCCL AICPU task 在双 stream overlap 下进入异常状态；
5. 后续算子观察到 `507018`，ubatch 线程退出；
6. EngineCore 最终退出，所有请求收到 `EngineDeadError`。

因此这是 **FlashComm2 OProj 通信块缺少 DBO 集成**，而不是通用 HCCL 线程安全问题。FC1-only 对照在相同压力下通过，也说明不能简单归因于“同一进程存在两个 ubatch 线程”。

## 次生缺陷

`npu_ubatch_wrapper.py::_run_ubatches()` 没有可靠传播子线程异常。子线程退出后 `results` 可能为空，主线程继续访问：

```python
sorted_results[i]
```

从而抛出：

```text
IndexError: list index out of range
```

该异常会掩盖原始 AICPU/HCCL 错误。它不是本次 device fault 的根因，但应独立修复。

## 修复建议

### 1. 将 FlashComm2 OProj 通信纳入 DBO row hook

参照 `OProjRowParallelOp`，在 `Flashcomm2OProjRowParallelOp.apply_impl()` 中覆盖完整通信块：

```python
forward_context = get_forward_context()
if forward_context.dbo_enabled:
    _dbo_call_linear_row_hook(forward_context, is_record=True)

input_parallel = otp_maybe_quant_comm(input_parallel)
output_parallel = self.quant_method.apply(self.layer, input_parallel, bias=bias_)

if self.tp_size > 1:
    output = self.comm_group.reduce_scatter(output_parallel, dim=0)
else:
    output = output_parallel

if not forward_context.flash_comm_v1_enabled:
    output = get_tp_group().all_gather(output, 0)

if forward_context.dbo_enabled:
    _dbo_call_linear_row_hook(forward_context, is_record=False)
```

hook 边界必须覆盖 FC2 AllToAll 以及紧随其后的 ReduceScatter/AllGather，避免把一个逻辑通信块拆开。

### 2. 修复前增加配置保护

在兼容性修复和 NPU 验证完成前，检测到以下组合时应禁用 FlashComm2 或直接报错：

```text
enable_dbo=True
enable_flashcomm2_parallel_size>0
```

当前安全 workaround：

```bash
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
```

FlashComm1 可以保留；本次 `DBO + FC1` 的 500 请求压力测试全部成功。

### 3. 传播 ubatch 子线程原始异常

主线程应收集每个 ubatch 的异常，在 merge results 前重新抛出，并校验结果数量：

```text
len(results) == num_ubatches
```

不要在子线程失败后继续执行 gather/cat。

### 4. 修正 demo 默认值

`deepseek-v2-dbo-server.sh` 的注释声明 FlashComm 在 DBO 下默认关闭，但当前工作区脚本将 FC1/FC2 默认值设为 1。默认值应与注释及已知兼容性状态一致。

## 验收标准

修复至少应通过：

1. DBO + FC2，FC1 关闭，O-Shard 关闭；
2. DBO + FC2，FC1 关闭，O-Shard 开启；
3. DBO + FC1 + FC2；
4. 输入长度 4096、500 requests、并发 96；
5. 重复运行至少 3 次；
6. 结果 100% 成功，plog 中无 `0x2a`、`507018`；
7. 对比 DBO disabled 的输出准确性；
8. profiler 验证 FC2 AllToAll 与相邻通信的 event 顺序；
9. 验证修复没有用全 stream synchronize 消除全部 overlap。

## 诊断产物

本次远端日志和 plog 位于：

```text
/data/workspace/codex-dbo-analysis-20260627/
```

主要目录：

```text
server-dbo-fc-aicpu.log
server-dbo-only-aicpu.log
server-dbo-fc1-aicpu.log
server-dbo-fc2-aicpu.log
server-dbo-fc2-nooshard-aicpu.log
plog/
plog-dbo-only/
plog-dbo-fc1/
plog-dbo-fc2/
plog-dbo-fc2-nooshard/
```
