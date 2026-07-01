# RFC: DBO + FlashComm1 冷启动 AOT/graph capture 触发 AddRmsNormBias shape mismatch

## 状态

- 状态：已复现，根因已缩小到 FC1 sequence-parallel shape contract
- 日期：2026-07-01
- 环境：remote-env 固定机 `218.106.157.54:32222`

## 摘要

在远端固定环境中，对 `DeepSeek-V2-Lite-Chat` 使用：

- `DBO=1`
- `FlashComm1=1`
- `FlashComm2=0`
- fresh `VLLM_CACHE_ROOT`
- fresh `compilation_config.cache_dir`
- `cudagraph_capture_sizes=[16]`

进行冷启动时，服务在 engine 初始化阶段失败，最早错误不是 Python timeout，也不是 HCCL watchdog，而是 worker warmup 里的设备侧算子报错：

```text
Input x2/x1 shape invalid, shape is not equal x1 shape.
call aclnnAddRmsNormBias failed
AddRmsNormBias do tiling failed, ret is -1
```

失败发生在：

```text
determine_available_memory
  -> profile_cudagraph_memory
    -> _dummy_run
      -> AOT compiled fn
        -> ACL graph / piecewise backend runtime
          -> torch.ops._C_ascend.npu_add_rms_norm_bias(...)
```

同一套参数下，仅将 `FlashComm1` 从 `1` 改为 `0`，服务可在冷缓存下正常完成 compile、graph capture 并启动监听。说明故障与 FC1 路径强相关。

## 复现记录

### 共同前提

使用同一远端环境初始化：

```bash
source /data/workspace/.venv-dbo/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export SOC_VERSION=ascend910b1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

为避免污染共享缓存，不删除已有缓存目录，而是为每次实验创建全新隔离目录，同时隔离：

- `VLLM_CACHE_ROOT`
- `--compilation-config.cache_dir`

### Case A: 失败复现（DBO=1, FC1=1）

实验标签：

```text
fc1_dbo_aot_confirm_20260701_031522
```

日志：

```text
/data/workspace/logs/fc1_dbo_aot_confirm_20260701_031522.log
```

缓存：

```text
/data/workspace/repro_cache/fc1_dbo_aot_confirm_20260701_031522/
```

关键命令：

```bash
vllm serve /data/models/DeepSeek-V2-Lite-Chat \
  --host 127.0.0.1 \
  --port 8013 \
  --dtype bfloat16 \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --all2all-backend deepep_low_latency \
  --dbo-prefill-token-threshold 1024 \
  --dbo-decode-token-threshold 1000000000 \
  --max-model-len 8192 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 256 \
  --disable-log-stats \
  --compilation-config '{"cudagraph_capture_sizes":[16],"cache_dir":"/data/workspace/repro_cache/fc1_dbo_aot_confirm_20260701_031522/compile_cache"}'
```

关键环境变量：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
export VLLM_ASCEND_ENABLE_DBO=1
export HCCL_OP_EXPANSION_MODE=AI_CPU
```

结果：

- `curl http://127.0.0.1:8013/v1/models` 失败；
- worker 在 warmup 阶段退出；
- APIServer 最终抛出 `Engine core initialization failed`。

最早设备错误：

```text
[ERROR] Input x2/x1 shape invalid, shape is not equal x1 shape.
[ERROR] Input shape invalid.
RuntimeError: call aclnnAddRmsNormBias failed
EZ9999 ... AddRmsNormBias do tiling failed, ret is -1
```

最关键调用链：

```text
determine_available_memory
  -> profile_cudagraph_memory
    -> _dummy_run
      -> DeepseekV2 forward
        -> AOT compiled fn
          -> ACL graph / piecewise backend
            -> torch.ops._C_ascend.npu_add_rms_norm_bias
```

### Case B: 对照成功（DBO=1, FC1=0）

实验标签：

```text
fc0_dbo_aot_20260701_031109
```

日志：

```text
/data/workspace/logs/fc0_dbo_aot_20260701_031109.log
```

唯一改动：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
```

其余保持一致，包括 fresh cache、DBO、模型、compile config、TP=2。

结果：

```text
init engine (profile, create kv cache, warmup model) took 46.98 s (compilation: 17.53 s)
Starting vLLM server on http://127.0.0.1:8012
Application startup complete.
GET /v1/models -> 200 OK
```

并且 worker 完成：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
Capturing CUDA graphs (decode, FULL)
Graph capturing finished in 3 secs
```

### Case C: FC1=1, DBO=0

实验标签：

```text
fc1_dbo0_aot_20260701_032050
```

该隔离实验在本轮只跑到了权重加载阶段后手动停止，未作为结论依据。

## 结论

在本轮实验中，可以确认：

1. 故障不是“缓存脏”导致。fresh AOT cache 与 fresh compile cache 下稳定复现。
2. 故障不是“服务太慢看起来像 hang”。最终会收敛成明确的设备侧错误。
3. 故障与 `FlashComm1` 强相关：`FC1=0` 成功，`FC1=1` 失败。
4. 失败点位于 cold compile / graph capture 的 warmup 路径，不是正式请求流量阶段。
5. 最早根因不是 HCCL、watchdog、EngineCore cancel 或 APIServer traceback；这些都是次级错误。

## 根因分析

### 1. 直接根因：AddRmsNormBias 两个输入 token 维度不一致

`npu_add_rms_norm_bias(x1, x2=residual, weight, ...)` 要求 `x1` 和 `x2` 的 token 维度一致。

当前最早错误：

```text
Input x2/x1 shape invalid, shape is not equal x1 shape.
```

说明 warmup 某条 FC1 sequence-parallel 路径中：

- 一侧张量保留了 TP communication padding 或 local shard padded 长度；
- 另一侧 residual 走的是 unpadded/chunked 长度；
- 两者在 AddRmsNormBias 处重新汇合时 shape contract 断裂。

### 2. 为什么只在 cold compile / AOT 暴露

冷启动时，vLLM 会在 engine 初始化中执行：

```text
determine_available_memory
  -> profile_cudagraph_memory
  -> _dummy_run
  -> AOT / piecewise compile
  -> graph capture
```

这条路径会稳定命中：

- FC1 开启后的 sequence-parallel fused pattern；
- compile 后的 runtime graph；
- capture size 固定为 `16` 的 warmup shape。

因此它比 eager 更容易暴露“局部 shard 长度”和“residual chunk 长度”不一致的问题。

### 3. 为什么不是纯 DBO hook trace 问题

这次最早失败点不是：

- `threading.get_ident()`
- `get_forward_context()`
- `get_tp_group()`
- fake impl 读 runtime context

而是 runtime warmup 已经进入编译后 callable，再在设备算子 `AddRmsNormBias` 处报 shape mismatch。

也就是说，本次复现说明至少还有一类问题独立存在：

> FC1 sequence-parallel runtime shape contract 本身不闭合。

这和之前 RFC 中讨论的 compile-safe / fake-impl 问题不是互斥关系，而是另一层 contract 问题。

### 4. 最可疑的 contract 断点

当前代码里已经出现一个明显的“partial fix”信号：

```python
o_proj_output = self.o_proj(...)[0]

## 第二轮修改与验证（2026-07-01）

### 修改 1：修正 FC1 residual 在 TP local shard 上的切片 contract

文件：

- `vllm_ascend/ops/register_custom_ops.py`

修改要点：

- 不再依赖 `pad_size + torch.chunk(...)` 去恢复 TP local residual；
- 改为按 `x.size(0)` 代表的 local logical token 数，显式切出 `[tp_rank * local_num_tokens : (tp_rank + 1) * local_num_tokens]`；
- 当 residual 总长度不足 `tp_size * local_num_tokens` 时，仅补齐到这个长度。

原因：

- 旧逻辑在 `x.shape[0]=5, residual.shape[0]=8, tp_size=2` 这类场景下，会把 residual chunk 成 `4 + 4`；
- 但当前 local shard 真实需要的是 `5` 个 token；
- 这正好解释了 `AddRmsNormBias` 的两个输入 token 维度不一致。

效果：

- 之前最早出现的 `call aclnnAddRmsNormBias failed` / `AddRmsNormBias do tiling failed` 在后续冷编译验证中不再出现。

### 修改 2：修正 MLA custom op 入口对 graph padded hidden_states 的 output contract

文件：

- `vllm_ascend/ops/mla.py`

新增 helper：

- `_resolve_mla_forward_inputs(...)`

修改要点：

- 当 `FlashComm1` 开启且 `_EXTRA_CTX.num_tokens` 已经表示 logical token 数时：
  - 先把 `hidden_states` 从 graph padded 长度裁到 logical token 数；
  - output 也按 logical token 数分配；
  - VL 首层特例仍保留，但改为基于 logical token 数计算 local output。

原因：

- 第二轮复现中，第一处 residual 问题修掉后，冷编译暴露出新的更后置错误：

```text
RuntimeError: The expanded size of the tensor (4096) must match the existing size (8)
Target sizes: [4096, 2048]. Tensor sizes: [8, 2048]
```

- 栈定位到：

```text
vllm_ascend/attention/mla_v1.py
output[...] = o_proj_output[:output.shape[0]]
```

- 根因不是这行赋值本身，而是：
  - `ops/mla.py` 仍按 `hidden_states.shape[0]`（graph padded 4096）分配 output；
  - `mla_v1.py` 内部 `o_proj_input` 却按 `_EXTRA_CTX.num_tokens`（logical 8）创建；
  - 同一前向内部 token 数语义分叉，最终在写回 output 时炸掉。

效果：

- 第二轮冷编译中，`4096 vs 8` 的 MLA expand 报错不再出现。

### 新的验证结果

冷编译实验标签：

```text
fc1_dbo_aot_fixed2_20260701_034400
```

日志：

```text
/data/workspace/logs/fc1_dbo_aot_fixed2_20260701_034400.log
```

缓存：

```text
/data/workspace/repro_cache/fc1_dbo_aot_fixed2_20260701_034400/
```

验证事实：

1. worker 两侧都完成权重加载；
2. compile cache 两侧都成功落盘：
   - `compile_cache/rank_0_0/...`
   - `compile_cache/rank_1_0/...`
3. AOT model 两侧都成功落盘：
   - `.../torch_aot_compile/.../rank_0_0/model`
   - `.../torch_aot_compile/.../rank_1_0/model`
4. 日志中不再出现：
   - `AddRmsNormBias`
   - `call aclnnAddRmsNormBias failed`
   - `expanded size of the tensor (4096) must match the existing size (8)`

### 当前新的后置阻塞点

冷编译不再在原始 shape mismatch 处失败，但服务仍未 ready。新的现象是 EngineCore 周期性打印：

```text
No available shared memory broadcast block found in 60 seconds.
```

表现：

- APIServer 端口未监听；
- worker 进程未退出，但 engine 初始化未完成；
- 更像是 compile/capture 之后的同步、广播或初始化收尾阶段卡住，而不是再次出现前面的 shape contract 错误。

## 当前判断

截至 2026-07-01 第二轮修改，已经可以确认：

1. 原始 bug 的第一处根因是 FC1 residual local shard 的 shape contract 错误；
2. 修掉后又暴露出 MLA custom op 对 graph padded hidden_states 的 output contract 错误；
3. 这两处 shape bug 均已被本轮修改消除；
4. 当前剩余问题已经推进到更后面的 engine 同步/广播阶段，属于新的 blocker，而不是原始 `AddRmsNormBias` 故障本身。

## 第三轮验证：定位 `name 'Min' is not defined` 与 FC1/DBO 分叉结论

### Case D: FC1=1, DBO=0 首次完整验证暴露 `Min is not defined`

实验标签：

```text
fc1_nodbo_aot_20260701_035634
```

日志：

```text
/data/workspace/logs/fc1_nodbo_aot_20260701_035634.log
```

现象：

- 该实验不是 DBO 死等；
- worker 在 compile 完成后明确抛出：

```text
RuntimeError: Worker failed with error 'name 'Min' is not defined'
```

关键调用链：

```text
determine_available_memory
  -> profile_run
    -> _dummy_run
      -> AOT compiled fn
        -> npugraph_ex compiler runtime
          -> File "<string>", line 83, in kernel
             NameError: name 'Min' is not defined
```

### `Min` 的精确根因

该错误不是算子内核本身抛出的 Python 名称错误，而是 compile cache 生成的 Python 图文件里直接出现了未导入的 `Min(...)` 符号。

定位到生成文件：

```text
/data/workspace/repro_cache/fc1_nodbo_aot_20260701_035634/compile_cache/rank_0_0/backbone/computation_graph.py
/data/workspace/repro_cache/fc1_nodbo_aot_20260701_035634/compile_cache/rank_1_0/backbone/computation_graph.py
```

文件头只有：

```python
from __future__ import annotations
import torch
```

没有：

```python
from sympy import Min
```

但图中出现了：

```python
"bf16[Min(16384, ((s72 + 1)//2)), 2048]"
"Sym(Min(16384, ((s72 + 1)//2)))"
```

这些 `Min(...)` 出现在：

```text
vllm_ascend/ops/mla.py:_resolve_mla_forward_inputs
```

附近对应的图节点里。根因是第二轮为修复 DBO 场景 MLA output contract 而新增的：

```python
hidden_states = hidden_states[:logical_num_tokens]
```

在 **非 DBO compiled 路径** 中引入了符号切片，torch_npu 生成图时把它表达成 `Sym(Min(...))`，而后续执行生成字符串时没有 `Min` 的定义，最终在 `<string>` 中炸成 `NameError`。

### 修改 3：仅在 DBO 路径启用 MLA logical-token 裁剪

文件：

- `vllm_ascend/ops/mla.py`

修改要点：

- `_resolve_mla_forward_inputs(...)` 中，仅当：
  - `flash_comm_v1_enabled`
  - `get_forward_context().dbo_enabled`
  - `_EXTRA_CTX.num_tokens is not None`

同时满足时，才执行：

```python
hidden_states = hidden_states[:logical_num_tokens]
```

原因：

- DBO microbatch 路径确实需要用 logical token 数覆盖 graph padded token 数；
- 非 DBO compile 路径不需要这样做，继续裁剪只会把 `Min(...)` 材料化到生成图里。

### Case E: FC1=1, DBO=0 修复 `Min` 后可成功启动

实验标签：

```text
fc1_nodbo_aot_fixmin_20260701_040451
```

日志：

```text
/data/workspace/logs/fc1_nodbo_aot_fixmin_20260701_040451.log
```

结果：

- `torch.compile took 19.76 s in total`
- 日志不再出现：

```text
name 'Min' is not defined
```

- 同时出现：

```text
Application startup complete.
```

结论：

- `FlashComm1=1, DBO=0` 在修复 `Min` 问题后可以正常启动；
- 因此当前剩余的“起不来”问题不属于 `FlashComm1` 单独路径，而是 `FlashComm1 + DBO` 组合路径。

## 截至当前的结论矩阵

| 组合 | 当前结果 | 备注 |
|---|---|---|
| FC1=0, DBO=1 | 成功启动 | 旧对照成功 |
| FC1=1, DBO=0 | 成功启动 | 修复 `Min is not defined` 后确认 |
| FC1=1, DBO=1 | 仍未完成启动 | 原始 residual/MLA shape bug 已修掉，但仍存在后置同步/等待问题 |

## 第四轮复验：当前代码下 FC1=1, DBO=1 的最新状态

实验标签：

```text
fc1_dbo_aot_recheck_20260701_041058
```

日志：

```text
/data/workspace/logs/fc1_dbo_aot_recheck_20260701_041058.log
```

结果：

- `torch.compile took 14.52 s in total`
- 服务未监听 `8020`
- APIServer 最终报：

```text
RuntimeError: Engine core initialization failed
```

最早明确错误重新落回：

```text
RuntimeError: The expanded size of the tensor (4096) must match the existing size (8)
Target sizes: [4096, 2048]. Tensor sizes: [8, 2048]
```

结论：

- `FlashComm1=1, DBO=1` 在当前代码下仍然失败；
- 失败点不是 `Min is not defined`，而是 DBO 组合路径里 MLA output contract 仍未完全闭合；
- 因此当前可以明确回答：
  - `FlashComm1` 单独开启可以；
  - `FlashComm1 + DBO` 还不可以。
output[...] = o_proj_output[:output.shape[0]]
```

位置：

```text
vllm_ascend/attention/mla_v1.py
```

这说明开发中已经观察到：

- FC1 / sequence-parallel collectives 可能返回比 graph-managed output 更长的 token 维；
- 需要在写回前显式丢弃 communication padding。

但本次复现仍然失败，说明：

1. 还有别的 producer 也在输出 padded local shard；
2. 或者同一 contract 在另一个 branch / fusion pass 中仍未对齐；
3. 或者 residual chunking 与 reduce-scatter 仍存在 ceil/floor 或 padded/unpadded 不一致。

高优先级可疑点：

- `vllm_ascend/attention/mla_v1.py`
- `vllm_ascend/ops/linear_op.py`
- `vllm_ascend/ops/register_custom_ops.py`
- `vllm_ascend/compilation/passes/sequence_parallelism.py`
- `vllm_ascend/compilation/passes/sequence_parallelism_moe.py`

尤其需要检查这些边界是否统一遵守：

```text
local_num_tokens = ceil(global_num_tokens / tp_size)
or
local_num_tokens = unpadded_runtime_tokens
```

不能有的地方用 padded local shard，有的地方用 unpadded sliced residual。

## 与现有代码/历史问题的联系

已有相关文档已经表明这类问题不是孤立事件：

- `testbench/MOE/dbo/rfc/rfc-dbo-flashcomm1-compile-shape-contract.md`
- `testbench/MOE/dbo/rfc/rfc-custom-op-fakeimpl-compile-safe.md`
- `bugs/dbo_torch_compile_analysis.md`

本次新证据补上了一个更具体的 runtime failure：

```text
FC1 cold AOT -> AddRmsNormBias shape mismatch
```

因此现在可以把问题分成两层：

1. compile-safe / fake-impl / runtime-context 边界；
2. FC1 sequence-parallel runtime tensor shape contract。

本次复现直接命中第 2 层。

## 解决方案建议

### 短期 workaround

如果目标是先 unblock 冷启动：

1. `DBO + FC1` 组合下临时关闭 FC1：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
```

2. 或者在该组合下临时关闭 compile/graph capture，回退 eager。

代价很明确：会损失目标优化路径的性能价值，但能先恢复可用性。

### 中期正确修复

需要把 FC1 SP 路径的 shape contract 明文化，并强制所有边界统一：

1. 定义单一 helper 表达 local token 语义：

```text
global real tokens
-> optional communication padding
-> local shard token count
-> residual chunk token count
```

2. 所有 producer 在进入 `npu_add_rms_norm_bias` 前必须输出同一 token contract：

- 要么都用 padded local shard；
- 要么都用 unpadded local shard；
- 不能混用。

3. 对所有 FC1 reduce-scatter / all-gather 相关 custom op：

- runtime impl；
- fake impl；
- compile pass replacement；

统一使用同一个 token 计数函数。

4. 对所有写回 graph-managed buffer 的位置，显式裁剪 communication padding。

### 建议的修复顺序

1. 先审计 `SequenceRowParallelOp` / `MLA o_proj` / MoE finalize 的输出 token 维；
2. 再审计 `maybe_chunk_residual` 与 `maybe_pad_and_reduce_fake` 的 contract；
3. 最后审计 sequence parallel fusion passes 中 `npu_add_rms_norm_bias` 两侧输入是否仍有 padded/unpadded 混用。

## 建议补充的回归测试

至少需要两类测试：

### 1. 冷启动系统回归

在固定 fresh cache 下验证：

- DBO=1, FC1=0 -> 成功；
- DBO=1, FC1=1 -> 成功（修复后目标）；

并且必须检查：

- worker init 完成；
- graph capture 完成；
- API server 真正开始监听。

### 2. shape contract 单测

针对 odd/even token、padding、TP=2：

- reduce-scatter 输出长度；
- residual chunk 长度；
- graph-managed output buffer 长度；
- AddRmsNormBias 两个输入长度；

显式断言相等。

## 本轮执行产物

远端日志：

```text
/data/workspace/logs/fc1_dbo_aot_confirm_20260701_031522.log
/data/workspace/logs/fc0_dbo_aot_20260701_031109.log
```

远端缓存：

```text
/data/workspace/repro_cache/fc1_dbo_aot_confirm_20260701_031522/
/data/workspace/repro_cache/fc0_dbo_aot_20260701_031109/
```

## 最终判断

当前更准确的表述不是“单纯 hang”，而是：

> `DBO + FlashComm1` 在 cold AOT / graph capture 启动路径中会触发 FC1 sequence-parallel shape contract 断裂，最终在 `aclnnAddRmsNormBias` 处以 tiling 失败暴露。

这已经足够指导后续修复：先围绕 FC1 local shard / residual chunk / padding contract 做定点审计，而不是继续把排障重点放在 APIServer、watchdog 或后续 Python 异常上。

## 2026-07-01 新增验证：MLA 挂点并未消失

在修复 `AddRmsNormBias` 残差切分问题，以及修复 `Min is not defined` 之后，重新对 `FC1=1 + DBO=1` 做 fresh-cache 冷启动验证，仍会在 MLA 路径失败。

### 关键观测

最新有效复现日志显示：

```text
RuntimeError: NPUModelRunner init failed, error is NPUModelRunner failed, error is MLA output mismatch:
dbo_enabled=False, fc1=True, forward_num_tokens=16, extra_num_tokens=16,
hidden_states=(8, 2048), output=(4096, 2048), o_proj_output=(8, 2048),
num_actual_tokens=16, num_decode_tokens=16, need_gather_q_kv=True
```

这说明：

1. 失败发生在 `profile_cudagraph_memory()` 的 warmup/profile forward；
2. 虽然整条命令启用了 `--enable-dbo`，但这一拍 forward 的 `forward_context.dbo_enabled=False`；
3. `flashcomm1` 仍然生效，`hidden_states` 已经是 local shard（8 tokens）；
4. 但是 MLA 写回 buffer 仍按 graph-padded shape 分配成了 `4096`；
5. `o_proj_output` 只有真实 local tokens（8），最终在 `output[...] = o_proj_output[:output.shape[0]]` 处暴露。

### 结论更新

当前主挂点已经从最初的 `AddRmsNormBias` 前移/收敛为：

> `FC1 + DBO` 冷启动时，MLA 在非 DBO 的 warmup/profile 子路径里混用了
> “local shard token contract”和“graph-padded output contract”。

更具体地说：

- 输入 `hidden_states`：8（local shard）
- MLA o_proj 输出：8（local shard）
- MLA graph-managed output buffer：4096（graph padded）

这三者 contract 不一致，导致 startup 失败。

### 对本轮修改的评价

本轮尝试在 `vllm_ascend/ops/mla.py::_resolve_mla_forward_inputs()` 中，把 FC1 下的 output token 计数改为优先使用 logical token count 推导 local token count。

但是实际冷启动回归表明：

- 该修改没有真正改变本次失败路径中的 MLA output buffer 大小；
- 最新 fresh-cache 验证仍然复现完全相同的 `4096 vs 8` mismatch。

因此当前 patch 还不足以解决根因。

### 下一步更可能有效的修复方向

1. 不要只在 helper 层修 output token 计数，要把 contract 校正推进到真正分配/返回 MLA output 的边界；
2. 审计 `AscendMultiHeadLatentAttention.forward()` 在 compile/profile 路径里，是否仍然依据 graph shape 分配 `output`；
3. 必要时同时审计 custom op fake impl / compile graph shape propagation，确认 `output` 的 token 维是否在 fake tensor 阶段就被固定成 padded shape；
4. 若上游 graph buffer 无法直接改小，则需要明确在 MLA 返回点把 padded buffer 切回 valid local tokens，再保证下游 residual / MLP / MoE 统一消费同一 contract。
