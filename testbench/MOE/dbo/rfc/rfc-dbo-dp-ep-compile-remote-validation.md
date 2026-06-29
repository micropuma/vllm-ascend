# RFC: DBO + DP/EP + torch.compile 的全局 ubatch 契约

## 状态

- 日期：2026-06-28（更新 2026-06-29）
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

vLLM-Ascend 为满足上游 DBO backend 校验，会在 A2 上把 `all2all_backend` 标记为 `deepep_low_latency`：

```python
# vllm_ascend/platform.py:684-691
if not vllm_config.compilation_config.pass_config.enable_sp:
    if parallel_config.enable_dbo:
        parallel_config.all2all_backend = "deepep_low_latency"
```

注意外层的 `if not ... enable_sp` 守卫。TP>1 时通常 FlashComm1 被开启，`enable_sp=True` → 整个块跳过，不设置 `all2all_backend`。**这就是 TP=2 场景不会触发本 bug 的原因**。

实际 NPU 通信走 Ascend 的 AllGather/ReduceScatter，这个 backend 值只是兼容上游校验。

上游 `all2all_utils.py` 中，`DeepEPLLPrepareAndFinalize` 的 import 和使用之间存在 platform guard 不一致：

- **import**（第 36 行）：`if current_platform.is_cuda_alike():` 守卫
- **使用**（第 92 行）：`if moe_parallel_config.use_deepep_ll_kernels:` **没有**平台检查

而 `use_deepep_ll_kernels` 的定义为：

```python
# vllm/model_executor/layers/fused_moe/config.py:1034-1046
@property
def use_all2all_kernels(self):
    return self.dp_size > 1 and self.use_ep

@property
def use_deepep_ll_kernels(self):
    return self.use_all2all_kernels and self.all2all_backend == "deepep_low_latency"
```

两个条件必须**同时满足**：

| 配置 | dp_size | use_ep | `use_all2all_kernels` | `all2all_backend` | `use_deepep_ll_kernels` | 触发? |
|------|---------|--------|----------------------|-------------------|------------------------|-------|
| TP=2, EP, SP 开 | - | - | - | 未设置（SP 跳过） | - | 否 |
| TP=2, EP, SP 关 | 1 | True | **False** | `"deepep_low_latency"` | **False** | 否 |
| **DP=2, EP=2** | **2** | True | **True** | `"deepep_low_latency"` | **True** | **是** |

这就是为什么**只有 DP 会触发 DeepEP 未导入**：`dp_size > 1` 是触发前置条件。

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

上游不应让 backend 字符串同时承担"通过 DBO 校验"和"选择 CUDA kernel"两种语义。更稳妥的方案是：

- DBO 校验允许 out-of-tree 平台声明自己的通信实现；
- kernel capability 由 platform capability 判断；
- DeepEP 类的导入和所有引用使用同一 capability guard；
- 增加 NPU/out-of-tree + DBO + EP 的模型初始化测试。

## 2. DP+DBO 到底出了什么 bug

### 2.1 单请求下为什么需要 idle rank

DP=2 时请求通常只被一个 DP replica 接收，但 EP group 跨两个 DP rank。active rank 执行 MoE EP collective 时，idle rank 必须通过 `execute_dummy_batch()` 执行匹配 collective，否则 active rank 永远等待。

`BalanceDPEngineCoreProc.run_busy_loop()` 的调度逻辑（`vllm_ascend/patch/platform/patch_balance_schedule.py:622-643`）：

```python
# 各 DP rank 独立拉取请求
self._process_input_queue()
executed = self._process_engine_step()

if not executed:
    if not local_unfinished_reqs and not self.engines_running:
        continue  # 都没活，跳过
    # 有别的 rank 在跑，但我没请求 → 执行 dummy batch
    self.execute_dummy_batch()
```

dummy batch 使用 `decode_token_per_req = 1`（`vllm_ascend/worker/worker.py:989`）：

```python
def execute_dummy_batch(self) -> None:
    self.model_runner._dummy_run(
        num_tokens=self.model_runner.decode_token_per_req,  # = 1
        uniform_decode=True
    )
```

### 2.2 根因：调用方传错参数给 `check_enable_ubatch`

`check_enable_ubatch` 本身签名是**正确**的（`vllm_ascend/worker/ubatch_utils.py:68-93`，自 commit `1be49998` 引入）：

```python
def check_enable_ubatch(
    num_tokens_unpadded: int,  # 用于阈值判断（eligibility）
    num_tokens_padded: int,    # 用于 is_last_ubatch_empty 检查
    ...
)
```

其中 `is_last_ubatch_empty` 检查 padding 后是否产生空 ubatch：

```python
def is_last_ubatch_empty(orig: int, padded: int, num_ubatches: int = 2) -> bool:
    return (padded // num_ubatches) * (num_ubatches - 1) >= orig
```

**但调用方在 commit `ccad1e77` 首次引入时传错了**：

```python
# 旧代码（ccad1e77）—— 两个参数都传 num_tokens_padded
should_ubatch = check_enable_ubatch(
    num_tokens_padded,    # ← 本应是 num_tokens（逻辑值）
    num_tokens_padded,    # ← 正确
    ...
)
```

在 graph 模式下 DP1 的 `num_tokens_padded` 已被 padding 到 4104，两个参数相同时检查被绕过：

```
is_last_ubatch_empty(orig=4104, padded=4104)
  → (4104 // 2) * 1 = 2052 >= 4104? → False → 没拦住
```

若传正确参数 `is_last_ubatch_empty(orig=1, padded=4104)` → `2052 >= 1` → True → 拦住。

**总结：`is_last_ubatch_empty` 检查本身没有 bug，是输入被污染了。**

### 2.3 eager 数据流

Eager 模式 `_sync_metadata_across_dp` 使用 `allow_dp_padding=False`，各 rank **保持自己的 token 数**：

```
                         DP0 (active)          DP1 (idle)
                         ────────────          ──────────
num_tokens:                  4103                   1

① SP padding:
   num_tokens_padded         4104                   1    (TP=1, 不变)

② cudagraph dispatch:
   mode                      NONE                  NONE
   batch_desc.num_tokens     4104                   1

③ DP sync (allow_dp_padding=False):
   num_tokens_padded         4104                   1    ← 各保持自己的

④ OLD check_enable_ubatch(4104, 4104)        check_enable_ubatch(1, 1)
   → True (4104 > 512)                        → False (1 < 32)

   结果: DP0 走 2 ubatch, DP1 走 1 batch → collective 数量不一致 → 死锁
```

### 2.4 compile/graph 数据流

Graph 模式 `allow_dp_padding=True`，**所有 rank padding 到 max**：

```
                         DP0 (active)          DP1 (idle)
                         ────────────          ──────────
num_tokens:                  4103                   1

① SP padding:
   num_tokens_padded         4104                   1

② cudagraph dispatch:
   mode                      FULL                  FULL
   batch_desc.num_tokens     4104                   1

③ DP sync (allow_dp_padding=True):
   max across DP = 4104
   num_tokens_padded         4104                 4104    ← DP1 被padding了!

④ OLD check_enable_ubatch(4104, 4104)        check_enable_ubatch(4104, 4104)
   → True                                     → True  ← BUG!

⑤ maybe_create_ubatch_slices:
   正常创建 2 ubatch          split_point=2052, num_scheduled_tokens=[1]
                              ubatch0: token(0,2052) ✓
                              ubatch1: token(2052,2052) ← 空的!
                              → AssertionError: Shape: 0
```

Graph 模式要求所有 rank batch size 一致才能正确 capture/replay，因此 DP sync 的 padding 必须在 batch execution 决策之前。**不能通过调顺序来绕过**。

### 2.5 为什么之前只看日志容易误判

- 没有 AICPU/device error；
- NPU AICore 利用率降为 0；
- EngineCore 只报告 60 秒无 shared-memory broadcast block；
- ubatch 子线程异常没有立即终止主请求；
- HTTP 200 只代表响应流建立，不代表 token 生成完成。

## 3. 如何修改

commit: `1ffe0850`，只改一个文件 `vllm_ascend/worker/model_runner_v1.py`（+29 / -4 行）

### 3.1 eligibility 使用逻辑 token

文件：`vllm_ascend/worker/model_runner_v1.py:2896-2901`

```python
# 旧代码（commit ccad1e77）
should_ubatch = check_enable_ubatch(
    num_tokens_padded,    # BUG: 应该传逻辑 token
    num_tokens_padded,
    ...
)

# 修复后
should_ubatch = check_enable_ubatch(
    num_tokens,           # 本 rank 逻辑 token 数，用于阈值判断
    num_tokens_padded,    # 只用于 is_last_ubatch_empty 检查
    uniform_decode=False,
    vllm_config=self.vllm_config,
    moe_comm_type=select_moe_comm_method(num_tokens_padded, self.vllm_config),
)
```

### 3.2 在 DP group 上统一 should_ubatch

文件：`vllm_ascend/worker/model_runner_v1.py:2903-2917`

```python
if (
    self.parallel_config.data_parallel_size > 1
    and self.parallel_config.enable_dbo
    and not should_skip_allreduce_across_dp_group(self.vllm_config, False)
):
    device, group = (
        ("npu", get_dp_group().device_group)
        if self.ascend_config.dp_allreduce_on_npu
        else ("cpu", get_dp_group().cpu_group)
    )
    ubatch_flag = torch.tensor(
        [int(should_ubatch)], device=device, dtype=torch.int32
    )
    dist.all_reduce(ubatch_flag, op=dist.ReduceOp.MIN, group=group)
    should_ubatch = bool(ubatch_flag.item())
```

**CPU/NPU 通信机制说明**：

vLLM 初始化 DP group 时会创建**两套 process group**：
- `device_group`（HCCL backend）：走 NPU 互联（HCCS），数据在 NPU 上
- `cpu_group`（gloo backend）：走 CPU 网络（TCP/RoCE），数据在 CPU 内存

默认 `dp_allreduce_on_npu=False`，使用 CPU group。每个进程在各自 CPU 上分配 tensor，gloo backend 通过网络完成 all-reduce，全程不经过 NPU，开销极小（只传一个 int32）。NPU 选项仅作为某些平台 gloo 有 bug 时的 fallback。

`ReduceOp.MIN` 的语义：

| 场景 | DP0 投票 | DP1 投票 | MIN 结果 | 全局行为 |
|------|---------|---------|---------|---------|
| 单请求（idle + active） | 1 (True) | 0 (False) | 0 (False) | 统一关闭 DBO |
| 双并发（两个大 prefill） | 1 (True) | 1 (True) | 1 (True) | 统一开启 DBO |

### 3.3 dummy run 保留同步结果

文件：`vllm_ascend/worker/model_runner_v1.py:3365, 3405-3411, 3550-3554`

**解包 `should_ubatch`**（第 3365 行）：

```python
# 旧代码
_cudagraph_mode, batch_desc, _, num_tokens_across_dp, _ = ...

# 修复后
_cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = ...
```

**创建 ubatch slices**（第 3405-3411 行，替换旧的 `None, None`）：

```python
# 旧代码
# vllm-ascend does not support ubatch now
ubatch_slices, ubatch_slices_padded = None, None

# 修复后
ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
    should_ubatch,
    num_scheduled_tokens,
    num_tokens_padded,
    num_reqs_padded,
    self.parallel_config.num_ubatches,
)
```

**传入 `set_ascend_forward_context`**（第 3550-3554 行）：

```python
ubatch_slices=(
    ubatch_slices_padded
    if cudagraph_runtime_mode == CUDAGraphMode.FULL
    else ubatch_slices
),
```

正常 inference 路径（第 2075-2081 行）已有相同逻辑，dummy 路径现在与之一致。

### 3.4 后续优化

#### 3.4.1 合并 all-reduce（可选方案）

当前实现保留独立 DP MIN all-reduce。理论上可以合并进 `_sync_metadata_across_dp`，但**无法简单前置** `check_enable_ubatch`——因为 `is_last_ubatch_empty` 依赖 DP sync **之后**的真实 padded 值，pre-sync 时小 side rank 不知道自己的 padded 会被拉多大。

反例：DP0 有 8000 token，DP1 有 3000 token，graph 模式下 DP1 pre-sync padded=3000 → `is_last_ubatch_empty(3000, 3000)` = False（认为合法），但 sync 后 DP1 padded 被拉到 8000 → `is_last_ubatch_empty(3000, 8000)` = True（实际会产生空 ubatch）。两个 rank pre-sync 判断都通过了，MIN 无法拯救。

若要合并，需将 `num_tokens_unpadded` 也打入 packed tensor，all-reduce 后各 rank 拿到所有人的 padded 和 unpadded，**本地**完成 `is_last_ubatch_empty` 判断。方案如下：

```python
# _sync_metadata_across_dp 改造
packed_tensor = torch.zeros(3, self.dp_size, device=device_str, dtype=torch.int32)
packed_tensor[0][self.dp_rank] = num_tokens_padded      # 已有
packed_tensor[1][self.dp_rank] = cudagraph_mode.value   # 已有
packed_tensor[2][self.dp_rank] = num_tokens_unpadded    # 新增：各 rank 逻辑 token
dist.all_reduce(packed_tensor, group=group)

# all-reduce 后各 rank 独立执行
def _compute_should_ubatch(packed_tensor, vllm_config):
    padded_all   = packed_tensor[0, :]   # 各 rank post-sync padded
    unpadded_all = packed_tensor[2, :]   # 各 rank 逻辑 token
    for r in range(dp_size):
        if not check_ubatch_thresholds(parallel_config, int(unpadded_all[r])):
            return False
        if is_last_ubatch_empty(int(unpadded_all[r]), int(padded_all[r])):
            return False
    return True
```

**为什么不合并？** 一次 4 字节标量 all-reduce 延迟为微秒级，相比每步 forward 百兆级 EP/DP 通信可忽略。合并会使 `_sync_metadata_across_dp` 从通用元数据同步函数变为"元数据 + DBO 决策"的混合体，违反单一职责，后续改 DBO 逻辑影响面更大。当前独立实现清晰、不耦合，**保留现状即可**。

#### 3.4.2 异常传播

ubatch 子线程异常应立即传播到主线程，避免再次表现为无诊断信息的请求挂起。

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

双并发 `input=8206` ≈ 每个 DP rank ~4103 token。两个 rank 都 `should_ubatch=True` → MIN(1, 1) = 1 → 统一开启 DBO，验证了大 batch 正确路径。

注意：当前验证关闭了 decode DBO（`DBO_DECODE_TOKEN_THRESHOLD=1000000000`），只覆盖了 prefill DBO。后续如需开启 decode DBO，建议补充"一个 rank decode（小 token）、另一个 rank prefill（大 token）"的混合阶段测试。

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

## 附录 A：相关 commit 历史

| commit | 说明 |
|--------|------|
| `1be49998` | 迁移 DBO 到 upstream，引入 `check_enable_ubatch`（签名正确） |
| `ccad1e77` | 首次调用 `check_enable_ubatch`，**调用方传错参数**（两个都传 `num_tokens_padded`） |
| `1ffe0850` | **修复**：改第一个参数为 `num_tokens`，新增 DP all-reduce，dummy run 保留 ubatch |
| `5b7bc85d` | 本文档及 DP=2 复现脚本 |

## 附录 B：关键代码位置索引

| 描述 | 文件 | 行号 |
|------|------|------|
| `all2all_backend` 设置 | `vllm_ascend/platform.py` | 684-691 |
| DeepEP import guard | `vllm/model_executor/layers/fused_moe/all2all_utils.py` | 36-41, 92-94 |
| `use_all2all_kernels` | `vllm/model_executor/layers/fused_moe/config.py` | 1034-1046 |
| `check_enable_ubatch` | `vllm_ascend/worker/ubatch_utils.py` | 68-93 |
| `is_last_ubatch_empty` | `vllm_ascend/worker/ubatch_utils.py` | 13-14 |
| `maybe_create_ubatch_slices` | `vllm_ascend/worker/ubatch_utils.py` | 96-118 |
| `_determine_batch_execution_and_padding` | `vllm_ascend/worker/model_runner_v1.py` | 2814-2933 |
| `_sync_metadata_across_dp` | `vllm_ascend/worker/model_runner_v1.py` | 626-674 |
| `_dummy_run` ubatch 恢复 | `vllm_ascend/worker/model_runner_v1.py` | 3365, 3405-3411, 3550-3554 |
| `execute_dummy_batch` | `vllm_ascend/worker/worker.py` | 987-989 |
| `dp_allreduce_on_npu` 配置 | `vllm_ascend/ascend_config.py` | 271 |
| DP engine busy loop | `vllm_ascend/patch/platform/patch_balance_schedule.py` | 622-643 |
