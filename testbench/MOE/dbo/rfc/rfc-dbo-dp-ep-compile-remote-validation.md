# RFC: DBO + DP/EP + torch.compile 远端验证与边界

## 状态

- 日期：2026-06-28
- 环境：2 × Ascend 910B3
- 模型：DeepSeek-V2-Lite-Chat
- 结论：两卡环境不能构造 `DP=2 + TP>1 + FlashComm1`；发现并绕过一个上游 DeepEP 平台兼容问题，但 DP=2 请求期仍出现阻塞，尚不能宣称 DP+EP+DBO 已正确支持。

## 验证目标

验证 TP shape-contract 修复是否需要扩展到 DP+EP，并实际运行：

```text
TP=1
DP=2
EP=2
DBO=on
torch.compile/ACL graph=on
```

对应启动脚本：

```text
testbench/MOE/dbo/demos/deepseek-v2-dbo-server-dp.sh
```

## 结论摘要

### 1. FlashComm1 的 DP-only 配置不可达

当 `TP=1, DP=2, FlashComm1=on` 时，配置校验直接退出：

```text
Assertion failed, Flash Comm v1 is only supported when tp_size > 1.
```

compile 的 sequence-parallel pass 也会在 `TP=1` 时被上游自动关闭。因此两卡机器不能验证真正的
`DP>1 + TP>1 + FlashComm1`，该矩阵至少需要四卡：

```text
TP=2, DP=2, EP=4
```

不能通过删除 `TP>1` 断言来“修复”，因为 FlashComm1 的 dense TP all-gather/reduce-scatter 本身依赖 TP group。

### 2. DP+EP+DBO 启动存在独立的 DeepEP 平台兼容问题

即使关闭 FlashComm1，worker 在模型构造期仍失败：

```text
NameError: name 'DeepEPLLPrepareAndFinalize' is not defined
```

调用链：

```text
FusedMoE
  -> maybe_roundup_layer_hidden_size
  -> DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size
```

上游 `all2all_utils.py` 只在 `current_platform.is_cuda_alike()` 时导入
`DeepEPLLPrepareAndFinalize`。vLLM-Ascend 为通过上游 DBO backend 校验，会把 A2 DBO 的 backend
字段改成 `deepep_low_latency`，从而令上游误判应执行 DeepEP-LL 专属 roundup。

实验性兼容修复：

```python
if (
    current_platform.is_cuda_alike()
    and moe_parallel_config.use_deepep_ll_kernels
):
    hidden_size = (
        DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(hidden_size)
    )
```

该修复只跳过 NPU 上不可用的 CUDA DeepEP shape roundup，不改变 Ascend A2 的实际 MoE 通信。

### 3. 修复启动阻塞后，compile 和 graph capture 成功

关键结果：

```text
tensor_parallel_size=1
data_parallel_size=2
EP ranks=2
enforce_eager=False
torch.compile: 17.19 s / 17.70 s
graph capture: 209 s
API server: 2/2 startup complete
```

说明 DP=2 的模型加载、torch.compile 和 ACL graph capture 本身可以完成。

### 4. 实际 4K 请求出现请求期阻塞

测试参数：

```text
INPUT_LEN=4096
OUTPUT_LEN=16
NUM_PROMPTS=4
MAX_CONCURRENCY=2
```

现象：

- API 返回 HTTP 200 响应头，但生成流不结束；
- benchmark 超过 60 秒无结果；
- 两个 EngineCore 连续报告没有可用 shared-memory broadcast block；
- 两张 NPU 的 AICore 利用率为 0；
- 未出现 shape tiling、AICPU 或 device-side error。

该现象更像 DP rank 间的 EP collective 参与/调度不一致，而不是 TP RFC 中的 fake shape floor-div 问题。
尤其是 benchmark 会先发送单请求探测；内部 DP 只把请求交给一个 replica，而 EP group 跨两个 DP rank，
需要确认空闲 rank 是否执行了匹配的 dummy batch/collective。

当前证据不足以把死锁根因定死，后续应记录每个 DP/EP rank 的 collective 序列和 DBO ubatch 生命周期。

## 与 TP shape-contract RFC 的关系

TP bug 的本质是 fake reduce-scatter 用 floor division，而 runtime 在 odd token 下执行 pad + ceil division。

DP+EP 不能直接复用 `TP size` 推导所有通信 shape：

```text
TP collective: group size = TP
EP collective: group size = EP = TP * DP（常见配置）
```

因此 generic fake implementation 若在 `is_ep_comm=True` 时仍固定使用
`get_tensor_model_parallel_world_size()`，理论上存在 fake/runtime shape 不一致风险。

但 DBO 的 MoE prepare/finalize 当前使用更明确的接口：

- all-gather 后调用 `maybe_unpad_after_all_gather(unpadded_length, padded_length, use_ep_comm)`；
- reduce-scatter 前调用 `maybe_prepare_for_reduce(prepared_length, padded_length, use_ep_comm)`。

这两个 fake implementation 直接返回显式长度，比 generic
`maybe_all_gather_and_maybe_unpad` / `maybe_pad_and_reduce` 更可靠。`# TODO: do unpad`
位于 generic `enable_sp_by_pass()` 路径，不能仅凭该 TODO 判断 DBO 路径已经发生同一 bug。

## 建议的统一修复方向

1. 保留 `FlashComm1 requires TP>1` 的配置保护。
2. 修复上游 DeepEP roundup 的平台条件，使“导入条件”和“使用条件”一致，并补 NPU out-of-tree regression test。
3. generic fake implementation 根据通信域选择 group size：

   ```text
   is_ep_comm=False -> TP group size
   is_ep_comm=True  -> EP group size
   ```

4. runtime 和 fake 共同使用显式 shape helper，统一 all-gather、reduce-scatter、padding 和 unpadding契约。
5. 增加四卡回归矩阵：

   ```text
   TP=2, DP=2, EP=4
   DBO on/off
   FlashComm1 on/off
   compile/eager
   odd/even per-rank token counts
   单请求和并发请求
   ```

6. 在请求级测试中校验输出 token 与 eager baseline，而不只检查服务启动和 HTTP 200。

## 日志

```text
/data/workspace/logs/deepseek-v2-dbo-server-dp-compile.log
/data/workspace/logs/deepseek-v2-dbo-server-dp-baseline.log
/data/workspace/logs/deepseek-v2-dbo-server-dp-a2.log
/data/workspace/logs/deepseek-v2-dbo-server-dp-naive.log
/data/workspace/logs/deepseek-v2-dbo-server-dp-compat-fixed.log
/data/workspace/logs/deepseek-v2-dbo-test-dp2-compile.log
```
