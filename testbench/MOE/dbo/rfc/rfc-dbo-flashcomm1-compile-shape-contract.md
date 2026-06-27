# RFC: DBO + FlashComm1 + torch.compile 的 ubatch shape contract

## 状态

- 状态：已实现并通过远端回归
- 日期：2026-06-27
- 影响范围：DBO、FlashComm1、torch.compile/ACL graph、TP > 1

## 摘要

DBO 将 batch 切成两个 micro-batch 后，`UBatchSlice.num_tokens` 可能包含
scheduler/graph padding，而 attention metadata 的 `num_actual_tokens` 表示真实逻辑
token。旧实现使用前者初始化 ubatch forward context，导致 FlashComm1 的 padding、
all-gather unpad、RoPE cache slice 和 FakeTensor shape 推导采用了错误长度。

修复统一了以下规则：

1. ubatch 的逻辑 token 数优先取 attention metadata 的 `num_actual_tokens`。
2. 没有 attention metadata 时才回退到 `UBatchSlice.num_tokens`。
3. RoPE cache 按逻辑 token 数切片。
4. reduce-scatter FakeTensor 输出使用 ceil division，与 runtime 的
   pad-to-TP-multiple 行为一致。

## 问题配置

```text
Model: DeepSeek-V2-Lite-Chat
Device: Ascend 910B1
TP: 2
DBO: enabled
FlashComm1: enabled
FlashComm2: disabled
torch.compile / ACL graph: enabled
all2all backend: deepep_low_latency
```

服务参数中不设置 `--enforce-eager`。

## 用户可见现象

服务可以完成模型加载和 graph capture，但首个 DBO prefill 请求失败。错误会沿模型
层逐步变化：

```text
MLA output: 1025 vs 1026
AddRmsNormBias: hidden/residual shape mismatch
InterleaveRope: x batch size differs from cos/sin
EngineDeadError
```

这些错误是同一个 shape contract 错误的级联表现，不是三个独立算子缺陷。

## 根因

### 三种 token 长度

DBO + graph + FlashComm1 同时存在时必须区分：

```text
padded_slice_tokens
    UBatchSlice 覆盖的长度，可能包含 scheduler/graph padding

logical_tokens
    attention metadata 的 num_actual_tokens

collective_tokens
    ceil(logical_tokens / TP) * TP
```

旧代码执行：

```python
new_forward_context.num_tokens = ubatch_slices[ubatch_num].num_tokens
```

这错误地令：

```text
logical_tokens := padded_slice_tokens
```

### 复现中的实际数据

单个 benchmark prompt 经 tokenizer 后为 4103 tokens。DBO 第二个 ubatch 的关键
shape 为：

```text
UBatchSlice/context tokens: 2051
attention/model logical tokens: 2049
TP size: 2
```

错误 context 令 FlashComm1 认为目标长度为 2051，但 dense MLP 实际处理 2049：

```text
layer-0 down_proj input = 2049
旧 FakeTensor floor shape = 2049 // 2 = 1024
部分 runtime 路径 = 1025
错误 context 期望 = ceil(2051 / 2) = 1026
```

下一层 all-gather 只能得到 2050，而 RoPE metadata/cache 按 2051 切片，因此最终
在 `npu_interleave_rope` 报 batch-size mismatch。

### 为什么 eager 模式不稳定复现

eager 模式主要依据 runtime tensor shape 执行。torch.compile 还需要 custom op 的
FakeTensor implementation 提前声明输出 shape。旧 fake 使用 floor division：

```python
num_tokens // tp_size
```

runtime 则会先 pad 到 TP 整数倍，语义为：

```python
ceil(num_tokens / tp_size)
```

奇数 ubatch 因而产生稳定的一 token 偏差。

## 修复设计

### 1. 统一逻辑 token 来源

新增纯函数：

```python
def _get_ubatch_num_tokens(attn_metadata, fallback_num_tokens):
    ...
```

优先读取每个 ubatch attention metadata 的 `num_actual_tokens`。metadata 不存在时
才使用 slice 长度，保持 profiling 或无 attention 路径兼容。

### 2. RoPE 使用相同逻辑长度

RoPE cache 的起点仍来自 ubatch token slice，但终点改为：

```python
token_slice.start + new_forward_context.num_tokens
```

从而保证 q/k tensor 与 cos/sin 的 batch 维相等。

### 3. FakeTensor 模拟 runtime padding

两个 reduce-scatter fake shape 改为：

```python
(num_tokens + tp_size - 1) // tp_size
```

覆盖：

- `maybe_pad_and_reduce`
- `matmul_and_reduce`

FlashComm1 enablement 使用 compile-safe snapshot，避免 fake propagation 在没有
active forward context 时读取 `_EXTRA_CTX`。

## 不采用的方案

### 在 MLA、Residual 或 RoPE 处裁剪

这只能移动报错位置，且可能裁掉真实 token，造成静默精度错误。

### 对缺失 token 补零

只有明确属于 scheduler padding 的位置才能补零。下游算子无法区分 padding 与真实
token，因此不能在 attention 或 linear 输出处猜测。

### 自动关闭 FlashComm1

可作为临时 workaround，但不是修复，并会失去预期性能路径。

## 测试

### 单元测试

```bash
pytest -q tests/ut/test_ascend_forward_context.py
```

结果：

```text
8 passed
```

覆盖 metadata 优先级、fallback 和奇偶 token 的 ceil reduce-scatter shape。

### 远端 fresh compile

测试必须隔离两类缓存：

```bash
export VLLM_CACHE_ROOT=/path/to/new/vllm-cache
--compilation-config \
  '{"cudagraph_capture_sizes":[16],"cache_dir":"/path/to/new/compile-cache"}'
```

仅设置 `compilation_config.cache_dir` 不够；顶层 AOT artifact 位于
`VLLM_CACHE_ROOT/torch_compile_cache/torch_aot_compile`。

验证矩阵：

| 输入 | 并发 | 请求数 | 结果 |
|---:|---:|---:|---|
| 4096 | 1 | 1 | 1 成功，0 失败 |
| 4096 | 16 | 32 | 32 成功，0 失败 |
| 4096 | 96 | 500 | 500 成功，0 失败 |

完整压力回归：

```text
Total input tokens: 2,051,936
Total generated tokens: 8,000
Successful requests: 500
Failed requests: 0
```

## 兼容性

- 无 metadata 路径继续使用原 slice 长度。
- 偶数 token 的 ceil division 与原 floor division 结果相同。
- eager、非 DBO、FlashComm1 关闭路径不改变。
- 不增加环境变量。

## 后续工作

1. 将 4103-token 的 odd-ubatch 场景加入 NPU nightly。
2. custom unpad op 可增加 debug-only assertion，禁止输入短于目标长度时静默返回。
3. DBO graph cache key 当前假设两个 ubatch 等长，建议单独 RFC 修正。
