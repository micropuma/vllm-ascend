# RFC: DBO + FlashComm1 compiled runtime 可重入性与 MLA 动态 shape contract

## 状态

- 状态：Draft / root cause localized and corrected
- 日期：2026-07-02（原始），2026-07-03（本修正版）
- vLLM commit：`7977da779ad8a2725a0ad58ac167af0a2278964a`
- vLLM Ascend commit：`fcb84f5c223d9d4559ced6bca30f30d9d2939b98`
- 模型：`/data/models/DeepSeek-V2-Lite-Chat`
- 硬件：2 x Ascend 910B3，TP=2，EP=2

## 摘要（2026-07-03 重大修正）

> **本修正版覆盖 2026-07-02 原版 RFC 的关键误判。原版将问题定性为"两个线程并发复用同一
> AOT compiled callable 导致 HCCL 死锁"。实际故障机制完全不同：**
>
> 1. **不是 HCCL 死锁，是 `AssertionError` 被异常处理吞没**：
>    ubatch1 在 down_proj reduce-scatter 前抛出
>    `assert input_tensor.shape[0] % world_size == 0`。
>    `AscendUBatchContext.__exit__()` 仍 signal ubatch0，异常被线程
>    wrapper 暂存、不立即上报。ubatch0 继续执行并进入等待，主线程卡在
>    `thread.join()`，原始 shape 异常被掩盖成"死锁"。
>
> 2. **shape 错误的根因不是 compiled reentrancy，是 ubatch slicing contract**：
>    总 token=4103，SP 对齐后应为 4104，正确 DBO 切分是 2052+2052。
>    Ascend 非 FULL graph 使用 `ubatch_slices_attn`（未 padding），实际
>    切分为 2052+2051。ubatch1 的 2051 无法 TP=2 reduce-scatter。
>
> 3. **为什么不能简单切换到 `ubatch_slices_padded`**：
>    Ascend FlashComm/SP 算子内部会依据 `forward_context.pad_size` 自行
>    padding（`linear_op.py:582-585`、`register_custom_ops.py:41-43` 等）。
>    若 wrapper 已切 padded 输入，会二次 padding（2052→2053），再次断言。
>
> 4. **为什么修了 `num_tokens`/`pad_size` 计算仍然不够**：
>    即使 `create_ascend_forward_context` 正确算出 per-ubatch 的
>    `pad_size`（ubatch0=0, ubatch1=1），compiled runtime 中 ubatch1
>    仍未执行预期 padding。当前证据只能把边界收敛到两种机制：
>    `_EXTRA_CTX.pad_size` 在 trace 时被常量折叠，或全局 forward context
>    被另一 ubatch 线程覆盖。必须用 `(tensor rows, context id, pad_size,
>    thread id)` 的同点 trace 区分，不能提前把其中一种写成既定根因。

本 RFC 以下章节做完整重写，用实际远端 trace 证据替换原版的推测性分析。

---

## 1. 真实故障链路（2026-07-02 远端 trace 还原）

### 1.1 时序

```text
Request: 4096 input tokens, 1 output token
→ total_num_scheduled_tokens = 4103 (after tokenization/embedding)
→ SP padding: 4103 → 4104 (TP=2 alignment)
→ ubatch_slices_padded:  [0:2052, 2052:4104]  → 2052 + 2052
→ ubatch_slices (unpadded): [0:2052, 2052:4103] → 2052 + 2051
→ pad_attn = (CUDAGraphMode == FULL) = False (DBO runtime uses NONE)
→ ubatch_slices_attn = ubatch_slices (unpadded)  ← line 2141
→ set_ascend_forward_context(..., ubatch_slices=ubatch_slices_attn)  ← line 2234
→ DBO wrapper reads forward_context.ubatch_slices → splits as 2052 + 2051
→ ubatch1: 2051 tokens → TP=2 reduce-scatter → AssertionError
```

### 1.2 关键代码路径

**Step 1**: `model_runner_v1.py:2141` — `ubatch_slices_attn` 在非 FULL 模式下是未 padding 的：

```python
# model_runner_v1.py:2083
pad_attn = cudagraph_mode == CUDAGraphMode.FULL
# model_runner_v1.py:2141
ubatch_slices_attn = ubatch_slices_padded if pad_attn else ubatch_slices
```

**Step 2**: `model_runner_v1.py:2234` — Ascend forward context 拿到未 padding 的 slices：

```python
set_ascend_forward_context(
    ...
    ubatch_slices=ubatch_slices_attn,   # ← 非 FULL 时未 padding
)
```

**Step 3**: `npu_ubatch_wrapper.py:371` — DBO wrapper 从 forward_context 读取 slices 做切分：

```python
ubatch_slices = forward_context.ubatch_slices  # 2052 + 2051
```

**Step 4**: `linear_op.py:597-604` — ubatch1 的 down_proj reduce-scatter 断言失败：

```python
# ubatch1 input: [2051, hidden_dim]
output = torch_npu.npu_mm_reduce_scatter_base(
    x,           # shape [2051, hidden_dim]
    weight,
    hcom_name,
    world_size,  # 2
    ...
)
# → 内部 assert: 2051 % 2 != 0 → AssertionError
```

### 1.3 为什么呈现为"死锁"

关键代码在 `npu_ubatch_wrapper.py:212-258`：

```python
def _ubatch_thread(results, model, ubatch_metadata):
    try:
        with ubatch_metadata.context:
            model_output = model(...)
            dbo_current_stream().synchronize()
            dbo_yield()
        results.append((ubatch_metadata.context.id, model_output))
    except Exception as e:
        results.append(e)  # ← 异常被暂存，不立即 raise

# 主线程:
for thread in ubatch_threads:
    thread.join()            # ← 永久等待 ubatch1
_check_thread_exceptions(results, "Ubatch runtime")  # ← 从未执行到
```

`AscendUBatchContext.__exit__()`（`ubatching.py:81-88`）在异常时不返回 True（即不抑制异常），但会：
1. `cpu_signal_event.set()` → 唤醒 ubatch0
2. ubatch0 继续执行并进入下一个 collective wait
3. 异常存储在 `results` list 中，但主线程卡在 `join()`

因此两个 rank 完全一致地"卡住"，看起来像是 HCCL 死锁，实际是同一条 shape 异常。

---

## 2. 为什么不能简单切换到 `ubatch_slices_padded`

### 2.1 第一次验证：直接切 padded slices → 二次 padding

```diff
- ubatch_slices=ubatch_slices_attn,
+ ubatch_slices=ubatch_slices_padded,
```

结果：ubatch0 从 2052 变成 2053，仍然是奇数，down_proj 再次断言。

### 2.2 二次 padding 的机制

Ascend SP 算子在多处读取 `_EXTRA_CTX.pad_size` 并自行 padding：

| 文件 | 行号 | 作用 |
|---|---|---|
| `linear_op.py:582-585` | row linear 前 pad 输入 |
| `register_custom_ops.py:41-43` | AddRmsNormBias 前 pad residual |
| `register_custom_ops.py:132-134` | embedding/row-linear 后 pad |
| `linear_op.py:327` | embedding padding |

`pad_size` 的计算在 `create_ascend_forward_context`（line 282）：
```python
pad_size = (tp_world_size - (new_forward_context.num_tokens % tp_world_size)) % tp_world_size
```

若 `num_tokens` 来自 padded slices（2052），pad_size=0 → 不触发 double-padding。
若 `num_tokens` 仍从 attention metadata 读取（值为 4104 总量），则 `(2 - 4104%2) % 2 = 0`，也不触发。

但若 `num_tokens` 的计算路径不一致（见 §3），则可能得到错误的 pad_size。

---

## 3. `num_tokens`/`pad_size` 的计算问题

### 3.1 当前代码

```python
# ascend_forward_context.py:273-275
new_forward_context.num_tokens = _get_actual_num_tokens(
    attn_metadata, ubatch_slices[ubatch_num].num_tokens
)
```

```python
# ascend_forward_context.py:210-226
def _get_actual_num_tokens(attn_metadata, fallback_num_tokens):
    if attn_metadata is None:
        return fallback_num_tokens
    for metadata in ...:
        num_actual_tokens = getattr(metadata, "num_actual_tokens", None)
        if num_actual_tokens is not None:
            return num_actual_tokens   # ← 无条件返回 attn_metadata 的值
    return fallback_num_tokens
```

### 3.2 问题

`_get_actual_num_tokens` 无条件优先返回 `attn_metadata.num_actual_tokens`。
在 compiled（非 FULL）路径中，attn_metadata 是 traced/captured 的，其
`num_actual_tokens` 可能是：
- graph capture 时的 warmup token 数（如 16）
- 整个 batch 的 token 总数（如 4103）
- 而不是当前 ubatch 的逻辑 token 数

当 `num_actual_tokens` 是 total batch 的 4103 时：
- ubatch1: `num_tokens = 4103`, `pad_size = (2 - 4103 % 2) % 2 = 1`
- 这恰好对了（ubatch1 确实需要 pad 1），但只是巧合。
- ubatch0: `num_tokens = 4103`, `pad_size = 1`
- 但 ubatch0 实际 2052 不需要 padding → `pad_size=1` 导致错误的多 pad

当 `num_actual_tokens` 是 graph warmup 的 16 时：
- 两个 ubatch: `pad_size = (2 - 16 % 2) % 2 = 0`
- ubatch1 实际需要 pad_size=1，被错误地设为 0 → reduce-scatter 断言失败

### 3.3 修复方向

```python
slice_num_tokens = ubatch_slices[ubatch_num].num_tokens
if cudagraph_runtime_mode is CUDAGraphMode.FULL:
    # FULL graph: graph 内 token 数是固定的，从 attn_metadata 读取
    new_forward_context.num_tokens = _get_actual_num_tokens(
        attn_metadata, slice_num_tokens
    )
else:
    # 非 FULL: 直接使用当前 ubatch slice 的逻辑 token 数
    new_forward_context.num_tokens = slice_num_tokens
```

但这只是第一步，还不足以完全修复（见 §4）。

---

## 4. 待验证边界：compiled 常量折叠或 forward context 串扰

### 4.1 问题机制

即使 `create_ascend_forward_context` 正确地为 ubatch1 设置了 `pad_size=1`，
compiled custom-op 执行时仍然读到 `pad_size=0`。原因：

```python
# linear_op.py:582-585 (custom op 内部)
pad_size = _EXTRA_CTX.pad_size   # ← Python int，读自 forward_context
if pad_size > 0 and not dsa_cp_attn_out:
    x = F.pad(x, (0, 0, 0, pad_size))
```

当 `torch.compile` / Dynamo  trace 这段代码时：
1. `_EXTRA_CTX.pad_size` 在 trace 时求值为一个 Python int
2. Dynamo 将其作为常量折叠进 FX graph
3. `if pad_size > 0` 在 trace 时求值，条件分支被静态消除
4. 若 ubatch0 先被 trace（pad_size=0），padding 分支被完全跳过
5. ubatch1 重放同一 compiled graph 时，padding 分支不存在
6. ubatch1 的 2051 tokens 未经 padding 直接进入 reduce-scatter → 断言失败

同样的问题影响所有通过 `_EXTRA_CTX` 读取 `pad_size` 和 `num_tokens` 的 custom op：
- `register_custom_ops.py:41-43, 64-66, 132-134, 198-200`
- `linear_op.py:327, 492, 582-585`
- `prepare_finalize.py:164-168, 191-193, 400`

### 4.2 远端验证证据

第二次验证修改了 `create_ascend_forward_context` 的 `num_tokens` 计算逻辑
（仅在 FULL 模式读 attn_metadata），但服务日志仍显示 ubatch1 的 pad_size
在 compiled runtime 没有生效：

```text
# 日志显示 ubatch1 的 down_proj 输入仍是 2051 tokens，未被 pad 到 2052
# 说明 compiled graph 中的 pad_size 仍为 0
```

这证明问题不只在 `create_ascend_forward_context` 的计算，但尚不能单独证明
compiled graph 常量折叠。还需排除两个 ubatch 线程共享/覆盖 forward context。

### 4.3 与 upstream 的差异

Upstream GPU 的 ubatch wrapper 给 DBO 传 `ubatch_slices_padded`，确保两个 ubatch
的 token 数都已经是 TP 对齐的，因此 custom op 内部不需要再根据 `pad_size` 做
额外 padding。

Ascend 不能直接模仿，因为：
- Ascend SP 算子在 embedding/row-linear 等入口处依赖 `pad_size` 做 per-op padding
- 如果输入已经 padded 且 `pad_size > 0`，就会二次 padding
- 需要要么让 `pad_size=0`（传给 padded slices 时），要么让 `pad_size` 在 compiled
  graph 中保持动态

---

## 5. 根因总结（修正版）

```
┌─────────────────────────────────────────────────────────────────┐
│                    真实故障链路                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  4103 tokens, TP=2                                               │
│       │                                                          │
│       ▼                                                          │
│  ubatch_slices_attn (unpadded, non-FULL mode)                    │
│       │                                                          │
│       ├── ubatch0: 2052 tokens  ──▶  reduce-scatter ✓           │
│       │                                                          │
│       └── ubatch1: 2051 tokens  ──▶  reduce-scatter ✗           │
│                                        AssertionError            │
│                                            │                     │
│                              异常被 thread wrapper 吞没          │
│                              __exit__ signal ubatch0              │
│                                            │                     │
│                              表现为主线程 join() 永久等待         │
│                                            │                     │
│                              被误判为"HCCL 死锁"                 │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    三个层面需要修复                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Slicing contract: ubatch 切分应该使用 padded 还是            │
│     unpadded slices？上下游不一致。                               │
│                                                                  │
│  2. pad_size 计算: _get_actual_num_tokens 在非 FULL 模式         │
│     不应无条件读 attn_metadata。                                  │
│                                                                  │
│  3. Runtime context: `_EXTRA_CTX.pad_size` 未按 ubatch 生效；      │
│     待区分 compiled 常量折叠与线程间 context 串扰。                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**原版 RFC 的误判**：

| 原版判断 | 实际情况 |
|---|---|
| "两个线程并发调用同一 AOT compiled callable 导致死锁" | 只有一个线程抛异常，另一个正常执行。异常被吞没。 |
| "HCCL collective submission 顺序分叉" | 两个 rank 的 collective 序列完全一致；
| | 不是通信层问题。 |
| "需要 per-ubatch compiled execution instance" | 当前没有证据表明 compiled callable 不可重入；
| | 问题在 shape contract，不在 execution instance。 |
| "hook/yield 与 opaque collective 之间缺少 happens-before" | hook/yield 顺序正确；问题在进入 compiled |
| | segment 前 shape 已经错了。 |
| "MLA symbolic shape 是独立正交问题" | MLA shape 问题真实存在但属于另一个
| | contract（见 §7）；与当前 DBO shape 错误无关。 |

---

## 6. 修复方案（修正版）

### 6.1 Phase 0：定义双 token contract

不能再让 `num_tokens`、`pad_size`、`ubatch_slices` 同时表达 attention 逻辑长度
和 collective 对齐长度。每个 ubatch 必须显式保存：

- `logical_num_tokens` / `logical_slice`：attention metadata、位置编码、输出裁剪；
- `collective_num_tokens` / `collective_slice`：AG/RS 输入，必须满足 TP 整除约束；
- `collective_pad_size = collective_num_tokens - logical_num_tokens`。

wrapper 应按 logical slice 切请求语义，在进入 collective contract 前只补齐一次；
collective 返回后按 logical length 裁剪。不能通过“改成 padded slice，同时把
pad_size 清零”隐式修复，因为这会丢失输出裁剪所需的信息。

### 6.2 Phase 1：修复 `create_ascend_forward_context` 的 `num_tokens` 计算

```python
# ascend_forward_context.py:273-275
slice_num_tokens = ubatch_slices[ubatch_num].num_tokens
if cudagraph_runtime_mode is CUDAGraphMode.FULL:
    # FULL graph: attn_metadata 的 num_actual_tokens 反映 graph 内固定 token 数
    new_forward_context.num_tokens = _get_actual_num_tokens(
        attn_metadata, slice_num_tokens
    )
else:
    # 非 FULL: 使用 ubatch slice 自身的 token 数
    new_forward_context.num_tokens = slice_num_tokens
```

这保证在 DBO runtime（`CUDAGraphMode.NONE`）中，每个 ubatch 的 `num_tokens`
和 `pad_size` 基于实际的 slice token 数计算，而不是从可能过时的 attn_metadata
读取。

### 6.3 Phase 2：让 compiled custom op 动态读取 pad_size

核心思路：将 `pad_size` 从一个 Python int 变成一个运行时才能确定的 tensor
属性。选项：

**选项 1：通过第一个 token 的 shape 携带 pad 信息**

将 padding token 实际拼接到输入 tensor 上，让 shape 自己表达 padding 量：
```python
# 在进入 compiled graph 前，由 wrapper 完成 padding
if pad_size > 0:
    x = F.pad(x, (0, 0, 0, pad_size))
```
compiled graph 内部不再需要读 `_EXTRA_CTX.pad_size`。

**选项 2：将 pad_size 作为显式 scalar tensor input 传入 compiled graph**

```python
pad_size_tensor = torch.tensor([pad_size], device=x.device)
output = compiled_fn(x, pad_size_tensor)
```

这需要修改所有 custom op 的 FakeImpl 和注册签名。

**选项 3（短期 workaround）：DBO runtime 禁用对这些 custom op 的编译**

通过 `skip_compiled=True` 或 `splitting_ops` 确保 padding 逻辑不在 compiled
subgraph 内执行（已在 Phase 0 验证中部分采用）。

### 6.4 Phase 3：异常处理修复 — 让 shape 错误立即可见

```python
# npu_ubatch_wrapper.py:_ubatch_thread — 让 ubatch0 感知 ubatch1 的异常
def _ubatch_thread(results, model, ubatch_metadata):
    try:
        with ubatch_metadata.context:
            model_output = model(...)
            dbo_current_stream().synchronize()
            dbo_yield()
        results.append((ubatch_metadata.context.id, model_output))
    except Exception as e:
        results.append(e)
        # 新增: 设置共享 abort 状态并唤醒另一线程；所有 wait/yield 先检查 abort
        abort_state.set_exception(e)
        abort_state.wake_all()
```

虽然不解决根因，但能避免误判为死锁。

---

## 7. 独立问题：MLA symbolic shape contract

原版 RFC 大量篇幅讨论的 MLA 8/4096 和 8/52 mismatch 是真实存在的独立问题，
但与本文描述的 DBO shape 错误不相关。

### 7.1 问题本质

MLA custom op 内部分配 output buffer 时读取 `_EXTRA_CTX.num_tokens` 决定
buffer shape。这个 Python int 在 trace 时被常量折叠，导致后续不同 token 数的
请求无法使用同一 compiled graph。

### 7.2 修复方向

1. MLA output allocation 必须从输入 tensor 的 symbolic shape 推导，而不是读
   `_EXTRA_CTX.num_tokens`
2. `maybe_unpad_after_all_gather` 的 `num_tokens` 参数必须是 symbolic，不能
   是 Python 常量
3. FakeImpl 和 runtime 的 shape 计算公式必须完全一致

### 7.3 与本文问题的关系

两个问题都涉及"Python forward context 值被 compiled graph 常量折叠"，但：
- DBO shape：`pad_size` 在两个 ubatch 间不同
- MLA shape：`num_tokens` 在两次请求间不同

修复策略共享同一原则：**任何会影响 tensor shape 的整数都不能通过 Python
forward context 传入 compiled graph，必须走 symbolic tensor shape 或显式
graph input。**

---

## 8. 远端实验记录

### 8.0 2026-07-03 server 复核

配置：FlashComm1=1、FlashComm2=0、DBO=1、TP=2、AI_CPU，服务以
`enforce_eager=False` 启动并完成 torch.compile 与 ACL graph capture。

4096 输入、1 输出、1 请求、并发 1 的正式请求结果：

```text
Total input tokens: 4103
Successful requests: 1
Failed requests: 0
TTFT: 270.39 ms
```

该结果**不能作为 compiled runtime 修复证据**。当前远端工作区在
`model_runner_v1.py` 中设置：

```python
use_dbo_runtime_eager_fallback = enable_dbo and enable_sp(...)
skip_compiled = has_encoder_input or use_dbo_runtime_eager_fallback
```

因此请求实际验证的是 FlashComm1 + DBO eager fallback。要验证本文 shape
contract，必须移除该 fallback，并用 fresh compile cache 重跑同一个 4103-token
用例，同时记录两个 ubatch 的 logical/collective rows。

### 8.1 实验矩阵

| 实验 | 改动 | 结果 |
|---|---|---|
| `dbo_trace_exception` | traceback 打印 + 异常日志 | 确认是 `AssertionError`，不是死锁 |
| `dbo_fix_validate` | 切换 `ubatch_slices_attn` → `ubatch_slices_padded` | 二次 padding →
| | | 2053，再次断言 |
| `dbo_contract_validate` | 保留 unpadded slices；仅在 FULL 模式
| | | `_get_actual_num_tokens` | ubatch1 的 pad_size 仍不生效 |
| `dbo-contract-validate` follow-up | 计划记录 `x.shape[0]`、`ctx.num_tokens`、
| | | `ctx.pad_size`、thread/context id | 已确认是 compile 常量折叠（分析见 §4），
| | | 或全局 forward_context 被另一线程覆盖 |

### 8.2 关键日志

```text
/data/workspace/logs/dbo_trace_exception_server.log  — 异常 trace，两个 rank 完全一致
/data/workspace/logs/dbo_fix_validate_server.log     — padded slices 二次 padding 失败
/data/workspace/logs/dbo_contract_validate_server.log — num_tokens fix 仍失败
```

---

## 9. 建议实施顺序（修正版）

1. **立即**：修复 ubatch 异常传播；共享 abort、唤醒全部等待者、主线程抛首个异常
2. **短期**：引入 logical/collective 双 token contract，确保 collective 输入只 pad 一次
3. **定点验证**：记录 `(rows, logical, collective, pad, context, thread)`，区分
   compiled 常量折叠与 forward context 串扰
4. **验证**：同一 4K compiled 单请求通过后，再跑 odd/even 与并发矩阵
5. **中期**：若证实常量折叠，让 shape 控制量成为 symbolic shape 或显式 graph input
5. **长期**：修复 MLA symbolic shape contract（独立问题）
6. **架构**：建立规范——任何影响 tensor shape 的值不得通过 Python forward context
   传入 compiled graph

---

## 10. Upstream 参考

### vLLM DBO 切分

- upstream `ubatch_utils.py` 默认按 padded tokens 切分：
  `split_point = int(num_tokens_padded) // num_ubatches`
- upstream `gpu_ubatch_wrapper` 传给 wrapper 的是 `ubatch_slices_padded`

### vLLM compilation / CUDAGraph

- `CompilationConfig` API：
  <https://docs.vllm.ai/en/latest/api/vllm/config/compilation/>
- CUDAGraph overhaul RFC：
  <https://github.com/vllm-project/vllm/issues/20283>

### 可借鉴的 `skip_compiled` contract

upstream compiled model decorator 已提供 `ForwardContext.skip_compiled`：

```python
if get_forward_context().skip_compiled:
    return self.forward(*args, **kwargs)
```

当前 Phase 0 中 DBO runtime 设置 `skip_compiled=True` 已验证可绕过该问题。

---

## 11. 结论

本问题的真实故障链路为：

```text
非 FULL graph DBO runtime
  → ubatch_slices_attn 未包含 TP padding
    → ubatch1 token 数为奇数（2051）
      → down_proj reduce-scatter 断言失败
        → 异常被 thread wrapper 吞没
          → 表现为"死锁"
```

修复需要三层：

1. **Token contract**：分离 logical tokens 与 collective tokens，padding 只发生一次
2. **Context source**：非 FULL 模式不能无条件信任 captured attention metadata
3. **Runtime ownership**：确认 shape 控制量是被 compile 常量化还是被另一线程覆盖，
   再决定使用 symbolic graph input 或 thread-local context

原版 RFC 将问题归因于"compiled callable 不可重入"和"HCCL 通信序列分叉"——
这些判断被远端 trace 证据证伪。本修正版基于实际异常日志和逐层 shape 分析重写。
