# RFC: DBO + FlashComm1 + torch.compile 的 ubatch shape contract

## 状态

- 状态：已实现并回归验证通过
- 日期：2026-06-28
- 影响范围：DBO、FlashComm1、torch.compile/ACL graph、TP > 1

## 问题配置

```text
Model: DeepSeek-V2-Lite-Chat
Device: Ascend 910B1
TP: 2
DBO: enabled (num_ubatches = 2)
FlashComm1: enabled
FlashComm2: disabled
torch.compile / ACL graph: enabled
all2all backend: deepep_low_latency
```

## 摘要

DBO + FlashComm1 的 shape contract 错误根因是**两层独立 padding 被混淆为一层**：

1. **Scheduler padding**：scheduler 为了让总 token 数对齐 TP 倍数，在 batch 末尾追加 padding
   request。这个 padding 贯穿整个前向计算，embedding 填 0，但以真实 tensor row 的形式存在。
2. **FlashComm padding**：单次 TP 通信（reduce_scatter）之前，临时在 tensor 维度上 pad，通信
   完成后立刻 unpad。pad/unpad 是同一层通信操作的内在闭环。

混淆发生在 `create_ascend_forward_context`：旧代码用 ubatch slice 的 padded `num_tokens`（已包含
scheduler padding）当作 forward context 的逻辑 token 数。FlashComm 据此算出 `pad_size = 0`，
跳过本应执行的 unpad。scheduler padding 于是**泄漏**到下游所有 layer 的 tensor shape 中，与
attention metadata 的 `num_actual_tokens` 产生系统性偏差。

最终 `AddRmsNormBias`（或其编译 pass 替换后的等价 fused op）收到两个首维不一致的 tensor，
tiling 检查失败，报 `EZ9999: Inner Error`。

## 两层 padding 全景

### 第一层：Scheduler padding

```text
真实 token:  [T0 .. T2050]              ← 2051 个
                 ↓ scheduler pad 到 TP=2 的倍数
padded batch: [T0 .. T2050, PAD]        ← 2052 个，最后一个 embedding 为 0
```

Scheduler padding 是**持久**的：PAD 以 row 的形式存在于每一层的输入 tensor 中。

### 第二层：FlashComm padding

```text
per-layer TP 通信:
  输入 [N, D]  (N 可能为奇数)
    ↓ F.pad 到 TP 的倍数
  [ceil(N/TP)*TP, D]
    ↓ reduce_scatter → all_gather
  [ceil(N/TP)*TP, D]
    ↓ x[:-pad_size]  unpad 回去
  [N, D]
```

FlashComm padding 是**瞬时**的：pad → 通信 → unpad，不让 padding row 泄漏到下一层。

### 关键：两层各管各的

```
                    Scheduler pad 的范围
                    ╭─────────────────────────────╮
batch:  [T0 .. T2050, PAD]
ubatch: [  ubatch 0  ][  ubatch 1 (含 PAD)  ]
                          ╰── 1025 real + 1 pad ─╯

ubatch 1 进入 FlashComm:
  输入 [1026, D] (scheduler 视角)
  但 forward context num_tokens 应该报 1025 (逻辑视角)
  → FlashComm pad: 1025 → 1026 → 通信 → unpad → 1025
  → scheduler 的 PAD 仍然在 tensor 里，但 FlashComm 不感知也不负责它
  → 下游 consistent 在 1025
```

**核心原则**：`forward_context.num_tokens` 必须反映逻辑 token 数（`attn_metadata.num_actual_tokens`），
让 FlashComm 独立管理自己的 pad/unpad 闭环。scheduler 的 padding 只影响 tensor 的实际 row 数量
（用于切片），不影响 forward context 记账。

## 用户可见现象

首个 DBO prefill 请求失败。错误的级联表现（按 layer 推进顺序）：

```text
MLA output shape:     1026 vs attention metadata 的 1025
AddRmsNormBias:       x1.shape[0]=1026, x2.shape[0]=1025 → tiling check 失败
EZ9999: Inner Error, AddRmsNormBias do tiling failed
```

不是三个独立 bug，是同一个 shape contract 错误在不同位置的三种表现。

## 根因：trace 推演

### 错误数据流

以 2051 真实 token、TP=2、DBO 2-ubatch 为例：

```text
scheduler:               2051 → pad → 2052
ubatch split:            ubatch[0] = 1026 (全 real)
                         ubatch[1] = 1026 (1025 real + 1 scheduler pad)

attention metadata:      ubatch[1].num_actual_tokens = 1025
ubatch slice:            ubatch[1].num_tokens         = 1026
```

### 旧代码（错误）

```python
# ascend_forward_context.py (旧)
new_forward_context.num_tokens = ubatch_slices[1].num_tokens  # = 1026
```

```text
pad_size = (2 - 1026 % 2) % 2 = 0

FlashComm 路径:
  [1026, D]
    → pad_size=0, 不 pad
    → reduce_scatter → [513, D]
    → all_gather → [1026, D]
    → pad_size=0, 不 unpad
    → 输出 [1026, D]  ← scheduler PAD 没有被 trim

attention 路径:
  num_actual_tokens = 1025
  某处 tensor trim 到 [1025, D]

下游 fused op:
  [1026, D] + [1025, D] → shape mismatch → EZ9999
```

### 关键误解

"ubatch slice 已经是 1026（能被 TP=2 整除），FlashComm 不需要再 pad"。

这个推理把 scheduler padding（持久、全局）当成了 FlashComm padding（瞬时、局部）。
Scheduler 加的 PAD row 会一直存在，而 FlashComm 的 unpad 是用来去除**自己**加的 padding，
不是用来去除 scheduler 的 padding。

当 `pad_size = 0` 时，FlashComm 既不 pad 也不 unpad——它认为不需要，但 scheduler 的 PAD 还在。
下游的 attention 按 `num_actual_tokens = 1025` 工作，产生 [1025, D] 的 tensor。两条路径汇合时，
shape 不一致。

### 正确的 pad_size 语义

```text
pad_size = (TP - num_actual_tokens % TP) % TP

当 num_actual_tokens = 1025:
  pad_size = (2 - 1) % 2 = 1
  含义: "为了完成 TP 通信，需要临时加 1 个 padding，通信后删掉它"

当 num_actual_tokens = 1026:
  pad_size = (2 - 0) % 2 = 0
  含义: "不需要临时 padding，直接从 1026 开始通信"
```

但问题在于，**这个 `num_actual_tokens` 必须与 attention 认为的逻辑 token 数一致**。
`ubatch_slices[1].num_tokens = 1026` 并不是 attention 认可的值——attention 只认
`num_actual_tokens = 1025`。用 1026 算出 `pad_size = 0`，是"用错了基准"，而非
"刚好不需要 pad"。

## 修复设计

### 1. 统一逻辑 token 来源：`_get_actual_num_tokens`

```python
def _get_actual_num_tokens(attn_metadata, fallback_num_tokens):
    """从 attention metadata 提取实际 token 数（不含 scheduler/graph padding）。"""
    if attn_metadata is None:
        return fallback_num_tokens
    metadata_values = (
        attn_metadata.values() if isinstance(attn_metadata, dict)
        else (attn_metadata,)
    )
    for metadata in metadata_values:
        n = getattr(metadata, "num_actual_tokens", None)
        if n is not None:
            return n
    return fallback_num_tokens
```

优先从 per-ubatch attention metadata 取 `num_actual_tokens`。没有 metadata 时才退回到 ubatch
slice 的 token count。

### 2. `create_ascend_forward_context` 中两处修正

```python
# num_tokens: 使用逻辑 token 数，让 FlashComm 的 pad_size 正确计算
new_forward_context.num_tokens = _get_actual_num_tokens(
    attn_metadata, ubatch_slices[ubatch_num].num_tokens
)

# token_slice: RoPE cache 仅覆盖到逻辑 token 边界
padded_token_slice = ubatch_slices[ubatch_num].token_slice
token_slice = slice(
    padded_token_slice.start,
    padded_token_slice.start + new_forward_context.num_tokens,
)
```

### 3. FakeTensor 与 runtime 的 reduce-scatter shape 一致

```python
def _get_reduce_scatter_num_tokens(num_tokens: int, tp_size: int) -> int:
    return (num_tokens + tp_size - 1) // tp_size   # ceiling division
```

应用于 `maybe_pad_and_reduce` 和 `matmul_and_reduce` 的 fake implementation，确保
compile-time shape trace 与 runtime 的 pad-then-divide 行为一致。

## 设计原则（总结）

```
┌─────────────────────────────────────────────────────────────┐
│  Scheduler padding  →  管 tensor row 的数量（用于切片）     │
│  FlashComm padding  →  管 TP 通信的 pad/unpad（瞬时闭环）    │
│  forward_context.num_tokens  →  必须以 attention metadata   │
│       的 num_actual_tokens 为准（逻辑 token 数）              │
│                                                             │
│  不要把 scheduler padding 当作 FlashComm padding 来用。      │
│  各层 padding 由引入它的那层负责管理。                        │
└─────────────────────────────────────────────────────────────┘
```

## 不采用的方案

### 在 MLA、Residual 或 RoPE 处裁剪

只能移动报错位置，且可能裁掉真实 token。

### 让 ubatch slice 不包含 scheduler padding

修改 `_pad_out_ubatch_slices` 去掉末尾 padding。会破坏 graph capture 的固定 shape
假设（两个 ubatch 必须等长），改动面太大。

### 自动关闭 FlashComm1

可作临时验证手段，但不是修复。

## 测试

### 单元测试

```bash
pytest -q tests/ut/test_ascend_forward_context.py
```

```text
test_get_actual_num_tokens[attn_dict-None-fallback]  PASSED
test_get_actual_num_tokens[attn_obj-None-fallback]   PASSED
test_get_actual_num_tokens[None-fallback]            PASSED
test_get_actual_num_tokens[empty_dict-fallback]      PASSED
test_get_reduce_scatter_num_tokens[2049-2-1025]      PASSED
test_get_reduce_scatter_num_tokens[2050-2-1025]      PASSED
test_get_reduce_scatter_num_tokens[2051-2-1026]      PASSED
test_get_reduce_scatter_num_tokens[3-4-1]            PASSED
8 passed
```

### 端到端

清理编译缓存后启动 DBO server + benchmark：

```bash
# terminal 1
cd testbench/MOE/dbo/demos
bash deepseek-v2-dbo-server.sh 2>&1 | tee dbo_server.log

# terminal 2
LABEL=dbo PORT=8001 INPUT_LEN=4096 OUTPUT_LEN=16 NUM_PROMPTS=500 MAX_CONCURRENCY=96 \
bash deepseek-v2-dbo-test.sh
```

预期：500 requests 全部成功，无 `EZ9999` 错误，无 EngineDeadError。

### 远端 fresh compile

测试必须隔离两类缓存：

```bash
export VLLM_CACHE_ROOT=/path/to/new/vllm-cache
--compilation-config \
  '{"cudagraph_capture_sizes":[16],"cache_dir":"/path/to/new/compile-cache"}'
```

验证矩阵：

| 输入 | 并发 | 请求数 | 结果 |
|---:|---:|---:|---|
| 4096 | 1 | 1 | 1 成功，0 失败 |
| 4096 | 16 | 32 | 32 成功，0 失败 |
| 4096 | 96 | 500 | 500 成功，0 失败 |

## 兼容性

- 无 metadata 路径回退到 ubatch slice 长度，不受影响
- 偶数 token 的 ceil division 与 floor division 等价，不受影响
- eager / 非 DBO / FlashComm1 关闭路径不受影响
- 不增加新环境变量

## 后续工作

1. 将 odd-ubatch 场景加入 NPU nightly 回归
2. custom unpad op 增加 debug-only assertion：输入短于目标长度时禁止静默返回
3. DBO graph cache key 假设两个 ubatch 等长，建议单独 RFC
