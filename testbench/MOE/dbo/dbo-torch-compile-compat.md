# DBO 多线程与 torch.compile 兼容性问题分析

## 1. 背景

DBO（Dual Batch Overlap）通过两个 CPU 线程交替执行两个 micro-batch 的 forward，
实现计算与通信的 overlap。同时，vLLM 默认开启 `VLLM_COMPILE` 模式，使用
`torch.compile(fullgraph=True)` 对 model forward 进行编译加速。

这两者在 flashcomm1 开启时存在系统性冲突。

---

## 2. 核心冲突：dynamo trace 发生在 DBO 子线程里

### 2.1 torch.compile 的 trace 时机

`torch.compile` 不是在调用 `torch.compile(model)` 时编译，而是在**第一次调用
`model(...)`** 时触发 dynamo 对整个 forward 函数做 fullgraph trace（graph capture）。

当开启 `VLLM_USE_AOT_COMPILE=True` 时（当前环境默认开启），这个 trace 是**同步阻塞**的，
第一次调用必须等 trace + compile 完成才返回。

### 2.2 DBO 的线程模型

`_run_ubatches`（`npu_ubatch_wrapper.py`）的执行流程：

```
主线程:
  with override_forward_context(None):   ← 把全局 _forward_context 清成 None
    thread0.start()
    thread1.start()
    ready_barrier.wait()
    cpu_wait_event[0].set()              ← 只唤醒 thread0

thread0:
  __enter__: ready_barrier.wait()
  __enter__: cpu_wait_event.wait()       ← 等主线程唤醒
  __enter__: _restore_context()          ← 恢复 _forward_context = ctx0
  model(...)                             ← 第一次调用，触发 dynamo trace
    -> fullgraph capture 开始
    -> 遍历整个 forward 函数
    -> 遇到任何 runtime 状态访问就炸
```

**关键点**：dynamo trace 发生在子线程里，此时整个进程的 runtime 状态（分布式 group、
forward context 等）都处于不稳定或不完整的状态。

### 2.3 为什么不开 flashcomm 时没问题

不开 flashcomm 时，forward 里所有 `if flash_comm_v1_enabled:` 的条件在 trace 时被
dynamo 求值为 `False`，整个分支被裁掉，dynamo 不进入这些分支，不会遇到有问题的代码。

开了 flashcomm 之后，这些条件变成 `True`，dynamo 进入了原本不会 trace 的代码路径，
这些路径里有大量的 runtime 状态依赖，全部炸开。

---

## 3. 具体炸点分类

按照错误类型分三类：

### 3.1 类型 A：fake impl 里读 runtime 状态（compile_all_ranges 阶段）

**位置**：`register_custom_ops.py` 的 fake impl 函数

**触发时机**：`PiecewiseBackend.compile_all_ranges()` 用 FakeTensor 重放 FX graph，
调用 fake impl 推断 output shape。

**炸点**：
- `_EXTRA_CTX.flash_comm_v1_enabled` → `get_forward_context()` → assert
- `get_tensor_model_parallel_world_size()` → `get_tp_group()` → assert `_TP is not None`

**已修**：用 `_FLASH_COMM_V1_SNAPSHOT` 和 `_TP_WORLD_SIZE_SNAPSHOT` 替换，
在 `set_ascend_forward_context` 时写入快照。

### 3.2 类型 B：real forward 里读 runtime 状态（fullgraph trace 阶段）

**位置**：被 dynamo trace 到的 forward 函数内部

**触发时机**：第一次调用 `model(...)` 触发 fullgraph trace，dynamo 遍历 forward 函数。

**炸点**：
- `mla.py:forward` 里 `_EXTRA_CTX.flash_comm_v1_enabled`
- `linear_op.py:apply_impl` 里 `get_forward_context()`
- `deepseek_v2.py:forward` 里 `get_pp_group()` → assert `_PP is not None`

**已修（部分）**：
- `mla.py`：`flash_comm_v1_enabled` 静态化为 `self.flash_comm_v1_enabled = enable_sp()`
- `linear_op.py`：`apply()` 加 `@torch.compiler.disable`

**未修**：`deepseek_v2.py` 是 upstream 代码，`get_pp_group()` 在 trace 时 `_PP is None`

### 3.3 类型 C：DBO hook 里的线程操作（fullgraph trace 阶段）

**位置**：`ubatching.py` 的 `_register_ubatch_function` 生成的 wrapper

**触发时机**：dynamo trace 进入 DBO hook（`dbo_linear_column_hook` 等），
hook 内部调 `threading.get_ident()`。

**为什么进入 hook**：`forward_context.dbo_enabled = True`（profile run 前的残留值），
dynamo 把这个条件求值为 `True`，进入 hook 分支。

**炸点**：`threading.get_ident()` 是 C builtin，dynamo 完全不认识。

**已修**：`wrapper` 函数加 `@torch.compiler.disable`

---

## 4. 根本矛盾

以上三类问题的根源是同一件事：

**DBO 要求 model forward 在子线程里执行，但 torch.compile 的 trace 也在第一次
forward 时发生，trace 过程会遍历整个 forward 函数，遇到任何不可 trace 的代码就炸。**

具体不可 trace 的代码有三类：
1. 读 runtime forward context（`get_forward_context()`）
2. 读 runtime 分布式 group（`get_tp_group()`、`get_pp_group()`）
3. 调线程相关 C builtin（`threading.get_ident()`）

这三类在 DBO 不触发（单线程 forward）时都没问题，因为没有 `override_forward_context(None)`，
runtime 状态始终完整。一旦 DBO 子线程触发 trace，这些代码就全部暴露出来。

---

## 5. 系统性修复方案

### 5.1 方案思路

核心原则：**让 dynamo trace 在 runtime 状态完整的环境里完成，不要在 DBO 子线程里
触发第一次 compile。**

有两个方向：

---

#### 方案 A：Pre-compile（推荐）

在进入 DBO 子线程之前，在主线程里提前触发一次 compile。

```
主线程（有完整 runtime 状态）:
  with set_ascend_forward_context(...):
    model(dummy_input)   ← 触发 trace + compile，此时 ctx/group 都在
    # compiled graph 已经缓存好了

DBO 子线程:
  model(real_input)      ← 直接用已编译的 graph，不再触发 trace
```

**实现位置**：`model_runner_v1.py` 的 `_dummy_run` 里，在第一次调用 `_model_forward`
（进入 DBO 路径）之前，先跑一次非 DBO 的 forward 触发 compile。

**优点**：一次性彻底解决所有类型的问题，不需要逐个修 forward 里的代码。

**缺点**：需要额外一次 dummy forward，增加启动时间；需要确保 pre-compile 用的
input shape 和 DBO 路径一致，否则可能触发 recompile。

---

#### 方案 B：逐点修复（当前正在做的）

对 dynamo trace 能进入的每一个问题代码点，用以下方式之一修复：

| 代码类型 | 修法 |
|---|---|
| fake impl 里读 runtime 状态 | 用编译前写入的 snapshot 替换（已做） |
| forward 里读 static config | 静态化为 module 属性（`self.xxx`），init 时写入 |
| forward 里读 per-step runtime 状态 | 用 `@torch.compiler.disable` 把这段代码挡在 graph 外 |
| DBO hook（线程操作） | 用 `@torch.compiler.disable` 包住 |
| upstream vLLM 代码（`get_pp_group` 等） | patch 或者在外层 disable |

**优点**：不需要额外的 dummy forward，改动局部。

**缺点**：这是打地鼠，每次开新功能可能引入新的炸点；`@torch.compiler.disable`
会让被禁用的代码不进入编译图，可能影响性能（取决于被 disable 的代码量）。

---

#### 方案 C：从设计上剥离 runtime 状态（长期）

这是你之前提到的正确方向：

> 把 flash_comm_v1_enabled / pad_size / DBO hook 等运行时状态从 compiled forward 里剥离，
> 要么 compile 前 snapshot，要么显式传参，要么放在 compiled graph 外层 dispatch。

具体做法：

1. **dispatch 层**：在 compiled graph 外层（`npu_ubatch_wrapper` 或 `model_runner`），
   根据 runtime 状态（flashcomm、dbo_enabled 等）决定调用哪条预编译路径。

2. **多路径编译**：为 `flashcomm=True` 和 `flashcomm=False` 各编译一个 graph，
   dispatch 层根据当前 step 的实际值选择对应的 graph。

3. **compiled graph 内部不读任何 runtime 状态**：所有影响 graph 结构的开关都在
   compile 时静态化，per-step 的动态值（如 pad_size）通过显式参数传入。

**优点**：根本上解决问题，graph 内部干净，无 runtime 依赖。

**缺点**：改动最大，涉及 `npu_ubatch_wrapper`、`linear_op`、`mla`、
`register_custom_ops`、`prepare_finalize` 等多个文件；需要重新梳理哪些值是
静态的，哪些是动态的。

---

### 5.2 当前建议

短期：继续方案 B，把现有的炸点逐个修掉，让服务能跑起来，验证 flashcomm 的性能收益。

中期：在理解清楚整个 compilation stack 之后，推进方案 C，把 runtime 状态从
compiled graph 里彻底剥离。

方案 A 可以作为方案 B 的补充——如果方案 B 遗漏了某个炸点，方案 A 的 pre-compile
可以作为兜底。

---

## 6. 已修改的文件汇总

| 文件 | 修改内容 | 类型 |
|---|---|---|
| `vllm_ascend/ops/mla.py` | `flash_comm_v1_enabled` 静态化为 `self.flash_comm_v1_enabled = enable_sp()` | 类型 B |
| `vllm_ascend/ops/register_custom_ops.py` | fake impl 里用 `_FLASH_COMM_V1_SNAPSHOT` 和 `_TP_WORLD_SIZE_SNAPSHOT` 替换 runtime 读取 | 类型 A |
| `vllm_ascend/worker/ubatching.py` | `_register_ubatch_function` 的 wrapper 加 `@torch.compiler.disable` | 类型 C |
| `vllm_ascend/ops/linear_op.py` | `CustomLinearOp.apply()` 加 `@torch.compiler.disable` | 类型 B |

## 7. 还未修的炸点

| 文件 | 问题 | 说明 |
|---|---|---|
| `deepseek_v2.py:1241` | `get_pp_group()` assert `_PP is not None` | upstream 代码，trace 时 `_PP` 是 None |
| `linear_op.py` 其他 `apply_impl` | `get_forward_context()` / `_EXTRA_CTX` | `apply()` 加了 disable 后应该已经覆盖 |
| `rotary_embedding.py:247` | `_EXTRA_CTX.flash_comm_v1_enabled` | 是否在 compiled graph 里待确认 |
| `dsa.py:165` | `get_forward_context().flash_comm_v1_enabled` | 同上 |
| `attention/mla_v1.py:1604` | `get_forward_context().flash_comm_v1_enabled` | 同上 |
