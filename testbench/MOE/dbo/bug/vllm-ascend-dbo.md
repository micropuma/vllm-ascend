# 上游 vLLM DBO 改造与 Ascend 迁移难度分析

## 1. 结论

你的判断基本成立：上游 vLLM 的 DBO 改造相对更容易维护和兼容 `torch.compile`，核心原因不是 DBO 本身简单，而是上游把 DBO 限制在少数受控边界内：

1. MoE runner 被注册成 opaque custom op。
2. DBO yield 主要发生在 fused MoE modular kernel 的 prepare/finalize 包装层。
3. DeepEP / NIXL EP 的 DBO 逻辑要求 async prepare/finalize。
4. shared experts overlap 被封装成独立的 `SharedExperts` 状态机。
5. 非 async 或不支持的路径明确 `assert not dbo_enabled()`。

因此，上游不是把 DBO 当成“任何 forward 代码里都可以插 hook”的通用机制，而是把它收敛成 MoE 通信/计算 overlap 的局部协议。

vllm-ascend 更难，是因为 Ascend 为了 FlashComm、sequence parallel、dense/MoE overlap，把 DBO hook 插入了更多普通 forward 路径。这些路径会被 Dynamo/FakeTensor/Inductor 看到，导致 compile 问题暴露面显著扩大。

## 2. 上游 DBO 的整体结构

上游 DBO 分两层：

1. 调度层：`UBatchWrapper` 把一个 batch 切成多个 ubatch，并用线程 + CPU event + CUDA stream/event 控制两个 ubatch 交替执行。
2. 插桩层：具体 MoE 通信路径在合适位置调用 `dbo_yield`、`dbo_register_recv_hook`、`dbo_switch_to_comm` 等函数。

调度层在 `vllm/v1/worker/ubatching.py` 中实现。`UBatchContext` 持有 compute stream、comm stream、CPU event、GPU event 和每个 ubatch 的 forward context。`dbo_enabled()` 本质上只是判断当前线程是否处于 ubatch context 内。

代码入口：

- [ubatching.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/v1/worker/ubatching.py:20)：`UBatchContext`
- [ubatching.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/v1/worker/ubatching.py:150)：`dbo_enabled`
- [ubatching.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/v1/worker/ubatching.py:170)：`dbo_yield` / stream switch wrapper
- [gpu_ubatch_wrapper.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/v1/worker/gpu_ubatch_wrapper.py:295)：真正启动 ubatch 线程

这个调度层本身不应该被 `torch.compile` trace。它是 runtime scheduling。

## 3. 上游 MoE compile 边界

上游 MoE runner 的关键改造是：MoE forward 不是普通 Python forward 被完整 trace，而是通过 custom op 进入。

关键代码：

- [moe_runner.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:60)：把 MoE layer 注册到 `static_forward_context`
- [moe_runner.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:118)：`_moe_forward` real impl 只负责找 layer 并调用 `_forward_impl`
- [moe_runner.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:135)：`_moe_forward_fake`
- [moe_runner.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:193)：注册 `torch.ops.vllm.moe_forward`
- [moe_runner.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:295)：非 CPU/TPU 平台选择 custom op 作为 forward entry

这里有两个重要设计点：

1. fake impl 是纯 shape 函数。它只根据输入 tensor 和显式参数 `hidden_dim_unpadded` 推导输出形状，不读取 forward context、通信 group、layer 对象。
2. real impl 可以读取 forward context 找真实 layer，但这发生在 custom op runtime 内，不是外层 compile 图需要展开的 Python 逻辑。

所以，上游 MoE 对 `torch.compile` 的兼容方式不是“让 Dynamo 编译整个 MoE Python 实现”，而是“把 MoE 变成图里的 opaque op”。外层图能继续编译，MoE 内部复杂调度不暴露给 FakeTensor/Inductor。

## 4. 上游 DBO 插桩点

### 4.1 fused MoE modular kernel

MoE modular kernel 是 DBO 的主插桩点。它把 prepare/finalize 包装成支持 async + receiver hook 的协议。

关键代码：

- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1122)：`_prepare` 是 prepare wrapper，明确说负责 DBO 和 async
- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1126)：不支持 async 的 prepare/finalize 路径直接 `assert not dbo_enabled()`
- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1149)：async prepare 前先处理上一个 recv hook
- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1169)：如果有 hook 且 DBO enabled，则注册到 ubatch context 并 `dbo_yield`
- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1287)：`_finalize` 是 finalize wrapper，明确负责 DBO、async 和 shared expert overlap
- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1299)：finalize 不支持 async 时同样 `assert not dbo_enabled()`
- [modular_kernel.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/modular_kernel.py:1330)：finalize hook 的 DBO 注册和 yield

这说明上游的 DBO 协议要求底层通信实现能拆成：

1. launch async communication
2. 返回 hook/receiver
3. 在合适的 ubatch 时机执行 hook/receiver

如果某条路径不能提供这个协议，就不允许 DBO。

### 4.2 DeepEP high-throughput prepare/finalize

DeepEP high-throughput 路径更细地控制 compute stream 和 comm stream。

关键代码：

- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:110)：yield 前捕获 compute stream 上的 DeepEP event
- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:116)：dispatch kernel 可能阻塞 CPU，所以先 yield，让另一个 ubatch 的 compute 入队
- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:119)：`dbo_yield_and_switch_from_compute_to_comm`
- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:160)：DBO 开启时不走 `async_finish`
- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:165)：handle 按当前 ubatch id 保存
- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:362)：combine 前再次 capture event + yield 到 comm
- [deepep_ht.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:395)：不支持的同步 receiver 路径 `assert not dbo_enabled()`

这是一种很强约束的改造：DBO 不是随便在 all2all 前后插两个 hook，而是通信库必须支持事件、handle、receiver、stream 切换语义。

### 4.3 DeepEP low-latency / NIXL EP

low-latency 和 NIXL EP 路径也遵循类似协议：dispatch 返回 hook + receiver，combine 返回 recv hook，并且 DBO 时通过 ubatch context 管理。

相关代码：

- [deepep_ll.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py:252)：按 ubatch id 保存 handle
- [deepep_ll.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py:386)：finalize 取当前 ubatch handle
- [deepep_ll.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py:387)：`do_recv_hook = dbo_enabled() or do_async`
- [deepep_ll.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py:398)：combine 前运行 pending recv hook
- [nixl_ep.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py:244)：NIXL EP 同样按 ubatch id 保存 handle
- [nixl_ep.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/prepare_finalize/nixl_ep.py:358)：NIXL EP finalize 中 `do_recv_hook = dbo_enabled() or do_async`

### 4.4 shared experts overlap

shared experts overlap 被单独封装为 `SharedExperts`，不是散落在模型 forward 里。

关键代码：

- [shared_experts.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:39)：`SharedExperts`
- [shared_experts.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:49)：DBO 下用 ubatch id 区分两个 output slot
- [shared_experts.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:89)：根据平台、阈值、MK 能力决定 overlap order
- [shared_experts.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:111)：aux stream 同步
- [shared_experts.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/shared_experts.py:144)：DBO 时用 `dbo_current_ubatch_id()` 选择 output slot
- [moe_runner.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-v0.20.2/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:799)：MoE runner 在 `_forward_impl` 开始处同步 shared experts stream

这里也体现了上游的边界控制：shared experts overlap 属于 MoE runner 内部状态，不是全局 forward hook。

## 5. 为什么上游相对容易兼容 torch.compile

上游容易的根因有四个。

第一，DBO 插桩集中在 MoE custom op 内部。外层 `torch.compile` 看到的是 `torch.ops.vllm.moe_forward`，不是 `dbo_yield`、thread local、stream switch、DeepEP receiver 这些 Python runtime 细节。

第二，fake impl 纯净。`_moe_forward_fake` 不读 live forward context，不读 TP/EP group，不访问 layer 对象。这让 FakeTensor propagation 稳定。

第三，不支持 async 的路径直接拒绝 DBO。上游没有试图让任意通信路径都兼容 DBO；如果 prepare/finalize 无法拆成 async hook/receiver，就 `assert not dbo_enabled()`。

第四，shared experts overlap 被局部状态机封装。它只处理 MoE shared experts 的输出缓存、aux stream、ubatch id，而不是污染 dense linear 或 attention 的普通 forward。

因此，上游“侵入点少”的准确表述是：DBO 的 runtime hook 只侵入少数 MoE 通信/overlap 抽象，而这些抽象又被 custom op 边界挡在外层 compile 图之外。

## 6. Ascend 为什么更难

vllm-ascend 的 DBO 插桩面明显更广。

### 6.1 Dense / sequence parallel linear 也插 DBO

Ascend 在普通 linear 路径里插了 DBO hook。

代码：

- [linear_op.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/linear_op.py:198)：`MLPColumnParallelOp` 读取 `get_forward_context().dbo_enabled`
- [linear_op.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/linear_op.py:200)：`dbo_linear_column_hook`
- [linear_op.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/linear_op.py:453)：`SequenceColumnParallelOp` 的 FlashComm + DBO 分支
- [linear_op.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/linear_op.py:457)：compiled path 内读取 `flash_comm_v1_enabled`

这类代码通常在模型普通 forward path 里，不像上游 MoE 那样天然被 `moe_forward` opaque custom op 包住。

### 6.2 MoE prepare/finalize 也直接读 context

Ascend MoE prepare/finalize 中 DBO 和 FlashComm 分支直接读 forward context。

代码：

- [prepare_finalize.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:390)：prepare 中读取 forward context
- [prepare_finalize.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:392)：`dbo_moe_prepare_hook`
- [prepare_finalize.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:393)：读 `flash_comm_v1_enabled`
- [prepare_finalize.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:527)：finalize 中读取 forward context
- [prepare_finalize.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:530)：`dbo_moe_finalize_hook`

如果 Ascend MoE 没有完全放到 opaque custom op 内，这些逻辑会被 Dynamo/FakeTensor 看见。

### 6.3 AllToAll dispatcher 也直接插 DBO

代码：

- [token_dispatcher.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:486)：dispatch 前读取 forward context
- [token_dispatcher.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:488)：dispatch DBO prepare hook
- [token_dispatcher.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:540)：combine 前读取 forward context
- [token_dispatcher.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:542)：combine DBO finalize hook

这部分和上游 DeepEP 类似，但上游要求 prepare/finalize 提供 async hook/receiver 协议；Ascend 当前更像是在现有同步/半同步路径外侧包 hook，compile 边界更不清晰。

### 6.4 Ascend custom op fake impl 仍读 runtime context

这是 compile 失败的直接高风险点。

代码：

- [register_custom_ops.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/register_custom_ops.py:123)：`_maybe_all_gather_and_maybe_unpad_fake`
- [register_custom_ops.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/register_custom_ops.py:126)：fake impl 读 `_EXTRA_CTX.flash_comm_v1_enabled`
- [register_custom_ops.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/register_custom_ops.py:128)：fake impl 读 TP world size
- [register_custom_ops.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/register_custom_ops.py:134)：`_maybe_pad_and_reduce_fake`
- [register_custom_ops.py](/mnt/home/douliyang/mlsys/vllm-analysis/vllm-ascend/vllm_ascend/ops/register_custom_ops.py:186)：`_matmul_and_reduce_impl_fake` 读取 forward context 和 layer

上游最关键的改造之一就是 fake impl 纯 shape 化。Ascend 这里还没达到这个要求。

## 7. 对“上游改造是否更容易”的判断

是，上游这套改造相对更容易，原因有三点。

第一，上游 DBO 的功能目标更窄。主要服务 MoE dispatch/combine、DeepEP/NIXL EP、shared experts overlap；没有把 dense linear、MLA preprocess/postprocess、sequence parallel all-gather/reduce-scatter 都纳入 DBO hook。

第二，上游先抽象出了 MoE runner 和 modular kernel。DBO 插在 `prepare_async/finalize_async` 这种语义清楚的位置，而不是插在任意通信 op 前后。

第三，上游先建立了 compile 边界。MoE runner 通过 `torch.ops.vllm.moe_forward` 成为 opaque custom op，fake impl 纯净，真实 runtime 复杂性被隔离。

Ascend 更难，是因为它要同时满足：

1. FlashComm overlap。
2. Sequence parallel overlap。
3. Dense linear communication overlap。
4. MoE prepare/finalize overlap。
5. MLA preprocess/postprocess overlap。
6. torch.compile fake tensor / graph partition / custom op 约束。

这些目标互相冲突：overlap 希望在更多细粒度位置插 hook；compile 希望 Python runtime/context/stream/thread 逻辑尽量不可见。

## 8. Ascend 如果要对齐上游，需要做什么

建议不要直接把现有 hook 原样保留后强行 compile，而是按上游思路重新划边界。

### 8.1 第一阶段：纯化 fake impl

所有 custom op fake impl 必须只依赖输入 tensor shape 和显式 primitive 参数。不能读：

1. `_EXTRA_CTX`
2. `get_forward_context()`
3. `get_tp_group()` / `get_ep_group()` / `get_dp_group()`
4. layer object
5. runtime flag，例如 `flash_comm_v1_enabled`

需要把 `flash_comm_enabled`、`tp_world_size`、`padded_length`、`pad_size`、`do_comm` 等变成显式参数，或者拆成不同 custom op variant。

### 8.2 第二阶段：先把 Ascend MoE 整体包成 opaque custom op

对齐上游 `moe_forward`，新增 Ascend MoE forward custom op：

1. fake impl 只返回 shape。
2. real impl 通过 layer name 找到 Ascend MoE layer。
3. real impl 内部调用现有 `prepare -> fused_experts -> finalize`。
4. DBO hook、FlashComm、EP/TP group 读取全部留在 real impl 内。

这一步的目标不是性能最优，而是先让外层 `torch.compile` 稳定。

### 8.3 第三阶段：把 dense/linear/MLA 的 DBO hook 收敛为受控 op

如果 dense linear / MLA 的 FlashComm overlap 必须保留，就不能让 hook 直接出现在普通 forward Python 里。需要二选一：

1. 把相关通信段封成 opaque custom op，例如 `ascend_flashcomm_allgather`、`ascend_flashcomm_reducescatter`。
2. 把 DBO 调度放到 compiled callable 外部，compiled graph 内只保留纯 tensor compute。

第一种更接近当前代码，改造成本较低；第二种架构更干净，但需要重做 overlap 调度边界。

### 8.4 第四阶段：建立 async prepare/finalize 协议

如果 Ascend 要像 DeepEP 那样稳定支持 DBO，通信实现最好提供类似协议：

1. `prepare_async()` 返回 hook/receiver 或 handle/receiver。
2. `finalize_async()` 返回 hook/receiver。
3. handle 按 ubatch id 保存。
4. 不支持 async 的路径直接禁止 DBO。

也就是说，不要让 DBO hook 只是“包在通信前后”的 side effect，而要让通信 API 本身表达 async launch、receiver、stream/event 依赖。

## 9. 最重要的技术判断

`torch.compile` 和 DBO 不是天然完全冲突，但它们的边界必须清楚：

1. DBO 的线程切换、CPU event、stream switch、recv hook 不应该暴露给 Dynamo。
2. 这些 runtime 逻辑可以放在 opaque custom op real impl 内，或者放在 compiled callable 外部。
3. fake impl 必须纯 shape 化。
4. 不支持 async/receiver 语义的通信路径，不应该强行启用 DBO。

上游能跑通，是因为它基本遵守了这些边界。Ascend 现在难，是因为 DBO + FlashComm 的插桩已经扩散到普通 forward path，同时 fake impl 又读取 runtime context，导致 compile 的每一层机制都会撞到动态状态。

