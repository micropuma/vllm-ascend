# vLLM 中的 torch.compile 与 Dynamo 捕获链路

本文先建立理解 vLLM `torch.compile` 所需的最小背景，再追踪 token
维度从普通 Tensor shape 变成 symbolic shape 的完整代码链路。本文讨论的
上游版本固定为仓库当前验证的 vLLM commit
`967c5c3bc38891f4465d3f4e99917ed837bb3833`，避免因上游代码变化造成行号和
行为混淆。[VERIFY: .github/vllm-main-verified.commit:1]

---

## 一、Dynamo 是什么

### 1.1 Dynamo 捕获的是 Python 执行，不只是 Tensor 算子

`torch.compile` 的前端是 TorchDynamo。模型第一次执行时，Dynamo 拦截
Python frame，沿实际执行路径分析 Python bytecode，把支持的 Tensor 运算转换为
FX graph。

可以把过程简化成：

```text
Python model.forward
        │
        ▼
TorchDynamo 分析 bytecode
        │
        ├─ Tensor 运算 ───────────────▶ FX graph node
        ├─ 可证明稳定的 Python 值 ───▶ graph constant / guard
        └─ 无法捕获的 Python 行为 ───▶ graph break 或 Unsupported
                                          （fullgraph 下直接报错）
```

vLLM 的模型类通过 `@support_torch_compile` 接入这套编译流程；当前仓库中的
`DeepseekV4Model` 和 `DeepSeekV4MTP` 是两个直接例子。
[VERIFY: vllm_ascend/models/deepseek_v4.py:996]
[VERIFY: vllm_ascend/models/deepseek_v4_mtp.py:200]

### 1.2 Graph、guard 和 constant

Dynamo 捕获一次 Python 路径后，需要保证后续调用仍满足这条图成立的前提。
这些前提由 guard 表达。例如某个 Python `bool` 决定是否进入 FlashComm
分支，Dynamo可以把该值特化为当前值，并为它建立 guard。

但如果某个值已经是图外计算好的 Python `int`：

```python
pad_size = _EXTRA_CTX.pad_size
```

Dynamo只能看到当前值，例如 `pad_size == 1`，看不到它在模型执行前曾由
`num_tokens` 计算得到。`_EXTRA_CTX` 本身只是对当前 `ForwardContext` 的属性代理。
[VERIFY: vllm_ascend/ascend_forward_context.py:494]

当前 `pad_size` 的图外计算和保存过程是：

```python
pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
forward_context.pad_size = pad_size
```

其中 `num_tokens` 是传入 context manager 的 Python `int`。因此，从
`_EXTRA_CTX.pad_size` 取回的仍是 Python `int`，不是与输入 Tensor shape
关联的 `torch.SymInt`。
[VERIFY: vllm_ascend/ascend_forward_context.py:58]
[VERIFY: vllm_ascend/ascend_forward_context.py:148]

### 1.3 Symbolic shape 的作用

如果输入 `x` 的 token 维被标记为动态，Dynamo捕获时可以把：

```text
x.shape = [4097, hidden_size]
```

表示为：

```text
x.shape = [s0, hidden_size]
```

其中 `s0` 是 symbolic integer。此后由 `x.shape[0]` 产生的 shape 运算可以继续
保留符号关系：

```python
pad_size = (-x.shape[0]) % tp_size
```

对应：

```text
pad_size = (-s0) mod tp_size
```

当前 Sequence Parallel row-parallel 路径正是从 `x.shape[0]` 计算 padding，
而不是读取 `_EXTRA_CTX.pad_size`。[VERIFY: vllm_ascend/ops/linear_op.py:579]

---

## 二、vLLM 如何让 token 维成为 symbol

### 2.1 第一步：模型声明 `@support_torch_compile`

当前 Ascend DeepSeek V4 模型入口为：

```python
@support_torch_compile
class DeepseekV4Model(nn.Module):
    ...
```

[VERIFY: vllm_ascend/models/deepseek_v4.py:996]

它的 `forward` 参数具有明确的 Tensor 类型标注：

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
):
```

[VERIFY: vllm_ascend/models/deepseek_v4.py:1093]

### 2.2 第二步：装饰器推导 dynamic argument dimensions

上游 `vllm/compilation/decorators.py` 中，`support_torch_compile()` 检查
`forward` 参数的类型标注。当参数是 `torch.Tensor`、可选 Tensor 或
`IntermediateTensors` 时，默认记录：

```python
inferred_dynamic_arg_dims[arg_name] = 0
```

即把这些输入的第 0 维作为动态维。对应上面的模型，可以近似理解为：

```python
dynamic_arg_dims = {
    "input_ids": 0,
    "positions": 0,
    "intermediate_tensors": 0,
    "inputs_embeds": 0,
}
```

该实现位于固定上游版本的
[`vllm/compilation/decorators.py`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/decorators.py#L201-L245)。
此时只是登记“哪些维度动态”，尚未创建 FX graph 或 `SymInt`。

### 2.3 第三步：首次编译前调用 `mark_dynamic`

模型第一次进入编译路径时，上游按以下顺序执行：

```text
model.__call__
    │
    ├─ 检查尚未完成首次编译
    ├─ _mark_dynamic_inputs(self, ds_type, *args, **kwargs)
    │    └─ torch._dynamo.mark_dynamic(tensor, dims)
    └─ 调用 torch.compile 包装后的模型
         └─ Dynamo 开始捕获 forward
```

关键顺序是“先 mark，后 capture”。上游源码明确在首次编译分支调用
`_mark_dynamic_inputs()`，随后才进入编译包装器：
[`vllm/compilation/decorators.py`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/decorators.py#L576-L590)。

普通 dynamic-shape 模式最终执行：

```python
dims = [dim for dim, _ in dim_shape_pairs]
torch._dynamo.mark_dynamic(arg, dims)
```

对应源码：
[`vllm/compilation/decorators.py`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/decorators.py#L414-L480)。

对于前面的 DeepSeek V4 输入，概念上相当于：

```python
torch._dynamo.mark_dynamic(input_ids, 0)
torch._dynamo.mark_dynamic(positions, 0)
torch._dynamo.mark_dynamic(inputs_embeds, 0)
```

`IntermediateTensors` 则遍历其中保存的 Tensor，对每个 Tensor 标记相同的动态维。

### 2.4 第四步：Dynamo 捕获 symbolic shape

`mark_dynamic` 不会改变 eager 模式下真实 Tensor 的 shape。输入为 4097 个
token 时，Python侧仍能看到 `input_ids.shape == [4097]`。它表达的是对
Dynamo 的约束：

> 捕获图时，不要把该维度特化为常量 4097。

Dynamo创建 FakeTensor 和 FX metadata 后，该维度才表现为类似 `s0` 的
`torch.SymInt`。因此模型内部：

```python
num_tokens = hidden_states.shape[0]
```

得到的是符号值，而下面的输出 shape 也继续依赖该符号：

```python
output = torch.empty((num_tokens, hidden_dim), ...)
```

当前 MLA wrapper 使用 `hidden_states.shape[0]` 决定 output token 维。
[VERIFY: vllm_ascend/ops/mla.py:155]
[VERIFY: vllm_ascend/ops/mla.py:174]

### 2.5 第五步：backend确认哪些输入包含 symbol

上游 compilation backend 会检查 FakeTensor 的各维：

```python
from torch.fx.experimental.symbolic_shapes import is_symbolic

sym_tensor_indices = [
    i
    for i, x in enumerate(fake_args)
    if isinstance(x, FakeTensor)
    and any(is_symbolic(d) for d in x.size())
]
```

对应源码：
[`vllm/compilation/backends.py`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/backends.py#L1299-L1318)。
这也是判断“动态维声明是否真正传播到 FX/FakeTensor”的直接代码证据。

---

## 三、symbol 如何在 vllm-ascend 中继续传播

仅在入口调用 `mark_dynamic` 还不够。下游 shape 必须继续由 Tensor shape
计算，symbol 才不会中途丢失。

### 3.1 ReduceScatter

当前 fake implementation 从输入 Tensor 取 token 维：

```python
num_tokens = _get_reduce_scatter_num_tokens(x.shape[0], tp_size)
```

其中：

```python
return (num_tokens + tp_size - 1) // tp_size
```

所以当 `x.shape[0] == s0` 时，输出第一维是：

```text
ceil(s0 / tp_size)
```

[VERIFY: vllm_ascend/ops/register_custom_ops.py:216]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:220]

### 3.2 AllGather

AllGather fake shape 使用：

```python
x.shape[0] * tp_size
```

因此输出第一维仍是 `s0` 的符号表达式。
[VERIFY: vllm_ascend/ops/register_custom_ops.py:197]

### 3.3 DBO hook

DBO同步 hook 的 fake implementation直接返回输入：

```python
def _dbo_hook_fake(x, is_record):
    return x
```

这保证 hook 前后的 symbolic shape 不变，也避免 fake propagation 读取动态
Python forward context。[VERIFY: vllm_ascend/ops/register_custom_ops.py:61]

整体链路为：

```text
真实输入 Tensor [4097, H]
        │
        ├─ mark_dynamic(dim=0)
        ▼
FakeTensor [s0, H]
        │
        ├─ pad = (-s0) mod TP
        ▼
[s0 + pad, H]
        │
        ├─ ReduceScatter
        ▼
[ceil(s0 / TP), H]
        │
        ├─ AllGather
        ▼
[symbolic expression, H]
```

---

## 四、为什么 `_EXTRA_CTX.pad_size` 不能替代 symbolic shape

下面两段代码在某次运行中可能得到相同数值，但对 Dynamo 的含义完全不同：

```python
# 图外 Python 状态
pad_size = _EXTRA_CTX.pad_size
```

```python
# 图内 Tensor shape 数据依赖
pad_size = (-x.shape[0]) % tp_size
```

前者的数据流是：

```text
scheduler Python int
    → context外部计算
    → ForwardContext属性
    → Dynamo读取当前具体值
    → constant或guard
```

后者的数据流是：

```text
dynamic Tensor dim
    → SymInt s0
    → FX中的符号算术
    → runtime shape expression
```

因此问题不在 `_EXTRA_CTX` getter 本身，而在于该值的来源不属于 FX graph 的
Tensor数据依赖。当前正确原则是：

> compiled graph 中决定 Tensor shape 的 token 数，应从 Tensor
> `shape/size` 派生；forward context 中的 Python token 数用于调度、运行时策略、
> graph dispatch 或最终业务裁剪。

当前代码仍存在 `_EXTRA_CTX.pad_size` 的 runtime 使用，不能笼统理解为该字段已被
删除。[VERIFY: vllm_ascend/ops/register_custom_ops.py:102]
[VERIFY: vllm_ascend/ops/register_custom_ops.py:170]

判断具体使用是否安全，需要继续确认它位于：

1. Dynamo正在捕获的 Python/FakeTensor shape 路径；还是
2. custom op 的真实 runtime implementation。

前者可能把首次值常量化；后者可以使用当前 forward context 处理当前真实 Tensor，
但 fake implementation 的输出 shape 仍必须独立、可推导，并与 runtime
implementation保持一致。

---

## 五、vLLM + vllm-ascend compilation 框架导读

这一章只建立从 Dynamo 到 ACL Graph replay 的主干，不展开每个 pass 和
attention backend。建议后续按文末的文件阅读顺序逐层深入。

### 5.1 总体分层

当前 compilation 链路可以分成五层：

```text
模型与输入
  │
  │ @support_torch_compile
  │ mark_dynamic(input, dim=0)
  ▼
TorchDynamo
  │ Python bytecode → 一张 FX Graph
  │ vLLM默认过滤掉运行时 guards
  ▼
VllmBackend
  │ 按 splitting_ops 切分 FX Graph
  │ splitting op 自身保留为 eager/runtime 边界
  ▼
PiecewiseBackend
  │ 为每个可编译子图建立 compile ranges
  │ 每个 range 调用平台 CompilerInterface
  ▼
AscendCompiler
  │ Ascend graph fusion passes
  │ npugraph_ex / torchair backend
  ▼
ACLGraphWrapper
  │ 根据 CUDAGraphMode + BatchDescriptor 分发
  ├─ 首次：torch.npu.graph capture
  └─ 后续：NPUGraph.replay
```

这里有两个必须区分的维度：

1. **piecewise compilation** 决定 FX graph 如何切分、每段如何编译；
2. **piecewise/full graph capture** 决定哪些已经可运行的 callable 被录制为
   ACL Graph。

二者相关但不等价。上游官方文档也明确说明 full graph capture 可以独立于
piecewise compilation存在，而 piecewise graph capture 依赖相应的切图结果。

### 5.2 第一层：模型装饰器与 Dynamo

模型类通过 `@support_torch_compile` 接入 vLLM compilation wrapper。当前
DeepSeek V4 模型是本仓库可直接阅读的例子。
[VERIFY: vllm_ascend/models/deepseek_v4.py:996]

装饰器主要完成：

1. 从 `forward` 类型标注推导 `dynamic_arg_dims`；
2. 把 `TorchCompileWithNoGuardsWrapper` 注入模型继承关系；
3. 首次调用前执行 `_mark_dynamic_inputs()`；
4. 通过 `torch.compile(fullgraph=True, backend=VllmBackend)` 启动捕获。

上游入口：

- [`support_torch_compile`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/decorators.py#L118-L250)
- [`TorchCompileWithNoGuardsWrapper`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/wrapper.py#L47-L154)

#### Guard策略

在非 `STOCK_TORCH_COMPILE` 模式下，vLLM默认使用
`torch.compiler.skip_all_guards_unsafe` 过滤所有 Dynamo guards。
`DynamicShapesConfig.evaluate_guards` 默认也是 `False`：

- [`wrapper.py` guard filter](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/wrapper.py#L98-L125)
- [`DynamicShapesConfig`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/config/compilation.py#L346-L370)

这意味着不能依赖普通 `torch.compile` 的 guard failure 自动重新捕获正确分支。
模型结构、Python bool和forward context若参与 compiled Python控制流，必须由
服务级不变量、显式 graph regime、descriptor分发或 custom-op runtime contract
保证正确。

`enable_cpp_symbolic_shape_guards=False` 是另一个独立设置：它关闭 symbolic
shape guard 的 C++ 编译优化以减少编译时间，不是决定是否保留 guard 的主开关：

- [`decorators.py` symbolic guard配置](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/decorators.py#L617-L628)

### 5.3 第二层：VllmBackend切分 FX graph

Dynamo完成 bytecode分析后，将完整 FX graph 和 example inputs 交给
`VllmBackend`。backend根据 `compilation_config.splitting_ops` 调用
`split_graph()`，把完整 graph 拆成：

- splitting op：作为不可合入相邻 compiled region 的边界；
- 普通子图：交给 `PiecewiseBackend` 编译。

上游关键位置：

- [`split_graph`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/backends.py#L548-L622)
- [`VllmBackend` 执行切图](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/backends.py#L1142-L1216)

典型设计意图是把 attention 等难以图捕获或需要runtime metadata的操作作为
切分边界，使两侧以 token-wise计算为主的区域仍可编译和捕获。具体 splitting
ops 由 compilation config 与平台/attention能力共同决定。

`PiecewiseCompileInterpreter` 遍历切分后的模块，为每个待编译子图创建
`PiecewiseBackend`，随后通过 `wrap_with_cudagraph_if_needed()` 决定是否再包一层
graph wrapper：

- [`PiecewiseCompileInterpreter`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/backends.py#L682-L771)

### 5.4 第三层：PiecewiseBackend按range编译

`PiecewiseBackend` 的职责不是再次做Dynamo捕获，而是管理已经切出的FX子图：

1. 从 `CompilationConfig` 取得 `compile_ranges`；
2. 为每个 range 建立 `RangeEntry`；
3. 单点 range 把 symbolic example inputs concretize到指定size；
4. 区间 range 继续使用带symbol的 FakeTensor inputs；
5. 调用平台 `CompilerManager`/`CompilerInterface` 得到 runnable；
6. runtime根据shape选择已编译的 range entry。

上游关键位置：

- [`PiecewiseBackend` 初始化与ranges](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/piecewise_backend.py#L86-L192)
- [`compile_all_ranges`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/piecewise_backend.py#L245-L277)
- [`create_concrete_args`](https://github.com/vllm-project/vllm/blob/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation/piecewise_backend.py#L37-L76)

因此“模型输入 token是symbol”与“某个capture size使用具体shape”可以同时成立：

```text
Dynamo总图：s0
    ├─ general range：[s0, H]
    ├─ single-size entry：[128, H]
    ├─ single-size entry：[256, H]
    └─ single-size entry：[512, H]
```

symbol负责表达通用关系；range和capture size负责生成可高效执行、可图捕获的具体
实例。

### 5.5 第四层：Ascend平台注入编译实现

vllm-ascend通过平台接口把三个实现交给上游框架：

```text
PassManager      → GraphFusionPassManager
CompilerInterface→ AscendCompiler
GraphWrapper     → ACLGraphWrapper
```

其中 pass manager和compiler注册位置为：
[VERIFY: vllm_ascend/platform.py:195]
[VERIFY: vllm_ascend/platform.py:203]

#### Ascend graph passes

`GraphFusionPassManager` 根据当前 `compile_range` 选择适用的pass，然后修改并
重新编译FX graph。默认配置包含 norm-quant、QKNorm-RoPE、
allreduce-RMSNorm等融合，也可以加入sequence parallel passes。
[VERIFY: vllm_ascend/compilation/graph_fusion_pass_manager.py:25]
[VERIFY: vllm_ascend/compilation/graph_fusion_pass_manager.py:36]

#### AscendCompiler

`AscendCompiler.compile()` 接收上游传下来的：

```text
FX GraphModule
example_inputs
compiler_config
compile_range
```

然后根据Ascend配置选择：

- `npugraph_ex_compile()`；
- 或 `fusion_pass_compile()`。

后者通过 AOTAutograd/FX compile路径执行 Ascend fusion pass manager；前者配置
npugraph_ex/torchair NPU backend，并把FX graph交给对应backend。
[VERIFY: vllm_ascend/compilation/compiler_interface.py:47]
[VERIFY: vllm_ascend/compilation/compiler_interface.py:226]
[VERIFY: vllm_ascend/compilation/compiler_interface.py:255]

这里应重点追踪 `compile_range`：同一个融合pass可能只对某些token range生效。
`GraphFusionPassManager` 在每次执行时从pass context读取该range，再调用
`pass_.is_applicable_for_range()`。
[VERIFY: vllm_ascend/compilation/graph_fusion_pass_manager.py:36]

### 5.6 第五层：运行时CUDAGraph分发与ACL Graph捕获

model runner先根据当前batch特征调用 `cudagraph_dispatcher.dispatch()`，得到：

```text
CUDAGraphMode
BatchDescriptor
```

必要时dispatcher会把实际token数pad到某个capture size；随后这两个值被写入
forward context，供外层/子图的 `ACLGraphWrapper` 使用。
[VERIFY: vllm_ascend/worker/model_runner_v1.py:2853]
[VERIFY: vllm_ascend/worker/model_runner_v1.py:2866]

初始化阶段，model runner还会：

1. 查询所有 attention backend 的 graph支持级别；
2. 解析最终 `cudagraph_mode` 与capture sizes；
3. 初始化dispatcher keys；
4. 为Ascend graph runtime准备按size索引的workspace/event/handle结构。

[VERIFY: vllm_ascend/worker/model_runner_v1.py:4828]
[VERIFY: vllm_ascend/worker/model_runner_v1.py:4849]

`ACLGraphWrapper.__call__()` 的runtime逻辑是：

```text
读取 forward_context
    ├─ runtime mode不匹配/NONE → 直接调用runnable
    └─ runtime mode匹配
         ├─ descriptor尚未捕获
         │    └─ with torch.npu.graph(...): runnable(...)
         └─ descriptor已有entry
              └─ entry.aclgraph.replay()
```

wrapper以 `BatchDescriptor` 为cache key；首次创建 `torch.npu.NPUGraph` 并在
`torch.npu.graph()` 内执行底层runnable，之后复用同一entry replay。
[VERIFY: vllm_ascend/compilation/acl_graph.py:64]
[VERIFY: vllm_ascend/compilation/acl_graph.py:152]
[VERIFY: vllm_ascend/compilation/acl_graph.py:166]
[VERIFY: vllm_ascend/compilation/acl_graph.py:206]
[VERIFY: vllm_ascend/compilation/acl_graph.py:257]

需要特别注意：`ACLGraphWrapper` 不负责把任意runtime shape复制进统一persistent
buffer。它信任外层已经选对runtime mode、descriptor、padding和输入地址；
DEBUG模式只额外检查capture与replay时Tensor地址一致。
[VERIFY: vllm_ascend/compilation/acl_graph.py:80]
[VERIFY: vllm_ascend/compilation/acl_graph.py:248]

### 5.7 建议的源码阅读顺序

后续深入时，建议按以下顺序阅读，避免一开始陷入NPU算子细节：

1. `vllm/compilation/decorators.py`
   - 看模型如何进入compile、dynamic dim如何声明。
2. `vllm/compilation/wrapper.py`
   - 看 `torch.compile` 参数、guard过滤和首次/后续调用。
3. `vllm/compilation/backends.py`
   - 看 Dynamo交付的整图如何按 splitting ops切分。
4. `vllm/compilation/piecewise_backend.py`
   - 看compile ranges、concrete args和runtime range dispatch。
5. `vllm_ascend/platform.py`
   - 看Ascend向上游注册哪些编译扩展点。
6. `vllm_ascend/compilation/compiler_interface.py`
   - 看FX graph如何进入npugraph_ex/torchair或fusion passes。
7. `vllm_ascend/compilation/graph_fusion_pass_manager.py`
   - 看Ascend passes如何按compile range运行。
8. `vllm_ascend/worker/model_runner_v1.py`
   - 看batch padding、mode与descriptor如何生成。
9. `vllm_ascend/compilation/acl_graph.py`
   - 看最终capture/cache/replay。

定位一次具体故障时，应沿相反方向检查：

```text
错误的ACL replay
  ← descriptor / padding是否正确
  ← compiled range是否正确
  ← fake impl shape是否正确
  ← FX graph中的symbol是否仍存在
  ← Dynamo是否捕获了错误Python分支
```

---

## 六、参考资料

### 6.1 PyTorch官方资料

1. [Dynamo Overview](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_overview.html)
   - Dynamo如何通过CPython frame evaluation与bytecode analysis提取FX graph。
2. [Dynamic Shapes Core Concepts](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html)
   - `SymInt`、hint、guard、runtime assert和动态shape的核心概念。
3. [Dynamic Shapes](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)
   - `mark_dynamic`、`maybe_mark_dynamic`、`mark_unbacked`的使用边界。
4. [torch.compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
   - `TORCH_LOGS`、graph break、recompile和guard诊断入口。
5. [torch.compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
   - `fullgraph`、`dynamic`、backend和options参数定义。

### 6.2 vLLM官方资料

1. [torch.compile integration](https://docs.vllm.ai/en/stable/design/torch_compile/)
   - vLLM切图、piecewise compilation与piecewise/full CUDAGraph的官方设计说明。
2. [How to debug the vLLM-torch.compile integration](https://docs.vllm.ai/en/stable/design/debug_vllm_compile/)
   - 如何分别关闭torch.compile、Inductor和CUDAGraph，以及如何用
     `TORCH_TRACE`/`tlparse`收集产物。
3. [CompilationConfig API](https://docs.vllm.ai/en/latest/api/vllm/config/compilation/)
   - `CompilationMode`、`CUDAGraphMode`、dynamic shapes、compile sizes和capture
     sizes配置定义。
4. [当前仓库固定的上游vLLM源码](https://github.com/vllm-project/vllm/tree/967c5c3bc38891f4465d3f4e99917ed837bb3833/vllm/compilation)
   - 分析当前vllm-ascend时应优先阅读该固定commit，而不是直接用持续变化的
     vLLM `main`。[VERIFY: .github/vllm-main-verified.commit:1]

### 6.3 当前仓库相关资料

1. [`flashcomm-dbo-compile-root-cause.md`](./flashcomm-dbo-compile-root-cause.md)
   - FlashComm + DBO进入Dynamo fullgraph时的具体故障链。
2. [`analysis-dbo-fc2-compile-startup.md`](./analysis-dbo-fc2-compile-startup.md)
   - DBO、compile、graph capture、dynamic shape与通信hook的联合分析。
3. [`flashcomm-source-code.md`](./flashcomm-source-code.md)
   - FlashComm token维通信、custom op real/fake implementation及调用路径。
4. [`analysis-flashcomm2-deep-dive.md`](./analysis-flashcomm2-deep-dive.md)
   - FlashComm2通信与执行路径的进一步分析。

### 6.4 推荐调试工具

```bash
# Dynamo dynamic-shape与guard日志
TORCH_LOGS="+dynamic,guards,recompiles" ...

# 生成Torch compile结构化trace
TORCH_TRACE=/tmp/torch_trace ...

# 浏览trace
tlparse /tmp/torch_trace
```

PyTorch dynamic-shape文档推荐使用 `TORCH_LOGS=dynamic` 检查symbol创建和guard
产生原因；vLLM调试文档推荐使用 `TORCH_TRACE` 与 `tlparse` 检查Dynamo graph、
piecewise split和后端编译产物。生产压测不应默认开启这些高开销日志。
