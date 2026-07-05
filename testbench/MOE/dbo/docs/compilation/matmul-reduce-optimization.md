# MatMul-Reduce 通算融合优化

vllm-ascend 通过 **编译期图重写 + 运行时融合算子** 两层机制，将 Transformer 模型中的 MatMul + 集合通信串行模式替换为通算融合模式，达到计算与通信并行掩盖的效果。

## 1. 整体架构

```
PyTorch 模型代码 (正常 TP)
        │
        │  torch.compile / dynamo 追踪
        ▼
  原始 FX Graph
        │
        ├─── 第1层: Custom Op 图边界 ──────────────┐
        │     通信操作被封装为 custom op             │ 编译期
        │     fake_impl 提供 shape 推断              │
        │     _FLASH_COMM_V1_SNAPSHOT 桥接编译态     │
        ├──────────────────────────────────────────┤
        │                                          │
        ├─── 第2层: Graph Pass 图重写 ──────────────┤
        │     all_reduce → reduce_scatter + all_gather
        │     all_gather 后推到 rms_norm 之后         │
        │     按 token range 选择性应用               │
        ├──────────────────────────────────────────┤
        │                                          │
        └─── 第3层: NPU 后端编译 ───────────────────┘
              npugraph_ex / torchair → 可执行图
```
        │
        │  运行时
        ▼
  npu_mm_reduce_scatter_base (融合 kernel)
  沿 M 轴 tiling，流水并行掩盖通信
```

## 2. 第1层：Custom Op 图边界

### 2.1 为什么需要 custom op

torch.compile 的 dynamo 无法追踪 HCCL 通信库内部实现。直接追踪会导致 graph break。将通信封装为 custom op 后，dynamo 将其视为**不透明节点**，保留在 FX 图中供后续 Pass 改写。

### 2.2 编译期与运行时的双实现

每个通信 custom op 都有两个实现：

| 实现 | 用途 | 何时调用 |
|------|------|----------|
| `_impl` | 真正做通信 | 运行时 |
| `_fake_impl` | 返回正确 shape 的空张量 | dynamo fake tensor 传播时 |

```python
# vllm_ascend/ops/register_custom_ops.py

def _matmul_and_reduce_impl(input_parallel, layer_name):
    self = forward_context.no_compile_layers[layer_name]
    return self.custom_op.matmul_and_reduce(input_parallel, bias_)  # 真正执行

def _matmul_and_reduce_impl_fake(input_parallel, layer_name):
    num_tokens = input_parallel.size(0)
    if _FLASH_COMM_V1_SNAPSHOT:
        num_tokens = num_tokens // self.tp_size  # reduce_scatter 后每 rank 只拿 1/tp
    return torch.empty(size=(num_tokens, self.output_size_per_partition), ...)
```

### 2.3 `_FLASH_COMM_V1_SNAPSHOT`：编译态上下文桥梁

这是关键设计。`PiecewiseBackend.compile_all_ranges()` 在 `set_forward_context()` 上下文外部触发编译，fake_impl 此时无法读取 `get_forward_context()`。

解决方法：在 `set_ascend_forward_context()` 中，将 `flash_comm_v1_enabled` 的值同时写入一个**模块级全局变量**：

```python
# ascend_forward_context.py
set_flash_comm_v1_snapshot(flash_comm_v1_enabled)  # 写入全局快照
# ... 前向传播，torch.compile 异步编译时读取快照 ...
```

编译期看快照，运行期看 context，互不干扰。

### 2.4 `no_compile_layers`：将层对象移出编译图

```python
# vllm_ascend/ops/linear.py
compilation_config.static_forward_context[layer.prefix] = self  # 注册
```

这样 custom op 可以通过 `forward_context.no_compile_layers[layer_name]` 访问到层的 weight、bias、tp_size 等信息，而不被 torch.compile 捕获。

---

## 3. 第2层：Graph Pass 图重写

### 3.1 FX Pattern Matching 机制

模型代码按正常 TP（all_reduce）编写，编译期通过 FX pattern matching **自动替换**为 SP 通信拓扑：

```
原始图 (PyTorch eager 写的):          编译后图 (Pass 改写):
┌─────┐  ┌───────────┐  ┌──────────┐   ┌─────┐  ┌──────────────┐  ┌──────────┐  ┌───────────┐
│ attn │─►│ all_reduce│─►│add_rms   │   │ attn │─►│reduce_scatter│─►│add_rms   │─►│ all_gather│
│ out  │  │ (全量通信) │  │norm_bias │   │ out  │  │ (1/tp 通信)  │  │norm_bias │  │ (恢复全量)│
└─────┘  └───────────┘  └──────────┘   └─────┘  └──────────────┘  └──────────┘  └───────────┘
 通信量: O(N)                              通信量: O(N/P)                    通信量: O(N/P)
```

**数学等价性**：
1. 通信上：`all_reduce` ≡ `reduce_scatter` + `all_gather`
2. 计算上：AddRMSNorm 对每行独立操作，可以在分片状态下直接计算
3. `all_gather` 放在末尾，把归一化结果重新拼完整

### 3.2 核心 Pass 列表

| Pass | 文件 | 功能 |
|------|------|------|
| `SequenceParallelismPass` | `compilation/passes/sequence_parallelism.py` | `all_reduce + rms_norm` → `reduce_scatter + rms_norm + all_gather` |
| `SequenceParallelismMoePass` | `compilation/passes/sequence_parallelism_moe.py` | MoE 的 `all_gather + rms_norm` → `rms_norm + all_gather`（后推 all_gather） |
| `MatmulAllReduceAddRMSNormPass` | `compilation/passes/allreduce_rmsnorm_fusion_pass.py` | `matmul + all_reduce + rms_norm` 三合一融合 |
| `AddRMSNormQuantFusionPass` | `compilation/passes/norm_quant_fusion_pass.py` | `rms_norm + quantize` 融合 |

### 3.3 SP Pass 实现细节

```python
# sequence_parallelism.py
class MiddleAllReduceRMSNormPattern:
    def register(self, pm_pass):
        # 原始 pattern：all_reduce → rms_norm（模型代码写的）
        def pattern(input, weight, residual):
            x = self._all_reduce(input)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, weight, None, self.eps)
            return result, residual

        # 替换 pattern：reduce_scatter → rms_norm → all_gather（编译器改写的）
        def replacement(input, weight, residual):
            reduce_scatter = self._reduce_scatter(input)
            residual = torch.ops.vllm.maybe_chunk_residual(reduce_scatter, residual)
            result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                reduce_scatter, residual, weight, None, self.eps)
            all_gather = self._all_gather(result)
            return all_gather, residual

        pm.register_replacement(pattern, replacement, ...)
```

### 3.4 MoE AllGather 后推

MoE 模型中，AllGather 被后推到 rms_norm 之后：

```python
# sequence_parallelism_moe.py

# 原始: all_gather → slice(num_tokens) → rms_norm
# 改写: rms_norm(分片状态直接算) → all_gather
def replacement(input, weight, residual, num_tokens):
    result, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
        input, residual, weight, None, self.eps)  # 直接在分片上算
    all_gather = self._all_gather(result)           # 算完再汇聚
    return all_gather, residual
```

还有 NoOp 消除：`all_gather + chunk(选 rank 自己的部分)` → 恒等映射，直接消掉。

### 3.5 阈值控制

编译期按 `compile_range`（token 数量）决定是否应用 Pass：

```python
def is_applicable_for_range(self, compile_range):
    return compile_range.start >= self.min_tokens  # 默认 1000，MoE 模型 1
```

小 token 数时通信量不大，改写反而增加 kernel launch 开销。

---

## 4. 第3层：融合 Kernel 实现（MMRS）

### 4.1 算子定位

`torch_npu.npu_mm_reduce_scatter_base` 属于昇腾 MC²（MatMul-Collective-Collective）通算融合算子家族：

| 算子 | 计算 + 通信 | 适用场景 |
|------|------------|----------|
| `AllGatherMatMul` | AllGather → MatMul | TP 的前向 / SP 的反向 |
| `MatMulReduceScatter` | MatMul → ReduceScatter | **TP 的 Row-Parallel 前向** / SP 的正向 |
| `MatMulAllReduce` | MatMul → AllReduce | 数据并行的梯度同步 |

### 4.2 重叠原理：M 轴 Tiling + 流水并行

输入数据沿 M 轴（序列维度）切分为多个子块，形成计算-通信流水线：

```
原始串行（无 overlap）：
  |══════════ MatMul (M×K @ K×N) ══════════|
                                             |══════ ReduceScatter ══════|
  总时间 = T_matmul + T_comm

MMRS 融合（沿 M 轴切为 5 块，流水并行）：
  Block0: [MatMul(0)] [通信(0) ─────────────────────────────────]
  Block1:            [MatMul(1)] [通信(1) ─────────────────────]
  Block2:                       [MatMul(2)] [通信(2) ─────────]
  Block3:                                  [MatMul(3)] [通信(3)]
  Block4:                                             [MatMul(4)] [通信(4)]
           ↑                                                    ↑
           │─── 流水重叠：Block i+1 算 · Block i 通信 ──────────│
```

**切分轴选 M 轴的原因**：ReduceScatter 的 HCCL 通信要求数据内存连续。K 轴切分会切断每行，导致内存不连续；M 轴切分则每行依然连续。

### 4.3 切分配平算法

基于 profiling 提前拟合 `CostMM(m)` 和 `CostComm(m)` 曲线，根据 **bound 类型** 选择策略：

```
计算 bound（K 较大，计算是瓶颈）：     通信 bound（K 较小，通信是瓶颈）：
目标 → 计算流连续，通信尾块要短          目标 → 通信流连续，计算头块要短

  A  A  A  ...  A  B                    B  A  A  A  ...  A
  └──── 长块 ────┘└┤                    └┤└───── 长块 ─────┘
                 短尾块                   短头块
```

**配平步骤**（以通信 bound 为例）：

1. **选短块 m0**：经验公式取 min(a,b,c)，满足 L2 Cache / 内存对齐
2. **解配平方程**：`CostMM(m1) = CostComm(m0) × 1.15`（1.15 为内存带宽冲突系数）
3. **128 对齐**：`m1 = align_down(m1, 128)`，再反推 `m0 = M - m1 × count`
4. **最终切分**：`{m0, m1, m1, ..., m1}`

**案例**（M=4096, K=3072, N=8192, fp16, 8 卡）：
- 切分方案：`{512, 896, 896, 896, 896}`
- 融合前：MatMul 803μs + ReduceScatter 1071μs = **1874μs**
- 融合后：**1262μs**
- 收益：**32.7%**

### 4.4 切分膨胀风险

切分会产生额外开销，严重时**融合比不融合更慢**：

| 膨胀来源 | 原因 |
|----------|------|
| 计算效率下降 | 小矩阵无法充分利用 Cube 计算单元 |
| 调度开销 | 块数越多，kernel launch + 管理开销越大 |
| 资源竞争 | 计算和通信并行时争夺 L2 Cache / 内存带宽 |

推荐实践：理论切分 + 实测迭代调整，找到最优方案。

### 4.5 接口与约束

```python
torch_npu.npu_mm_reduce_scatter_base(
    input,         # (M, K), fp16/bf16
    x2,            # (K, N), fp16/bf16, 可转置
    hcom,          # HCCL 通信域 handle
    world_size,    # 2 / 4 / 8
    reduce_op="sum",
    bias=None,     # 当前仅支持 None
    comm_turn=0,   # 当前仅支持 0
)
```

关键约束：
- M 必须整除 world_size
- K ∈ [256, 65536)
- 仅支持 HCCS 全互联组网（A2 训练产品）
- bias 当前不支持非零输入

---

## 5. 与 DBO 的关系

| 维度 | MMRS 融合 | DBO |
|------|----------|-----|
| **粒度** | 单 kernel 内部的 M 轴流水 | 两 ubatch 跨层流水线 |
| **重叠对象** | MatMul 计算 ↔ ReduceScatter 通信 | Layer N 的通信 ↔ Layer N+1 的计算 |
| **实现层** | NPU 算子内部的细粒度调度 | CPU 线程 + stream event 同步 |
| **适用条件** | tp ≤ 8, M % world_size == 0 | 需要 batch 拆分为两个 ubatch |

两者互补：MMRS 消除 Row-Parallel 层内部的通信 tail，DBO 消除层与层之间的通信 bubble。

```
Layer N:
  ColumnParallel: [AllGather] ← DBO 重叠
  RowParallel:    [mm_reduce_scatter_base] ← MMRS 内部流水（此处无 DBO 钩子）
  MoE prepare:    [EP AllGather] ← DBO 重叠
  Expert:         [compute]
  MoE finalize:   [EP ReduceScatter] ← DBO 重叠
```

---

## 6. 启用方式

```bash
# FlashComm1（开启 SP + MMRS）
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# 可选：FlashComm2（o_proj 重编队）
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2

# 启动服务
vllm serve <model> --tensor-parallel-size 8
```

编译配置：

```python
# GraphFusionPassManager.configure() 中
config.compilation_config.pass_config.enable_sp  # 开启 SequenceParallelism 图 Pass
```

---

## 7. 上游参考

| 资源 | 链接 |
|------|------|
| torch_npu API 文档 | [npu_mm_reduce_scatter_base](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/apiref/apilist/ptaoplist_000764.html) |
| CANN aclnn 接口 | [aclnnMatmulReduceScatter](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/aolapi/context/aclnnMatmulReduceScatter.md) |
| MC² 算子源码 | [ops-transformer/mc2/matmul_reduce_scatter](https://gitcode.com/cann/ops-transformer/tree/9.0.0-beta.2/mc2/matmul_reduce_scatter) |
| Kernel 实现 | [matmul_reduce_scatter_base.h](https://gitcode.com/cann/ops-transformer/blob/b95d84f4d025e709287320b59cf15453199c35fa/mc2/matmul_reduce_scatter/op_kernel/matmul_reduce_scatter_base.h) |
| 切分算法最佳实践 | [基于Ascend C的MC²通算融合算子性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20250325-1) |
| FlashComm 架构设计 (CSDN) | [vllm-ascend计算通信优化方案FlashComm](https://hwcomputing.csdn.net/6a0d698b10ee7a33f273e68c.html) |
