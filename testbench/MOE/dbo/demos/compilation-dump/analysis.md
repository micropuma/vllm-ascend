# 编译各阶段分析指南

> 以 `cache/20260705_094422/`（`DUMP_MODE=all`, FC1=1, FC2=0, DBO=1, HCCL_MODE=AI_CPU, TP=2, DeepSeek-V2-Lite）为例。

整个编译链路按时间顺序分为 **5 个阶段**。前 3 个对应 README 中的三层编译图，第 4 个是 Graph Capture，阶段 0 是前置初始化。

---

## 目录

- [阶段 0：模型初始化 (Model Init)](#阶段-0模型初始化-model-init)
- [阶段 1：Dynamo 捕获 (Layer 1 — FX Graph)](#阶段-1dynamo-捕获-layer-1--fx-graph)
- [阶段 2：vllm-ascend 切图与编译 (Layer 2 — Subgraphs)](#阶段-2vllm-ascend-切图与编译-layer-2--subgraphs)
- [阶段 3：Backend AOT 编译 (Layer 3 — ACL Graph)](#阶段-3backend-aot-编译-layer-3--acl-graph)
- [阶段 4：NPU Graph Capture](#阶段-4npu-graph-capture)
- [编译全流程时间线](#编译全流程时间线)
- [各阶段日志搜索速查](#各阶段日志搜索速查)

---

## 阶段 0：模型初始化 (Model Init)

**时间**：`09:44:31` — `09:45:52`（约 81s）

**发生了什么**：vLLM serve 进程启动 → Ascend plugin 注册 → 权重加载 → `compilation_config` 生成 → EngineCore 初始化。

### 日志关键行

```bash
grep -n "Platform plugin\|Using OOT custom backend\|AscendCompiler hash\|Initializing a V1\|compilation_config\|model loaded" \
  logs/server_20260705_094422.log
```

**实际输出**（来自 `logs/server_20260705_094422.log`）：

```
L83:  INFO: compilation_config: {
        'backend': '...AscendCompiler',
        'splitting_ops': ['vllm::unified_attention_with_output', ..., 'vllm::mla_forward'],
        'compile_ranges_endpoints': [16384],
        'cudagraph_capture_sizes': [16],
        'cudagraph_mode': FULL_AND_PIECEWISE,
        'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, ...},
        'fast_moe_cold_start': True,
      }
L136: INFO: Initializing a V1 LLM engine (v0.22.1)
L192: INFO: Using OOT custom backend for compilation.
```

### 产物

| 文件 | 说明 |
|---|---|
| [`cache/20260705_094422/vllm_cache/modelinfos/vllm-model_executor-models-deepseek_v2-DeepseekV2ForCausalLM.json`](cache/20260705_094422/vllm_cache/modelinfos/vllm-model_executor-models-deepseek_v2-DeepseekV2ForCausalLM.json) | 模型结构 hash + 元信息 |

### 值得关注的参数

从 `compilation_config` 中重点关注：

| 参数 | 本次值 | 含义 |
|---|---|---|
| `splitting_ops` | 18 个 op（含 `vllm::mla_forward`, `vllm::deepseek_v4_attention`） | 在这些 op 处将完整 FX Graph 切为独立编译段 |
| `compile_ranges_endpoints` | `[16384]` | 以 token 数 16384 为界切分 piecewise compile range |
| `cudagraph_capture_sizes` | `[16]` | 只 capture batch_size=16 的 graph |
| `pass_config.fuse_norm_quant` | `True` | 启用 norm_quant fusion pass |
| `fast_moe_cold_start` | `True` | 启用快速 MoE 冷启动 |

---

## 阶段 1：Dynamo 捕获 (Layer 1 — FX Graph)

**时间**：`09:45:46` — `09:46:28`（约 42s，从 torch_trace 首条事件到 compile 开始）

**发生了什么**：TorchDynamo 逐条 trace Python bytecode → 构建完整 FX Graph → 推导 symbolic shapes → 生成 guard 条件。

### 日志关键行

```bash
grep -n "fullgraph\|recompile\|graph_break\|torch.compile\|create_aot" \
  logs/server_20260705_094422.log
```

```bash
# 检查 recompile 次数（>0 说明有 dynamic shape 变化）
grep -c "recompile" logs/server_20260705_094422.log
```

### 产物：TORCH_TRACE（Chrome Trace 格式）

| 文件 | 大小 | 说明 |
|---|---|---|
| [`cache/20260705_094422/torch_trace/dedicated_log_torch_trace_o0enkxkx.log`](cache/20260705_094422/torch_trace/dedicated_log_torch_trace_o0enkxkx.log) | 1.7 MB, 14518 行 | rank_0 的 Dynamo trace |
| [`cache/20260705_094422/torch_trace/dedicated_log_torch_trace_vtgnwnv2.log`](cache/20260705_094422/torch_trace/dedicated_log_torch_trace_vtgnwnv2.log) | 1.7 MB, 14518 行 | rank_1 的 Dynamo trace |

### 如何分析

**1. Chrome Trace 可视化**：

```bash
# 用 tlparse 打开
tlparse cache/20260705_094422/torch_trace/

# 或者直接用 Chrome 打开 JSON（trace 文件是 JSONL 格式，每行一个事件）
# 需要先把 JSONL 转为 Chrome Trace 数组格式：
python3 -c "
import json
events = []
with open('cache/20260705_094422/torch_trace/dedicated_log_torch_trace_o0enkxkx.log') as f:
    for line in f:
        line = line.strip()
        if line.startswith('V'):
            # 提取 JSON 部分（跳过 protobuf header）
            idx = line.index('{')
            events.append(json.loads(line[idx:]))
with open('/tmp/trace.json', 'w') as f:
    json.dump(events, f)
"
# 然后在 chrome://tracing 中加载 /tmp/trace.json
```

**2. 关键事件提取**：

实际 trace 中包含的关键 Dynamo 阶段（来自 trace 文件前 50 行）：

```json
{"name": "create_aot_dispatcher_function", "cat": "dynamo_timed", ...}
{"name": "aot_collect_metadata",           "cat": "dynamo_timed", ...}
{"name": "aot_trace_joint_graph",          "cat": "dynamo_timed", ...}
```

每个子图编译都会经过 `create_aot → collect_metadata → trace_joint → compile_fx → inductor_compile` 的流水线。

**3. 排查 Dynamo 问题**：

| 问题 | 排查方法 |
|---|---|
| Graph break 导致非 fullgraph | `grep "graph_break" server log`，检查 `TORCH_LOGS="+graph_breaks"` 输出 |
| 多次 recompile（dynamic shape 不稳定） | `grep "recompile" server log`，配合 `TORCH_LOGS="+guards"` 看 guard 变化原因 |
| DBO hook 被常量化剪枝 | 在 `load_model()` 后插桩 dump FX Graph code，搜索 `dbo_linear_column_hook` |

---

## 阶段 2：vllm-ascend 切图与编译 (Layer 2 — Subgraphs)

**时间**：`09:46:28` — `09:47:08`（约 40s，从 hash factors 到最后一个 subgraph 保存）

**发生了什么**：

```
完整 FX Graph
  → VllmBackend 按 splitting_ops (18个) 切分为大段
  → PiecewiseBackend 按 compile_ranges=[16384] 细切
  → Ascend fusion passes (fuse_norm_quant, fuse_act_quant)
  → 每个子图独立 torch.compile(AscendCompiler)
  → 保存到 vllm_compile_cache/
```

### 产物总览

```
cache/20260705_094422/vllm_compile_cache/
├── rank_0_0/backbone/                     # TP rank 0 的编译缓存
│   ├── computation_graph.py               # 1800 行，完整 computation graph Python 代码
│   ├── vllm_compile_cache.py              # 子图索引表
│   ├── cache_key_factors.json             # hash 因子
│   └── artifact_compile_range_1_16384_subgraph_{0..27}  # 28 个编译产物
└── rank_1_0/backbone/                     # TP rank 1（结构完全相同）
    ├── computation_graph.py
    ├── vllm_compile_cache.py
    ├── cache_key_factors.json
    └── artifact_compile_range_1_16384_subgraph_{0..27}  # 28 个编译产物
```

### 子图数量

```bash
find cache/20260705_094422/vllm_compile_cache/ -name "artifact_compile_range_*" | wc -l
# → 56（2 ranks × 28 subgraphs）
```

每个 rank 各 28 个子图，说明 DeepSeek-V2-Lite（27 layers）的模型在 `compile_range=(1,16384)` 下被切为 28 个编译单元。

### 关键文件分析

#### [`computation_graph.py`](cache/20260705_094422/vllm_compile_cache/rank_0_0/backbone/computation_graph.py)（1800 行）

这是缓存的完整 computation graph 的 Python 代码。可以查看 Dynamo 捕获后的图结构：

```bash
# 看整体结构（forward 参数 = 模型输入 + 所有权重）
head -5 cache/20260705_094422/vllm_compile_cache/rank_0_0/backbone/computation_graph.py
```

实际内容（第 1-4 行）：
```python
class GraphModule(torch.nn.Module):
    def forward(self, s72: "Sym(s72)",                    # symbolic batch size
                     L_input_ids_: "i32[s72]",             # input tokens
                     L_self_modules_embed_tokens_...: "bf16[51200, 2048]",  # embedding weight
                     L_self_modules_layers_modules_0_...: "bf16[2048]",     # layer 0 norm weight
                     ...):
```

图中包含 54 个 submod（`submod_0` 到 `submod_54`），对应 28 个编译子图 + embedding + lm_head 等。

**查看子图边界**：
```bash
grep "class submod_" cache/20260705_094422/vllm_compile_cache/rank_0_0/backbone/computation_graph.py
# → submod_0, submod_1, submod_2, ..., submod_54
```

**看最后一个 submod（submod_54）**（对应 layer 26 的 post_attention → MoE → final norm）：

来自 [`computation_graph.py:1755-1800`](cache/20260705_094422/vllm_compile_cache/rank_0_0/backbone/computation_graph.py#L1755)：

```python
class submod_54(torch.nn.Module):
    def forward(self, ...):
        # MLA attention output reshape
        view = output_79.view(-1, 2048)
        # residual add
        maybe_chunk_residual = torch.ops.vllm.maybe_chunk_residual(view, residual_105)
        # RMSNorm fusion (Ascend custom op): norm + residual
        npu_add_rms_norm_bias = torch.ops._C_ascend.npu_add_rms_norm_bias(...)
        # Gate linear (fp32)
        to = view_1.to(torch.float32)
        linear = torch._C._nn.linear(to, gate_weight, None)
        # MoE forward
        moe_forward_shared = torch.ops.vllm.moe_forward_shared(view_1, linear, ...)
        # AllReduce TP
        maybe_all_reduce_tensor_model_parallel = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(add)
        # Final RMSNorm
        npu_add_rms_norm_bias_1 = torch.ops._C_ascend.npu_add_rms_norm_bias(...)
        return getitem_5
```

**从 `computation_graph.py` 可以分析**：
- 哪些 op 被 Dynamo 捕获到了（及哪些被 graph break 排除了）
- Ascend custom op 的分布（`torch.ops._C_ascend.*`, `torch.ops.vllm.*`）
- 权重参数的 shape 和 dtype

#### [`vllm_compile_cache.py`](cache/20260705_094422/vllm_compile_cache/rank_0_0/backbone/vllm_compile_cache.py)

子图索引表，映射 `(compile_range, subgraph_idx, backend)` → `artifact 文件路径`：

```python
{ ((1, 16384), 0, 'AscendCompiler'):  {'graph_handle': ('artifact_compile_range_1_16384_subgraph_0', '.../artifact_compile_range_1_16384_subgraph_0')},
  ((1, 16384), 1, 'AscendCompiler'):  {'graph_handle': ('artifact_compile_range_1_16384_subgraph_1', '.../artifact_compile_range_1_16384_subgraph_1')},
  ...
  ((1, 16384), 27, 'AscendCompiler'): {'graph_handle': ('artifact_compile_range_1_16384_subgraph_27', '.../artifact_compile_range_1_16384_subgraph_27')}}
```

#### [`cache_key_factors.json`](cache/20260705_094422/vllm_compile_cache/rank_0_0/backbone/cache_key_factors.json)

```json
{
  "code_hash": "571226a170c7434df17f4a74197b2c026046906915b39c1a22e8ff4f44522680",
  "compiler_hash": "f33afdc634",
  "config_hash": "28c1b4edd4",
  "env": { ... }   // 所有 VLLM_* 环境变量的快照
}
```

**值得注意的 env 变量**（从实际缓存中提取）：

| 变量 | 值 | 关联 |
|---|---|---|
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 未出现在 env 中 | 说明 FC1 不在 hash 因子里 — 确认了已知缺陷 |
| `VLLM_USE_AOT_COMPILE` | `true` | AOT 编译模式 |
| `VLLM_USE_STANDALONE_COMPILE` | `true` | 独立编译进程 |
| `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` | `1024` | DBO 多 stream GEMM 阈值 |
| `VLLM_DBO_COMM_SMS` | `20` | DBO communication SMs |

### 日志中的 compile 事件

```bash
grep "Saved compiled graph to cache" logs/server_20260705_094422.log | head -5
```

实际输出：
```
09:46:32 Saved compiled graph to cache: .../artifact_compile_range_1_16384_subgraph_0 (rank_0)
09:46:32 Saved compiled graph to cache: .../artifact_compile_range_1_16384_subgraph_0 (rank_1)
09:46:33 Saved compiled graph to cache: .../artifact_compile_range_1_16384_subgraph_1 (rank_0)
09:46:33 Saved compiled graph to cache: .../artifact_compile_range_1_16384_subgraph_1 (rank_1)
...
```

两个 rank 并行编译，每个子图 ~1-2s，总共 28 个子图 × 2 ranks ≈ 40s。

---

## 阶段 3：Backend AOT 编译 (Layer 3 — ACL Graph)

**时间**：与阶段 2 交织（每个子图 compile 后立即做 AOT）

**发生了什么**：每个子图 → `AscendCompiler.compile()` → `enable_npugraph_ex` 开启 → `torch.npu.graph` capture → ACL Graph 序列化为 `.model` 文件。

### 产物

```
cache/20260705_094422/vllm_cache/torch_compile_cache/torch_aot_compile/
└── c2b64376b86bd8b02053bc88e4ea844e1da0d529641926301d8d5ccad0360a8f/   ← AOT hash
    ├── rank_0_0/model    (2.4 MB, binary)
    └── rank_1_0/model    (2.4 MB, binary)
```

```bash
find cache/20260705_094422 -path "*/torch_aot_compile/*" -name "model" -exec ls -lh {} \;
# -rw-r--r-- ... 2.4M ... rank_0_0/model
# -rw-r--r-- ... 2.4M ... rank_1_0/model
```

### 如何分析

| 关注点 | 方法 |
|---|---|
| AOT hash 是否匹配配置 | 对比不同 FC1/FC2/HCCL_MODE 下的 hash 目录名是否不同 |
| AOT 产物大小是否合理 | 2.4 MB per rank（DeepSeek-V2-Lite TP=2）|
| 文件类型 | `file` 命令显示为 `data`（torch binary serialization） |
| compile 是否完整 | 两个 rank 的 `model` 文件都存在且非 0 字节 |

### AscendCompiler hash

从日志中确认 hash 构成：

```bash
grep "AscendCompiler hash factors" logs/server_20260705_094422.log
# AscendCompiler hash factors: {
#   'torch_npu_version': '2.10.0',
#   'enable_npugraph_ex': True,
#   'enable_static_kernel': False
# }
```

> **已知缺陷**：hash 只包含 3 个因子，不包括 DBO/FC1/FC2/TP/model arch。不同配置可能生成相同 cache hash 导致错误命中。因此必须通过 `server.sh` 的冷启动目录隔离。

### triton_cache 目录

Ascend NPU 上不使用 Triton kernel，但某些 dispatch kernel（如 `_compute_slot_mapping_kernel`）仍会走 triton cache 路径：

```
cache/20260705_094422/triton_cache/
├── 83CaCO.../_compute_slot_mapping_kernel.ttir    # Triton IR
├── 83CaCO.../_compute_slot_mapping_kernel.ttadapter  # Ascend adapter
├── 83CaCO.../_compute_slot_mapping_kernel.npubin     # NPU binary
└── jzHx-Y.../npu_utils.so                            # NPU utility shared lib
```

这些是 NPU 上的辅助 kernel（非 MoE/attention 主干），由 `torch_npu` 自动编译。

---

## 阶段 4：NPU Graph Capture

**时间**：`09:47:08` — `09:47:12`（约 4s）

### 日志关键行

```bash
grep -n "Capturing CUDA graphs\|Graph capturing finished\|Application startup" \
  logs/server_20260705_094422.log
```

实际输出：
```
L1615: Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████| 1/1 [00:01<00:00, 1.58s/it]
L1616: Capturing CUDA graphs (decode, FULL):             100%|████| 1/1 [00:01<00:00, 1.51s/it]
L1619: Graph capturing finished in 4 secs, took 0.24 GiB
L1691: Application startup complete.
```

### 发生了什么

`cudagraph_mode=FULL_AND_PIECEWISE`，`capture_sizes=[16]`：

1. **PIECEWISE capture**（1.58s）：对 28 个 piecewise 子图分别 capture batch=16 的 NPU graph
2. **FULL capture**（1.51s）：capture 完整的 decode batch=16 graph
3. **总计** 4s，占用 0.24 GiB NPU 显存

### 验证

```bash
# 确认 capture 成功
grep -c "Graph capturing finished" logs/server_20260705_094422.log
# → 2（每个 rank 一次）

# 确认最终 ready
grep "Application startup complete" logs/server_20260705_094422.log
# → Application startup complete.
```

---

## 编译全流程时间线

以本次运行 `20260705_094422` 的实际时间戳为例：

```
09:44:31  启动 vllm serve
09:44:38  Platform plugin ascend activated
09:44:41  compilation_config 生成
09:45:00  Enabled custom fusions: norm_quant, act_quant
09:45:18  Initializing a V1 LLM engine        ← 阶段 0 开始
09:45:46  Dynamo trace 开始 (第一个 chromium event)  ← 阶段 1 开始
09:45:52  Using OOT custom backend
09:46:28  AscendCompiler hash factors          ← 阶段 2 开始
09:46:32  subgraph_0 编译完成（rank_0, rank_1）
09:46:33  subgraph_1
  ...
09:46:57  subgraph_27（最后一个）               ← 阶段 2 结束
09:47:08  Graph capture 开始                   ← 阶段 4 开始
09:47:12  Graph capturing finished in 4 secs   ← 阶段 4 结束
09:47:??  Application startup complete         ← server ready
```

**各阶段耗时**：

| 阶段 | 内容 | 耗时（约） | 从日志看 |
|---|---|---|---|
| 0 | 模型加载 + Engine 初始化 | ~80s | `Initializing a V1 LLM engine` → `Using OOT custom backend` |
| 1 | Dynamo trace (FX Graph) | ~40s | torch_trace 首条 → compile 开始 |
| 2 | 切图 + torch.compile (28 subgraphs × 2 ranks) | ~40s | `AscendCompiler hash` → 最后一个 `Saved compiled graph` |
| 3 | Backend AOT (npugraph_ex → ACL) | 与阶段2交织 | AOT hash 目录中的 `model` 文件时间戳 |
| 4 | NPU Graph Capture | ~4s | `Capturing CUDA graphs` → `Graph capturing finished` |
| **合计** | | **~165s** | |

> 阶段 1-3 内部有交织，总和大于实际墙钟时间。实际从 `Initializing engine` (09:45:18) 到 `Application startup complete` (09:47:12+) 约为 ~114s。

---

## 各阶段日志搜索速查

以下命令直接使用 `cache/20260705_094422/` 和对应的 server log：

```bash
RUN="cache/20260705_094422"
LOG="logs/server_20260705_094422.log"

# === 阶段 0 ===
echo "=== 阶段 0: 模型初始化 ==="
grep -n "Platform plugin\|Using OOT custom backend\|AscendCompiler hash\|Initializing a V1" "$LOG" | head -10
echo ""
echo "compilation_config 中的 splitting_ops:"
grep "splitting_ops" "$LOG" | head -1

# === 阶段 1 ===
echo ""
echo "=== 阶段 1: Dynamo 捕获 ==="
echo "TORCH_TRACE 产物:"
ls -lh "$RUN/torch_trace/"
echo ""
echo "Dynamo 关键事件数:"
grep -c "create_aot_dispatcher\|aot_trace_joint\|inductor_compile" "$RUN/torch_trace/"*.log 2>/dev/null || echo "(需手动解析 trace 文件)"

# === 阶段 2 ===
echo ""
echo "=== 阶段 2: 切图编译 ==="
echo "子图数量:"
find "$RUN/vllm_compile_cache/" -name "artifact_compile_range_*" | wc -l
echo ""
echo "cache_key_factors:"
cat "$RUN/vllm_compile_cache/rank_0_0/backbone/cache_key_factors.json" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'  code_hash:     {d[\"code_hash\"][:16]}...')
print(f'  compiler_hash: {d[\"compiler_hash\"]}')
print(f'  config_hash:   {d[\"config_hash\"]}')
"
echo ""
echo "computation_graph 的 submod 数量:"
grep -c "class submod_" "$RUN/vllm_compile_cache/rank_0_0/backbone/computation_graph.py"

# === 阶段 3 ===
echo ""
echo "=== 阶段 3: Backend AOT ==="
echo "AOT compile 产物:"
find "$RUN" -path "*/torch_aot_compile/*" -name "model" -exec ls -lh {} \;
echo ""
echo "AOT hash:"
find "$RUN" -path "*/torch_aot_compile/*" -name "model" | head -1 | sed 's|.*/torch_aot_compile/||' | cut -d'/' -f1

# === 阶段 4 ===
echo ""
echo "=== 阶段 4: Graph Capture ==="
grep -n "Capturing CUDA\|Graph capturing finished\|Application startup" "$LOG"
```

---

## 相关文档

- [README.md](./README.md) — 工具套件总览与快速开始
- `testbench/MOE/dbo/docs/compilation/torch-compile-guide.md` — Dynamo → VllmBackend → PiecewiseBackend → ACL Graph 全链路
- `testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md` — 冷启动 shape contract 断裂的定位与修复
- `testbench/MOE/dbo/docs/analysis/analysis-dbo-fc2-compile-startup.md` — DBO + compile + FlashComm 交叉分析
