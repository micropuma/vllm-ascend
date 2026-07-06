# Compilation Dump 工具套件

冷启动 + 渐进式编译图 dump，用于分析 DeepSeek-V2 DBO 在 Ascend vLLM 上的完整编译链路。

## 三层编译图

```
Layer 1: Dynamo 捕获图 (FX Graph)
  Python bytecode → TorchDynamo trace → 完整 FX Graph (含 symbolic shapes)

Layer 2: vllm-ascend 编译切图 (PiecewiseBackend + AscendCompiler)
  完整 FX Graph → VllmBackend 按 splitting_ops 切分 → PiecewiseBackend 按 compile_ranges 编译
  → Ascend fusion passes (norm-quant, QKNorm-RoPE, allreduce-RMSNorm 等)

Layer 3: 后端图编译 (npugraph_ex / torchair → ACL Graph)
  切好子图 → AscendCompiler → npugraph_ex/torchair NPU backend → ACL Graph (torch.npu.graph)
```

> **注意**：Ascend NPU 上不使用 Triton。脚本中的 `TRITON_CACHE_DIR` 仅为兼容性保留。
> 第三层实际是 torch_npu 的 npugraph_ex / torchair 编译和 ACL Graph capture/replay。

## 编译各阶段分析指南

详细分析请参见 [analysis.md](./analysis.md)。该文档结合 `cache/` 目录中的实际 dump 产物，按 5 个阶段逐层分解编译链路。

## 文件说明

| 文件 | 作用 |
|---|---|
| `server.sh` | 冷启动 DBO server，按 DUMP_MODE 采集不同层级的编译产物 |
| `test.sh` | 发压 benchmark + DBO 触发验证 + 结果对比 |
| `logs/` | server 和 test 日志默认落地点 |
| `cache/` | 冷启动缓存隔离目录（每次 RUN_ID 独立子目录） |
| `results/` | benchmark 结果 JSON |
| `profile/` | torch_npu profiler 数据 |

## Warmup 实验入口

如果你要评估“首个抖动”而不是普通吞吐，直接看 `warmup-exp/`。

### 这个目录适合做什么

- `compile range` 导致的编译抖动
- `ACLGraph` 首次命中 capture 导致的抖动
- DBO `on/off` 对首回合抖动的影响
- `warmup` 是否能把首回合抖动消掉

### 脚本说明

- [`warmup-exp/server.sh`](./warmup-exp/server.sh)：启动实验 server，默认创建全新 `RUN_ID` 和独立 cache
- [`warmup-exp/test.sh`](./warmup-exp/test.sh)：按固定序列发压，支持 `MODE=observe|warmup`

### 推荐跑法

compile range 抖动：

```bash
cd testbench/MOE/dbo/demos/compilation-dump/warmup-exp
EXPERIMENT=compile_range DUMP_MODE=compile DBO=0 bash server.sh
EXPERIMENT=compile_range MODE=observe bash test.sh
```

ACLGraph 首次 capture 抖动：

```bash
cd testbench/MOE/dbo/demos/compilation-dump/warmup-exp
EXPERIMENT=aclgraph ENABLE_PROFILER=1 DBO=0 bash server.sh
EXPERIMENT=aclgraph MODE=observe bash test.sh
```

warmup 对照组：

```bash
cd testbench/MOE/dbo/demos/compilation-dump/warmup-exp
EXPERIMENT=compile_range MODE=warmup bash test.sh
EXPERIMENT=aclgraph MODE=warmup bash test.sh
```

DBO 对照组：

```bash
cd testbench/MOE/dbo/demos/compilation-dump/warmup-exp
EXPERIMENT=aclgraph DBO=0 bash server.sh
EXPERIMENT=aclgraph DBO=1 bash server.sh
```

### 为什么必须用干净 cache

这类实验测的是“第一次触发”的额外成本。如果复用旧 cache：

- compile range 可能已经被编译过；
- ACLGraph bucket 可能已经被 capture 过；
- 你看到的就不再是首回合抖动，而是稳态 replay。

因此默认必须让每次 server 启动生成新的 `RUN_ID`，不要手动复用旧的 `cache/<RUN_ID>/`。

## 快速开始

### 1. 最小验证（单请求，确认能跑通）

```bash
# 终端 1 — 启动 server（all dump，capture size 小）
CAPTURE_SIZES='[16]' DUMP_MODE=all bash server.sh

# 等 server ready（看到 "Application startup complete"），然后终端 2:
PORT=8001 PRESET=quick-verify bash test.sh
```

### 2. 按 DUMP_MODE 渐进式采集

```bash
# 只 dump Dynamo 图（最小 overhead）
DUMP_MODE=dynamo bash server.sh

# 只 dump vllm-ascend 编译切图产物
DUMP_MODE=compile bash server.sh

# 只 dump backend AOT 编译产物
DUMP_MODE=backend bash server.sh

# 全部 dump（默认）
DUMP_MODE=all bash server.sh

# 不 dump，仅冷启动（性能测试用）
DUMP_MODE=none bash server.sh
```

### 3. 不同 FlashComm / DBO 组合

```bash
# baseline: 关 FC1, 关 FC2, 开 DBO
FC1=0 FC2=0 DBO=1 bash server.sh

# FC1 only
FC1=1 FC2=0 DBO=1 bash server.sh

# FC1 + FC2 + AIV (注意: AI_CPU + FC2 未充分验证)
FC1=1 FC2=1 DBO=1 HCCL_MODE=AIV bash server.sh
```

### 4. 采集 profiler

```bash
# server 端
ENABLE_PROFILER=1 DUMP_MODE=none bash server.sh

# test 端
bash test.sh --profile
```

## 关键环境变量

### server.sh

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DUMP_MODE` | `all` | 采集模式：`none` / `dynamo` / `compile` / `backend` / `all` |
| `RUN_ID` | 时间戳 | 本次运行的唯一标识，cache 和日志用此命名 |
| `COLD_ROOT` | `cache/<RUN_ID>` | 冷启动 cache 根目录，所有缓存产物放在这里 |
| `CAPTURE_SIZES` | `[16]` | CUDA graph capture sizes，小值加速冷编译验证 |
| `FC1` | `1` | FlashComm1 开关 |
| `FC2` | `0` | FlashComm2 parallel size |
| `FC2_OSHARED` | `0` | FlashComm2 O-shared 开关 |
| `DBO` | `1` | DBO 开关 |
| `HCCL_MODE` | `AI_CPU` | HCCL expansion mode：`AI_CPU` / `AIV` |
| `ENABLE_PROFILER` | `0` | 设为 `1` 开启 torch_npu profiler |
| `MODEL` | `DeepSeek-V2-Lite-Chat` | 模型路径 |
| `PORT` | `8001` | 服务端口 |
| `TP` | `2` | Tensor parallel size |

### test.sh

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRESET` | `prefill4k` | 压测预设：`quick-verify` / `small` / `prefill4k` / `prefill8k` / `custom` |
| `LABEL` | `compilation-dump` | 结果文件标签 |
| `PORT` | `8001` | 目标 server 端口 |
| `INPUT_LEN` | (preset 决定) | prompt token 数，需 ≥ 1024 才能触发 DBO prefill 阈值 |
| `OUTPUT_LEN` | (preset 决定) | 生成 token 数 |
| `NUM_PROMPTS` | (preset 决定) | 总请求数 |
| `MAX_CONCURRENCY` | (preset 决定) | 最大并发数 |

### Presets

| Preset | INPUT_LEN | OUTPUT_LEN | NUM_PROMPTS | MAX_CONCURRENCY | 用途 |
|---|---|---|---|---|---|
| `quick-verify` | 1024 | 1 | 1 | 1 | 快速验证服务可用 |
| `small` | 4096 | 1 | 32 | 16 | 并发正确性回归 |
| `prefill4k` | 4096 | 16 | 500 | 96 | 正式压测（默认） |
| `prefill8k` | 8192 | 16 | 500 | 96 | 长 prompt 压测 |
| `custom` | 手动指定 | 手动指定 | 手动指定 | 手动指定 | 自定义 |

## Dump 产物清单

### DUMP_MODE=dynamo (Layer 1)

| 产物 | 路径 | 如何查看 |
|---|---|---|
| TORCH_TRACE (结构化 trace) | `cache/<RUN_ID>/torch_trace/` | `tlparse cache/<RUN_ID>/torch_trace/` |
| Dynamo 日志 (guards/recompiles) | `logs/server_<RUN_ID>.log` | 搜索 `TORCH_LOGS` 相关行 |
| FX Graph code (动态 dump) | 需代码插桩 | 见"进阶：代码级 FX Graph dump" |

### DUMP_MODE=compile (Layer 2)

| 产物 | 路径 | 如何查看 |
|---|---|---|
| vLLM compile cache | `cache/<RUN_ID>/vllm_compile_cache/` | `find ... -name "*.py"` 查看 computation_graph.py |
| torch inductor cache | `cache/<RUN_ID>/torch_inductor_cache/` | 编译后的 Triton/Inductor kernel |
| VLLM cache root | `cache/<RUN_ID>/vllm_cache/` | piecewise backend 缓存 |

### DUMP_MODE=backend (Layer 3)

| 产物 | 路径 | 如何查看 |
|---|---|---|
| AOT compile 产物 | `cache/<RUN_ID>/vllm_cache/torch_aot_compile/` | 包含 `rank_*/model` 序列化文件 |
| graph capture 日志 | `logs/server_<RUN_ID>.log` | 搜索 `"Capturing CUDA graphs"` 和 `"Graph capturing finished"` |

### DUMP_MODE=all

三层产物全部输出。日志还会包含 TORCH_LOGS 的动态 shape / guard 分析。

## 冷启动验证

### 如何保证冷启动

核心原则：**所有缓存路径指向全新空目录**。

`server.sh` 默认行为：
1. 生成唯一的 `RUN_ID`（时间戳精度到秒）
2. 在 `cache/<RUN_ID>/` 下创建所有缓存子目录
3. 启动前检查 compile cache 目录是否为空，非空时打印 WARNING
4. 通过 `--compilation-config` 显式传入 `cache_dir`

### 验证要点

```bash
# 1. 确认 compile cache 目录为空（server.sh 启动时会自动检查）
# 输出: "✓ Compile cache dir is empty — true cold start confirmed."

# 2. 确认 compile 实际发生了（非缓存命中）
grep "torch.compile took" logs/server_*.log
# 应看到耗时 > 10s 的编译过程

# 3. 确认 graph capture 重新执行
grep "Graph capturing finished" logs/server_*.log
# 应看到 capture 耗时（非 0 秒命中）

# 4. 不同的配置组合使用不同 cache 目录（RUN_ID 隔离）
# cache/20260705_143022/  ← FC1=1, FC2=0
# cache/20260705_143501/  ← FC1=1, FC2=1
```

### 已知缺陷

当前 `AscendCompiler.compute_hash()` 只包含 `torch_npu.__version__`、`enable_npugraph_ex`、`enable_static_kernel`。

**不包含** DBO、FC1、FC2、TP、model arch。不同配置可能生成相同 cache hash，导致错误命中。

因此 **必须** 通过本工具的全新目录隔离，不能仅依赖 hash 机制。

## 验证 DBO 是否触发

```bash
# 使用 test.sh 的 --verify 模式
bash test.sh --verify

# 或手动搜索
grep -n "should_ubatch: True" logs/server_*.log
grep -n "AllgatherTemplate\|DeepseekAllgather\|select_dbo_templates" logs/server_*.log
```

DBO 触发需同时满足 4 个条件：
1. `--enable-dbo` 已传
2. batch token 数 ≥ `dbo_prefill_token_threshold`（默认 1024）
3. MoE 通信模式不为 MC2
4. padding 后第二个 ubatch 非空

## 常用操作速查

```bash
# === 一键 dump 所有图的冷启动 + 验证 ===

# 1. 冷启动 server（dump all，快速 capture size）
CAPTURE_SIZES='[16]' DUMP_MODE=all bash server.sh

# 2. 另一个终端验证 server 就绪后发压
PRESET=quick-verify bash test.sh

# 3. 检查 dump 产物
tlparse cache/$(ls -t cache/ | head -1)/torch_trace/   # Dynamo FX Graph
find cache/$(ls -t cache/ | head -1) -name "computation_graph.py"  # compile cache
find cache/$(ls -t cache/ | head -1) -path "*/torch_aot_compile/*" -name "model"  # AOT compile

# 4. 验证 DBO
bash test.sh --verify


# === 不同编译配置的对比 dump ===

# Run A: FC1=1, DBO=1, AI_CPU
FC1=1 FC2=0 DBO=1 HCCL_MODE=AI_CPU DUMP_MODE=compile bash server.sh
# (等 ready) → 发压 → 停 server

# Run B: FC1=1, DBO=1, AIV
FC1=1 FC2=0 DBO=1 HCCL_MODE=AIV DUMP_MODE=compile bash server.sh
# (等 ready) → 发压 → 停 server

# 对比 compile cache
diff -r cache/<run_a>/vllm_compile_cache/ cache/<run_b>/vllm_compile_cache/


# === 仅做性能测试（不 dump，降低 overhead） ===

DUMP_MODE=none bash server.sh
PRESET=prefill4k bash test.sh


# === Profile 采集 ===

ENABLE_PROFILER=1 DUMP_MODE=none bash server.sh
bash test.sh --profile
tensorboard --logdir profile/dbo_profile/
```

## 进阶：代码级 FX Graph dump

当 `TORCH_TRACE` 不够细粒度时，可以直接在代码中 dump FX Graph（来自 `testbench/MOE/dbo/bug/flashcomm-dbo-compile-bugs.md` Bug 4 检测方法）：

```python
# 在 model_runner_v1.py 的 load_model() 之后插入:
if hasattr(self.model, 'runnable'):
    gm = self.model.runnable
    if hasattr(gm, 'code'):
        with open('/tmp/fx_graph_code.py', 'w') as f:
            f.write(gm.code)

# 然后搜索 DBO hook 是否被 Dynamo 剪枝:
# grep "dbo_linear_column_hook\|dbo_moe_prepare_hook" /tmp/fx_graph_code.py
# 找不到 → DBO 分支在编译期被常量化剪掉
```

检查 recompile 计数（来自同文档）：

```python
import torch._dynamo.utils as dynamo_utils
print(f"Recompile count: {dynamo_utils.get_dynamo_compile_time()}")
```

## 故障排查

### 常见错误与对应文档

| 错误 | 根因 | 参考文档 |
|---|---|---|
| `AddRmsNormBias do tiling failed` | FC1 SP shape contract 断裂 | `testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md` |
| `name 'Min' is not defined` | compile cache 生成图文件缺 `from sympy import Min` | 同上 RFC Case D |
| `expanded size (4096) must match (8)` | MLA output contract 混用 graph padded / logical token | 同上 RFC 第二轮 |
| `aicpu exception 507018` | FC2 + AI_CPU AllToAll kernel 异常 | `testbench/MOE/dbo/docs/analysis/analysis-dbo-fc2-compile-startup.md` |
| `HCCL watchdog terminated` | 跨 rank collective 配对破坏 | 同上 |
| DBO 未触发 | token 数不足阈值 / MC2 / 最后 ubatch 为空 | `testbench/MOE/dbo/dbo.md` |
| compile 后 DBO hook 被跳过 | Dynamo 将 `dbo_enabled=False` 常量化剪枝 | `testbench/MOE/dbo/bug/flashcomm-dbo-compile-bugs.md` Bug 4 |

### 稳定配置矩阵

来自 `testbench/MOE/dbo/docs/analysis/analysis-dbo-fc2-compile-startup.md` 第七章和 `testbench/MOE/dbo/bug/dbo-compiled-fullgraph-fc2-aiv-validation-20260703.md`：

| DBO | FC1 | FC2 | HCCL_MODE | 冷启动状态 | 500×96 |
|---|---|---|---|---|---|
| 1 | 0 | 0 | AIV | ✅ | ✅ |
| 1 | 0 | 0 | AI_CPU | ✅ | ✅ |
| 1 | 1 | 0 | AIV | ✅ | ✅ |
| 1 | 1 | 0 | AI_CPU | ✅ | ✅ |
| 1 | 0 | 1 | AIV | ✅ | ✅ |
| 1 | 1 | 1 | AIV | ✅ | ✅ |
| 1 | 1 | 1 | AI_CPU | ⚠️ 未验证 | ⚠️ 未验证 |

> 注意：以上结论严格绑定 DeepSeek-V2-Lite / A2 / TP=2 / prefill-only DBO。
> 不能外推到 TP>2 / DP>1 / PP>1 / A3 / decode DBO / O-Shard。

## 最小阅读路线

如果想深入理解编译链路，推荐按以下顺序阅读 `testbench/` 下的文档：

1. `testbench/MOE/dbo/dbo.md` — DBO 机制与触发条件
2. `testbench/MOE/dbo/docs/compilation/torch-compile-guide.md` — Dynamo → VllmBackend → PiecewiseBackend → ACLGraph 全链路
3. `testbench/MOE/dbo/docs/analysis/analysis-dbo-fc2-compile-startup.md` — DBO + compile + FlashComm 交叉分析（最长最全）
4. `testbench/MOE/dbo/dbo-torch-compile-compat.md` — DBO 多线程与 compile 的冲突分析
5. `testbench/MOE/dbo/rfc/rfc-dbo-cold-start-problem.md` — 冷启动 shape contract 断裂的定位与修复
6. `testbench/MOE/dbo/bug/flashcomm-dbo-compile-bugs.md` — 8 个已知 bug 的详细根因
7. `testbench/MOE/dbo/docs/flashcomm/flashcomm.md` — FlashComm 机制
8. `testbench/MOE/dbo/bug/vllm-ascend-dbo.md` — Ascend DBO vs 上游 vLLM DBO 难度对比
