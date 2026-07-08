# DBO Overlap Template Writer Skill 使用指南

本文档介绍如何使用 `dbo-overlap-template-writer` skill 的 `analyze_ascend_profiling.py` 脚本，基于 Ascend NPU Profiler 输出进行 DBO overlap 分析。

## 1. 快速开始：六个子命令

```bash
SKILL_DIR=.claude/skills/dbo-overlap-template-writer
SCRIPT=$SKILL_DIR/scripts/analyze_ascend_profiling.py
PROFILE=profile/flashcomm1_aicpu

# ① triage — 全局概览（最常用）
python3 $SCRIPT triage --input $PROFILE --num-layers 27 --top-k 30

# ② breakdown — 按 kernel 类别详细分解
python3 $SCRIPT breakdown --input $PROFILE --top-k 50

# ③ comm — 通信算子的频次和时长分析
python3 $SCRIPT comm --input $PROFILE --num-layers 27

# ④ compare — 两个 profile 的 step 级对比
python3 $SCRIPT compare --input-a profile/no-dbo --input-b profile/flashcomm1_aicpu

# ⑤ stack — 通信算子的 Python 调用栈溯源
python3 $SCRIPT stack --input $PROFILE --top-comm-types 3 --max-stacks 50

# ⑥ wait — kernel 级 Wait Time 分布和 Stream 分析 (新增)
python3 $SCRIPT wait --input $PROFILE --kernel allgatherAicpu
python3 $SCRIPT wait --input-a $PROFILE --input-b profile/flashcomm1_aiv --kernel allgather
```

| 子命令 | 输入数据源 | 输出什么 | 使用场景 |
|---|---|---|---|
| `triage` | `op_statistic.csv` + `step_trace_time.csv` + `communication.json` + `metadata` | 全局概览：step 级 overlap 率、kernel 分类汇总、通信类型统计 | **每次分析的第一步** |
| `breakdown` | `op_statistic.csv` | 按 kernel 类别详细分解 | 需要看 gemm/attention/norm 等分类占比时 |
| `comm` | `communication.json` | 每种通信类型的频次、平摊到每层的通信量 | 分析通信瓶颈 |
| `compare` | 两个 profile 的 `op_statistic.csv` + `step_trace_time.csv` | step 级指标对比 + 分类对比 | A/B test（DBO vs no-DBO, FC1 vs no-FC1） |
| `stack` | `trace_view.json` + `communication.json` | 每个通信算子的 Python 调用栈和源码位置 | 定位通信来自哪个 hook/op |
| `wait` | `kernel_details.csv` | Wait Time 分位数（P50/P90/P95/P99）、直方图、Stream 级分布 | 分析 kernel 调度延迟和通信抖动 |

---

## 2. 数据源参考

skill 脚本读取 Ascend Profiler 输出的以下文件：

```
<profiling_root>/
├── ASCEND_PROFILER_OUTPUT/
│   ├── op_statistic.csv          ← triage/breakdown/compare 的主数据源
│   ├── kernel_details.csv        ← wait 命令的数据源（逐 kernel 记录）
│   ├── communication.json        ← comm 命令的数据源
│   ├── communication_matrix.json ← triage 的带宽数据
│   ├── step_trace_time.csv       ← triage/compare 的 overlap 指标
│   ├── trace_view.json           ← stack 命令的调用栈数据
│   └── analysis.db               ← SQLite 数据库（部分命令的备用数据源）
├── profiler_info_<rank>.json     ← CANN/torch_npu 版本, rank 信息
└── profiler_metadata.json        ← 并行组配置 (TP/EP/DP group)
```

**关键文件的列结构**：

`op_statistic.csv`（聚合后，一行 = 一种 kernel 类型）：
```
OP Type | Core Type | Count | Total Time(us) | Avg Time(us) | Max Time(us)
```
→ `triage` 用它做 kernel 分类和 Top-K 排名。**没有 Wait Time 和分位数**。

`kernel_details.csv`（逐 kernel，一行 = 一次 kernel 调用）：
```
Name | Stream ID | Start Time(us) | Duration(us) | Wait Time(us) | ...
```
→ `wait` 用它做 Wait Time 分位数、Stream 级分布、时序分析。**保留完整分布信息**。

`step_trace_time.csv`（一行 = 一个 device 的 step 级指标）：
```
computing | communication_not_overlapped | overlapped | communication | free
```
→ `triage`/`compare` 直接读这行做 overlap 率计算。**CANN profiler 硬件计数器，非估算**。

---

## 3. 分析示例一：DBO Overlap 边界分析

### 3.1 问题

DBO overlap 的 record/wait 点应该放在模型的什么位置？如何从 profiling 数据反推？

### 3.2 实验设计

| Profile | DBO | FC1 | HCCL | 作用 |
|---|---|---|---|---|
| `no-dbo` | OFF | ON | AI_CPU | 观察无 DBO 时通信在哪里暴露 |
| `flashcomm1_aicpu` | ON | ON | AI_CPU | 验证 DBO overlap 效果 |

### 3.3 分析步骤

**Step 1：triage 看全局**

```bash
python3 $SCRIPT triage --input profile/no-dbo --num-layers 27 --top-k 30
```

关键输出：
```
Step-Level Overlap:
  Communication exposed:   2701ms (25.6%)  ← 1/4 通信暴露
  Overlapped:              1460ms (35.1%)  ← 仅靠 HCCL 异步

Communication Summary:
  AllReduce: 1813 ops × avg 2.29ms ← 主要通信类型
```

→ 25.6% 通信暴露 → 需要 DBO。通信主要是 AllReduce。

**Step 2：从模型代码标出每层通信序列**

DeepSeek-V2-Lite 是 MLA+MoE 架构（A2 AllGather 模式），代码位置：
- MLA: `vllm_ascend/attention/mla_v1.py`
- Column/Row Parallel: `vllm_ascend/ops/linear_op.py`
- MoE Prepare/Finalize: `vllm_ascend/ops/fused_moe/prepare_finalize.py`

每层通信序列：
```
[MLA AllGather] → [MLA compute] → [o_proj ReduceScatter] → [RMSNorm + MoE Gate]
→ [MoE Prepare AllGather] → [Expert compute] → [MoE Finalize ReduceScatter]
```

**Step 3：按 Skill 的 Principle 确定 overlap 边界**

- **Principle 1（连续通信合并）**：o_proj RS 和 MoE Prep AG 之间间隔 ~2ms（实测 RS→AG 间隙 P50=2084μs），但这 ~2ms 是 RMSNorm + MoE Gate + token permutation —— **纯计算，没有通信算子**。DBO 的 hook 只能包裹通信算子，纯计算段没有可插入 hook 的点。因此只能用一对 ATTN_POST event 包裹整个 [RS + 2ms compute + AG]。
- **Principle 2（计算-通信平衡）**：ubatch0 的通信块（o_proj RS~850μs + MoE Prep AG~850μs）与 ubatch1 的 MLA compute（~2000μs）在时间轴上重叠

**Step 4：映射到 DBO hook**

```
Event      record 位置            wait 位置
ATTN_PRE   dbo_moe_finalize(T)   dbo_mla_preprocess(F)
ATTN_POST  dbo_linear_row(T)     dbo_moe_prepare(F)
```

**Step 5：compare 验证效果**

```bash
python3 $SCRIPT compare --input-a profile/no-dbo --input-b profile/flashcomm1_aicpu
```

```
Comm exposed: 2701ms → 2054ms (−24.0%)
Overlap:      1460ms → 3224ms (+120.8%)
```

---

## 4. 分析示例二：FlashComm1 效果分析

### 4.1 问题

FC1 开了之后到底改变了什么？为什么 overlap 率从 41% 升到 73%？

### 4.2 实验设计

| Profile | DBO | FC1 | HCCL | 作用 |
|---|---|---|---|---|
| `no-flashcomm1` | ON | OFF | AIV | FC1=OFF baseline |
| `flashcomm1_aiv` | ON | ON | AIV | FC1=ON |

### 4.3 分析步骤

**Step 1：compare 看全局差异**

```bash
python3 $SCRIPT compare --input-a profile/no-flashcomm1 --input-b profile/flashcomm1_aiv
```

```
Comm exposed: 1897ms → 1359ms (−28.3%)
Overlap:      1325ms → 3740ms (+182%)
Free/idle:    1484ms →  871ms (−41.3%)
```

→ FC1 大幅改善 overlap。但根因是什么？

**Step 2：看代码找到 FC1 的改动**

FC1=ON 的核心改动只有一处——`linear_op.py:407-409`：
```python
if not _EXTRA_CTX.flash_comm_v1_enabled:
    output = get_tp_group().all_gather(output, 0)  # ← FC1 去掉这个
```

Row Parallel 已经做了 `ReduceScatter`，FC1=OFF 又多做了一次 `AllGather`。`ReduceScatter + AllGather = AllReduce`——FC1 通过去掉末尾冗余 AllGather，把一大坨 AllReduce 变成两段独立通信（ReduceScatter → 下一层的 AllGather(input)），DBO hook 可以分别包裹。

**Step 3：stack 确认通信来源**

```bash
python3 $SCRIPT stack --input profile/no-flashcomm1 --top-comm-types 3 --max-stacks 50
```

可以看到 FC1=OFF 时 AllReduce 的 Python 调用栈，确认来自 `OProjRowParallelOp.apply_impl`。

---

## 5. 分析示例三：AI_CPU vs AIV Wait Time 分析

### 5.1 问题

AI_CPU 的 `allgatherAicpuKernel` wait time 为什么剧烈抖动（P50=0μs, P95=951μs）？

### 5.2 分析步骤

**Step 1：wait 命令看分位数**

```bash
python3 $SCRIPT wait --input profile/flashcomm1_aicpu --kernel allgatherAicpu
```

```
Wait Time:
  avg  = 126.1 us      ← 平均等待比执行时间(103μs)还长!
  P50  =   0.0 us      ← 一半立即执行
  P90  = 264.8 us
  P95  = 950.9 us      ← 5% 等了近 1ms
  P99  = 1439.1 us
  max  = 4170.7 us

Wait Time histogram:
  =0us:       2664 (73.2%)   ← 大部分立即执行
  >=1ms:       169 ( 4.6%)   ← 但有 169 次等了超过 1ms
```

**Step 2：wait compare 对比 AIV**

```bash
python3 $SCRIPT wait \
  --input-a profile/flashcomm1_aicpu \
  --input-b profile/flashcomm1_aiv \
  --kernel allgather
```

```
              AI_CPU          AIV
Wait P50       0.0us         0.0us
Wait P90     264.8us         2.8us    ← 关键差异!
Wait P95     950.9us         8.1us
Wait>=1ms   169 (4.6%)      6 (0.1%)
```

→ AIV 的 wait time 几乎为零（P95=2.8μs），AI_CPU 有系统性的长尾。

**Step 3：Stream 级分析定位**

```bash
python3 $SCRIPT wait --input profile/flashcomm1_aicpu --kernel allgatherAicpu --show-streams 5
```

```
Stream    Kernels   Unique    AvgWait    P95Wait  Top Kernel
11          3481        1      131.9      978.3   allgatherAicpuKernel
3958         107        1        0.6        3.5   allgatherAicpuKernel
```

→ 96% 的 allgather 在 Stream 11 上（纯通信流），P95 wait=978μs。其他 stream 上的 allgather 几乎没有 wait。**抖动集中在主通信流上**。

**Step 4：确认 compute stream**

```bash
python3 $SCRIPT wait --input profile/flashcomm1_aicpu --show-streams 5
```

```
Stream    Kernels   Top Kernel
46         27689    Slice, Cast, MatMul...  ← ubatch0 compute
40         19050    Slice, Cast, MatMul...  ← ubatch1 compute
11          3481    allgatherAicpuKernel    ← 通信
```

→ DBO 的三层架构：Stream 11（通信）→ Stream 40 + 46（计算）

---

## 6. `wait` 子命令详解

### 6.1 为什么需要这个命令？

`triage` 读的是 `op_statistic.csv`（聚合后的统计表），只有 avg/max，**没有**：
- Wait Time（调度等待时间）
- 分位数分布（P50/P90/P95/P99）
- Stream ID 分组
- 逐 kernel 的时序关系

`wait` 读的是 `kernel_details.csv`（逐 kernel 记录），补上了这些盲区。

### 6.2 用法

```bash
# 单 profile：看某个 kernel 的 Wait Time 分布
python3 $SCRIPT wait --input <dir> --kernel <name>

# 对比两个 profile
python3 $SCRIPT wait --input-a <dir_a> --input-b <dir_b> --kernel <name>

# 看 Stream 分布
python3 $SCRIPT wait --input <dir> --show-streams 20
```

`--kernel` 是大小写不敏感的**子串匹配**。例如 `--kernel allgather` 在 AI_CPU 模式下匹配 `allgatherAicpuKernel`，在 AIV 模式下匹配 `hcom_allGather__*`。

### 6.3 输出解读

```
Wait Time histogram:
  =0us:       2664 (73.2%)   ← 提交后立即执行
  <10us:        30 ( 0.8%)   ← 几乎没等
  10-100us:     16 ( 0.4%)
  100us-1ms:   762 (20.9%)   ← 等了 0.1~1ms
  >=1ms:       169 ( 4.6%)   ← 等了超过 1ms
```

- **如果 =0us 占比高（>70%）且 >=1ms 极少（<1%）** → 通信调度正常，无系统性问题
- **如果 >=1ms 占比较高（>5%）** → 通信调度存在系统性问题
- **如果 P95 很高但 >=1ms 极少** → 极端异常值，可能是跨 rank 同步或 driver 罕见的调度延迟

---

## 7. 分析流程总结

```
                     ┌─────────────────┐
                     │ 1. triage       │  ← 每次分析的起点
                     │   全局概览       │
                     └───────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ 2. compare │  │ 3. comm    │  │ 4. wait    │
     │   A/B 对比  │  │   通信分析  │  │   Wait分布  │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           ↓               ↓               ↓
     step级差异      通信类型+频次    Wait分位数+Stream
     定位哪个维度     定位哪个通信    定位调度是否异常
     效果最大        是瓶颈          抖动是否系统性

              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │ 5. stack   │  │ 6. SQLite  │  │ kernel_     │
     │   调用栈    │  │   深度查询  │  │ details.csv │
     └────────────┘  └────────────┘  └────────────┘
     通信来自哪个    自定义查询      逐kernel时序分析
     Python代码     灵活分析        重叠窗口可视化
```
