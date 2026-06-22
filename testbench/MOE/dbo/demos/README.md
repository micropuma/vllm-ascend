# DeepSeek-V2 DBO 测试套件

DBO（Dual Batch Overlap）把一个大 batch 拆成两个 microbatch，让通信（AllGather/AllToAll）和计算（Attention/MoE 矩阵乘）在 NPU 上真正并发，从而隐藏通信延迟。

**DBO 只在 server 高并发模式下有意义**，offline `llm.generate()` 的 batch 太小，overhead 反而更大。

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `deepseek-v2-dbo-server.sh` | 启动开启 DBO 的 vllm server（默认端口 8001） |
| `deepseek-v2-server.sh` | 启动关闭 DBO 的 baseline server（默认端口 8000） |
| `deepseek-v2-dbo-test.sh` | 发压 + 对比 + profiler 采集 |
| `bench.sh` | **一键对标**：顺序跑 baseline 和 DBO，输出 speedup 对比 |
| `deepseek-v2-offline-dbo.py` | offline 模式验证 DBO 是否触发（不适合看性能） |

---

## 快速开始：一键对标

```bash
# 只需一个命令，脚本会引导你依次启动两个 server
bash bench.sh
```

脚本流程：
1. 提示你启动 baseline server → 等就绪 → 自动发压
2. 提示你重启为 DBO server → 等就绪 → 自动发压
3. 打印 speedup 对比表

---

## 手动分步运行

### 1. 启动 server

```bash
# 终端 1 — baseline（端口 8000）
PORT=8000 bash deepseek-v2-server.sh

# 终端 1 — 或 DBO（端口 8001）
PORT=8001 bash deepseek-v2-dbo-server.sh
```

### 2. 发压

```bash
# 对 baseline 发压
LABEL=baseline PORT=8000 bash deepseek-v2-dbo-test.sh

# 对 DBO 发压
LABEL=dbo PORT=8001 bash deepseek-v2-dbo-test.sh

# 打印对比
bash deepseek-v2-dbo-test.sh --compare
```

### 3. 采集 Profiler

server 启动时加 `ENABLE_PROFILER=1`，然后发压时用 `--profile` 模式：

```bash
# 终端 1 — 启动带 profiler 的 DBO server
ENABLE_PROFILER=1 PORT=8001 bash deepseek-v2-dbo-server.sh

# 终端 2 — 发压并自动 start/stop profiler
LABEL=dbo PORT=8001 bash deepseek-v2-dbo-test.sh --profile
```

profiler 数据落到 `/data/workspace/vllm-ascend-dbo/profile/dbo_profile/`，用 Tensorboard 打开：

```bash
tensorboard --logdir /data/workspace/vllm-ascend-dbo/profile/dbo_profile/
```

---

## 关键参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL` | `/data/models/DeepSeek-V2-Lite-Chat` | 模型路径 |
| `INPUT_LEN` | `1024` | prompt token 数，**需 ≥ 512** 才触发 DBO prefill 阈值 |
| `OUTPUT_LEN` | `128` | 生成 token 数 |
| `NUM_PROMPTS` | `200` | 总请求数 |
| `MAX_CONCURRENCY` | `64` | 并发数，越大 batch 越大，DBO 收益越明显 |
| `ENABLE_PROFILER` | `0` | server 启动时设为 `1` 开启 torch_npu profiler |

---

## DBO 触发条件

同时满足以下 4 个条件时，每次 forward 会启用 DBO：

1. server 启动时传了 `--enable-dbo`
2. 当前 step 的 batch token 数 ≥ `dbo_prefill_token_threshold`（默认 512）
3. MoE 通信模式不为 MC2（TP=2/EP=2 小规模下自动满足）
4. padding 后第二个 microbatch 非空

**验证是否触发**：在 server 日志里搜索 `should_ubatch: True`

---

## 环境要求

- 2 张 Ascend NPU（TP=2）
- `HCCL_OP_EXPANSION_MODE=AI_CPU`（已在 server 脚本中设置，让通信跑在 AI_CPU 核上，是 DBO 并发生效的硬件前提）
