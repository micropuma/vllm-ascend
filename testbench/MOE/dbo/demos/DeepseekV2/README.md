# DeepSeek-V2 DBO 性能测试

DBO（Dual Batch Overlap）把大 batch 拆成两个 microbatch，让通信和计算在 NPU 上并发执行，隐藏通信延迟。

所有性能对比统一通过 `auto_benchmark.sh` 完成。

## 快速实验

```bash
cd /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate
source /data/workspace/vllm-dbo-v0221/env.sh

# DP 快速对比: dp_baseline vs dp_dbo
TEST_GROUPS=p0_deepseek_dp bash auto_benchmark.sh

# TP 快速对比: baseline vs fc1 vs dbo vs dbo_fc1
TEST_GROUPS=p0_deepseek_tp CONFIGS=baseline,fc1,dbo,dbo_fc1 bash auto_benchmark.sh

# TP 全矩阵
TEST_GROUPS=p0_deepseek_tp bash auto_benchmark.sh
```

自定义模型：

```bash
MODEL=/path/to/DeepSeek-V2-Lite-Chat TEST_GROUPS=p0_deepseek_tp bash auto_benchmark.sh
```

## 手动分步（调试用）

```bash
# 终端 1 — 启动 server
PORT=8001 bash deepseek-v2-server.sh           # baseline
PORT=8001 bash deepseek-v2-dbo-server.sh       # DBO
PORT=8001 bash deepseek-v2-dbo-server-dp.sh    # DP

# 终端 2 — 发压
LABEL=baseline PORT=8001 bash deepseek-v2-dbo-test.sh
LABEL=dbo PORT=8001 bash deepseek-v2-dbo-test.sh

# 打印对比
bash deepseek-v2-dbo-test.sh --compare
```

## 关键参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL` | `/data/models/DeepSeek-V2-Lite-Chat` | 模型路径 |
| `PORT` | `8001` | 服务端口 |
| `TP` | `2` | 张量并行度 |
| `DBO_PREFILL_TOKEN_THRESHOLD` | `1024` | DBO prefill 触发阈值 |

## 环境

- 2 张 Ascend NPU
- `HCCL_OP_EXPANSION_MODE=AI_CPU`
- 详见 `../README.md`
