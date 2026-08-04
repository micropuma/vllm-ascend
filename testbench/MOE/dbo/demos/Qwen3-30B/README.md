# Qwen3-30B DBO 性能测试

性能对比统一通过 `auto_benchmark.sh` 完成。

## 快速实验

```bash
cd /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate
source /data/workspace/vllm-dbo-v0221/env.sh

# Qwen3 TP 快速对比: baseline vs fc1 vs dbo vs dbo_fc1
TEST_GROUPS=p1_qwen_tp CONFIGS=qwen_baseline,qwen_fc1,qwen_dbo,qwen_dbo_fc1 bash auto_benchmark.sh

# Qwen3 TP 全矩阵
TEST_GROUPS=p1_qwen_tp bash auto_benchmark.sh
```

自定义模型：

```bash
QWEN_MODEL=/path/to/Qwen3-30B TEST_GROUPS=p1_qwen_tp bash auto_benchmark.sh
```

## 手动分步（调试用）

```bash
# 终端 1 — 启动 server
PORT=8001 bash Qwen3-30B-server.sh           # baseline
PORT=8001 bash Qwen3-30B-dbo-server.sh       # DBO

# 终端 2 — 发压
LABEL=baseline PORT=8001 bash Qwen3-30B-dbo-test.sh
LABEL=dbo PORT=8001 bash Qwen3-30B-dbo-test.sh
```

## 关键参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL` | `/data/models/Qwen3-30B/Qwen3-30B` | 模型路径 |
| `TP` | `2` | 张量并行度 |
| `DBO_PREFILL_TOKEN_THRESHOLD` | `1024` | DBO prefill 触发阈值 |

## 精度测试

本模型使用 `enable_thinking: false` 进行精度评测，详见 `../precision/`。
