# DBO 自动性能测试

`auto_benchmark.sh` 面向 DBO、FlashComm 和并行方式的批量性能对比。脚本会按配置依次启动 server、运行 plain benchmark、停止进程并回收 HBM。DBO 配置默认会自动先运行同通信设置的 baseline，便于生成配对结果。

## 1. 环境准备

要求：

- 两张可用的 Ascend NPU（默认逻辑设备 `0,1`）。
- 已准备模型：默认 `/data/models/DeepSeek-V2-Lite-Chat`；Qwen 组默认 `/data/models/Qwen3-30B/Qwen3-30B`。
- vLLM、vllm-ascend、CANN 和 `torch_npu` 已安装在同一运行环境。
- 默认环境脚本和虚拟环境分别为 `/data/workspace/vllm-dbo-v0221/env.sh` 和 `/data/workspace/vllm-dbo-v0221/.venv-dbo`。

脚本默认会加载这两个环境文件；不需要脚本加载时设置 `SOURCE_ENV=0`。所有路径均可通过环境变量覆盖：

```bash
cd /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate
source /data/workspace/vllm-dbo-v0221/env.sh
```

**自定义模型路径：**

```bash
# DeepSeek
MODEL=/path/to/DeepSeek-V2-Lite-Chat TEST_GROUPS=p0_deepseek_tp bash auto_benchmark.sh

# Qwen3
QWEN_MODEL=/path/to/Qwen3-30B TEST_GROUPS=p1_qwen_tp bash auto_benchmark.sh
```

**自定义环境路径：**

```bash
VENV_ROOT=/path/to/venv ENV_SCRIPT=/path/to/env.sh TEST_GROUPS=all bash auto_benchmark.sh
```

运行前确认端口 `8001` 和目标 NPU 没有其他 vLLM 进程。默认使用共享编译缓存；需要每个 case 独立冷缓存时设置 `USE_CASE_CACHE=1`。

## 2. 快速复现

### DP 实验（DeepSeek，TP=1 / DP=2）

baseline DP vs baseline DP + DBO，只测 DBO 在数据并行下的收益：

```bash
TEST_GROUPS=p0_deepseek_dp bash auto_benchmark.sh
```

运行 `dp_baseline` 和 `dp_dbo`（DBO 配置自动先跑 `dp_dbo_baseline` 再跑 `dp_dbo`）。

如需 shared expert overlap：

```bash
TEST_GROUPS=p0_deepseek_dp_shared_expert bash auto_benchmark.sh
```

### TP 实验（DeepSeek，TP=2）

baseline vs FlashComm1 vs DBO vs FlashComm1+DBO 四组正交：

```bash
TEST_GROUPS=p0_deepseek_tp CONFIGS=baseline,fc1,dbo,dbo_fc1 bash auto_benchmark.sh
```

| 配置 | DBO | FC1 | 说明 |
|------|:---:|:---:|------|
| `baseline` | 0 | 0 | 纯 baseline |
| `fc1` | 0 | 1 | 仅 FlashComm1 |
| `dbo` | 1 | 0 | 仅 DBO |
| `dbo_fc1` | 1 | 1 | DBO + FlashComm1 |

如需 AIV 或 FlashComm2 变体，追加 `dbo_fc1_aiv`、`dbo_fc2` 等配置名。

### Qwen3-30B TP 实验

```bash
TEST_GROUPS=p1_qwen_tp CONFIGS=qwen_baseline,qwen_fc1,qwen_dbo,qwen_dbo_fc1 bash auto_benchmark.sh
```

### 全量

```bash
TEST_GROUPS=all bash auto_benchmark.sh
```

建议先 dry-run 确认命令和路径：`TEST_GROUPS=all DRY_RUN=1 bash auto_benchmark.sh`

## 3. 测试组

当前共有 6 个测试组。`p0/p1/p2` 表示优先级和覆盖范围。

| 测试组 | 默认 workload | 覆盖内容 | 默认配置 |
|---|---|---|---|
| `p0_deepseek_tp` | `prefill4k` | DeepSeek TP=2，baseline、DBO、FlashComm1、FlashComm2、AIV 组合 | `baseline dbo fc1 fc1_aiv dbo_fc1 dbo_fc1_aiv dbo_fc2` |
| `p0_deepseek_dp` | `prefill4k` | DeepSeek TP=1/DP=2，普通 DP 下 DBO 对照 | `dp_baseline dp_dbo` |
| `p0_deepseek_dp_shared_expert` | `prefill4k` | DP=2，并开启 shared-expert overlap | `dp_shared_baseline dp_shared_dbo` |
| `p1_deepseek_small_batch` | `prefill4k16` | DeepSeek 小批次，观察 DBO 额外开销 | `baseline dbo dp_baseline dp_dbo` |
| `p1_qwen_tp` | `prefill4k` | Qwen3-30B TP=2，DBO/FlashComm 组合 | `qwen_baseline qwen_dbo qwen_fc1 qwen_fc1_aiv qwen_dbo_fc1 qwen_dbo_fc1_aiv qwen_dbo_fc2` |
| `p2_deepseek_decode` | `decode` | DeepSeek DP=2 decode 场景；DBO decode threshold 默认 32 | `dp_dbo` |

`all` 会按上述顺序运行全部 6 组。兼容别名：`p0` 包含全部 P0 组，`p1` 包含两个 P1 组，`p2` 等于 decode 组，`qwen` 等于 Qwen TP 组。

### Workload preset

| preset | 输入/输出 token | 请求数/最大并发 |
|---|---:|---:|
| `prefill4k` | 4096/16 | 500/96 |
| `prefill4k16` | 4096/16 | 16/16 |
| `decode` | 128/128 | 128/64 |
| `ttft4k` | 4096/1 | 500/96 |
| `prefill8k` | 8192/16 | 500/96 |
| `quick` | 1024/128 | 200/64 |

## 4. 运行命令

### 全量测试

```bash
TEST_GROUPS=all bash auto_benchmark.sh
```

只做命令和结果路径检查，不启动 server：

```bash
TEST_GROUPS=all DRY_RUN=1 bash auto_benchmark.sh
```

默认 `DEBUG_VALIDATION=0`，只进行性能测试。需要额外检查 DBO worker 的 `should_ubatch: True` 证据时：

```bash
TEST_GROUPS=all DEBUG_VALIDATION=1 bash auto_benchmark.sh
```

### 各测试组示例

```bash
# DeepSeek TP 全矩阵
TEST_GROUPS=p0_deepseek_tp bash auto_benchmark.sh

# 普通 DP baseline vs DBO
TEST_GROUPS=p0_deepseek_dp bash auto_benchmark.sh

# DP shared expert overlap
TEST_GROUPS=p0_deepseek_dp_shared_expert bash auto_benchmark.sh

# 小批次 DBO 开销
TEST_GROUPS=p1_deepseek_small_batch bash auto_benchmark.sh

# Qwen3-30B TP 矩阵
TEST_GROUPS=p1_qwen_tp bash auto_benchmark.sh

# DeepSeek decode
TEST_GROUPS=p2_deepseek_decode bash auto_benchmark.sh
```

覆盖 workload 或只跑指定配置：

```bash
TEST_GROUPS=p0_deepseek_tp BENCH_PRESETS=prefill8k bash auto_benchmark.sh
TEST_GROUPS=p1_qwen_tp CONFIGS=qwen_baseline,qwen_dbo_fc1 bash auto_benchmark.sh
```

常用参数：`MODEL`、`QWEN_MODEL`、`DEVICES`、`PORT`、`OUT_DIR`、`SERVER_START_TIMEOUT`、`INTER_CONFIG_SLEEP`、`PERFORMANCE_RUN=0`。`TEST_GROUPS` 和 `BENCH_PRESETS` 支持逗号或空格分隔。

## 5. 查看结果

默认输出根目录：

```text
testbench/MOE/dbo/testbench/results/
```

每次运行会生成：

- `auto_benchmark_<timestamp>.log`：完整控制台日志、配置、每个 case 的指标和失败信息。
- `benchmark_report_<timestamp>.md`：自动生成的 baseline/当前配置对比报告。
- `perf/<config>_in..._out..._np..._c..._<timestamp>_perf.json`：单个 case 的原始性能指标。
- `runs/<timestamp>/`：按 case 保存的 server 日志和运行缓存（开启相应日志/缓存隔离时）。

快速查看最近一轮：

```bash
ls -lt testbench/MOE/dbo/testbench/results/auto_benchmark_*.log | head
ls -lt testbench/MOE/dbo/testbench/results/benchmark_report_*.md | head
sed -n '1,220p' testbench/MOE/dbo/testbench/results/benchmark_report_<timestamp>.md
```
