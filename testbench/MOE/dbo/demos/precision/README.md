# DBO 精度测试框架

本目录提供通过本地 OpenAI-compatible 服务进行 baseline/DBO 配对精度验证的工具。测试只使用本地数据集和
`127.0.0.1` 服务，不访问外部推理 endpoint。

## 环境

```bash
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate
source /data/workspace/vllm-dbo-v0221/env.sh
cd /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos/precision
```

## 完整测试

### DeepSeek-V2-Lite-Chat（默认）

```bash
python run_precision.py --task gpqa_diamond --limit 64
python run_precision.py --task all
```

### Qwen3-30B-A3B

```bash
python run_precision.py --model-family qwen3 --task all
```

自定义模型路径：

```bash
python run_precision.py --model-family qwen3 \
    --model /data/models/Qwen3-30B/Qwen3-30B --task all
```

DeepSeek 自定义路径：

```bash
python run_precision.py --model /data/models/DeepSeek-V2-Lite-Chat --task all
```

DBO server 的 cold-start 可能较慢，可增加 `--ready-timeout 3600`。

## Quick Gate

快速门禁检查固定请求波次的生成 token 和输出 logprob：

```bash
bash quick/test_dbo_precision.sh
bash quick/test_dbo_fc1_precision.sh
```

完整 runner 会依次执行 baseline、停止服务、冷启动 DBO、检查所有 TP rank 的 `should_ubatch: True`，然后生成
配对比较报告。默认允许 baseline 到 DBO 的准确率下降不超过 `0.5pp`。

## 结果

每次运行写入 `results/<run-id>/`，主要文件包括：

```text
manifest.json                 commit、editable import 和运行环境
orchestrator.json             commit、模型路径和测试策略
baseline_<task>.json          baseline 原始输出与 logprob
dbo_<task>.json               DBO 原始输出与 logprob
baseline_server_config.json   baseline 实际服务配置
dbo_server_config.json        DBO 实际服务配置
dbo_trigger.json              DBO 触发证据
compare_<task>.json           配对比较结果
```

精度分叉时，先检查 server config 和 DBO trigger，再使用 `diff_artifacts.py` 定位第一个 token 分叉。精度测试
不是性能测试，吞吐和延迟应使用 `testbench/MOE/dbo/demos/` 下的 benchmark 脚本测量。

## 开发检查

```bash
cd testbench/MOE/dbo/demos/precision
python -m pytest
python -m mypy --config-file pyproject.toml .
```
