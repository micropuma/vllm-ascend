# Skill：使用 Ascend PyTorch Profiler 分析 vLLM-Ascend 算子性能

## 1. Skill 目标

本 Skill 用于快速采集 vLLM-Ascend 在线推理服务的性能数据，并通过 **Ascend PyTorch Profiler** 查看算子级性能分析结果。

主要目标：

- 采集 vLLM-Ascend 推理过程中的 NPU 算子性能数据
- 查看算子耗时、算子统计、kernel 执行信息
- 生成 `trace_view.json`，用于 MindStudio Insight 可视化分析
- 辅助定位推理服务中的算子级性能瓶颈

---

## 2. 适用场景

适合使用 Ascend PyTorch Profiler 的场景：

- 想分析某个模型在 Ascend NPU 上的算子耗时
- 想查看 NPU kernel 执行时间
- 想分析算子调用频率、耗时占比、执行时序
- 想对比不同模型、不同参数配置下的算子性能
- 想进一步定位 vLLM-Ascend 推理瓶颈是否来自算子执行

不适合的场景：

- 如果你主要想分析 vLLM 服务框架内部流程，例如 Request、BatchSchedule、KV Cache、ModelExecute 等，应优先使用 MS Service Profiler。
- 如果你只想看端到端吞吐、首 token 延迟、平均延迟，可以先用 benchmark 脚本或压测工具。

---

## 3. 前置条件

确保环境中已经具备：

- vLLM-Ascend 可正常运行
- Ascend NPU 驱动和 CANN 环境正常
- `torch_npu` 已安装
- 模型可以被 vLLM-Ascend 正常加载
- 可以正常访问 OpenAI-compatible API

检查环境：

```bash
python -c "import torch; import torch_npu; print(torch.__version__); print(torch_npu.__version__)"
```

检查 vLLM 是否可用：

```bash
python -m vllm.entrypoints.openai.api_server --help
```

---

## 4. 核心思路

Ascend PyTorch Profiler 的最小使用流程如下：

```text
启动 vLLM 服务并配置 profiler
        ↓
调用 /start_profile 开始采集
        ↓
发送真实推理请求
        ↓
调用 /stop_profile 停止采集
        ↓
使用 torch_npu.profiler.profiler.analyse 解析数据
        ↓
查看 ASCEND_PROFILER_OUTPUT 下的 CSV 和 trace 文件
```

---

## 5. 最小变量配置

建议先设置模型路径和服务端口：

```bash
export MODEL=/path/to/your/model
export HOST=0.0.0.0
export PORT=8080
export PROF_DIR=./vllm_profile
```

如果你使用的是本地模型，例如：

```bash
export MODEL=/root/models/Qwen2.5-0.5B-Instruct
```

---

## 6. 启动 vLLM 服务并开启 Profiler 能力

启动 vLLM 在线服务时，通过 `--profiler-config` 指定 profiler 类型和输出目录：

```bash
python -m vllm.entrypoints.openai.api_server \
  --host ${HOST} \
  --port ${PORT} \
  --model ${MODEL} \
  --dtype bfloat16 \
  --max-model-len 256 \
  --max-num-seqs 128 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}'
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--profiler-config` | vLLM profiler 配置 |
| `profiler: "torch"` | 使用 PyTorch Profiler |
| `torch_profiler_dir` | profiler 数据保存目录 |
| `torch_profiler_with_stack: false` | 不采集 Python 调用栈，减少数据量 |
| `--dtype bfloat16` | 使用 BF16 推理 |
| `--max-model-len` | 最大上下文长度 |
| `--max-num-seqs` | 最大并发序列数 |

注意：

- `torch_profiler_with_stack: false` 建议默认关闭，否则数据量会明显增加。
- `--profiler-config` 是在线服务推荐使用方式。
- 服务启动后不要立刻采集，建议等模型加载完成、服务稳定后再执行 `/start_profile`。

---

## 7. 开始采集

新开一个终端，执行：

```bash
curl -X POST http://localhost:${PORT}/start_profile
```

如果端口是 `8080`，则为：

```bash
curl -X POST http://localhost:8080/start_profile
```

成功后，vLLM 服务会开始记录 profiler 数据。

---

## 8. 发送推理请求

发送真实业务请求，例如：

```bash
curl http://localhost:${PORT}/v1/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"San Francisco is a\",
    \"max_tokens\": 7,
    \"temperature\": 0
  }"
```

如果模型字段不接受环境变量形式，也可以直接写模型路径：

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/your/model",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'
```

建议发送多次请求，采集几秒到十几秒即可：

```bash
for i in $(seq 1 5); do
  curl http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"San Francisco is a\",
      \"max_tokens\": 7,
      \"temperature\": 0
    }"
  echo
done
```

注意：

- 请求数量不需要太多。
- Profiler 会带来额外开销，不建议长时间采集。
- 推荐在业务请求开始前调用 `/start_profile`，请求结束后立刻调用 `/stop_profile`。

---

## 9. 停止采集

执行：

```bash
curl -X POST http://localhost:${PORT}/stop_profile
```

如果端口是 `8080`，则为：

```bash
curl -X POST http://localhost:8080/stop_profile
```

停止后，性能数据会写入 `torch_profiler_dir` 指定的目录，例如：

```bash
./vllm_profile
```

---

## 10. 查看采集目录

进入 profiler 输出目录：

```bash
cd ./vllm_profile
ls
```

通常可以看到类似目录：

```text
localhost.localdomain_XXXXX_ascend_pt
```

其中 `*_ascend_pt` 就是 Ascend PyTorch Profiler 的原始采集目录。

---

## 11. 分析 Profiler 数据

使用 `torch_npu.profiler.profiler.analyse` 解析数据。

### 方法一：手动指定目录

```python
from torch_npu.profiler.profiler import analyse

analyse("./vllm_profile/localhost.localdomain_XXXXX_ascend_pt/")
```

将路径替换为实际生成的目录。

---

### 方法二：Shell 自动查找目录

推荐使用这个方式：

```bash
cd /path/to/workdir

PROFILE_DIR=$(ls -d ./vllm_profile/*_ascend_pt | head -n 1)

python - <<PY
from torch_npu.profiler.profiler import analyse

profile_dir = "${PROFILE_DIR}"
print(f"Analyse profile dir: {profile_dir}")
analyse(profile_dir)
PY
```

如果有多个采集目录，可以查看：

```bash
ls -d ./vllm_profile/*_ascend_pt
```

然后选择需要分析的目录。

---

## 12. 查看分析结果

分析完成后，进入：

```bash
cd ./vllm_profile/*_ascend_pt/ASCEND_PROFILER_OUTPUT
ls
```

重点关注以下文件：

| 文件 | 作用 |
|---|---|
| `operator_details.csv` | 算子耗时详情 |
| `op_statistic.csv` | 算子统计和耗时占比 |
| `kernel_details.csv` | kernel 级执行信息 |
| `trace_view.json` | Chrome Trace / MindStudio Insight 时间线文件 |
| `step_trace_time.csv` | step 级调度和执行时间 |
| `api_statistic.csv` | API 调用统计 |
| `analysis.db` | 数据库格式性能数据 |

---

## 13. 推荐分析顺序

### 13.1 先看 op_statistic.csv

用于快速判断哪些算子最耗时：

```bash
head -n 20 op_statistic.csv
```

重点关注：

- 算子名称
- 调用次数
- 总耗时
- 平均耗时
- 耗时占比

适合回答：

- 哪些算子是热点？
- 是否有异常高频算子？
- 主要耗时集中在 attention、matmul、norm，还是其他算子？

---

### 13.2 再看 operator_details.csv

用于查看每一次算子调用的具体信息：

```bash
head -n 20 operator_details.csv
```

重点关注：

- 单次算子耗时
- 输入输出 shape
- 算子执行位置
- 是否存在长尾算子调用

适合回答：

- 某类算子是不是每次都很慢？
- 是否只有某几个 step 出现异常？
- prefill 和 decode 中算子表现是否不同？

---

### 13.3 再看 kernel_details.csv

用于分析更底层 kernel 执行数据：

```bash
head -n 20 kernel_details.csv
```

重点关注：

- kernel 名称
- kernel 执行耗时
- kernel 调用次数
- kernel 与上层算子的对应关系

适合回答：

- 上层算子最终调用了哪些 NPU kernel？
- kernel 执行时间是否符合预期？
- 是否存在大量小 kernel 导致调度开销偏高？

---

### 13.4 最后看 trace_view.json

`trace_view.json` 适合做时间线分析。

文件位置：

```bash
./vllm_profile/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json
```

可以导入：

- MindStudio Insight
- Chrome Trace Viewer
- Perfetto UI

重点观察：

- 算子执行时序
- host 和 device 之间是否有明显空洞
- 是否存在算子串行化
- 是否有异常长耗时 kernel
- prefill 和 decode 阶段执行模式是否不同

---

## 14. 最简完整流程

### 14.1 终端 1：启动服务

```bash
export MODEL=/path/to/your/model
export HOST=0.0.0.0
export PORT=8080

python -m vllm.entrypoints.openai.api_server \
  --host ${HOST} \
  --port ${PORT} \
  --model ${MODEL} \
  --dtype bfloat16 \
  --max-model-len 256 \
  --max-num-seqs 128 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}'
```

---

### 14.2 终端 2：启动采集

```bash
export MODEL=/path/to/your/model
export PORT=8080

curl -X POST http://localhost:${PORT}/start_profile
```

---

### 14.3 终端 2：发送请求

```bash
for i in $(seq 1 5); do
  curl http://localhost:${PORT}/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"San Francisco is a\",
      \"max_tokens\": 7,
      \"temperature\": 0
    }"
  echo
done
```

---

### 14.4 终端 2：停止采集

```bash
curl -X POST http://localhost:${PORT}/stop_profile
```

---

### 14.5 终端 2：分析数据

```bash
PROFILE_DIR=$(ls -d ./vllm_profile/*_ascend_pt | head -n 1)

python - <<PY
from torch_npu.profiler.profiler import analyse

profile_dir = "${PROFILE_DIR}"
print(f"Analyse profile dir: {profile_dir}")
analyse(profile_dir)
PY
```

---

### 14.6 终端 2：查看结果

```bash
cd ${PROFILE_DIR}/ASCEND_PROFILER_OUTPUT
ls
```

重点查看：

```bash
ls operator_details.csv op_statistic.csv kernel_details.csv trace_view.json
```

---

## 15. 常见问题

### 15.1 没有生成 vllm_profile 目录

检查：

```bash
ls
```

确认服务启动命令中包含：

```bash
--profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}'
```

同时确认：

- 服务已经成功启动
- 调用了 `/start_profile`
- 调用了 `/stop_profile`
- 中间确实发送了推理请求
- 当前目录具有写权限

---

### 15.2 start_profile 返回失败

检查服务端口是否正确：

```bash
curl http://localhost:8080/v1/models
```

如果服务使用的是 8000 端口，则需要改成：

```bash
curl -X POST http://localhost:8000/start_profile
```

注意：启动服务时的 `--port` 必须和 curl 请求使用的端口一致。

---

### 15.3 数据量太大

解决方法：

1. 关闭调用栈采集：

```json
"torch_profiler_with_stack": false
```

2. 缩短采集窗口：

```bash
curl -X POST http://localhost:8080/start_profile
# 只发送少量请求
curl -X POST http://localhost:8080/stop_profile
```

3. 减少请求数量：

```bash
for i in $(seq 1 3); do
  ...
done
```

---

### 15.4 analyse 找不到目录

检查实际目录名：

```bash
ls -d ./vllm_profile/*
```

如果路径不是 `*_ascend_pt`，可以手动指定：

```python
from torch_npu.profiler.profiler import analyse

analyse("/absolute/path/to/actual_ascend_pt_dir")
```

---

### 15.5 没有 ASCEND_PROFILER_OUTPUT

可能原因：

- 没有执行 `analyse`
- `analyse` 的输入路径错误
- 采集过程中没有有效推理请求
- `torch_npu` 或 Ascend profiler 环境异常

检查：

```bash
find ./vllm_profile -maxdepth 3 -type d
```

重新分析：

```bash
PROFILE_DIR=$(ls -d ./vllm_profile/*_ascend_pt | head -n 1)

python - <<PY
from torch_npu.profiler.profiler import analyse
analyse("${PROFILE_DIR}")
PY
```

---

## 16. 性能分析建议

### 16.1 采集时间不要过长

推荐：

```text
3 秒到 15 秒
```

不要长时间开启 profiler，否则：

- 数据量会很大
- 分析速度变慢
- trace 文件难以打开
- 可能影响服务性能

---

### 16.2 先小模型跑通流程

建议先用小模型，例如：

```bash
Qwen/Qwen2.5-0.5B-Instruct
```

或者本地小模型路径。

确认完整流程跑通后，再切换到目标模型。

---

### 16.3 先不开 Python 调用栈

默认建议：

```json
"torch_profiler_with_stack": false
```

如果后续需要追踪算子来自哪段 Python 代码，再改成：

```json
"torch_profiler_with_stack": true
```

注意：开启调用栈会显著增加数据量和 profiler 开销。

---

### 16.4 对比不同配置时保持请求一致

如果要比较不同配置，例如：

- 不同 `max_model_len`
- 不同 `max_num_seqs`
- 不同 batch 压力
- 不同模型
- 不同 dtype
- 不同 vLLM-Ascend 版本

需要保持请求输入尽量一致，否则 profiler 结果不可直接对比。

---

## 17. 与 MS Service Profiler 的区别

| 工具 | 关注点 | 适合分析 |
|---|---|---|
| Ascend PyTorch Profiler | 算子级、kernel 级 | 算子耗时、kernel 执行、算子统计 |
| MS Service Profiler | 服务框架级 | 请求、调度、KV Cache、ModelExecute 流程 |

推荐组合方式：

1. 先用 MS Service Profiler 判断瓶颈属于请求、调度、KV Cache 还是模型执行。
2. 如果瓶颈集中在模型执行，再用 Ascend PyTorch Profiler 深入分析算子和 kernel。
3. 如果瓶颈集中在调度或 KV Cache，则优先看 MS Service Profiler 的 `batch.csv`、`kvcache.csv` 和 `chrome_tracing.json`。

---

## 18. 最小 Checklist

开始前确认：

```text
[ ] vLLM-Ascend 服务可以正常启动
[ ] torch_npu 可以正常 import
[ ] 服务启动命令包含 --profiler-config
[ ] torch_profiler_dir 路径正确
[ ] curl 使用的端口和服务端口一致
[ ] 已调用 /start_profile
[ ] 已发送真实推理请求
[ ] 已调用 /stop_profile
[ ] vllm_profile 下生成 *_ascend_pt 目录
[ ] 已使用 analyse 解析数据
[ ] ASCEND_PROFILER_OUTPUT 下生成分析结果
```

---

## 19. 一句话总结

Ascend PyTorch Profiler 的最小使用流程是：

```text
启动服务时添加 --profiler-config
        ↓
curl /start_profile
        ↓
发送推理请求
        ↓
curl /stop_profile
        ↓
analyse(*_ascend_pt)
        ↓
查看 operator_details.csv、op_statistic.csv、kernel_details.csv 和 trace_view.json
```

它适合做 **vLLM-Ascend 的算子级性能诊断**，尤其适合定位模型执行阶段中的 NPU 算子瓶颈。