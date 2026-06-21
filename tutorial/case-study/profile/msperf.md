# vLLM-Ascend MS Service Profiler 最简使用文档

## 1. 文档目的

本文档用于快速说明如何在 vLLM-Ascend 推理服务中使用 **MS Service Profiler** 进行性能分析。

MS Service Profiler 主要用于采集和分析 vLLM-Ascend 服务框架内部的执行流程，包括：

- 请求处理流程
- Batch 调度流程
- Model Execute 执行流程
- KV Cache 管理流程
- Communication 通信流程
- 可选的算子下发与执行耗时

如果你的目标是分析 **vLLM-Ascend 服务框架本身的性能瓶颈**，例如请求调度、KV Cache 分配、模型执行入口耗时等，应优先使用 MS Service Profiler。

如果你的目标是分析 **PyTorch / NPU 算子级性能**，例如某个算子耗时、kernel 执行时间、算子调用栈等，应使用 Ascend PyTorch Profiler。

---

## 2. 工具定位

vLLM-Ascend 官方性能分析方案主要包括两类：

| 工具 | 主要用途 | 采集粒度 | 输出格式 |
|---|---|---|---|
| Ascend PyTorch Profiler | 算子级性能分析 | PyTorch / NPU 算子 | ascend_pt、CSV、trace |
| MS Service Profiler | 服务框架性能分析 | vLLM 服务内部函数、请求、调度、KV Cache | Chrome Trace、CSV、DB |

本文档只介绍 **MS Service Profiler** 的最简使用流程。

---

## 3. 环境要求

建议环境：

- 已安装 vLLM-Ascend
- 已安装 CANN Toolkit
- NPU 环境可正常运行 vLLM 服务
- 能够正常执行 `vllm serve`
- 能够正常访问 OpenAI-compatible API，例如 `/v1/completions`

检查 vLLM 服务是否可用：

```bash
vllm --help
```

检查 `msserviceprofiler` 是否可用：

```bash
msserviceprofiler --help
```

如果命令不存在，可参考下一节进行安装或升级。

---

## 4. 安装或升级 MS Service Profiler

`msserviceprofiler` 通常已随 CANN Toolkit 预安装。

如果当前环境中没有该命令，或者需要升级，可以从源码构建：

```bash
git clone https://gitcode.com/Ascend/msserviceprofiler.git
cd msserviceprofiler
bash scripts/build_and_upgrade.sh
```

安装完成后检查：

```bash
msserviceprofiler --help
```

如果能够正常显示帮助信息，说明工具已安装完成。

---

## 5. 最简目录准备

创建一个专门用于保存 profiling 配置和结果的目录：

```bash
mkdir -p /root/prof_test
cd /root/prof_test
```

后续所有 profiling 配置文件和输出结果都放在该目录下。

---

## 6. 设置环境变量

在启动 vLLM 服务之前，必须设置以下环境变量：

```bash
export SERVICE_PROF_CONFIG_PATH=msserviceprofiler_config.json
export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml
```

含义如下：

| 环境变量 | 说明 |
|---|---|
| `SERVICE_PROF_CONFIG_PATH` | 指定 MS Service Profiler 主配置文件 |
| `PROFILING_SYMBOLS_PATH` | 指定需要插桩分析的 Python 函数符号配置文件 |

注意：

- 这两个环境变量必须在 `vllm serve` 启动之前设置。
- 如果配置文件不存在，工具通常会自动生成默认配置文件。
- 为了保证流程可控，建议手动创建 `msserviceprofiler_config.json`。

---

## 7. 创建最简配置文件

创建 `msserviceprofiler_config.json`：

```bash
cat > msserviceprofiler_config.json <<'EOF'
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "acl_prof_task_time_level": "",
  "timelimit": 0,
  "domain": ""
}
EOF
```

字段说明：

| 字段 | 说明 |
|---|---|
| `enable` | 是否开启采集。`1` 表示开启，`0` 表示关闭 |
| `prof_dir` | profiling 原始数据保存目录 |
| `profiler_level` | profiling 等级，最简使用 `INFO` |
| `acl_task_time` | 是否采集算子下发和执行耗时，最简使用设为 `0` |
| `acl_prof_task_time_level` | ACL profiling 级别，最简使用留空 |
| `timelimit` | 采集时间限制，`0` 表示不限制 |
| `domain` | 指定采集域，空字符串表示采集全部域 |

最简场景下，保持上述配置即可。

---

## 8. 启动 vLLM 服务

示例使用 `Qwen/Qwen2.5-0.5B-Instruct`：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

如果使用本地模型路径，例如：

```bash
vllm serve /root/models/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

注意：

- `SERVICE_PROF_CONFIG_PATH` 和 `PROFILING_SYMBOLS_PATH` 必须在启动服务前设置。
- 服务启动后再设置环境变量不会生效。
- profiling 数据会写入配置文件中指定的 `prof_dir`，即 `./vllm_prof`。

---

## 9. 发送测试请求

新开一个终端，发送 completion 请求：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Beijing is a",
    "max_tokens": 5,
    "temperature": 0
  }' | python3 -m json.tool
```

如果启动服务时使用的是本地模型路径，则请求中的 `model` 字段需要与服务端识别的模型名保持一致。

例如：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/models/Qwen2.5-0.5B-Instruct",
    "prompt": "Beijing is a",
    "max_tokens": 5,
    "temperature": 0
  }' | python3 -m json.tool
```

建议至少发送多次请求，避免采集数据过少：

```bash
for i in $(seq 1 5); do
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "prompt": "Beijing is a",
      "max_tokens": 5,
      "temperature": 0
    }' | python3 -m json.tool
done
```

---

## 10. 查看原始 profiling 数据

回到启动服务所在目录，查看输出目录：

```bash
cd /root/prof_test
ls ./vllm_prof
```

正常情况下，`vllm_prof` 下会生成一个按启动时间命名的目录。

示例：

```bash
./vllm_prof/20260521-153012
```

进入该目录：

```bash
cd ./vllm_prof/<启动时间目录>
```

例如：

```bash
cd ./vllm_prof/20260521-153012
```

---

## 11. 解析 profiling 数据

执行解析命令：

```bash
msserviceprofiler parse --input-path=./ --output-path output
```

解析完成后，会生成 `output` 目录：

```bash
ls output
```

---

## 12. 输出文件说明

常见输出文件如下：

| 文件 | 说明 |
|---|---|
| `chrome_tracing.json` | Chrome Trace 格式时间线文件，可用于可视化 |
| `profiler.db` | 数据库格式 profiling 数据 |
| `request.csv` | 请求级性能数据 |
| `kvcache.csv` | KV Cache 相关数据 |
| `batch.csv` | Batch 调度相关数据 |

重点关注：

```bash
output/chrome_tracing.json
```

该文件可以导入以下工具查看：

- MindStudio Insight
- Chrome Trace Viewer
- Perfetto UI

---

## 13. 最简完整命令流程

### 13.1 终端 1：启动服务

```bash
mkdir -p /root/prof_test
cd /root/prof_test

export SERVICE_PROF_CONFIG_PATH=msserviceprofiler_config.json
export PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml

cat > msserviceprofiler_config.json <<'EOF'
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "acl_prof_task_time_level": "",
  "timelimit": 0,
  "domain": ""
}
EOF

vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### 13.2 终端 2：发送请求

```bash
for i in $(seq 1 5); do
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "prompt": "Beijing is a",
      "max_tokens": 5,
      "temperature": 0
    }' | python3 -m json.tool
done
```

### 13.3 终端 2：解析数据

```bash
cd /root/prof_test/vllm_prof
ls

cd <启动时间目录>

msserviceprofiler parse --input-path=./ --output-path output

ls output
```

---

## 14. 如何只采集指定模块

如果 profiling 数据太多，可以通过 `domain` 限制采集范围。

例如只采集 Request、KVCache 和 BatchSchedule：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "acl_prof_task_time_level": "",
  "timelimit": 0,
  "domain": "Request;KVCache;BatchSchedule"
}
```

常见 domain：

| domain | 说明 |
|---|---|
| `Request` | 请求处理 |
| `KVCache` | KV Cache 管理 |
| `ModelExecute` | 模型执行 |
| `BatchSchedule` | Batch 调度 |
| `Communication` | 通信 |

如果只关注 KV Cache，可以使用：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "domain": "KVCache"
}
```

如果只关注调度，可以使用：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "domain": "BatchSchedule"
}
```

---

## 15. 如何采集算子下发与执行延迟

默认最简配置：

```json
"acl_task_time": 0
```

这表示不采集 ACL 算子下发和执行延迟。

如果需要进一步分析算子下发与执行时间，可以开启：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 1,
  "acl_prof_task_time_level": "L0",
  "timelimit": 0,
  "domain": ""
}
```

参数含义：

| 配置 | 含义 |
|---|---|
| `acl_task_time: 0` | 不采集算子下发与执行时间 |
| `acl_task_time: 1` | 使用 ACL Profiling 采集 task time |
| `acl_prof_task_time_level: "L0"` | 低开销，只采集算子下发和执行延迟 |
| `acl_prof_task_time_level: "L1"` | 更详细，包含 AscendCL 接口、内存拷贝、算子基本信息等 |

建议：

- 首次分析只使用 `acl_task_time: 0`
- 先定位服务框架中的大致瓶颈
- 确认需要进一步分析算子后，再开启 `acl_task_time`

---

## 16. 推荐分析顺序

建议按照以下顺序分析输出结果：

### 16.1 先看 request.csv

目标：

- 判断请求整体耗时
- 判断是否存在请求长尾
- 判断不同请求之间耗时是否稳定

查看：

```bash
cat output/request.csv
```

或者：

```bash
head -n 20 output/request.csv
```

### 16.2 再看 batch.csv

目标：

- 判断 batch 调度是否正常
- 判断每个 batch 的执行耗时
- 判断是否存在 batch 过小、调度过频繁等问题

查看：

```bash
head -n 20 output/batch.csv
```

### 16.3 再看 kvcache.csv

目标：

- 分析 KV Cache 分配、释放和管理行为
- 判断是否存在 KV Cache 相关异常
- 对 kv-cache-size、block 分配、请求调度问题进行辅助定位

查看：

```bash
head -n 20 output/kvcache.csv
```

### 16.4 最后看 chrome_tracing.json

目标：

- 在时间线上观察整体执行流程
- 查看 Request、BatchSchedule、ModelExecute、KVCache 等模块之间的先后关系
- 判断是否存在长时间阻塞、串行化或异常空洞

文件路径：

```bash
output/chrome_tracing.json
```

---

## 17. 常见问题

### 17.1 没有生成 profiling 数据

检查环境变量：

```bash
echo $SERVICE_PROF_CONFIG_PATH
echo $PROFILING_SYMBOLS_PATH
```

检查配置文件：

```bash
cat msserviceprofiler_config.json
```

确认配置中包含：

```json
"enable": 1
```

还需要确认：

- 环境变量是在 `vllm serve` 之前设置的
- 请求确实发送到了当前 vLLM 服务
- vLLM 服务没有提前退出
- 当前目录下有写权限

---

### 17.2 output 目录内容很少

可能原因：

- 请求数量太少
- 服务运行时间太短
- `domain` 设置过窄
- profiling 数据还没有完整落盘
- 采集的函数符号过少

建议：

- 多发送几次请求
- 不要一启动服务就立刻停止
- 先将 `domain` 设置为空字符串
- 使用默认 `service_profiling_symbols.yaml`

---

### 17.3 trace 文件很大

如果 `chrome_tracing.json` 太大，可以限制 domain。

例如只看 KV Cache：

```json
"domain": "KVCache"
```

只看调度：

```json
"domain": "BatchSchedule"
```

只看模型执行：

```json
"domain": "ModelExecute"
```

---

### 17.4 修改配置后没有生效

MS Service Profiler 的配置在服务启动时加载。

如果修改了以下文件：

- `msserviceprofiler_config.json`
- `service_profiling_symbols.yaml`

需要重启 vLLM 服务。

---

### 17.5 如何关闭 profiling

将配置文件中的 `enable` 改为 `0`：

```json
{
  "enable": 0
}
```

或者使用命令：

```bash
sed -i 's/"enable": *1/"enable": 0/' msserviceprofiler_config.json
```

然后重启 vLLM 服务。

---

## 18. 针对 vLLM-Ascend 性能优化的建议用法

如果你正在分析 vLLM-Ascend 的推理性能，可以按照下面思路使用 MS Service Profiler。

### 18.1 分析请求整体耗时

重点看：

```bash
output/request.csv
```

判断：

- 请求总耗时是否异常
- 首 token 延迟是否较高
- 请求之间是否存在明显长尾

### 18.2 分析调度行为

重点看：

```bash
output/batch.csv
```

判断：

- batch 是否过小
- batch 调度是否频繁
- prefill 和 decode 是否存在明显阻塞
- 是否存在调度阶段耗时异常

### 18.3 分析 KV Cache 行为

重点看：

```bash
output/kvcache.csv
```

判断：

- KV Cache 分配是否频繁
- KV Cache 释放是否正常
- 是否存在 cache 空间不足导致的异常行为
- 修改 kv-cache-size 后是否改变调度和分配行为

### 18.4 分析模型执行阶段

重点看：

```bash
output/chrome_tracing.json
```

以及：

```json
"domain": "ModelExecute"
```

判断：

- model execute 是否占据主要时间
- 调度阶段和模型执行阶段之间是否存在空洞
- 模型执行是否和请求调度存在串行化问题

---

## 19. 推荐最小配置

如果只是第一次跑通，推荐使用：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "acl_prof_task_time_level": "",
  "timelimit": 0,
  "domain": ""
}
```

如果只想看 KV Cache：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "domain": "KVCache"
}
```

如果只想看调度：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 0,
  "domain": "BatchSchedule"
}
```

如果想进一步看算子下发和执行延迟：

```json
{
  "enable": 1,
  "prof_dir": "./vllm_prof",
  "profiler_level": "INFO",
  "acl_task_time": 1,
  "acl_prof_task_time_level": "L0",
  "timelimit": 0,
  "domain": ""
}
```

---

## 20. 总结

MS Service Profiler 的最简流程是：

```text
设置环境变量
    ↓
创建 msserviceprofiler_config.json
    ↓
enable 设置为 1
    ↓
启动 vLLM 服务
    ↓
发送推理请求
    ↓
进入 profiling 输出目录
    ↓
执行 msserviceprofiler parse
    ↓
查看 CSV 和 chrome_tracing.json
```

最关键的三个输出文件是：

```text
request.csv
batch.csv
kvcache.csv
```

最关键的可视化文件是：

```text
chrome_tracing.json
```

一句话总结：

> MS Service Profiler 适合分析 vLLM-Ascend 服务框架内部的请求、调度、KV Cache 和模型执行流程；第一次使用时，建议先不开启算子级采集，只用默认框架级 profiling 跑通完整流程。