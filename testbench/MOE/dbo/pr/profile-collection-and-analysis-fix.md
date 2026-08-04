# [Fix][DBO] 修复 Ascend Profiler 原始数据消费与 flush 判断问题

## 问题概述

Qwen3/DeepSeek DBO profile 目录中出现以下现象：

```text
PROF_*/host/
PROF_*/device_0/
PROF_*/device_1/
```

目录存在但为空，`ASCEND_PROFILER_OUTPUT` 只有部分文件，例如
`trace_view.json`、`memory_record.csv` 和 `operator_memory.csv`，缺少：

```text
step_trace_time.csv
op_statistic.csv
kernel_details.csv
communication.json
communication_matrix.json
```

随后再次执行 `torch_npu.profiler.analyse()` 时出现：

```text
Get localtime diff failed, the start info path is not exist.
```

这不是 DBO hook 或通信算子本身的性能问题，而是 profile 采集/分析生命周期管理问题。

## 根因

### 1. CANN handler 未初始化到正确输出目录

`torch_npu.profiler.tensorboard_trace_handler()` 在构造时就调用
`ProfPathCreator.init()`，不是在 `profiler.start()` 时才读取输出目录。
此前 wrapper 删除了该 handler，并在 `_start()` 才设置 `ASCEND_WORK_PATH`，
因此只能创建空的 `PROF_*/host`、`device_*` 目录，离线 `analyse()` 没有
CANN 原始数据可解析。torch_npu/CANN 版本没有变化，变化的是 wrapper 的
初始化顺序和导出路径。

现在在 wrapper 构造阶段设置 `ASCEND_WORK_PATH`，恢复
`tensorboard_trace_handler(..., analyse_flag=False)` 作为 CANN 导出边界，
再由所有 worker 退出后的脚本执行一次离线 `analyse()`。

### 2. 原始 CANN 数据被重复分析消费

`torch_npu.profiler.analyse()` 会解析并消费 `PROF_*` 下的 CANN 原始
`host/device` 数据。第一次分析后，原始目录为空是正常行为。

现有 profile 的时间戳证明了这一点：

- `host`、`device_1` 在 profile 启动阶段创建；
- `device_0` 随后写入；
- 第一次分析生成了 `memory_record.csv` 等结果；
- 再次分析同一个 `*_ascend_pt` 目录时，原始 `start_info` 已经不存在，
  因此 CANN parser 报错。

因此，不能对同一个 profile run 重复执行 `analyse()`。缺失的 CSV 不能
通过对已分析目录再次分析恢复，必须重新采集。

### 3. 旧版 `start_info.done` 等待条件不适用于 CANN 9

旧版 `deepseek-v2-dbo-test.sh` 等待：

```text
host/start_info.done
```

CANN 9 不保证生成这个历史 marker。脚本可能在真正的 profiler flush
完成判断前超时，或者直接进入不完整分析。

当前使用每个 rank 的：

```text
profiler_info_*.json.end_info
```

作为 worker flush 边界，并在此后执行一次离线 `analyse()`。

### 4. stop 时处于 `RECORD`，CANN 明确报告不完整 finalize

这次新采集的 `baseline_qwen_20260803T120521Z` 日志明确记录：

```text
Incorrect schedule: Stop profiler while current state is RECORD
which may result in incomplete parsed data.
```

无 schedule 时 torch-npu 默认状态是 `RECORD`。直接调用 `stop()` 会走
`RECORD -> None`，CANN 虽然写出 framework trace 和 `profiler_info`，但
不保证生成完整的 device timeline，因此 `analyse()` 只能生成部分输出。

之前可用的 DeepSeek profiler 配置使用：

```json
{
  "profiler": "torch",
  "torch_profiler_with_stack": true,
  "torch_profiler_record_shapes": true,
  "max_iterations": 20
}
```

之前可用脚本实际依赖 worker 内置的 `on_trace_ready -> analyse()` 路径；
wrapper 改成 `analyse_flag=False` 后，stop 之后再从外部分析，暴露了上述
不完整 finalize 问题。

当前统一恢复为：

- benchmark 请求 warmup 在 `/start_profile` 之前执行；
- profiler wrapper 恢复使用 torch-npu handler 完成 CANN 导出，但关闭
  worker 内 daemon 分析（`analyse_flag=False`）；
- 外层脚本在所有 worker 退出后执行一次离线 `analyse()`；
- 不改变 benchmark warmup 行为；
- server 只使用 `max_iterations=${PROFILER_MAX_ITERATIONS}`；
- 外层脚本检测 `analyse.done` 后复用结果，不重复调用 `analyse()`。

## 修改范围

已同步修改以下脚本：

- `testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-server.sh`
- `testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh`
- `testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server-dp.sh`
- `testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh`
- `testbench/MOE/dbo/demos/Qwen3-30B/Qwen3-30B-server.sh`
- `testbench/MOE/dbo/demos/Qwen3-30B/Qwen3-30B-dbo-server.sh`
- `testbench/MOE/dbo/demos/profile-scripts/e2e_profile.sh`
- `testbench/MOE/dbo/demos/profile-scripts/e2e_dp_profile.sh`
- `testbench/MOE/dbo/demos/profile-scripts/qwen3_30b_profile.sh`
- `testbench/MOE/dbo/demos/profile-scripts/profile_analyse.sh`

关键行为：

1. 每个 rank 都被发现、等待和分析，不再只取最后一个 rank。
2. 已存在 `ASCEND_PROFILER_OUTPUT/analyse.done` 时不重复分析。
3. 已分析但缺少核心 CSV 时直接失败，并提示重新采集。
4. 所有脚本要求以下核心产物非空：

   ```text
   trace_view.json
   step_trace_time.csv
   op_statistic.csv
   kernel_details.csv
   communication.json
   communication_matrix.json
   ```

5. `torch_npu_profiler` wrapper 传递 `record_shapes`，保持 server 配置与
   实际 profiler 行为一致。

## 推荐运行方式

使用新的 profile tag 或新的目录，不复用已经存在的 `*_ascend_pt`：

```bash
PROFILE_CASE=baseline PROFILE_TAG=qwen-baseline-$(date -u +%Y%m%dT%H%M%SZ) \
bash testbench/MOE/dbo/demos/profile-scripts/qwen3_30b_profile.sh

# Run DBO separately, with another tag (and after the baseline process has exited).
PROFILE_CASE=dbo PROFILE_TAG=qwen-dbo-$(date -u +%Y%m%dT%H%M%SZ) \
bash testbench/MOE/dbo/demos/profile-scripts/qwen3_30b_profile.sh
```

默认输出目录是：

```text
testbench/MOE/dbo/demos/profile/
```

DeepSeek 手动流程：

```bash
ENABLE_PROFILER=1 PORT=8001 \
bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server.sh

LABEL=dbo PORT=8001 BENCH_PRESET=prefill4k \
bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-test.sh --profile
```

分析完成后不要再次对同一个 run 调用 `analyse()`。如需重新分析，必须
确认没有 `analyse.done` 且 `PROF_*` 原始文件仍然存在；否则重新采集。

## 验证状态

已完成：

- 所有相关 shell 脚本 `bash -n` 检查；
- server profiler 配置与之前可用的 DeepSeek 配置对齐；
- CANN 9 的 `end_info` flush 判断替代旧 `start_info.done` 判断；
- 重复 `analyse()` 保护和完整 CSV gate。

已额外验证：

- 使用当前 `torch_npu 2.10.0 / CANN 9.0.0`，直接通过修复后的
  `TorchNPUProfilerWrapper` 采集 NPU matmul，`host/` 和 `device_0/` 均有
  非空 raw 文件；
- 离线 `analyse()` 成功生成 `step_trace_time.csv`、`op_statistic.csv`、
  `kernel_details.csv`、`analysis.db` 等完整产物。

旧 profile 目录无法恢复，必须使用新目录验证完整 Qwen baseline/DBO pair。

## 结论

该问题属于 profiler 原始数据生命周期和脚本 flush/分析逻辑问题，不能
据此判断 DBO 没有 overlap 或通信性能退化。重新采集得到完整的
`step_trace_time.csv`、通信和算子统计后，才可以继续进行 DBO baseline
对比和性能归因。
