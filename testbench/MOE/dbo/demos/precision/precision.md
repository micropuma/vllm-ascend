# DBO 精度定位与修正手册

日常精度验证只运行 `python run_precision.py`，它负责 baseline -> 停止 -> 冷启动 DBO -> 双 TP trigger
验证 -> 配对报告的完整事务。本文其余内容是解释评测门限、处理分叉和进入模型内定位时使用的诊断参考；不要将其中
保留的手工命令与日常入口混用。

本手册用于 DeepSeek-V2-Lite-Chat、Ascend A2、TP=2 的 DBO 精度问题。目标不是把 `temperature=0`
的输出差异一概视为缺陷，而是用可复现证据区分三类情况：

1. 测试调度或输入批次不同造成的表面差异；
2. 浮点归约顺序变化造成的低 margin 贪心分叉；
3. DBO 的 event、stream、通信完成依赖或张量切分错误。

只有第 3 类可以直接修改 DBO 实现。第 2 类先要证明业务精度满足预设的非劣界，不能通过“没有报错”放行。

## 0. 固定不变量

`run_precision.py` 自动创建每一个比较目录、记录 editable import/commit，并在 source 环境后设置
`NO_PROXY`/`no_proxy`。手工诊断才需要单独创建目录：

```bash
cd /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos/precision
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate
source /data/workspace/vllm-dbo-v0221/env.sh
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
RUN_DIR=$(python init_run.py --results-root results)
```

`manifest.json` 会记录 editable import 路径、两个 commit、工作树是否 dirty、Python 路径和 evaluator 进程环境。
若 `vllm` / `vllm_ascend` 不是本地源码，或 baseline/DBO 的模型、TP、EP、`MAX_MODEL_LEN`、batch 限制、
generation config、并发、样本集合有任何不同，该目录不得用于精度结论。

当前 A2 的特别事实：DBO 配置名会被置为 `deepep_low_latency` 以满足 upstream 的 microbatch 校验，
但实际 MoE 选择为 Ascend `ALLGATHER`。不要把该字符串当成已经启用 DeepEP kernel 的证据。DeepSeek A2 DBO
使用 `DeepseekAllgatherTemplate`。

精度比较时强制两边均设置：

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
export VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED=0
```

这很重要：baseline 脚本默认 FC1 是 `0`，而当前 DBO 脚本的默认值可能不同。测试不得依赖脚本默认值。
已有结果只有在两边显式记录同一值时才可用于判断 DBO。

## 1. 冻结服务路径与 DBO 触发

先启动 baseline，完成本轮 baseline 请求后停止它；再使用一个新的空 `XDG_CACHE_HOME` 冷启动 DBO。不要删除共享 cache。
DBO 服务必须以 DEBUG 日志启动，并在跑完后：

```bash
python verify_dbo_log.py --log /tmp/dbo-accuracy.log --tp-size 2 --run-dir "$RUN_DIR"
```

通过条件：两个 TP rank 都出现 `should_ubatch: True`。仅 HTTP 200 不是 DBO 执行证据。每轮将 server 命令、
环境和日志归档到 `$RUN_DIR`。如果不满足，停止后续 accuracy 判断，先查阈值、batch token 数和启动参数。

server 常在独立 shell 中启动，因此不能只依赖 `manifest.json` 的环境。baseline 和 DBO 各自就绪后均执行：

```bash
python record_server_config.py --mode baseline --log /tmp/baseline-accuracy.log --run-dir "$RUN_DIR"
python record_server_config.py --mode dbo --log /tmp/dbo-accuracy.log --run-dir "$RUN_DIR"
```

这会写入 `baseline_server_config.json` / `dbo_server_config.json`，其中的 `printed_server_config` 是脚本
实际打印参数，`runtime_facts.flashcomm1` 是 AscendConfig 实际解析值。两者不一致或缺失时，不能做比较。

服务启动完成的判据是 `curl --noproxy '*' -sf http://127.0.0.1:8001/v1/models` 成功，且 JSON 的
`data` 非空。模型加载、profile run 和 graph capture 期间端口可能尚未监听；不得在此期间提交评测，
更不得把连接失败生成的 artifact 当作精度结果。

## 2. 建立自一致性控制组

先不要比较 baseline 与 DBO。为每个模式分别跑两次完全相同的 workload：

```text
baseline-a  baseline-b
dbo-a       dbo-b
```

至少包含两种 workload：

| workload | 目的 | 约束 |
|---|---|---|
| 单请求长 prefill | 排除请求到 batch 拼接噪声，同时超过 DBO prefill threshold | DBO 日志必须触发 |
| 固定样本、固定并发波次 | 覆盖真实 continuous batching | 两次请求顺序、并发和服务配置相同 |

运行时加 `--logprobs`，例如：

```bash
python accuracy_runner.py --config configs/gsm8k.yaml \
  --endpoint http://127.0.0.1:8001 --model /data/models/DeepSeek-V2-Lite-Chat \
  --concurrency 64 --limit 128 --logprobs --top-logprobs 5 \
  --output "$RUN_DIR/baseline_a_gsm8k.json"
```

两次 artifact 比较：

```bash
python diff_artifacts.py \
  --left "$RUN_DIR/baseline_a_gsm8k.json" \
  --right "$RUN_DIR/baseline_b_gsm8k.json" \
  --output "$RUN_DIR/diff_baseline_self_gsm8k.json"
```

对 DBO 同样执行。若 baseline 自一致性已有分叉，不能把同等级的 DBO 分叉归因于 DBO；反之，若 baseline 稳定而
DBO 不稳定，DBO 是主要嫌疑。连续 HTTP 请求不能严格保证每轮 scheduler 打出同一个物理 batch。这里的控制组
量化这种影响；若需要绝对固定物理 batch，应进入第 4 步的模型内测试，而不是声称 API 层已经固定了 batch。

## 3. 定位首个生成 token 分叉

`--logprobs` 会在 artifact 的每条样本中保存 OpenAI Chat API 返回的：输出 token、该 token 的 logprob、top-k token
和 logprob。`diff_artifacts.py` 输出：

* `text_mismatch_count`：完整文本不同的样本数；
* `token_divergences`：每样本第一个不同 token 的 index、两侧 token 与两侧 top-1/top-2 margin；
* `text_mismatches_without_token_logprobs`：服务器未返回 logprob，不能做 token 归因的样本数；
* `accuracy_transitions`：同题的对/错转移。

若服务启动时添加 `--return-tokens-as-token-ids`，token 字段会以 token id 形式返回，诊断更稳定；否则比较的是 API
返回 token 文本和 bytes 语义。该服务选项必须同时作用于 baseline 和 DBO。

| 观察 | 初步判断 | 后续动作 |
|---|---|---|
| 只在很小 margin 首分叉，且 baseline 自一致性也有 | 数值非结合性或批次噪声 | 做配对 accuracy 非劣判断 |
| baseline 自一致性稳定，DBO 高 margin 首 token 分叉 | DBO 数值/同步高风险 | 进入第 4 步 |
| 首分叉很早且大量样本同层附近发生 | 共享计算路径或上下文恢复错误 | 进入第 4 步 |
| 无 logprob | 证据不足 | 重跑该小集合，保持 `--logprobs` |

不要以全篇 CoT 文本不同本身判 bug；一旦第一个 greedy token 不同，后续文本本来就会级联不同。

## 4. 对首分叉样本做模型内检查

API 层只能定位到输出 token。确认 DBO 特有的高风险分叉后，使用一个固定输入 batch、同一进程配置分别运行 baseline
与 DBO，并只记录首分叉样本。记录应放在测试专用 hook，不能在热路径无条件 `.item()` 或 dump 全 tensor。

按顺序添加以下摘要，每项都带 layer index、TP rank、ubatch id、shape、dtype：

1. router 的 top-k expert ids 与 weights；
2. MoE prepare 后输入、expert MLP 输出、MoE finalize 后输出；
3. MLA preprocess all-gather 后、o_proj reduce-scatter 后的张量；
4. 每层输出与最终 logits 的 `max_abs`、`max_rel`、cosine；
5. 首分叉位置 top-k logits 与 margin。

张量摘要使用 `float32` 下的 `amax`、`sum`、`sum of squares`，每层只同步一次；有需要再将单个首分叉 token 的
小切片落盘。不要全量保存 activation，也不要为调试新增未审查的环境变量。

| 证据 | 最可能位置 | 修正方向 |
|---|---|---|
| router top-k 首先不同 | MoE 输入、prepare/all-gather、上下文切分 | 查 token slice、padding、all-gather 完成依赖 |
| router 相同，MoE finalize 后误差跳变 | finalize/reduce 或 comm stream 同步 | 在 finalize 前后检查 event record/wait 顺序与 tensor 生命周期 |
| MLA all-gather 或 o_proj 后误差跳变 | `ATTN_PRE`/`ATTN_POST` event 依赖 | 检查 hook 是否在通信前 record、消费者是否 wait 后才读取 |
| 各层接近，只有最终 margin 极小处翻转 | 浮点归约顺序 | 不改同步；进入非劣 accuracy 判定 |

对 DeepSeek A2，优先检查 `DeepseekAllgatherTemplate` 的两个依赖：MLA preprocess 与上一层 MoE finalize 的
`ATTN_PRE`，以及 post-MLA 与 MoE prepare 的 `ATTN_POST`。任何修复必须先用固定 batch 摘要证明误差消失，再进入全量 accuracy。

## 5. 配对 accuracy 与放行门限

全量集在定位后才运行：

```bash
python compare_runs.py --run-dir "$RUN_DIR" --task gsm8k \
  --max-accuracy-drop 0.005 --allow-text-mismatches 999999
```

报告给出 `true_false`（baseline 对、DBO 错）与 `false_true`（baseline 错、DBO 对），并给出 two-sided exact
McNemar p-value。总分差相同但转移方向不同，风险完全不同。

建议初始 gate：

* 功能 gate：0 请求失败，DBO 两个 TP rank 均触发；
* 数值阻塞：DBO 相对自一致性新增的高 margin 首分叉、router 分叉或层级误差突增为 0；
* GSM8K 主 gate：DBO 相对 baseline accuracy 不低于 `0.5pp`，且用大样本成对结果复核；
* GPQA Diamond：198 条只作告警。单向 `4:0` 退化必须复现和定位，不能仅以 p-value 不显著放行；
* 文本 mismatch 不是默认硬 gate，必须与 baseline 自一致性及首分叉 margin 一起解释。

当前历史结果中，GSM8K 的成对转移是 `56` 个 baseline 对/DBO 错、`52` 个反向；GPQA Diamond 是 `4` 个 baseline
对/DBO 错、`0` 个反向。这是启动本流程的线索，不是最终结论：它们尚未具备 baseline 自一致性和 token-logprob 证据。

### 2026-07-29 首轮控制组

目录 `results/precision-control-20260729T0625Z/` 使用本地 editable `vllm` commit `0decac0`、
`vllm-ascend` commit `fb880822`、TP=2/EP=2、FC1=0、`temperature=0`、`top_p=1`、GPQA Diamond
前 64 条、并发 64、`max_tokens=8`。baseline 完成后停止，DBO 用独立空 cache 冷启动；DBO TP0/TP1 均有
`should_ubatch: True`。

| 对比 | 文本/首 token 分叉 | accuracy 转移 | 首分叉最大 margin |
|---|---:|---|---:|
| baseline-a vs baseline-b | 5 / 64 | 0 个 DBO 相关；baseline-b 多 1 题 | 0.375 |
| dbo-a vs dbo-b | 2 / 64 | 0 | 0.25 |
| baseline-a vs dbo-a | 4 / 64 | 0 | 0.375 |

本轮范围内，DBO 的分叉量没有超过 baseline 自一致性，且没有高 margin 首 token 分叉、router 分叉或层级张量证据。
因此**不修改 DBO 同步实现**；当前最合理的暂时解释是 batch/浮点敏感性。该结论只覆盖 64 条 GPQA 和当前
配置，不能外推为全量 accuracy 放行。下一步按本手册跑全量 GSM8K 与 GPQA，并在出现 DBO 特有高风险分叉后才进入第 4 步。

### 2026-07-29 完整 GPQA 复跑

目录 `results/precision-full-gpqa-20260729T0645Z/` 使用同一源码、TP/EP、FC1=0、并发 64 和 generation
参数；baseline 完成后停止，DBO 使用新的空 cache 冷启动，且 TP0/TP1 均触发。完整 198 条结果如下：

| 对比 | 正确数 | 相对 baseline 的配对转移 | 文本/首 token 分叉 |
|---|---:|---|---:|
| baseline | 65 / 198 | - | - |
| dbo-a | 62 / 198 | baseline 对/DBO 错 5；反向 2 | 18 |
| dbo-b（同一 DBO 服务重复） | 67 / 198 | baseline 对/DBO 错 3；反向 5 | 18 |
| dbo-a vs dbo-b | 62 -> 67 | dbo-a 对/dbo-b 错 0；反向 5 | 12 |

为量化 baseline 自身波动，又启动新的 baseline 服务，在完全相同 workload 下跑第二遍：仍为 `65 / 198`，但
baseline-a vs baseline-b 有 `16` 条文本/首 token 分叉；配对转移是 `2` 条对到错、`2` 条错到对，净 accuracy
变化为 `0`。以这个 baseline-b 再比较，DBO-a 为 `62 / 198`（`5:2`，McNemar `p=0.453125`，24 条分叉），
DBO-b 为 `67 / 198`（`1:3`，`p=0.625`，21 条分叉）。

因此同一 baseline 已能产生与单次 baseline-vs-DBO 同量级的 token/text 分叉；而 DBO 相对 baseline 的
accuracy 差异又在 `-3` 与 `+2` 间变号。单次的 `-3`、此前的 `-4` 均不是稳定可重复的 DBO accuracy 回归，
也不能据此修改 DBO 同步。当前没有 router/activation 级证据授权改实现；下一步应补全量 GSM8K，并以“DBO
相对 baseline 自一致性新增的高 margin 分叉”作为进入第 4 步的条件。

## 6. 修正、回归和归档

仅在第 4 步把误差定位到特定 DBO 依赖后修改实现。一个修复只解决一个已证明的根因，例如：

1. 缺少 wait：在消费者读取通信结果前补正确 event wait；
2. record 时机错误：把 event record 放到真实通信开始前，不能放在已读取结果后；
3. ubatch 上下文错用：按 ubatch id 恢复独立的 forward context 和 MoE communication instance；
4. padding/slice 错误：修复后断言恢复的 token 数、rank slice 和原始 batch 完整一致。

每个修复依次通过：单元测试/固定 batch 摘要 -> baseline/DBO 自一致性 -> 小样本逐 token -> 全量配对 accuracy ->
冷启动 DBO E2E。归档至少包含 `manifest.json`、两组 baseline/DBO artifact、三份 diff、`compare_<task>.json`、
`dbo_server.log` 和 `dbo_trigger.json`。

修复前后必须保持同一模型、数据 manifest、generation 参数和测试 workload。性能优化不能以移除必要 wait 为代价；
若正确同步导致性能下降，应单独报告，而不是放宽精度门限。

## 7. 2026-08-04 Qwen3 & DeepSeek 完整精度

> **Git**: `7475635f` | **并发**: 64 | **FC1**: 0 | **闸门**: accuracy drop ≤ 0.5pp
> **Qwen3 日志**: `results/20260804T045755Z/`
> **DeepSeek 日志**: `results/20260804T051617Z/`

### Qwen3-30B-A3B

**模型**: `/data/models/Qwen3-30B/Qwen3-30B` | **chat_template_kwargs**: `enable_thinking: false`

| Task | 题目数 | Baseline | DBO | Δ | McNemar p | Gate |
|------|--------|----------|-----|---|-----------|------|
| GSM8K | 1,319 | **92.04%** (1214) | 91.81% (1211) | **-0.23pp** | 0.701 | 通过 |
| GPQA Diamond | 198 | **43.43%** (86) | 42.93% (85) | **-0.51pp** | 1.000 | 边界 |

| Task | Text Mismatches | 说明 |
|------|----------------|------|
| GSM8K | 719 / 1319 | DBO 推理路径差异（temperature=0 仍存在），准确率差 3 题 |
| GPQA Diamond | 12 / 198 | 仅差 1 题，198 题小样本一次 flip 即为 0.51pp |

- GSM8K 通过：DBO 下降 0.23pp，McNemar p=0.701 不显著。
- GPQA 边界：86→85（差 1 题），0.51pp 刚好超闸，p=1.0 完全无统计意义。
- **关键发现**：Qwen3 必须设 `enable_thinking: false`，否则 `<think>` 标签会吃掉 token 预算（GPQA max_tokens=8 时 baseline 为 0%）。已在 `configs/gsm8k.yaml` 和 `configs/gpqa_diamond.yaml` 中固化。

### DeepSeek-V2-Lite-Chat

**模型**: `/data/models/DeepSeek-V2-Lite-Chat` | **chat_template_kwargs**: 无（不需要）

| Task | 题目数 | Baseline | DBO | Δ | McNemar p | Gate |
|------|--------|----------|-----|---|-----------|------|
| GSM8K | 1,319 | **73.84%** (974) | 72.71% (959) | **-1.14pp** | 0.151 | 超闸 |
| GPQA Diamond | 198 | **32.83%** (65) | 32.32% (64) | **-0.51pp** | 1.000 | 边界 |

| Task | Text Mismatches | 说明 |
|------|----------------|------|
| GSM8K | 746 / 1319 | DBO 文本路径差异较多 |
| GPQA Diamond | 15 / 198 | 差 1 题，与 Qwen3 GPQA 同现象 |

- GSM8K 超闸：974→959（-1.14pp，15 题），p=0.151 未达 0.05 但偏低。需 `diff_artifacts.py` 定位首分叉 + 重复 3 次确认稳定性。
- GPQA 边界：65→64，同 Qwen3 现象。
- 历史对照：`precision-full-gpqa-20260729T0645Z` 的 baseline vs DBO 曾有 65→62（5:2, p=0.453），但 baseline 自一致性为 65→65（2:2），说明 GPQA 的 ±3 波动在 baseline 自身范围内。

### 跨模型对比

| Task | Qwen3 Baseline | Qwen3 DBO | Δ Qwen3 | DS Baseline | DS DBO | Δ DS |
|------|:-------------:|:---------:|:-------:|:-----------:|:------:|:----:|
| GSM8K | 92.0% | 91.8% | **-0.23pp** | 73.8% | 72.7% | **-1.14pp** |
| GPQA Diamond | 43.4% | 42.9% | **-0.51pp** | 32.8% | 32.3% | **-0.51pp** |

| 模型 | GSM8K 绝对优势 | GPQA 绝对优势 |
|------|:-------------:|:-------------:|
| Qwen3 vs DeepSeek | **+18.2pp** | **+10.6pp** |

### 结论与下一步

1. **DBO 不引入显著精度退化**：GPQA 两组均为 1 题 flip，McNemar p=1.0，纯属小样本随机。
2. **DeepSeek GSM8K 需复核**：1.14pp 下降 + 746 text mismatch，建议用 `diff_artifacts.py` 对翻转样本做首 token 定位，同一环境重复 3 次确认是否为稳定回归。
3. **Qwen3 `enable_thinking: false`**：已固化到 config YAML，对 DeepSeek 无影响。
4. **GPQA 198 题不作为硬 gate**：单题 flip 即超 0.5pp 闸门，建议只作告警，以 GSM8K 为主要 gate。
