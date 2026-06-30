# DeepSeek-V2 DBO 与 FlashComm 测试矩阵

## 测试环境

- 日期：2026-06-30
- 模型：`/data/models/DeepSeek-V2-Lite-Chat`
- 设备：2 x Ascend 910B3；TP=2
- vLLM：0.22.1；vLLM Ascend：`f8bbd90d00495427286cec514c1a11417d5233a8`
- 输入/输出：4096/16 tokens；请求数/并发：500/96
- FC2 O-Shard：关闭；DBO prefill/decode threshold：1024/1000000000

所有成功的 DBO 测试均在服务日志中出现 `should_ubatch: True`。

## Benchmark 结果

| DBO | FC1 | FC2 | HCCL模式 | 成功 | QPS | 输出tok/s | 平均TTFT ms | P99 TTFT ms | 平均TPOT ms | P99 TPOT ms | 平均E2E ms | P99 E2E ms |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 关 | 关 | 关 | AIV | 500/500 | 6.207 | 99.309 | 6144.92 | 13887.16 | 605.81 | 661.64 | 15232.06 | 23625.50 |
| 关 | 开 | 关 | AIV | 500/500 | 6.232 | 99.718 | 6079.69 | 13729.71 | 605.59 | 659.75 | 15163.52 | 23448.37 |
| 关 | 关 | 开 | AIV | 500/500 | 6.396 | 102.332 | 5932.04 | 13728.69 | 589.65 | 642.42 | 14776.74 | 23162.80 |
| 开 | 关 | 关 | AIV | 500/500 | 6.403 | 102.454 | 5958.78 | 13815.27 | 586.74 | 651.00 | 14759.81 | 23203.33 |
| 开 | 关 | 关 | AI_CPU | 500/500 | 6.590 | 105.433 | 5782.68 | 13267.25 | 570.86 | 637.00 | 14345.65 | 22354.92 |
| 开 | 开 | 关 | AIV | 500/500 | 6.555 | 104.877 | 5777.55 | 13301.00 | 574.06 | 637.96 | 14388.49 | 22425.52 |
| 开 | 开 | 关 | AI_CPU | 500/500 | 7.609 | 121.739 | 5027.23 | 11690.74 | 491.76 | 544.90 | 12403.58 | 19614.06 |
| 开 | 关 | 开 | AIV | 500/500 | 6.506 | 104.095 | 5808.21 | 13237.30 | 581.15 | 642.28 | 14525.48 | 22513.15 |
| 开 | 开 | 开 | AIV | 500/500 | 6.619 | 105.906 | 5749.78 | 13250.99 | 566.62 | 635.16 | 14249.13 | 22323.51 |

## 已知失败组合

所有 `FlashComm2=开启` 且 `HCCL_OP_EXPANSION_MODE=AI_CPU` 的组合无法完成请求。两rank最小复现在第一次 `dist.all_to_all_single()` 即报：

```text
RunAicpuRpcSrvLaunchV2_alltoall
errorCode=0x2a
runtime result=507018
```

相同复现切换 AIV 后通过。DBO、FC1、ACL Graph、消息大小和tensor layout均不是必要条件。故障边界位于CANN/HCCL 9.0.0的AI_CPU AllToAll路径；修复前无法取得有效的FC2+AI_CPU性能数据。

## 数据与日志

- JSON：`/data/workspace/vllm-ascend/testbench/MOE/dbo/results/matrix_*_20260630.json`
- Server/Test日志：`/data/workspace/logs/matrix-20260630/`
- AI_CPU最小复现：`/data/workspace/logs/verify-minimal-aicpu-20260630.log`
- AIV对照：`/data/workspace/logs/verify-minimal-aiv-20260630.log`

## 初步结论

- DBO且FlashComm全关时，AI_CPU吞吐比AIV高约2.9%。
- 最快有效组合为DBO+FC1+AI_CPU：7.609 QPS、121.739 output tok/s。
- AIV下DBO+FC1+FC2为6.619 QPS，比DBO-only+AIV高约3.4%。
- 当前均为单轮数据；稳定结论需每组至少重复三轮并统计方差。
