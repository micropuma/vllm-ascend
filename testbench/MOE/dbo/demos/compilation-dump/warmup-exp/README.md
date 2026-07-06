# Warmup Experiments

这个目录用于把 `compilation-dump/` 里的冷启动 / 编译链路分析，进一步拆成两个可重复的 warmup 实验。

目标不是做泛化 benchmark，而是回答两个更具体的问题：

1. compile range 变化是否会引入真实的编译抖动；
2. ACLGraph 首次 capture 是否会引入真实的 capture 抖动。

详细分析和执行计划见 [`analysis-and-plan.md`](./analysis-and-plan.md)。

## 脚本

- [`server.sh`](./server.sh)：启动服务，默认使用全新 `RUN_ID` 和独立 cache 目录。
- [`test.sh`](./test.sh)：按序列发压，支持 `MODE=observe|warmup`。
- [`run_matrix.sh`](./run_matrix.sh)：一键跑核心矩阵，默认 4 组；`FULL=1` 扩展到 8 组。

## 建议用法

compile range 实验：

```bash
EXPERIMENT=compile_range DUMP_MODE=compile bash server.sh
EXPERIMENT=compile_range MODE=observe bash test.sh
```

ACLGraph 实验：

```bash
EXPERIMENT=aclgraph DUMP_MODE=backend ENABLE_PROFILER=1 bash server.sh
EXPERIMENT=aclgraph MODE=observe bash test.sh
```

对照组建议：

- `MODE=warmup`：在正式序列前先把同样的 shape 跑一遍；
- `COMPILE_RANGES=...`：手动缩小或合并 compile range；
- `CAPTURE_SIZES=...`：手动缩小或重排 ACLGraph bucket。

如果要比较 DBO on/off，直接在 server 上切 `DBO=1/0`，但务必保持 `RUN_ID` 和 cache
目录独立，不要复用旧 cache。

### 一键矩阵

```bash
cd testbench/MOE/dbo/demos/compilation-dump/warmup-exp
bash run_matrix.sh
```

如果要把 DBO 也纳入矩阵：

```bash
cd testbench/MOE/dbo/demos/compilation-dump/warmup-exp
FULL=1 bash run_matrix.sh
```
