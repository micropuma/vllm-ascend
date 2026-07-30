# Qwen3-30B DBO benchmark

This directory adapts the common DeepSeek DBO server and benchmark workflow for
the local Qwen3-30B-A3B checkpoint at `/data/models/Qwen3-30B/Qwen3-30B`.
The checkpoint is Qwen3 MoE, so both modes use TP=2 and expert parallelism.

Run the matched baseline and cold-DBO `prefill4k` comparison:

```bash
bash run-e2e.sh
```

The workload is 4096 input tokens, 16 output tokens, 500 requests, and 96-way
concurrency. Results are placed in `results/`; isolated DBO cache and server
logs are retained under `/data/tmp/qwen3-30b-dbo-<run-id>/`. The driver fails
unless both TP workers log `should_ubatch: True`.

For manual runs, source the DBO environment, launch either server script, then
run `LABEL=baseline|dbo PORT=8001 bash Qwen3-30B-dbo-test.sh`.
