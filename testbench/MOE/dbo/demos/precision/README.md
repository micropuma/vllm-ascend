# DBO Offline Precision

`run_precision.py` is the daily entry point. It uses only `/data/datasets` and
the local `127.0.0.1` vLLM service; it neither downloads a dataset nor uses an
external inference API. A local HTTP service is intentional: DBO changes
continuous batching and must be evaluated through the serving path.

## One Command

Activate the same editable environment used by the server, then run both
datasets:

```bash
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate
cd /data/workspace/vllm-dbo-v0221/vllm-ascend/testbench/MOE/dbo/demos/precision
python run_precision.py
```

For a quick smoke test before a full run:

```bash
python run_precision.py --task gpqa_diamond --limit 64
```

## Quick Regression Gate

Run this after a DBO performance change before starting a longer accuracy run:

```bash
bash quick/test_dbo_precision.sh
bash quick/test_dbo_fc1_precision.sh
```

The quick gate uses a concurrent wave of 16 unique, tokenizer-verified 2K
prefills. It requires `should_ubatch: True` from both TP ranks, requires every
HTTP request and output-logprob response to succeed, and fails on a generated
token difference or sampled-token logprob difference above `1e-3`. Its result
artifacts are saved under `precision/results/quick-*`. It reuses the normal
compile caches for turnaround time, so it is a correctness gate rather than a
cold-start or performance measurement.

The runner forces `NO_PROXY`/`no_proxy` for local traffic and forces
`VLLM_ASCEND_ENABLE_FLASHCOMM1=0` in both modes. It performs this sequence:

```text
baseline server -> evaluate -> stop
new empty DBO cache -> DBO server -> evaluate -> verify TP0 and TP1 trigger -> stop
paired accuracy reports
```

The default accuracy gate permits at most a `0.5pp` baseline-to-DBO decline.
Override it only with an explicit policy, for example
`--max-accuracy-drop 0.0`. Text mismatches are retained for diagnosis rather
than used as a gate because concurrent serving can have baseline self-variance.

## Artifacts

Every invocation writes `results/<run-id>/`:

```text
manifest.json                 checked-out commits and editable import paths
orchestrator.json             exact runner policy and cold DBO cache root
baseline_<task>.json          raw outputs, parsed answers, optional logprobs
dbo_<task>.json               matching DBO results
baseline_server_config.json   printed and runtime baseline facts
dbo_server_config.json        printed and runtime DBO facts
dbo_trigger.json              required DBO evidence from both TP ranks
compare_<task>.json           paired transitions and McNemar p-value
```

Use `diff_artifacts.py` only to inspect a completed pair or repeated runs. Use
`rescore_artifact.py` only after changing an answer parser; it does not rerun
inference.

## Development Checks

The framework is Python-only, so no C++ kernel or binding is introduced here.
Python code follows Google-style docstrings and type annotations. Run:

```bash
python -m pytest
python -m mypy --config-file pyproject.toml .
```
