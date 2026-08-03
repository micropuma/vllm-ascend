# DP DBO token-skew: 4096 vs 2048

## Question and controlled setup

This report isolates a single decision divergence between the current Ascend
implementation and upstream vLLM:

| DP0 | DP1 | Current Ascend | Upstream vLLM |
|---:|---:|---|---|
| 4096 real tokens | 2048 real tokens | enables DBO | disables DBO |

Hardware/model/configuration used for the measurements:

- 2 x Ascend 910B3; DP=2, TP=1, EP=2; `deepep_low_latency`.
- DeepSeek-V2-Lite-Chat, BF16.
- DBO prefill threshold 1024; decode threshold `1_000_000_000`.
- `--enforce-eager` for both A/B configurations. This removes graph-capture
  noise; it is not a claim about graph-mode throughput.
- One pair means one 4096-token request routed to DP0 concurrently with one
  2048-token request routed to DP1. Each request generates one token.
- Every pair changes its first input token. This invalidates the chained
  prefix-cache block hashes while preserving the exact prompt lengths.

The request driver is
`testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-dp-skew-test.py` and routes through
`X-data-parallel-rank`.

## Reproduce the plain A/B

Run these commands from the repository root. Start the server in one terminal;
run the client commands only after `/v1/models` responds. Stop the server
between configurations. `DBO_ENABLED` is the only server-side variable that
changes.

```bash
source /data/workspace/vllm-dbo-v0221/env.sh
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate

# Terminal A: DBO-on server.
DBO_ENABLED=1 ENFORCE_EAGER=1 VLLM_LOGGING_LEVEL=INFO \
  bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server-dp.sh

# Terminal B: wait for readiness, warm up, then measure three independent runs.
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 2; done
python testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-dp-skew-test.py \
  --repeats 3 --token-offset 0
for run in 0 1 2; do
  python testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-dp-skew-test.py \
    --repeats 15 --token-offset "$((1000 + run * 100))"
done
```

Terminate Terminal A cleanly, then repeat exactly the same commands with
`DBO_ENABLED=0`. Do not use a profiler in this section: these numbers are the
performance result. The client prints one latency for each concurrent request
pair; aggregate the three printed means, rather than mixing all individual
requests.

`--token-offset` is important. It changes the first token of every prompt;
prefix-cache block hashes are chained, so a measurement pair cannot reuse a
prefill from warmup or an earlier measured pair.

## Reproduce the profile and export MindStudio JSON

Use a fresh output directory and a DBO-on server. Profile latency is
diagnostic only and must not be compared with the plain A/B values.

```bash
source /data/workspace/vllm-dbo-v0221/env.sh
source /data/workspace/vllm-dbo-v0221/.venv-dbo/bin/activate

export ARTIFACT_ROOT=/data/tmp/dp-dbo-profile-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ARTIFACT_ROOT"

# Terminal A
ENABLE_PROFILER=1 PROFILER_MODE=operator PROFILER_MAX_ITERATIONS=20 \
ENFORCE_EAGER=1 VLLM_LOGGING_LEVEL=DEBUG TORCH_PROFILER_DIR="$ARTIFACT_ROOT" \
LABEL=skew_4096_2048 \
  bash testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-server-dp.sh

# Terminal B, after readiness
until curl -sf http://127.0.0.1:8001/v1/models >/dev/null; do sleep 2; done
python testbench/MOE/dbo/demos/DeepseekV2/deepseek-v2-dbo-dp-skew-test.py \
  --profile --repeats 8 --output-tokens 1 --token-offset 1000

# After stop_profile returns, analyse each rank directory.
for profile_dir in "$ARTIFACT_ROOT"/*_ascend_pt; do
  python - "$profile_dir" <<'PY'
import sys
from torch_npu.profiler import profiler
profiler.analyse(sys.argv[1], max_process_number=8)
PY
done
```

The normal import target is
`$ARTIFACT_ROOT/*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json`.
Validate it first with `jq empty <trace_view.json>`. On the CANN version used
for this report, the exporter occasionally emits a complete event sequence but
omits its final `]`. Only if the raw file ends in `}` and its recovered copy is
one byte longer should the following mechanical recovery be used:

```bash
raw=/path/to/ASCEND_PROFILER_OUTPUT/trace_view.json
recovered=/path/to/recovered_trace_view.json
cp "$raw" "$recovered"
printf ']' >> "$recovered"
jq empty "$recovered"
cmp -n "$(stat -c%s "$raw")" "$raw" "$recovered"
```

The last `cmp` must succeed: it proves that the recovered trace is exactly the
raw trace plus one closing byte. If it fails, do not import the repaired file;
retain the raw profiler directory and collect again.

## Eligibility divergence

Current Ascend first synchronizes only metadata, then synchronizes a local
boolean DBO candidate with `MIN`. For this input both local candidates are
true, so DBO remains true. The microbatch shapes are:

```text
DP0: 4096 -> [2048, 2048]
DP1: 2048 -> [1024, 1024]
```

Upstream `coordinate_batch_across_dp()` collects original token counts,
locally padded token counts, DBO candidates, and graph mode in one DP
allreduce. It then rejects DBO when the last global microbatch would contain
no real token on the shortest rank:

```text
min(real) <= max(padded) / 2
2048     <= 4096 / 2
```

Thus upstream disables DBO for this exact case. It does not create a DP1
second microbatch consisting entirely of padding.

## Plain A/B result

Profiler was disabled. Each configuration had a three-pair warmup, then three
independent runs of 15 cache-free request pairs. All 90 measured requests per
configuration were accepted at the intended 4096/2048 prompt lengths.

| Configuration | Run 1 mean pair latency | Run 2 | Run 3 | Mean of runs | Stddev |
|---|---:|---:|---:|---:|---:|
| Current DBO on | 426.61 ms | 437.37 ms | 436.74 ms | **433.57 ms** | 6.04 ms |
| DBO off | 333.59 ms | 316.68 ms | 319.45 ms | **323.24 ms** | 9.07 ms |

`DBO on` is **34.13% slower** than `DBO off` for this controlled skew case.
Since upstream disables DBO for this case, the DBO-off row is the relevant
behavioural comparator (though it is not a binary built from upstream).

Raw client logs:

```text
/data/tmp/dp-dbo-profile-20260731/plain_skew_dbo_on_20260731T102330Z/
/data/tmp/dp-dbo-profile-20260731/plain_skew_dbo_off_20260731T102648Z/
```

## Profile evidence and interpretation

The DBO-on controlled trace contains host scopes added around the two DP
collectives. Its largest metadata rendezvous waits are:

| Trace rank | `dp.sync_metadata.cpu_allreduce` max |
|---|---:|
| DP0 | 142.044 ms |
| DP1 | 176.968 ms |

The long range alternates rank: this is rendezvous waiting, not evidence that
one NPU is persistently slower or that Gloo transfers 142--177 ms of payload.
The peer has not reached the matching allreduce yet. DBO's unequal ubatch
shapes make the preceding forward paths structurally different, and the
resulting waiting is exposed at the next DP coordination point.

Do not interpret this trace as a profiler-based latency claim. The plain A/B
above is the performance evidence; the trace identifies the mechanism.

## MindStudio Insight walkthrough

The generated CANN exporter omitted only the final `]` of both Chrome trace
arrays. The original output is retained unchanged. The following recovered
copies append exactly that delimiter and pass `jq empty`; import these copies:

```text
/data/tmp/dp-dbo-profile-20260731/skew_4096_2048_eager_visual_20260731T101014Z/skew_4096_2048_dp0_recovered_trace_view.json
/data/tmp/dp-dbo-profile-20260731/skew_4096_2048_eager_visual_20260731T101014Z/skew_4096_2048_dp1_recovered_trace_view.json
```

1. Open MindStudio Insight, create/open a trace project, and drag one
   `*_recovered_trace_view.json` into the timeline. Open both files in
   separate tabs (or import both if the installed Insight version supports
   multi-trace comparison).
2. Use timeline search for `dp.sync_metadata.cpu_allreduce`. Select the long
   event in DP0, then find the matching-order event in DP1. The difference
   between their start timestamps is the rendezvous delay.
3. Search `gloo:all_reduce`. It is nested beneath the DP scope and confirms
   that the long host scope is a collective rendezvous.
4. Search `dp.sync_should_ubatch.cpu_allreduce`. It is short here (roughly
   0.3--0.5 ms), demonstrating that the expensive delay is not the boolean
   DBO vote itself.
5. Search `HcclAllGather`, `vllm::all_gather`, or `c10d::_allgather_base_`
   to inspect the EP/communication work surrounding the forward. Zoom into a
   long DP metadata scope and compare its neighbours across rank traces.

The original `*_ascend_pt/FRAMEWORK` and `PROF_*` folders remain available if
you want to rerun CANN analysis locally. `trace_view.json` is the Chrome trace
format that MindStudio Insight accepts.

## What this does and does not prove

This proves an actual, reproducible performance regression for current DBO in
this specified DP token-skew workload, and identifies a DP synchronization
mechanism consistent with the semantic divergence from upstream.

It does **not** prove that all DBO workloads regress: an earlier balanced
high-concurrency workload showed positive aggregate DBO throughput. The
result applies when all ranks independently cross the threshold but their
physical microbatch shapes differ enough that upstream's empty-last-ubatch
guard would reject DBO.
