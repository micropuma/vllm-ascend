#!/usr/bin/env python3
"""Force an unequal prefill batch onto each DP rank for DBO diagnosis.

The OpenAI server accepts ``X-data-parallel-rank`` as a router-provided
header.  Sending the two requests concurrently therefore makes the next
prefill forward contain a controlled ``rank0_tokens`` vs ``rank1_tokens``
case.  Token IDs avoid tokenizer/chat-template variance.

This is a diagnostic workload, not a throughput benchmark.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = response.read()
        return json.loads(body) if body else {}


def completion(
    base_url: str,
    rank: int,
    token_count: int,
    output_tokens: int,
    repeat_index: int,
) -> tuple[int, int]:
    # Change the first token for every pair. Prefix-cache block hashes are
    # chained, so this prevents a later repeat from reusing the 4K prefill of
    # an earlier repeat while preserving the exact requested prompt length.
    prompt_token_ids = [100] * token_count
    prompt_token_ids[0] = 100 + repeat_index
    payload = {
        "model": "/data/models/DeepSeek-V2-Lite-Chat",
        "prompt": prompt_token_ids,
        "add_special_tokens": False,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    response = post_json(
        f"{base_url}/v1/completions",
        payload,
        {
            "X-data-parallel-rank": str(rank),
            "X-Request-Id": f"dp-skew-{repeat_index}-r{rank}",
        },
    )
    return rank, response["usage"]["prompt_tokens"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--rank0-tokens", type=int, default=4096)
    parser.add_argument("--rank1-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--token-offset",
        type=int,
        default=0,
        help="Unique first-token offset, for cache-free warmup/measurement phases.",
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.profile:
        post_json(f"{args.base_url}/start_profile", {}, {})
    try:
        results = []
        pair_latencies_s = []
        for repeat_index in range(args.repeats):
            start_time = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = (
                    executor.submit(
                        completion,
                        args.base_url,
                        0,
                        args.rank0_tokens,
                        args.output_tokens,
                        repeat_index + args.token_offset,
                    ),
                    executor.submit(
                        completion,
                        args.base_url,
                        1,
                        args.rank1_tokens,
                        args.output_tokens,
                        repeat_index + args.token_offset,
                    ),
                )
                results.extend(future.result() for future in futures)
            pair_latencies_s.append(time.perf_counter() - start_time)
    finally:
        if args.profile:
            post_json(f"{args.base_url}/stop_profile", {}, {})

    for rank, prompt_tokens in sorted(results):
        print(f"DP{rank}: server accepted {prompt_tokens} prompt tokens")
    if pair_latencies_s:
        sorted_latencies_s = sorted(pair_latencies_s)
        p50_index = (len(sorted_latencies_s) - 1) // 2
        p95_index = min(
            len(sorted_latencies_s) - 1,
            int(len(sorted_latencies_s) * 0.95),
        )
        print(
            "pair latency (both ranks completed): "
            f"mean={statistics.mean(pair_latencies_s) * 1000:.2f} ms, "
            f"p50={sorted_latencies_s[p50_index] * 1000:.2f} ms, "
            f"p95={sorted_latencies_s[p95_index] * 1000:.2f} ms"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"dp-skew request failed: {error}", file=sys.stderr)
        raise
