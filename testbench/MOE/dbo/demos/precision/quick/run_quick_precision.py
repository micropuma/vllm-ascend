#!/usr/bin/env python3
"""Fast, deterministic DBO serving correctness gate for DeepSeek-V2.

This is deliberately a regression gate, not an accuracy benchmark. It sends a
single concurrent long-prefill wave, requires DBO evidence from both TP ranks,
and rejects output-token or sampled-token-logprob drift against non-DBO.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


DEFAULT_MODEL = "/data/models/DeepSeek-V2-Lite-Chat"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_PROMPTS = 16
DEFAULT_INPUT_LEN = 2048
DEFAULT_OUTPUT_LEN = 16
DEFAULT_TIMEOUT = 300.0
DEFAULT_LOGPROB_ATOL = 1e-3
DEFAULT_TP_SIZE = 2
DEFAULT_DP_SIZE = 1
READY_TIMEOUT = 1800.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashcomm1", choices=(0, 1), type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument("--dp-size", type=int, default=DEFAULT_DP_SIZE)
    parser.add_argument("--dp-local", type=int)
    parser.add_argument(
        "--server-script",
        type=Path,
        help="Use this launcher for both baseline and DBO; it must honor DBO_ENABLED.",
    )
    parser.add_argument("--model", default=os.getenv("MODEL", DEFAULT_MODEL))
    parser.add_argument("--host", default=os.getenv("HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", DEFAULT_PORT)))
    parser.add_argument("--num-prompts", type=int, default=int(os.getenv("NUM_PROMPTS", DEFAULT_PROMPTS)))
    parser.add_argument("--input-len", type=int, default=int(os.getenv("INPUT_LEN", DEFAULT_INPUT_LEN)))
    parser.add_argument("--output-len", type=int, default=int(os.getenv("OUTPUT_LEN", DEFAULT_OUTPUT_LEN)))
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", DEFAULT_PROMPTS)))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--logprob-atol", type=float, default=DEFAULT_LOGPROB_ATOL)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.tp_size < 1 or args.dp_size < 1:
        parser.error("--tp-size and --dp-size must be positive")
    if args.dp_local is None:
        args.dp_local = args.dp_size
    if args.dp_local < 1 or args.dp_local > args.dp_size:
        parser.error("--dp-local must be in [1, dp-size]")
    if args.dp_size > 1:
        if args.tp_size != 1:
            parser.error("the DP gate currently supports TP=1 only")
        if args.flashcomm1:
            parser.error("FlashComm1 is unsupported for the TP=1 DP gate")
        if args.server_script is None:
            parser.error("--server-script is required for DP so baseline honors DBO_ENABLED=0")
    if args.input_len < 1024:
        parser.error("--input-len must be >= 1024 so it is DBO-prefill eligible")
    if args.num_prompts < 2 or args.concurrency < 2:
        parser.error("--num-prompts and --concurrency must be >= 2 to exercise batching")
    if args.concurrency > args.num_prompts:
        parser.error("--concurrency cannot exceed --num-prompts")
    if args.output_len < 1 or args.logprob_atol < 0:
        parser.error("--output-len must be positive and --logprob-atol non-negative")
    return args


def ensure_port_unused(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        if connection.connect_ex((host, port)) == 0:
            raise RuntimeError(f"{host}:{port} is already in use; do not mix quick runs with another server")


def build_prompts(model: str, count: int, target_tokens: int, seed: int) -> tuple[list[str], list[int]]:
    """Make unique chat prompts whose templated token counts reach the target."""
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    def prompt_and_length(request_id: int, word_count: int) -> tuple[str, int]:
        words = [f"request_{request_id}_seed_{seed}"]
        words.extend(f"datum_{request_id}_{seed + offset}" for offset in range(word_count))
        prompt = " ".join(words)
        tokenized = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return prompt, len(tokenized["input_ids"])

    lower, upper = 0, 64
    while prompt_and_length(0, upper)[1] < target_tokens:
        lower, upper = upper, upper * 2
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if prompt_and_length(0, middle)[1] < target_tokens:
            lower = middle
        else:
            upper = middle

    prompts, lengths = [], []
    for request_id in range(count):
        prompt, token_length = prompt_and_length(request_id, upper)
        if token_length < target_tokens:
            prompt, token_length = prompt_and_length(request_id, upper + 1)
        prompts.append(prompt)
        lengths.append(token_length)
    return prompts, lengths


def server_environment(mode: str, args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "VLLM_ASCEND_ENABLE_DBO": "1" if mode == "dbo" else "0",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": str(args.flashcomm1),
        "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "0",
        "VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED": "0",
        "VLLM_LOGGING_LEVEL": "DEBUG" if mode == "dbo" else "INFO",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    })
    environment.update({
        "DBO_ENABLED": "1" if mode == "dbo" else "0",
        "TP": str(args.tp_size),
        "DP": str(args.dp_size),
        "DP_LOCAL": str(args.dp_local),
    })
    return environment


def start_server(mode: str, demos_dir: Path, out_dir: Path, args: argparse.Namespace) -> tuple[subprocess.Popen[bytes], Path]:
    script = args.server_script or Path(
        "deepseek-v2-dbo-server.sh" if mode == "dbo" else "deepseek-v2-server.sh"
    )
    if not script.is_absolute():
        script = demos_dir / script
    if not script.is_file():
        raise RuntimeError(f"server script does not exist: {script}")
    log_path = out_dir / f"{mode}_server.log"
    environment = server_environment(mode, args)
    if mode == "dbo":
        # Isolate the DBO instance's cache without deleting shared caches.
        dbo_xdg_cache = out_dir / "dbo-xdg-cache"
        dbo_xdg_cache.mkdir()
        environment["XDG_CACHE_HOME"] = str(dbo_xdg_cache)
    environment.update({"MODEL": args.model, "HOST": args.host, "PORT": str(args.port)})
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        ["bash", str(script)], cwd=demos_dir, env=environment,
        stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
    )
    log_file.close()
    return process, log_path


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)


def wait_for_server(endpoint: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT
    with httpx.Client(timeout=5.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with code {process.returncode}")
            try:
                response = client.get(f"{endpoint}/v1/models")
                if response.is_success and response.json().get("data"):
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2)
    raise RuntimeError(f"server was not ready within {READY_TIMEOUT}s")


async def request_wave(endpoint: str, args: argparse.Namespace, prompts: list[str]) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, trust_env=False) as client:
        async def request_one(index: int, prompt: str) -> dict[str, Any]:
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.output_len,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": args.seed + index,
                "logprobs": True,
                "top_logprobs": 5,
            }
            async with semaphore:
                try:
                    response = await client.post(f"{endpoint}/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["logprobs"]["content"]
                    if not isinstance(content, list) or not content:
                        raise ValueError("server did not return output token logprobs")
                    tokens = [{"token": item.get("token"), "bytes": item.get("bytes"),
                               "logprob": item.get("logprob")} for item in content]
                    if any(not isinstance(item["token"], str) or not isinstance(item["logprob"], (int, float))
                           for item in tokens):
                        raise ValueError("malformed output token logprobs")
                    return {"request_id": index, "text": choice["message"]["content"], "tokens": tokens}
                except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
                    return {"request_id": index, "error": f"{type(error).__name__}: {error}"}
        return await asyncio.gather(*(request_one(index, prompt) for index, prompt in enumerate(prompts)))


def require_dbo_trigger(log_path: Path, tp_size: int = DEFAULT_TP_SIZE,
                        dp_size: int = DEFAULT_DP_SIZE) -> dict[str, Any]:
    """Require an eligible DBO batch from every logical worker.

    vLLM's process prefix is ``Worker_TP<tp>`` for TP-only. The observed
    TP=1 DP executor uses ``Worker_DP<dp>_EP<ep>`` (no explicit TP field),
    while other executors may use ``Worker_DP<dp>_TP<tp>``. Keep the logical
    worker IDs in the artifact so a naming change cannot weaken this gate.
    """
    text = log_path.read_text(errors="replace")
    triggered: set[tuple[int, int]] = set()
    prefix = re.compile(
        r"Worker_(?:(?:DP(?P<dp>\d+)(?:_TP(?P<dp_tp>\d+))?)|TP(?P<tp>\d+))"
    )
    for line in text.splitlines():
        if "should_ubatch: True" not in line:
            continue
        match = prefix.search(line)
        if match is None:
            continue
        dp_rank = int(match.group("dp") or 0)
        tp_rank = int(match.group("dp_tp") or match.group("tp") or 0)
        triggered.add((dp_rank, tp_rank))
    expected = {(dp_rank, tp_rank) for dp_rank in range(dp_size) for tp_rank in range(tp_size)}
    report = {
        "triggered_workers": [f"dp{dp}_tp{tp}" for dp, tp in sorted(triggered)],
        "missing_workers": [f"dp{dp}_tp{tp}" for dp, tp in sorted(expected - triggered)],
    }
    if report["missing_workers"]:
        raise RuntimeError(
            f"DBO did not trigger on workers {report['missing_workers']}; "
            f"observed {report['triggered_workers']}"
        )
    return report


def compare_runs(baseline: list[dict[str, Any]], dbo: list[dict[str, Any]], atol: float) -> dict[str, Any]:
    failures: list[str] = []
    max_logprob_delta = 0.0
    for left, right in zip(baseline, dbo):
        request_id = left["request_id"]
        if "error" in left or "error" in right:
            failures.append(f"request {request_id}: baseline={left.get('error')} dbo={right.get('error')}")
            continue
        if left["text"] != right["text"]:
            failures.append(f"request {request_id}: generated text differs")
            continue
        left_tokens, right_tokens = left["tokens"], right["tokens"]
        if len(left_tokens) != len(right_tokens):
            failures.append(f"request {request_id}: output token count differs")
            continue
        for token_id, (left_token, right_token) in enumerate(zip(left_tokens, right_tokens)):
            if (left_token["token"], left_token["bytes"]) != (right_token["token"], right_token["bytes"]):
                failures.append(f"request {request_id}, token {token_id}: generated token differs")
                break
            delta = abs(float(left_token["logprob"]) - float(right_token["logprob"]))
            max_logprob_delta = max(max_logprob_delta, delta)
            if delta > atol:
                failures.append(f"request {request_id}, token {token_id}: logprob delta {delta:.6g} > {atol:.6g}")
                break
    return {"passed": not failures, "failure_count": len(failures), "failures": failures[:20],
            "max_sampled_token_logprob_delta": max_logprob_delta, "logprob_atol": atol}


def run_mode(mode: str, demos_dir: Path, out_dir: Path, args: argparse.Namespace, prompts: list[str]) -> tuple[list[dict[str, Any]], Path]:
    process, log_path = start_server(mode, demos_dir, out_dir, args)
    try:
        wait_for_server(f"http://{args.host}:{args.port}", process)
        results = asyncio.run(request_wave(f"http://{args.host}:{args.port}", args, prompts))
        if mode == "dbo":
            require_dbo_trigger(log_path, args.tp_size, args.dp_size)
        return results, log_path
    finally:
        stop_server(process)


def main() -> int:
    args = parse_args()
    ensure_port_unused(args.host, args.port)
    quick_dir = Path(__file__).resolve().parent
    demos_dir = quick_dir.parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or quick_dir.parent / "results" / f"quick-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    prompts, prompt_lengths = build_prompts(args.model, args.num_prompts, args.input_len, args.seed)
    (out_dir / "workload.json").write_text(json.dumps({"args": vars(args), "prompt_token_lengths": prompt_lengths}, default=str, indent=2) + "\n")
    baseline, _ = run_mode("baseline", demos_dir, out_dir, args, prompts)
    dbo, dbo_log = run_mode("dbo", demos_dir, out_dir, args, prompts)
    comparison = compare_runs(baseline, dbo, args.logprob_atol)
    comparison["dbo_trigger"] = require_dbo_trigger(dbo_log, args.tp_size, args.dp_size)
    (out_dir / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "dbo.json").write_text(json.dumps(dbo, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"artifacts={out_dir}")
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
