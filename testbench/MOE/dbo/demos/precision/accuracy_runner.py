#!/usr/bin/env python3
"""Run an offline-dataset accuracy evaluation against a local vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

GSM8K_GOLD_ANSWER_PATTERN = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")
GSM8K_BOXED_ANSWER_PATTERN = re.compile(
    r"\\boxed\s*\{\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*\}"
)
GSM8K_ANSWER_TEXT_PATTERN = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:|=)\s*\$?\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
GSM8K_TRAILING_NUMBER_PATTERN = re.compile(r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*[.!]?\s*$")
CHOICE_PATTERN = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
CHOICE_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Example:
    sample_id: str
    messages: list[dict[str, str]]
    expected: str
    metadata: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True,
                        help="Local OpenAI-compatible base URL, e.g. http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path,
                        help="Explicit artifact path. Prefer --run-dir with --mode for paired runs.")
    parser.add_argument("--run-dir", type=Path,
                        help="Run directory created by init_run.py.")
    parser.add_argument("--mode", choices=("baseline", "dbo"),
                        help="Required with --run-dir; selects the artifact filename.")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--logprobs", action="store_true",
                        help="Request and archive per-output-token logprobs for differential diagnosis.")
    parser.add_argument("--top-logprobs", type=int, default=5,
                        help="Number of alternatives requested with --logprobs (0-20).")
    args = parser.parse_args()
    if args.output and (args.run_dir or args.mode):
        parser.error("--output cannot be combined with --run-dir or --mode")
    if not args.output and not (args.run_dir and args.mode):
        parser.error("provide --output or both --run-dir and --mode")
    if not 0 <= args.top_logprobs <= 20:
        parser.error("--top-logprobs must be in [0, 20]")
    return args


def stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def normalize_logprobs(body: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Keep only stable OpenAI chat logprob fields needed for offline diffs."""
    try:
        content = body["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, "response does not contain choices[0].logprobs.content"
    if not isinstance(content, list):
        return None, "choices[0].logprobs.content is not a list"
    normalized = []
    for item in content:
        if not isinstance(item, dict):
            return None, "logprob item is not an object"
        alternatives = item.get("top_logprobs", [])
        if not isinstance(alternatives, list):
            alternatives = []
        normalized.append({
            "token": item.get("token"),
            "logprob": item.get("logprob"),
            "top_logprobs": [
                {"token": candidate.get("token"), "logprob": candidate.get("logprob")}
                for candidate in alternatives if isinstance(candidate, dict)
            ],
        })
    return normalized, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(dataset_path: Path) -> dict[str, Any]:
    files = sorted(path for path in dataset_path.rglob("*") if path.is_file())
    return {
        "path": str(dataset_path),
        "files": [{"path": str(path.relative_to(dataset_path)), "sha256": sha256_file(path)}
                  for path in files],
    }


def normalize_number(value: str) -> str | None:
    match = GSM8K_GOLD_ANSWER_PATTERN.search(value)
    return match.group(1).replace(",", "") if match else None


def parse_gsm8k_answer(value: str) -> str | None:
    """Extract the final scalar answer without treating arbitrary reasoning as an answer."""
    for pattern in (GSM8K_GOLD_ANSWER_PATTERN, GSM8K_BOXED_ANSWER_PATTERN,
                    GSM8K_ANSWER_TEXT_PATTERN, GSM8K_TRAILING_NUMBER_PATTERN):
        matches = pattern.findall(value)
        if matches:
            return matches[-1].replace(",", "")
    return None


def parse_choice(value: str) -> str | None:
    matches = CHOICE_PATTERN.findall(value.upper())
    return matches[-1] if matches else None


def choice_order(record_id: str, seed: int) -> list[int]:
    order = list(range(4))
    state = stable_seed(record_id, seed)
    for index in range(len(order) - 1, 0, -1):
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        swap_index = state % (index + 1)
        order[index], order[swap_index] = order[swap_index], order[index]
    return order


def build_gsm8k_examples(dataset: Any, config: dict[str, Any]) -> list[Example]:
    fewshot = dataset[config["fewshot_split"]].select(range(config["num_fewshot"]))
    examples = []
    for index, row in enumerate(dataset[config["split"]]):
        messages = [{"role": "system", "content": config["system_prompt"]}]
        for shot in fewshot:
            messages.extend((
                {"role": "user", "content": shot["question"]},
                {"role": "assistant", "content": shot["answer"]},
            ))
        messages.append({"role": "user", "content": row["question"]})
        expected = normalize_number(row["answer"])
        if expected is None:
            raise ValueError(f"GSM8K sample {index} has no final answer marker")
        examples.append(Example(str(index), messages, expected, {}))
    return examples


def build_gpqa_examples(dataset: Any, config: dict[str, Any]) -> list[Example]:
    examples = []
    seed = config["generation"]["seed"]
    for index, row in enumerate(dataset[config["split"]]):
        record_id = str(row.get("Record ID") or index)
        choices = [row["Correct Answer"], row["Incorrect Answer 1"],
                   row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
        order = choice_order(record_id, seed)
        correct_label = CHOICE_LABELS[order.index(0)]
        formatted_choices = "\n".join(
            f"{CHOICE_LABELS[position]}. {choices[choice_index]}"
            for position, choice_index in enumerate(order)
        )
        user_content = f"Question:\n{row['Question']}\n\nOptions:\n{formatted_choices}\n\nAnswer:"
        examples.append(Example(
            record_id,
            [{"role": "system", "content": config["system_prompt"]},
             {"role": "user", "content": user_content}],
            correct_label,
            {"subdomain": str(row.get("Subdomain", "")), "choice_order": "".join(map(str, order))},
        ))
    return examples


def load_examples(config: dict[str, Any]) -> list[Example]:
    try:
        from datasets import load_from_disk
    except ModuleNotFoundError as error:
        raise RuntimeError("Install precision/requirements.txt in an evaluation environment.") from error
    dataset = load_from_disk(config["dataset_path"])
    adapter = config["adapter"]
    if adapter == "gsm8k":
        return build_gsm8k_examples(dataset, config)
    if adapter == "gpqa":
        return build_gpqa_examples(dataset, config)
    raise ValueError(f"Unsupported adapter: {adapter}")


async def request_one(client: httpx.AsyncClient, endpoint: str, model: str,
                      generation: dict[str, Any], example: Example,
                      request_seed: int, include_logprobs: bool,
                      top_logprobs: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": example.messages,
        "max_tokens": generation["max_tokens"],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "seed": request_seed,
    }
    if include_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    started = time.perf_counter()
    try:
        response = await client.post(f"{endpoint.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        token_logprobs, logprobs_error = normalize_logprobs(body) if include_logprobs else (None, None)
        return {"text": text, "latency_s": time.perf_counter() - started,
                "usage": body.get("usage", {}), "token_logprobs": token_logprobs,
                "logprobs_error": logprobs_error, "error": None}
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        return {"text": None, "latency_s": time.perf_counter() - started,
                "usage": {}, "token_logprobs": None, "logprobs_error": None,
                "error": f"{type(error).__name__}: {error}"}


async def evaluate(examples: list[Example], endpoint: str, model: str,
                   generation: dict[str, Any], concurrency: int,
                   timeout: float, adapter: str, include_logprobs: bool,
                   top_logprobs: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        async def guarded(index: int, example: Example) -> dict[str, Any]:
            async with semaphore:
                result = await request_one(client, endpoint, model, generation, example,
                                           generation["seed"] + index, include_logprobs,
                                           top_logprobs)
            parsed = parse_gsm8k_answer(result["text"] or "") if adapter == "gsm8k" else parse_choice(result["text"] or "")
            return {"sample_id": example.sample_id, "expected": example.expected,
                    "parsed": parsed, "correct": parsed == example.expected,
                    "metadata": example.metadata, **result}
        return await asyncio.gather(*(guarded(index, example) for index, example in enumerate(examples)))


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    examples = load_examples(config)
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("No samples selected")
    results = asyncio.run(evaluate(examples, args.endpoint, args.model, config["generation"],
                                   args.concurrency, args.request_timeout, config["adapter"],
                                   args.logprobs, args.top_logprobs))
    successful = [result for result in results if result["error"] is None]
    correct = sum(result["correct"] for result in successful)
    output_path = args.output or args.run_dir / f"{args.mode}_{config['name']}.json"
    report = {
        "schema_version": 1,
        "config": config,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "dataset_manifest": dataset_manifest(Path(config["dataset_path"])),
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "diagnostics": {"logprobs_requested": args.logprobs,
                        "top_logprobs_requested": args.top_logprobs if args.logprobs else 0},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"total": len(results), "successful": len(successful),
                    "failed": len(results) - len(successful), "correct": correct,
                    "accuracy": correct / len(successful) if successful else 0.0},
        "samples": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"artifact={output_path}")
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
