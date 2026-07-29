#!/usr/bin/env python3
"""Compare two precision artifacts sample by sample and locate token divergence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def token_sequence(row: dict[str, Any]) -> list[str] | None:
    tokens = row.get("token_logprobs")
    if not isinstance(tokens, list):
        return None
    values: list[str] = []
    for item in tokens:
        if not isinstance(item, dict):
            return None
        token = item.get("token")
        if not isinstance(token, str):
            return None
        values.append(token)
    return values


def margin(item: dict[str, Any] | None) -> float | None:
    if not isinstance(item, dict):
        return None
    choices = item.get("top_logprobs")
    if not isinstance(choices, list) or len(choices) < 2:
        return None
    try:
        return float(choices[0]["logprob"]) - float(choices[1]["logprob"])
    except (KeyError, TypeError, ValueError):
        return None


def first_divergence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    left_tokens, right_tokens = token_sequence(left), token_sequence(right)
    if left_tokens is None or right_tokens is None:
        return None
    for index, (left_token, right_token) in enumerate(zip(left_tokens, right_tokens)):
        if left_token != right_token:
            return {"index": index, "left_token": left_token, "right_token": right_token,
                    "left_margin": margin(left["token_logprobs"][index]),
                    "right_margin": margin(right["token_logprobs"][index])}
    if len(left_tokens) != len(right_tokens):
        index = min(len(left_tokens), len(right_tokens))
        return {"index": index,
                "left_token": left_tokens[index] if index < len(left_tokens) else None,
                "right_token": right_tokens[index] if index < len(right_tokens) else None,
                "left_margin": margin(left["token_logprobs"][index]) if index < len(left_tokens) else None,
                "right_margin": margin(right["token_logprobs"][index]) if index < len(right_tokens) else None}
    return None


def ensure_comparable(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("config_sha256", "dataset_manifest", "model", "concurrency"):
        if left.get(key) != right.get(key):
            raise ValueError(f"incomparable artifacts: {key} differs")


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    ensure_comparable(left, right)
    left_rows = {row["sample_id"]: row for row in left["samples"]}
    right_rows = {row["sample_id"]: row for row in right["samples"]}
    if left_rows.keys() != right_rows.keys():
        raise ValueError("incomparable artifacts: sample IDs differ")
    transitions = {"true_true": 0, "true_false": 0, "false_true": 0, "false_false": 0}
    text_mismatches, token_differences = [], []
    unavailable_logprobs = 0
    for sample_id in sorted(left_rows):
        left_row, right_row = left_rows[sample_id], right_rows[sample_id]
        transition = f"{str(bool(left_row['correct'])).lower()}_{str(bool(right_row['correct'])).lower()}"
        transitions[transition] += 1
        if left_row.get("text") != right_row.get("text"):
            text_mismatches.append(sample_id)
        divergence = first_divergence(left_row, right_row)
        if divergence is not None:
            token_differences.append({"sample_id": sample_id, **divergence})
        elif left_row.get("text") != right_row.get("text") and (
            token_sequence(left_row) is None or token_sequence(right_row) is None
        ):
            unavailable_logprobs += 1
    return {
        "schema_version": 1,
        "left": left["summary"],
        "right": right["summary"],
        "accuracy_drop": left["summary"]["accuracy"] - right["summary"]["accuracy"],
        "accuracy_transitions": transitions,
        "text_mismatch_count": len(text_mismatches),
        "text_mismatch_sample_ids": text_mismatches,
        "token_divergence_count": len(token_differences),
        "token_divergences": token_differences,
        "text_mismatches_without_token_logprobs": unavailable_logprobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(load(args.left), load(args.right))
    report.update({"left_artifact": str(args.left), "right_artifact": str(args.right),
                   "created_at": datetime.now(timezone.utc).isoformat()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "accuracy_drop", "accuracy_transitions", "text_mismatch_count",
        "token_divergence_count", "text_mismatches_without_token_logprobs")}, ensure_ascii=False, indent=2))
    print(f"artifact={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
