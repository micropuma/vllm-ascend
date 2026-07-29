#!/usr/bin/env python3
"""Recompute accuracy fields from stored raw outputs after parser updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from accuracy_runner import parse_choice, parse_gsm8k_answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.artifact.read_text())
    adapter = report["config"]["adapter"]
    parser_fn = parse_gsm8k_answer if adapter == "gsm8k" else parse_choice
    successful = 0
    correct = 0
    for row in report["samples"]:
        row["parsed"] = parser_fn(row["text"] or "")
        row["correct"] = row["error"] is None and row["parsed"] == row["expected"]
        if row["error"] is None:
            successful += 1
            correct += row["correct"]
    report["summary"] = {
        "total": len(report["samples"]),
        "successful": successful,
        "failed": len(report["samples"]) - successful,
        "correct": correct,
        "accuracy": correct / successful if successful else 0.0,
    }
    report["parser_version"] = 2
    args.artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
