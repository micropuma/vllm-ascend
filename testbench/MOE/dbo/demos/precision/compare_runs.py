#!/usr/bin/env python3
"""Compare baseline and DBO artifacts with paired accuracy and token diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from diff_artifacts import compare


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def exact_mcnemar_two_sided(baseline_only_correct: int, dbo_only_correct: int) -> float:
    """Exact two-sided McNemar p-value without adding a statistics dependency."""
    discordant = baseline_only_correct + dbo_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only_correct, dbo_only_correct)
    numerator = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * numerator / (2 ** discordant))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--dbo", type=Path)
    parser.add_argument("--run-dir", type=Path,
                        help="Use baseline_<task>.json and dbo_<task>.json from one run directory.")
    parser.add_argument("--task", help="Required with --run-dir, e.g. gsm8k.")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--allow-text-mismatches", type=int, default=0)
    parser.add_argument("--max-token-divergences", type=int,
                        help="Optional hard cap for samples with a first token divergence.")
    args = parser.parse_args()
    if args.run_dir:
        if args.baseline or args.dbo or not args.task:
            parser.error("--run-dir requires --task and cannot be combined with explicit artifacts")
        baseline_path = args.run_dir / f"baseline_{args.task}.json"
        dbo_path = args.run_dir / f"dbo_{args.task}.json"
    elif args.baseline and args.dbo:
        baseline_path, dbo_path = args.baseline, args.dbo
    else:
        parser.error("provide --baseline and --dbo, or --run-dir and --task")
    baseline, dbo = load(baseline_path), load(dbo_path)
    try:
        report = compare(baseline, dbo)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    transitions = report["accuracy_transitions"]
    report.update({
        "baseline": report.pop("left"),
        "dbo": report.pop("right"),
        "text_mismatches": report["text_mismatch_count"],
        "mismatch_sample_ids": report["text_mismatch_sample_ids"][:20],
        "mcnemar_two_sided_pvalue": exact_mcnemar_two_sided(
            transitions["true_false"], transitions["false_true"]),
        "baseline_artifact": str(baseline_path), "dbo_artifact": str(dbo_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps(report, indent=2))
    passed = (baseline["summary"]["failed"] == 0 and dbo["summary"]["failed"] == 0
              and report["accuracy_drop"] <= args.max_accuracy_drop
              and report["text_mismatch_count"] <= args.allow_text_mismatches
              and (args.max_token_divergences is None
                   or report["token_divergence_count"] <= args.max_token_divergences))
    if args.run_dir:
        output_path = args.run_dir / f"compare_{args.task}.json"
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"artifact={output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
