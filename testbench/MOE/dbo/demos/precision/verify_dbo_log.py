#!/usr/bin/env python3
"""Require DBO trigger evidence from every expected TP worker in a server log."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--run-dir", type=Path,
                        help="Copy the DBO log and trigger report into this run directory.")
    args = parser.parse_args()
    triggered = set(re.findall(r"Worker_TP(\d+).*should_ubatch: True", args.log.read_text(errors="replace")))
    expected = {str(index) for index in range(args.tp_size)}
    missing = expected - triggered
    report = {"log": str(args.log), "tp_size": args.tp_size,
              "triggered_tp_ranks": sorted(triggered), "missing_tp_ranks": sorted(missing)}
    print(json.dumps(report, indent=2))
    if args.run_dir:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.log, args.run_dir / "dbo_server.log")
        (args.run_dir / "dbo_trigger.json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        print(f"missing_tp_ranks={sorted(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
