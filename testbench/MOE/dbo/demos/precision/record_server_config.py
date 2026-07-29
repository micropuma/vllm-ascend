#!/usr/bin/env python3
"""Archive the server configuration actually printed by a baseline or DBO launch."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


PRINTED_CONFIG_PATTERN = re.compile(r"^\s{2}([^=\s]+)\s+=\s*(.*?)\s*$")
RUNTIME_FACTS = {
    "flashcomm1": re.compile(r"AscendConfig\.enable_flashcomm1 .* value (True|False)"),
    "dbo_enabled": re.compile(r"'enable_dbo': (True|False)"),
    "moe_allgather_noop": re.compile(r"MoE communication uses ALL_GATHER so this is a no-op"),
}


def parse_server_log(text: str) -> dict[str, object]:
    printed = {}
    for line in text.splitlines():
        match = PRINTED_CONFIG_PATTERN.match(line)
        if match:
            printed[match.group(1)] = match.group(2)
    runtime = {}
    for name, pattern in RUNTIME_FACTS.items():
        matches = pattern.findall(text)
        runtime[name] = matches[-1] if matches else None
    return {"printed_server_config": printed, "runtime_facts": runtime}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "dbo"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    text = args.log.read_text(errors="replace")
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "source_log": str(args.log),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **parse_server_log(text),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.log, args.run_dir / f"{args.mode}_server.log")
    output = args.run_dir / f"{args.mode}_server_config.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"artifact={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
