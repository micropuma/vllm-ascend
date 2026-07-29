#!/usr/bin/env python3
"""Create one self-contained baseline/DBO accuracy result directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def git_dirty(repo: Path) -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        return None


def editable_imports() -> dict[str, str]:
    command = [sys.executable, "-c", "import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)"]
    try:
        vllm_path, ascend_path = subprocess.check_output(command, text=True).splitlines()
        return {"vllm": vllm_path, "vllm_ascend": ascend_path}
    except (subprocess.CalledProcessError, ValueError):
        return {"vllm": "unknown", "vllm_ascend": "unknown"}


def create_run(results_root: Path, run_id: str | None = None,
               allow_existing_directory: bool = False) -> Path:
    """Create a result directory and immutable source/environment manifest.

    Args:
        results_root: Parent directory for result directories.
        run_id: Optional stable directory name; UTC time is used when omitted.
        allow_existing_directory: Permit an existing directory only when it has
            no manifest. This supports recovery of an interrupted manual run.

    Returns:
        Newly created result directory.

    Raises:
        FileExistsError: A directory or manifest already exists for ``run_id``.
    """
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / resolved_run_id
    if run_dir.exists() and not allow_existing_directory:
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=allow_existing_directory)
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"manifest already exists: {run_dir / 'manifest.json'}")
    repo_root = Path(__file__).resolve().parents[5]
    vllm_root = repo_root.parent / "vllm"
    env_names = (
        "ASCEND_RT_VISIBLE_DEVICES", "HCCL_OP_EXPANSION_MODE",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE",
        "VLLM_ASCEND_ENABLE_FLASHCOMM2_OSHARED", "VLLM_ASCEND_ENABLE_DBO",
        "VLLM_LOGGING_LEVEL", "NO_PROXY", "no_proxy",
    )
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "run_id": resolved_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vllm_commit": git_commit(vllm_root),
        "vllm_ascend_commit": git_commit(repo_root),
        "vllm_dirty": git_dirty(vllm_root),
        "vllm_ascend_dirty": git_dirty(repo_root),
        "python_executable": sys.executable,
        "editable_imports": editable_imports(),
        "relevant_environment": {name: os.environ.get(name) for name in env_names},
    }
    (run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", help="Optional stable ID; defaults to a UTC timestamp.")
    parser.add_argument("--allow-existing", action="store_true",
                        help="Write a missing manifest into an existing run directory; never overwrite one.")
    args = parser.parse_args()
    if args.allow_existing:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        # This compatibility path is only for recovering an interrupted manual run.
        create_run(args.results_root, run_id, allow_existing_directory=True)
    else:
        run_dir = create_run(args.results_root, args.run_id)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
