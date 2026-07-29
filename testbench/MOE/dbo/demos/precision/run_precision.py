#!/usr/bin/env python3
"""Run one reproducible offline accuracy comparison for the DBO optimization.

The runner owns the complete serving sequence: baseline evaluation, baseline
shutdown, cold DBO startup, DBO trigger validation, and paired comparison.
It never downloads a dataset or contacts a remote inference endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Final, Sequence

import httpx

from init_run import create_run


DEFAULT_MODEL: Final[str] = "/data/models/DeepSeek-V2-Lite-Chat"
DEFAULT_ENDPOINT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8001
DEFAULT_READY_TIMEOUT_S: Final[float] = 1_800.0
DEFAULT_REQUEST_TIMEOUT_S: Final[float] = 600.0
DEFAULT_ACCURACY_DROP: Final[float] = 0.005
DEFAULT_CONCURRENCY: Final[int] = 64


@dataclass(frozen=True)
class Task:
    """Describes one local dataset configuration.

    Attributes:
        name: Stable artifact suffix and command-line task value.
        config_name: YAML configuration file name under ``configs/``.
    """

    name: str
    config_name: str


TASKS: Final[dict[str, Task]] = {
    "gsm8k": Task("gsm8k", "gsm8k.yaml"),
    "gpqa_diamond": Task("gpqa_diamond", "gpqa_diamond.yaml"),
}


@dataclass
class ManagedServer:
    """Owns a server subprocess and its live log stream."""

    mode: str
    process: subprocess.Popen[bytes]
    log_file: BinaryIO
    log_path: Path

    def stop(self) -> None:
        """Terminate this process group, escalating only after a timeout."""
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.log_file.close()


def parse_args() -> argparse.Namespace:
    """Parse the command-line interface for a full DBO precision run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", *TASKS),
        default="all",
        help="Dataset to run; default evaluates both local datasets.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, help="Limit each dataset for a smoke run.")
    parser.add_argument("--run-id", help="Stable result directory name; defaults to UTC time.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_S)
    parser.add_argument("--max-accuracy-drop", type=float, default=DEFAULT_ACCURACY_DROP)
    parser.add_argument("--no-logprobs", action="store_true",
                        help="Skip token diagnostics to reduce artifact size.")
    parser.add_argument("--top-logprobs", type=int, default=5)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.ready_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    if not 0 <= args.top_logprobs <= 20:
        parser.error("--top-logprobs must be in [0, 20]")
    return args


def selected_tasks(task_name: str) -> list[Task]:
    """Return the dataset configurations requested by ``--task``.

    Args:
        task_name: ``all`` or a key in :data:`TASKS`.

    Returns:
        Dataset tasks in a stable order.
    """
    if task_name == "all":
        return list(TASKS.values())
    return [TASKS[task_name]]


def build_environment(mode: str, cache_root: Path | None) -> dict[str, str]:
    """Build the controlled environment for one server mode.

    Args:
        mode: Either ``baseline`` or ``dbo``.
        cache_root: New cache root used only for the cold DBO server.

    Returns:
        Environment variables for the child server process.
    """
    environment = os.environ.copy()
    environment.update({
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0",
        "VLLM_ASCEND_ENABLE_DBO": "1" if mode == "dbo" else "0",
        "VLLM_LOGGING_LEVEL": "DEBUG" if mode == "dbo" else "INFO",
    })
    if cache_root is not None:
        environment.update({
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "VLLM_COMPILE_CACHE_PATH": str(cache_root / "vllm-compile"),
        })
    return environment


def start_server(mode: str, demo_root: Path, environment_script: Path,
                 port: int, environment: dict[str, str], run_dir: Path) -> ManagedServer:
    """Start a server script and redirect all output to a run-local log.

    Args:
        mode: ``baseline`` or ``dbo``.
        demo_root: Directory containing both server scripts.
        environment_script: Environment setup shell script for vLLM Ascend.
        port: Local API port.
        environment: Prepared environment for the child process.
        run_dir: Directory that receives the live server log.

    Returns:
        A process owner that must be stopped by the caller.
    """
    script_name = "deepseek-v2-server.sh" if mode == "baseline" else "deepseek-v2-dbo-server.sh"
    log_path = run_dir / f"{mode}_server_live.log"
    log_file = log_path.open("wb")
    child_environment = environment.copy()
    child_environment["PORT"] = str(port)
    controlled_keys = (
        "PORT", "VLLM_ASCEND_ENABLE_FLASHCOMM1", "VLLM_ASCEND_ENABLE_DBO",
        "VLLM_LOGGING_LEVEL", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR",
        "VLLM_COMPILE_CACHE_PATH",
    )
    assignments = [
        f"{key}={child_environment[key]}" for key in controlled_keys if key in child_environment
    ]
    command = [
        "bash", "-c",
        "source \"$1\"; shift; export NO_PROXY=127.0.0.1,localhost; "
        "export no_proxy=127.0.0.1,localhost; exec env \"$@\"",
        "precision-server", str(environment_script), *assignments,
        "bash", str(demo_root / script_name),
    ]
    process = subprocess.Popen(
        command,
        cwd=demo_root,
        env=child_environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ManagedServer(mode, process, log_file, log_path)


def wait_for_server(endpoint: str, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    """Wait until the local OpenAI endpoint exposes at least one model.

    Args:
        endpoint: Base endpoint without ``/v1/models``.
        process: Server process checked for an early failure.
        timeout_s: Maximum readiness wait in seconds.

    Raises:
        RuntimeError: The server exits or does not become ready before timeout.
    """
    deadline = time.monotonic() + timeout_s
    models_url = f"{endpoint}/v1/models"
    with httpx.Client(timeout=5.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with code {process.returncode}")
            try:
                response = client.get(models_url)
                response.raise_for_status()
                data = response.json().get("data")
                if isinstance(data, list) and data:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2)
    raise RuntimeError(f"server did not become ready within {timeout_s} seconds: {models_url}")


def ensure_port_unused(port: int) -> None:
    """Reject a port already occupied by another local process.

    Args:
        port: Loopback TCP port that the runner will use.

    Raises:
        RuntimeError: Another process is listening on ``port``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        if connection.connect_ex((DEFAULT_ENDPOINT_HOST, port)) == 0:
            raise RuntimeError(
                f"port {port} is already in use; stop the existing server before this run")


def run_command(command: Sequence[str], cwd: Path) -> None:
    """Run a checked command and display it for reproducibility.

    Args:
        command: Argument-vector command to execute.
        cwd: Working directory for the command.
    """
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def evaluate_tasks(tasks: Sequence[Task], mode: str, args: argparse.Namespace,
                   demo_root: Path, run_dir: Path, endpoint: str) -> None:
    """Evaluate every selected dataset against one running server.

    Args:
        tasks: Dataset configurations to submit.
        mode: Artifact mode, either ``baseline`` or ``dbo``.
        args: Parsed CLI options.
        demo_root: Directory containing evaluator scripts and configs.
        run_dir: Artifact destination.
        endpoint: Ready local OpenAI endpoint.
    """
    for task in tasks:
        command = [
            sys.executable, "accuracy_runner.py", "--config", str(Path("configs") / task.config_name),
            "--endpoint", endpoint, "--model", args.model, "--concurrency", str(args.concurrency),
            "--request-timeout", str(args.request_timeout), "--run-dir", str(run_dir), "--mode", mode,
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        if not args.no_logprobs:
            command.extend(("--logprobs", "--top-logprobs", str(args.top_logprobs)))
        run_command(command, demo_root / "precision")


def archive_server_facts(mode: str, log_path: Path, run_dir: Path, demo_root: Path) -> None:
    """Archive printed configuration and runtime facts from a server log."""
    run_command([
        sys.executable, "record_server_config.py", "--mode", mode,
        "--log", str(log_path), "--run-dir", str(run_dir),
    ], demo_root / "precision")


def write_orchestrator_metadata(run_dir: Path, args: argparse.Namespace,
                                cache_root: Path) -> None:
    """Record runner policy that is not represented by dataset artifacts."""
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": vars(args),
        "policy": {
            "sequence": "baseline -> stop -> cold dbo -> verify both TP ranks -> compare",
            "flashcomm1": "forced disabled in both modes",
            "dbo_cache_root": str(cache_root),
            "network": "local datasets and local 127.0.0.1 inference only",
        },
    }
    serializable = report.copy()
    serializable["command"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (run_dir / "orchestrator.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n")


def compare_tasks(tasks: Sequence[Task], args: argparse.Namespace,
                  demo_root: Path, run_dir: Path) -> bool:
    """Write paired reports and return whether every accuracy gate passes."""
    passed = True
    for task in tasks:
        command = [
            sys.executable, "compare_runs.py", "--run-dir", str(run_dir), "--task", task.name,
            "--max-accuracy-drop", str(args.max_accuracy_drop),
            "--allow-text-mismatches", "999999",
        ]
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=demo_root / "precision", check=False)
        passed = passed and result.returncode == 0
    return passed


def main() -> int:
    """Execute the baseline-to-DBO precision validation transaction."""
    args = parse_args()
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    demo_root = Path(__file__).resolve().parent.parent
    repo_root = demo_root.parents[3]
    environment_script = repo_root.parent / "env.sh"
    if not environment_script.is_file():
        raise FileNotFoundError(f"Missing vLLM environment script: {environment_script}")
    tasks = selected_tasks(args.task)
    ensure_port_unused(args.port)
    run_dir = create_run(args.results_root.resolve(), args.run_id)
    endpoint = f"http://{DEFAULT_ENDPOINT_HOST}:{args.port}"
    cache_root = Path(tempfile.mkdtemp(prefix=f"dbo-precision-{run_dir.name}-", dir="/data/tmp"))
    for cache_name in ("xdg", "torchinductor", "vllm-compile"):
        (cache_root / cache_name).mkdir()
    write_orchestrator_metadata(run_dir, args, cache_root)

    baseline = start_server(
        "baseline", demo_root, environment_script, args.port,
        build_environment("baseline", None), run_dir)
    try:
        wait_for_server(endpoint, baseline.process, args.ready_timeout)
        evaluate_tasks(tasks, "baseline", args, demo_root, run_dir, endpoint)
    finally:
        baseline.stop()
    archive_server_facts("baseline", baseline.log_path, run_dir, demo_root)

    dbo = start_server(
        "dbo", demo_root, environment_script, args.port,
        build_environment("dbo", cache_root), run_dir)
    try:
        wait_for_server(endpoint, dbo.process, args.ready_timeout)
        evaluate_tasks(tasks, "dbo", args, demo_root, run_dir, endpoint)
        run_command([
            sys.executable, "verify_dbo_log.py", "--log", str(dbo.log_path),
            "--tp-size", "2", "--run-dir", str(run_dir),
        ], demo_root / "precision")
    finally:
        dbo.stop()
    archive_server_facts("dbo", dbo.log_path, run_dir, demo_root)

    passed = compare_tasks(tasks, args, demo_root, run_dir)
    print(f"run_dir={run_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
