"""Unit tests for the quick DBO regression gate's hard failure conditions."""

from __future__ import annotations

import importlib.util
from pathlib import Path


QUICK_RUNNER = Path(__file__).parents[1] / "quick" / "run_quick_precision.py"
SPEC = importlib.util.spec_from_file_location("quick_precision", QUICK_RUNNER)
assert SPEC is not None and SPEC.loader is not None
quick_precision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quick_precision)


def response(token: str = "A", logprob: float = -0.1) -> list[dict]:
    return [{"request_id": 0, "text": token,
             "tokens": [{"token": token, "bytes": [ord(token)], "logprob": logprob}]}]


def test_compare_runs_accepts_identical_token_sequences() -> None:
    assert quick_precision.compare_runs(response(), response(), 1e-3)["passed"]


def test_compare_runs_rejects_token_or_logprob_drift() -> None:
    assert not quick_precision.compare_runs(response(), response("B"), 1e-3)["passed"]
    assert not quick_precision.compare_runs(response(), response(logprob=-0.2), 1e-3)["passed"]


def test_require_dbo_trigger_requires_every_rank(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("Worker_TP0 should_ubatch: True\nWorker_TP1 should_ubatch: True\n")
    assert quick_precision.require_dbo_trigger(log) == {
        "triggered_workers": ["dp0_tp0", "dp0_tp1"], "missing_workers": []}
    log.write_text("Worker_TP0 should_ubatch: True\n")
    try:
        quick_precision.require_dbo_trigger(log)
    except RuntimeError as error:
        assert "dp0_tp1" in str(error)
    else:
        raise AssertionError("missing TP rank must fail the quick gate")


def test_require_dbo_trigger_requires_every_dp_worker(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "Worker_DP0_EP0 should_ubatch: True\n"
        "Worker_DP1_EP1 should_ubatch: True\n"
    )
    assert quick_precision.require_dbo_trigger(log, tp_size=1, dp_size=2) == {
        "triggered_workers": ["dp0_tp0", "dp1_tp0"], "missing_workers": []}
