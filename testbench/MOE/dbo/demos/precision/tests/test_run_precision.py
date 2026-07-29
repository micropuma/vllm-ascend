"""Unit tests for local DBO precision runner policy."""

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "run_precision.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_precision", MODULE_PATH)
assert SPEC is not None
run_precision = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = run_precision
SPEC.loader.exec_module(run_precision)


def test_selected_tasks_are_stable_and_complete() -> None:
    assert [task.name for task in run_precision.selected_tasks("all")] == [
        "gsm8k", "gpqa_diamond"
    ]


def test_server_environment_forces_local_and_flashcomm_policy() -> None:
    baseline = run_precision.build_environment("baseline", None)
    dbo = run_precision.build_environment("dbo", Path("/tmp/precision-cache"))

    assert baseline["NO_PROXY"] == "127.0.0.1,localhost"
    assert baseline["no_proxy"] == "127.0.0.1,localhost"
    assert baseline["VLLM_ASCEND_ENABLE_FLASHCOMM1"] == "0"
    assert baseline["VLLM_ASCEND_ENABLE_DBO"] == "0"
    assert dbo["VLLM_ASCEND_ENABLE_DBO"] == "1"
    assert dbo["XDG_CACHE_HOME"].endswith("precision-cache/xdg")
