import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "record_server_config.py"
SPEC = importlib.util.spec_from_file_location("record_server_config", MODULE_PATH)
assert SPEC is not None
record_server_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = record_server_config
SPEC.loader.exec_module(record_server_config)


def test_parse_server_log_uses_printed_and_runtime_values() -> None:
    report = record_server_config.parse_server_log(
        "  TP                                  = 2\n"
        "  VLLM_ASCEND_ENABLE_FLASHCOMM1        = 0\n"
        "AscendConfig.enable_flashcomm1 falls back to environment variable X with value False.\n"
        "MoE communication uses ALL_GATHER so this is a no-op.\n"
    )
    assert report["printed_server_config"] == {
        "TP": "2", "VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"
    }
    assert report["runtime_facts"] == {
        "flashcomm1": "False", "dbo_enabled": None,
        "moe_allgather_noop": "MoE communication uses ALL_GATHER so this is a no-op"
    }
