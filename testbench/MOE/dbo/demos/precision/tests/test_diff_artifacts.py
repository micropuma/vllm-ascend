import importlib.util
from pathlib import Path
import sys
from typing import Any


MODULE_PATH = Path(__file__).parents[1] / "diff_artifacts.py"
SPEC = importlib.util.spec_from_file_location("diff_artifacts", MODULE_PATH)
assert SPEC is not None
diff_artifacts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = diff_artifacts
SPEC.loader.exec_module(diff_artifacts)


def row(sample_id: str, correct: bool, text: str,
        tokens: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"sample_id": sample_id, "correct": correct, "text": text, "token_logprobs": tokens}


def artifact(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {"config_sha256": "config", "dataset_manifest": {"files": []}, "model": "model",
            "concurrency": 1, "summary": {"accuracy": 0.5, "failed": 0}, "samples": samples}


def test_compare_finds_first_token_divergence_and_transitions() -> None:
    left_tokens = [{"token": "A", "top_logprobs": [{"logprob": -0.1}, {"logprob": -1.1}]},
                   {"token": "B", "top_logprobs": [{"logprob": -0.2}, {"logprob": -1.2}]}]
    right_tokens = [{"token": "A", "top_logprobs": [{"logprob": -0.1}, {"logprob": -1.1}]},
                    {"token": "C", "top_logprobs": [{"logprob": -0.3}, {"logprob": -0.4}]}]
    report = diff_artifacts.compare(artifact([row("one", True, "AB", left_tokens)]),
                                    artifact([row("one", False, "AC", right_tokens)]))
    assert report["accuracy_transitions"] == {"true_true": 0, "true_false": 1,
                                               "false_true": 0, "false_false": 0}
    assert report["token_divergences"] == [{"sample_id": "one", "index": 1,
                                             "left_token": "B", "right_token": "C",
                                             "left_margin": 1.0, "right_margin": 0.10000000000000003}]
