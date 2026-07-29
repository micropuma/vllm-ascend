import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "accuracy_runner.py"
SPEC = importlib.util.spec_from_file_location("accuracy_runner", MODULE_PATH)
assert SPEC is not None
accuracy_runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = accuracy_runner
SPEC.loader.exec_module(accuracy_runner)


def test_normalize_gsm8k_answer() -> None:
    assert accuracy_runner.normalize_number("work\n#### 1,024") == "1024"
    assert accuracy_runner.normalize_number("no marker") is None


def test_parse_gsm8k_model_answer_formats() -> None:
    assert accuracy_runner.parse_gsm8k_answer("Answer: \\boxed{1,024}.") == "1024"
    assert accuracy_runner.parse_gsm8k_answer("The answer is: 540") == "540"
    assert accuracy_runner.parse_gsm8k_answer("After calculation, 18.") == "18"


def test_parse_choice_uses_last_option() -> None:
    assert accuracy_runner.parse_choice("I considered A, final answer: C") == "C"
    assert accuracy_runner.parse_choice("unknown") is None


def test_gpqa_choice_order_is_stable_per_record() -> None:
    first = accuracy_runner.choice_order("record-1", 42)
    assert first == accuracy_runner.choice_order("record-1", 42)
    assert sorted(first) == [0, 1, 2, 3]


def test_normalize_chat_logprobs() -> None:
    body = {"choices": [{"logprobs": {"content": [{
        "token": "A", "logprob": -0.1,
        "top_logprobs": [{"token": "A", "logprob": -0.1}, {"token": "B", "logprob": -2.0}],
    }]}}]}
    rows, error = accuracy_runner.normalize_logprobs(body)
    assert error is None
    assert rows == [{"token": "A", "logprob": -0.1,
                     "top_logprobs": [{"token": "A", "logprob": -0.1},
                                      {"token": "B", "logprob": -2.0}]}]


def test_normalize_chat_logprobs_reports_missing_response_field() -> None:
    rows, error = accuracy_runner.normalize_logprobs({"choices": [{}]})
    assert rows is None
    assert error is not None
