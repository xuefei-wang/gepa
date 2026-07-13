import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationBatch:
    outputs: list
    scores: list[float]
    trajectories: list | None = None
    objective_scores: list[dict[str, float]] | None = None


class _GEPAAdapter:
    def __class_getitem__(cls, item):
        return cls


class _LM:
    pass


class _MaxCandidateProposalsStopper:
    def __init__(self, *args, **kwargs):
        pass

sys.modules.setdefault(
    "litellm",
    types.SimpleNamespace(completion=lambda *args, **kwargs: None, suppress_debug_info=False),
)
sys.modules.setdefault(
    "gepa",
    types.SimpleNamespace(
        EvaluationBatch=EvaluationBatch,
        GEPAAdapter=_GEPAAdapter,
        optimize=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault("gepa.lm", types.SimpleNamespace(LM=_LM))
sys.modules.setdefault(
    "gepa.utils",
    types.SimpleNamespace(MaxCandidateProposalsStopper=_MaxCandidateProposalsStopper),
)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "swarms_polyglot" / "train_polyglot_policy.py"
_SPEC = importlib.util.spec_from_file_location("swarms_polyglot_train_policy", _SCRIPT_PATH)
train_policy = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = train_policy
_SPEC.loader.exec_module(train_policy)

PolyglotPolicyAdapter = train_policy.PolyglotPolicyAdapter
_information_regime_summary = train_policy._information_regime_summary


def _adapter():
    return PolyglotPolicyAdapter(
        model_id="test-model",
        workers=1,
        max_llm_calls_per_task=1,
        max_turns=1,
        max_tool_calls=1,
        bash_timeout_sec=1,
        temperature=0.0,
        max_reflection_tasks=1,
        random_seed=0,
        docker_image="unused",
    )


def _eval_batch():
    return EvaluationBatch(
        outputs=[],
        scores=[0.0],
        trajectories=[
            {
                "instance_id": "python__demo",
                "language": "python",
                "exercise_name": "demo",
                "success": False,
                "native_score": 0.0,
                "tool_calls": 3,
                "total_cost": 0.0,
                "final_summary": "could not solve",
                "parse_error": None,
                "exec_error": None,
                "test_stdout_tail": "SECRET_STDOUT_TAIL",
                "test_stderr_tail": "SECRET_STDERR_TAIL",
                "history": [{"phase": "finish", "summary": "own transcript"}],
                "raw_response": "own model output",
            }
        ],
        objective_scores=[{"test_score": 0.0}],
    )


def test_reflective_dataset_withholds_test_output_tails_by_default(monkeypatch):
    monkeypatch.delenv("KCSI_GEPA_LEAK_TEST_OUTPUT", raising=False)

    rows = _adapter().make_reflective_dataset({}, _eval_batch(), ["coding_solver_policy"])["coding_solver_policy"]

    assert rows[0]["Test Stdout Tail"] == ""
    assert rows[0]["Test Stderr Tail"] == ""
    assert "SECRET_STDOUT_TAIL" not in str(rows[0])
    assert "SECRET_STDERR_TAIL" not in str(rows[0])
    assert rows[0]["Tool History"] == [{"phase": "finish", "summary": "own transcript"}]
    assert rows[0]["Model Raw Output"] == "own model output"


def test_reflective_dataset_leak_flag_restores_test_output_tails(monkeypatch):
    monkeypatch.setenv("KCSI_GEPA_LEAK_TEST_OUTPUT", "1")

    rows = _adapter().make_reflective_dataset({}, _eval_batch(), ["coding_solver_policy"])["coding_solver_policy"]

    assert rows[0]["Test Stdout Tail"] == "SECRET_STDOUT_TAIL"
    assert rows[0]["Test Stderr Tail"] == "SECRET_STDERR_TAIL"


def test_information_regime_summary_records_selected_flags(monkeypatch):
    monkeypatch.delenv("KCSI_GEPA_LEAK_POLYGLOT_TESTS", raising=False)
    monkeypatch.delenv("KCSI_GEPA_LEAK_TEST_OUTPUT", raising=False)
    assert _information_regime_summary()["polyglot_hidden_tests_visible_to_solver"] is False
    assert _information_regime_summary()["reflection_test_output_tails_visible"] is False

    monkeypatch.setenv("KCSI_GEPA_LEAK_TEST_OUTPUT", "yes")
    assert _information_regime_summary()["reflection_test_output_tails_visible"] is True
