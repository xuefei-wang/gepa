#!/usr/bin/env python3
"""Train-50 ARC prompt-optimization baseline for Swarms comparisons.

This script adapts GEPA to optimize a single persistent ARC solver policy over a
committed ARC 50-task map. Unlike Swarms, which improves by accumulating
cross-task knowledge, this baseline improves by rewriting one shared textual
policy over repeated train-set evaluations.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import litellm
    from litellm import completion
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "This GEPA ARC baseline requires `litellm`. Install GEPA with its "
        "full extras or otherwise ensure `litellm` is available."
    ) from exc

from gepa import EvaluationBatch, GEPAAdapter, optimize
from gepa.utils import MaxCandidateProposalsStopper

litellm.suppress_debug_info = True


SEED_ARC_SOLVER_POLICY = """You are solving ARC tasks with a fixed, disciplined protocol.

1. Observe every training pair before deciding on any rule.
2. Prefer one transformation that explains all training pairs over a rule that only fits one example.
3. Always check grid size, object count, object positions, colors, symmetry, connectivity, repetition, and unchanged regions.
4. Treat unchanged cells as evidence, not noise.
5. Before trusting a rule, simulate it against every training pair and confirm that it reproduces each output exactly.
6. If two candidate rules compete, prefer the one with fewer special cases.
7. Do not use blind guesses as exploration. Submit only candidates that survive all consistency checks.
8. Return one prediction for every training example and up to two attempts for every test input.
9. Output strict JSON only in the required schema with integer grids.
"""


@dataclass
class ArcExample:
    task_id: str
    benchmark: str
    split: str
    train_in: list[list[list[int]]]
    train_out: list[list[list[int]]]
    test_in: list[list[list[int]]]
    test_out: list[list[list[int]]]


@dataclass
class TrackedLLM:
    model_id: str
    max_llm_calls: int
    temperature: float = 0.2
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(float(call.get("cost", 0.0)) for call in self.calls)

    def __call__(self, prompt: str) -> str:
        if len(self.calls) >= self.max_llm_calls:
            raise RuntimeError(f"LLM budget exhausted ({self.max_llm_calls} calls)")

        resp = completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        message = resp.choices[0].message
        content = message.content or ""
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = 0.0
        usage = getattr(resp, "usage", None)
        self.calls.append(
            {
                "prompt": prompt,
                "response": content,
                "cost": cost,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            }
        )
        return content


def infer_litellm_model(model: str | None = None, provider: str | None = None) -> str:
    provider = (provider or os.environ.get("MODEL_PROVIDER") or "").strip()
    model = (model or os.environ.get("MODEL") or "").strip()
    if not model:
        raise ValueError(
            "No model configured. Set MODEL / MODEL_PROVIDER in the environment or pass --model."
        )
    if "/" in model or not provider:
        return model
    return f"{provider}/{model}"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty model output")

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced if fenced else [text]

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : idx + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        break
        start = text.find("{", start + 1)

    raise ValueError("No JSON object found in model output")


def compare_grid(pred: Any, gold: list[list[int]]) -> tuple[bool, str]:
    if not isinstance(pred, list):
        return False, f"prediction must be a 2D list, got {type(pred).__name__}"
    if not pred or not isinstance(pred[0], list):
        return False, "prediction must be a non-empty 2D list"
    rows = len(pred)
    cols = len(pred[0])
    if any(not isinstance(row, list) for row in pred):
        return False, "prediction rows must all be lists"
    if any(len(row) != cols for row in pred):
        return False, "prediction rows have inconsistent lengths"
    gold_shape = (len(gold), len(gold[0]))
    pred_shape = (rows, cols)
    if pred_shape != gold_shape:
        return False, f"shape {pred_shape} != expected {gold_shape}"

    wrong = []
    for i in range(rows):
        for j in range(cols):
            try:
                if int(pred[i][j]) != gold[i][j]:
                    wrong.append((i, j))
            except Exception:
                wrong.append((i, j))
    if not wrong:
        return True, "correct"
    if len(wrong) <= 8:
        return False, f"wrong cells: {wrong}"
    return False, f"wrong at {len(wrong)} cells"


def evaluate_predictions(preds: list[Any], golds: list[list[list[int]]]) -> tuple[float, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for idx, gold in enumerate(golds):
        pred = preds[idx] if idx < len(preds) else None
        if pred is None:
            correct, feedback = False, "missing prediction"
        else:
            correct, feedback = compare_grid(pred, gold)
        results.append(
            {
                "idx": idx,
                "correct": correct,
                "feedback": feedback,
                "gold": gold,
                "prediction": pred,
            }
        )
    score = sum(1 for row in results if row["correct"]) / len(results) if results else 0.0
    return score, results


def _leak_arc_test_gold() -> bool:
    """Whether to expose hidden ARC test gold to the reflection LM.

    Default ``False``: the GEPA reflection LM (the adaptive component) must
    not receive the hidden test answer, matching KCSI's ARC information
    regime (only ``{test_index, correct}`` for hidden test pairs). Set
    ``KCSI_GEPA_ARC_LEAK_TEST_GOLD=1`` to restore the pre-fix behavior for
    reproducing earlier, information-leaky baseline numbers.
    """
    val = os.environ.get("KCSI_GEPA_ARC_LEAK_TEST_GOLD", "")
    # Strict allow-list (fail closed): only an explicit truthy value enables
    # the leak, so a typo/ambiguous value (e.g. "off", "2") keeps the
    # integrity-preserving default. Matches _leak_polyglot_tests and the
    # KCSI in-repo OpenEvolve gates.
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _test_feedback_row(item: dict[str, Any]) -> dict[str, Any]:
    """Reflective-dataset row for a hidden *test* pair.

    The gold output grid and the differing-cell positions are derived from
    the hidden ARC answer, so by default they are withheld from the
    reflection LM; only ``{idx, correct}`` plus the solver's own prediction
    (in-channel) are surfaced. Train pairs keep full detail elsewhere.
    """
    row = {
        "idx": item.get("idx"),
        "correct": item.get("correct"),
        "prediction": item.get("prediction"),
    }
    if _leak_arc_test_gold():
        row["feedback"] = item.get("feedback")
        row["gold"] = item.get("gold")
    else:
        row["feedback"] = "correct" if item.get("correct") else "incorrect"
    return row



def evaluate_test_attempts(preds: list[Any], golds: list[list[list[int]]]) -> tuple[float, list[dict[str, Any]]]:
    normalized: list[list[Any]] = []
    for pred in preds:
        if isinstance(pred, list) and pred and isinstance(pred[0], list) and pred and pred[0] and isinstance(pred[0][0], list):
            normalized.append(pred[:2])
        elif pred is None:
            normalized.append([])
        else:
            normalized.append([pred])

    attempt1 = [attempts[0] if attempts else None for attempts in normalized]
    attempt2 = [attempts[1] if len(attempts) > 1 else None for attempts in normalized]
    _, results1 = evaluate_predictions(attempt1, golds)
    _, results2 = evaluate_predictions(attempt2, golds)

    results: list[dict[str, Any]] = []
    for idx in range(len(golds)):
        r1 = results1[idx]
        r2 = results2[idx]
        correct = r1["correct"] or r2["correct"]
        if r1["correct"]:
            chosen = r1
        elif r2["correct"]:
            chosen = r2
        else:
            chosen = r1
        results.append(
            {
                "idx": idx,
                "correct": correct,
                "feedback": chosen["feedback"],
                "gold": chosen["gold"],
                "prediction": chosen["prediction"],
                "attempt_1": attempt1[idx],
                "attempt_2": attempt2[idx],
            }
        )

    score = 1.0 if results and all(row["correct"] for row in results) else 0.0
    return score, results


def build_arc_solver_prompt(policy: str, example: ArcExample) -> str:
    train_examples = [
        {"input": inp, "output": out}
        for inp, out in zip(example.train_in, example.train_out, strict=True)
    ]
    test_examples = [{"input": inp} for inp in example.test_in]
    return (
        "You are solving one ARC task.\n\n"
        "Follow the ARC solver policy exactly.\n\n"
        "## ARC Solver Policy\n"
        f"{policy.strip()}\n\n"
        "## Task Data\n"
        f"- task_id: {example.task_id}\n"
        f"- benchmark: {example.benchmark}\n"
        f"- split: {example.split}\n"
        f"- train_examples: {len(example.train_in)}\n"
        f"- test_inputs: {len(example.test_in)}\n\n"
        "Training pairs:\n"
        f"{json.dumps(train_examples, ensure_ascii=True)}\n\n"
        "Test inputs:\n"
        f"{json.dumps(test_examples, ensure_ascii=True)}\n\n"
        "Return strict JSON only in this schema:\n"
        "{\"train\": [grid, ...], \"test\": [[attempt1, attempt2], ...]}\n"
        "Rules:\n"
        "- `train` must contain exactly one predicted output grid for each training pair.\n"
        "- `test` must contain one list per test input, with one or two candidate output grids.\n"
        "- Every grid must be a nested list of integers.\n"
        "- Do not include any prose, markdown, or explanation.\n"
    )


def run_policy_on_example(
    example: ArcExample,
    *,
    policy: str,
    model_id: str,
    max_llm_calls_per_task: int,
    temperature: float,
) -> dict[str, Any]:
    llm = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls_per_task, temperature=temperature)
    prompt = build_arc_solver_prompt(policy, example)
    parsed: dict[str, Any] | None = None
    raw_response = ""
    parse_error: str | None = None
    exec_error: str | None = None

    try:
        raw_response = llm(prompt)
        parsed = extract_json_object(raw_response)
        train_preds = parsed.get("train", []) if isinstance(parsed, dict) else []
        test_preds = parsed.get("test", []) if isinstance(parsed, dict) else []
    except Exception as exc:
        parse_error = str(exc)
        train_preds = []
        test_preds = []

    try:
        training_score, train_results = evaluate_predictions(train_preds, example.train_out)
        test_score, test_results = evaluate_test_attempts(test_preds, example.test_out)
    except Exception as exc:
        exec_error = str(exc)
        training_score = 0.0
        test_score = 0.0
        train_results = []
        test_results = []

    return {
        "task_id": example.task_id,
        "benchmark": example.benchmark,
        "training_score": training_score,
        "test_score": test_score,
        "raw_response": raw_response,
        "parsed": parsed,
        "parse_error": parse_error,
        "exec_error": exec_error,
        "train_results": train_results,
        "test_results": test_results,
        "prompt": prompt,
        "llm_calls": llm.calls,
        "total_cost": llm.total_cost,
        "success": bool(test_score == 1.0),
    }


class ArcPolicyAdapter(GEPAAdapter[ArcExample, dict[str, Any], dict[str, Any]]):
    def __init__(
        self,
        *,
        model_id: str,
        workers: int,
        max_llm_calls_per_task: int,
        temperature: float,
        max_reflection_tasks: int,
        random_seed: int,
    ):
        self.model_id = model_id
        self.workers = workers
        self.max_llm_calls_per_task = max_llm_calls_per_task
        self.temperature = temperature
        self.max_reflection_tasks = max_reflection_tasks
        self.random = random.Random(random_seed)

    def evaluate(
        self,
        batch: list[ArcExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        policy = candidate["arc_solver_policy"]
        outputs: list[dict[str, Any]] = [None] * len(batch)  # type: ignore[list-item]
        scores: list[float] = [0.0] * len(batch)
        trajectories: list[dict[str, Any] | None] = [None] * len(batch)
        objective_scores: list[dict[str, float]] = [{"train_score": 0.0, "test_score": 0.0} for _ in batch]

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_idx = {
                executor.submit(
                    run_policy_on_example,
                    example,
                    policy=policy,
                    model_id=self.model_id,
                    max_llm_calls_per_task=self.max_llm_calls_per_task,
                    temperature=self.temperature,
                ): idx
                for idx, example in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - network/runtime dependent
                    example = batch[idx]
                    result = {
                        "task_id": example.task_id,
                        "benchmark": example.benchmark,
                        "training_score": 0.0,
                        "test_score": 0.0,
                        "raw_response": "",
                        "parsed": None,
                        "parse_error": str(exc),
                        "exec_error": None,
                        "train_results": [],
                        "test_results": [],
                        "prompt": "",
                        "llm_calls": [],
                        "total_cost": 0.0,
                        "success": False,
                    }
                outputs[idx] = {
                    "task_id": result["task_id"],
                    "parsed": result["parsed"],
                    "raw_response": result["raw_response"],
                }
                scores[idx] = float(result["test_score"])
                objective_scores[idx] = {
                    "train_score": float(result["training_score"]),
                    "test_score": float(result["test_score"]),
                }
                if capture_traces:
                    trajectories[idx] = result

        final_trajectories = [t if t is not None else {} for t in trajectories] if capture_traces else None
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=final_trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        assert eval_batch.trajectories is not None
        component = components_to_update[0]
        rows = list(eval_batch.trajectories)
        rows.sort(key=lambda row: (1 if row.get("success") else 0, row.get("training_score", 0.0)))

        selected = rows[: self.max_reflection_tasks]
        reflective_rows: list[dict[str, Any]] = []
        for row in selected:
            reflective_rows.append(
                {
                    "Task ID": row.get("task_id"),
                    "Outcome": "solved" if row.get("success") else "failed",
                    "Training Score": row.get("training_score"),
                    "Test Score": row.get("test_score"),
                    "Train Feedback": [
                        {
                            "idx": item.get("idx"),
                            "correct": item.get("correct"),
                            "feedback": item.get("feedback"),
                            "gold": item.get("gold"),
                            "prediction": item.get("prediction"),
                        }
                        for item in row.get("train_results", [])
                    ],
                    "Test Feedback": [
                        _test_feedback_row(item) for item in row.get("test_results", [])
                    ],
                    "Model Raw Output": row.get("raw_response", "")[:4000],
                    "Parser Error": row.get("parse_error"),
                    "Execution Error": row.get("exec_error"),
                    "Total Cost": row.get("total_cost", 0.0),
                }
            )

        return {
            component: reflective_rows,
        }


class IterationMetricsLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "iteration_metrics.jsonl"

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        scores = [float(score) for score in event["scores_by_val_id"].values()]
        solved = sum(1 for score in scores if score == 1.0)
        row = {
            "iteration": int(event["iteration"]),
            "candidate_idx": int(event["candidate_idx"]),
            "num_examples": int(event["num_examples_evaluated"]),
            "solved": int(solved),
            "solve_rate": (solved / len(scores)) if scores else 0.0,
            "average_score": float(event["average_score"]),
            "is_best_program": bool(event["is_best_program"]),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[iter={row['iteration']}] solved={row['solved']}/{row['num_examples']} "
            f"solve_rate={row['solve_rate']:.1%} avg_score={row['average_score']:.3f} "
            f"best={row['is_best_program']}"
        )


def load_arc_examples(task_map_path: Path, repo_root: Path) -> list[ArcExample]:
    data = json.loads(task_map_path.read_text())
    benchmark = str(data.get("benchmark") or "arc")
    split = str(data.get("split") or "unknown")
    tasks = data.get("tasks") or []
    examples: list[ArcExample] = []
    missing: list[str] = []

    for row in tasks:
        rel = row.get("source_file")
        task_id = row.get("task_id")
        if not rel or not task_id:
            continue
        source_file = (repo_root / rel).resolve()
        if not source_file.exists():
            missing.append(str(source_file))
            continue
        task_data = json.loads(source_file.read_text())
        train = task_data.get("train") or []
        test = task_data.get("test") or []
        examples.append(
            ArcExample(
                task_id=str(task_id),
                benchmark=benchmark,
                split=split,
                train_in=[item["input"] for item in train],
                train_out=[item["output"] for item in train],
                test_in=[item["input"] for item in test],
                test_out=[item["output"] for item in test],
            )
        )

    if missing:
        sample = "\n".join(missing[:5])
        raise FileNotFoundError(
            "ARC source files referenced by the task map are missing.\n"
            f"Repo root checked: {repo_root}\n"
            f"Missing examples (first few):\n{sample}\n\n"
            "Sync the ARC benchmark source data into this checkout, or pass a "
            "--source-root that points at a checkout with the benchmark files present."
        )
    if not examples:
        raise ValueError(f"No ARC examples loaded from task map: {task_map_path}")
    return examples


def default_task_map(benchmark: str) -> Path:
    if benchmark == "arc1":
        return Path("benchmarks/arc1/task_maps/arc1_train_50_seed0.json")
    if benchmark == "arc2":
        return Path("benchmarks/arc2/task_maps/arc2_train_50_seed0.json")
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def summarize_scores(eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    solved = int(sum(1 for score in eval_batch.scores if float(score) == 1.0))
    train_scores = [float(obj.get("train_score", 0.0)) for obj in (eval_batch.objective_scores or [])]
    avg_train_score = sum(train_scores) / len(train_scores) if train_scores else 0.0
    return {
        "solved": solved,
        "total": len(eval_batch.scores),
        "solve_rate": solved / len(eval_batch.scores) if eval_batch.scores else 0.0,
        "avg_train_score": avg_train_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA ARC train-50 policy optimization baseline")
    parser.add_argument("--benchmark", choices=["arc1", "arc2"], default="arc1")
    parser.add_argument("--task-map", type=Path, default=None)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Repository root that contains benchmark source files referenced by the task map.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--reflection-model", type=str, default=None)
    parser.add_argument("--reflection-provider", type=str, default=None)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-llm-calls-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-reflection-tasks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    task_map = args.task_map or default_task_map(args.benchmark)
    task_map = task_map.resolve()
    source_root = args.source_root.resolve()
    model_id = infer_litellm_model(args.model, args.provider)
    reflection_model = infer_litellm_model(args.reflection_model or args.model, args.reflection_provider or args.provider)

    examples = load_arc_examples(task_map, source_root)
    reflection_minibatch_size = len(examples)
    run_dir = args.run_dir or (
        Path("results/internal_ddl")
        / f"gepa_{args.benchmark}_{model_id.split('/')[-1].replace(':', '_')}_train50"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = ArcPolicyAdapter(
        model_id=model_id,
        workers=args.workers,
        max_llm_calls_per_task=args.max_llm_calls_per_task,
        temperature=args.temperature,
        max_reflection_tasks=args.max_reflection_tasks,
        random_seed=args.seed,
    )

    baseline_eval = adapter.evaluate(examples, {"arc_solver_policy": SEED_ARC_SOLVER_POLICY}, capture_traces=True)
    baseline_summary = summarize_scores(baseline_eval)
    print(
        f"[baseline] solved={baseline_summary['solved']}/{baseline_summary['total']} "
        f"solve_rate={baseline_summary['solve_rate']:.1%} avg_train_score={baseline_summary['avg_train_score']:.3f}"
    )

    callback = IterationMetricsLogger(run_dir)

    result = optimize(
        seed_candidate={"arc_solver_policy": SEED_ARC_SOLVER_POLICY},
        trainset=examples,
        valset=None,
        adapter=adapter,
        reflection_lm=reflection_model,
        reflection_minibatch_size=reflection_minibatch_size,
        run_dir=str(run_dir),
        stop_callbacks=MaxCandidateProposalsStopper(args.generations),
        perfect_score=1.0,
        skip_perfect_score=False,
        seed=args.seed,
        display_progress_bar=False,
        callbacks=[callback],
    )

    best_candidate = result.best_candidate
    final_eval = adapter.evaluate(examples, best_candidate, capture_traces=True)
    final_summary = summarize_scores(final_eval)
    print(
        f"[final] solved={final_summary['solved']}/{final_summary['total']} "
        f"solve_rate={final_summary['solve_rate']:.1%} avg_train_score={final_summary['avg_train_score']:.3f}"
    )

    summary = {
        "benchmark": args.benchmark,
        "task_map": str(task_map),
        "source_root": str(source_root),
        "model_id": model_id,
        "reflection_model": reflection_model,
        "generations": args.generations,
        "workers": args.workers,
        "max_llm_calls_per_task": args.max_llm_calls_per_task,
        "max_reflection_tasks": args.max_reflection_tasks,
        "baseline": baseline_summary,
        "final": final_summary,
        "best_candidate": best_candidate,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "best_arc_solver_policy.txt").write_text(best_candidate["arc_solver_policy"].strip() + "\n")


if __name__ == "__main__":
    main()
