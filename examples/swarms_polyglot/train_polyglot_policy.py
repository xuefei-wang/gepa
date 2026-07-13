#!/usr/bin/env python3
"""GEPA Polyglot train-50 coding-policy baseline for Swarms comparisons.

This baseline optimizes a single persistent coding policy over a 50-task
Polyglot pool. Each task is executed by a minimal coding loop that can use only
two tools:

* ``bash``   - runs commands inside the task container/workspace
* ``editor`` - views and edits files inside the mounted workspace

Unlike Swarms, which improves by accumulating reusable cross-task knowledge,
this baseline improves by rewriting one shared textual coding policy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import litellm
    from litellm import completion
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "This GEPA Polyglot baseline requires `litellm`. Install GEPA with its "
        "full extras or otherwise ensure `litellm` is available."
    ) from exc

from gepa import EvaluationBatch, GEPAAdapter, optimize
from gepa.lm import LM
from gepa.utils import MaxCandidateProposalsStopper

litellm.suppress_debug_info = True


SEED_CODING_SOLVER_POLICY = """You are solving a repository coding task with a fixed, disciplined protocol.

1. Read the problem statement carefully and identify the required behavioral change before editing code.
2. Inspect the relevant test files and the most likely implementation files before making a patch.
3. Prefer minimal, local changes over broad refactors.
4. Use the bash tool for lightweight inspection and test execution only; avoid long noisy commands.
5. Use the editor tool for exact file changes. Do not rewrite large files unless necessary.
6. After each meaningful edit, run the narrowest relevant test command or file-level check you can justify.
7. If a hypothesis fails, revise it quickly rather than compounding speculative edits.
8. Stop once the tests pass or you have one coherent final patch. Do not continue polishing after success.
9. When uncertain, prioritize correctness over style and keep the patch as small as possible.
10. Respond only with valid JSON in the required tool/finish schema.
"""


_DEFAULT_POLYGLOT_DOCKER_IMAGE = os.environ.get("POLYGLOT_DOCKER_IMAGE", "swarms-polyglot-eval:latest")

_ALLOWED_TEST_CMD_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^python3?\s+-m\s+pytest\b"),
    "rust": re.compile(r"^cargo\s+test\b"),
    "go": re.compile(r"^go\s+test\b"),
    "javascript": re.compile(r"^(npm\s+test|node\s+|jest\b|npx\s+jest\b)"),
    "java": re.compile(r"^(gradle\s+test|mvn\s+test|java\b)"),
    "cpp": re.compile(r"^(make\s+test|cmake\b|ctest\b|cd\s+\S+|g\+\+(?:\s|$)|c\+\+(?:\s|$))"),
}
_SHELL_INJECTION_RE = re.compile(r"[`$(){}<>]|(?<![&|])\|(?![|])")
_CHAIN_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;)\s*")


@dataclass
class PolyglotExample:
    instance_id: str
    language: str
    exercise_name: str
    problem_statement: str
    starter_code: dict[str, str]
    test_files: dict[str, str]
    build_files: dict[str, str]
    test_command: str
    meta_config: dict[str, Any]


@dataclass
class TrackedLLM:
    model_id: str
    max_llm_calls: int
    temperature: float = 0.2
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(float(call.get("cost", 0.0)) for call in self.calls)

    @property
    def total_tokens_in(self) -> int:
        total = 0
        for call in self.calls:
            usage = call.get("usage")
            if isinstance(usage, dict):
                total += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        return total

    @property
    def total_tokens_out(self) -> int:
        total = 0
        for call in self.calls:
            usage = call.get("usage")
            if isinstance(usage, dict):
                total += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return total

    @staticmethod
    def _normalize_usage(usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        normalized: dict[str, Any] = {}
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                normalized[key] = value
        return normalized

    @staticmethod
    def _extract_cost(resp: Any) -> float:
        try:
            cost = litellm.completion_cost(completion_response=resp)
            if cost is not None:
                return float(cost)
        except Exception:
            pass
        hidden = getattr(resp, "_hidden_params", None)
        if isinstance(hidden, dict):
            response_cost = hidden.get("response_cost")
            if response_cost is not None:
                try:
                    return float(response_cost)
                except Exception:
                    pass
        return 0.0

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
        cost = self._extract_cost(resp)
        usage = self._normalize_usage(getattr(resp, "usage", None))
        self.calls.append(
            {
                "prompt": prompt,
                "response": content,
                "cost": cost,
                "usage": usage,
            }
        )
        return content


def infer_litellm_model(model: str | None = None, provider: str | None = None) -> str:
    provider = (provider or os.environ.get("MODEL_PROVIDER") or "").strip()
    model = (model or os.environ.get("MODEL") or "").strip()
    if not model:
        raise ValueError("No model configured. Set MODEL / MODEL_PROVIDER in the environment or pass --model.")
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


def _validate_test_command(test_command: str, language: str) -> None:
    cmd = test_command.strip()
    pattern = _ALLOWED_TEST_CMD_PATTERNS.get(language)
    if pattern is None:
        raise ValueError(f"No allowlisted test-command pattern for language {language!r}")
    if _SHELL_INJECTION_RE.search(cmd):
        raise ValueError(f"test_command contains shell metacharacters: {test_command!r}")
    if re.search(r"(?<!&)&(?!&)", cmd):
        raise ValueError(f"test_command contains shell metacharacters: {test_command!r}")
    for step in _CHAIN_SPLIT_RE.split(cmd):
        step = step.strip()
        if not step:
            raise ValueError(f"test_command has empty chain step: {test_command!r}")
        if not pattern.match(step):
            raise ValueError(f"test_command rejected for language {language!r}: {test_command!r}")


def _validate_safe_path(base: Path, name: str) -> Path:
    base_path = base.resolve()
    p = (base_path / name).resolve()
    try:
        p.relative_to(base_path)
    except ValueError as exc:
        raise ValueError(f"Unsafe file path in task metadata or tool input: {name!r}") from exc
    return p


def _safe_write(base: Path, name: str, content: str) -> None:
    target = _validate_safe_path(base, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _setup_command(language: str) -> str:
    setups: dict[str, str] = {
        "javascript": "npm install --silent 2>/dev/null",
        "java": "[ -f settings.gradle ] || echo 'rootProject.name=\"exercise\"' > settings.gradle",
        "cpp": 'for f in *.cpp; do h="${f%.cpp}.h"; [ -f "$h" ] || touch "$h"; done 2>/dev/null; true',
    }
    return setups.get(language, "")


def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n<response clipped>\n" + text[-tail:]


class WorkspaceEditor:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir.resolve()

    def _resolve(self, rel_path: str) -> Path:
        if rel_path.startswith("/"):
            rel_path = rel_path.lstrip("/")
        return _validate_safe_path(self.base_dir, rel_path)

    def _format_output(self, content: str, path: str, start_line: int = 1) -> str:
        content = content.expandtabs()
        lines = [f"{idx + start_line:6}\t{line}" for idx, line in enumerate(_truncate(content, 8000).splitlines())]
        return f"cat -n {path}\n" + "\n".join(lines)

    def tool(self, command: str, path: str, **kwargs: Any) -> str:
        path_obj = self._resolve(path)
        if command == "view":
            return self.view(path_obj, kwargs.get("view_range"))
        if command == "create":
            return self.create(path_obj, kwargs.get("file_text"))
        if command == "str_replace":
            return self.str_replace(path_obj, kwargs.get("old_str"), kwargs.get("new_str", ""))
        if command == "insert":
            return self.insert(path_obj, kwargs.get("insert_line"), kwargs.get("new_str", ""))
        raise ValueError(f"Unknown editor command: {command}")

    def view(self, path_obj: Path, view_range: Any = None) -> str:
        if path_obj.is_dir():
            entries = sorted(
                str(p.relative_to(self.base_dir))
                for p in path_obj.rglob("*")
                if p.is_file() and len(p.relative_to(path_obj).parts) <= 2
            )
            return _truncate("\n".join(entries), 4000)
        content = path_obj.read_text(encoding="utf-8")
        if isinstance(view_range, list) and len(view_range) == 2:
            start, end = int(view_range[0]), int(view_range[1])
            lines = content.splitlines()
            if end == -1:
                snippet = "\n".join(lines[start - 1 :])
            else:
                snippet = "\n".join(lines[start - 1 : end])
            return self._format_output(snippet, str(path_obj.relative_to(self.base_dir)), start)
        return self._format_output(content, str(path_obj.relative_to(self.base_dir)))

    def create(self, path_obj: Path, file_text: Any) -> str:
        if path_obj.exists():
            raise ValueError(f"File already exists: {path_obj}")
        if not isinstance(file_text, str):
            raise ValueError("editor.create requires string file_text")
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(file_text, encoding="utf-8")
        return f"created {path_obj.relative_to(self.base_dir)}"

    def str_replace(self, path_obj: Path, old_str: Any, new_str: Any) -> str:
        if not path_obj.exists():
            raise ValueError(f"File not found: {path_obj}")
        if not isinstance(old_str, str):
            raise ValueError("editor.str_replace requires string old_str")
        if not isinstance(new_str, str):
            raise ValueError("editor.str_replace requires string new_str")
        content = path_obj.read_text(encoding="utf-8")
        count = content.count(old_str)
        if count == 0:
            raise ValueError("old_str did not appear verbatim in the file")
        if count > 1:
            raise ValueError("old_str is not unique in the file")
        path_obj.write_text(content.replace(old_str, new_str), encoding="utf-8")
        return f"edited {path_obj.relative_to(self.base_dir)}"

    def insert(self, path_obj: Path, insert_line: Any, new_str: Any) -> str:
        if not path_obj.exists():
            raise ValueError(f"File not found: {path_obj}")
        if not isinstance(insert_line, int):
            raise ValueError("editor.insert requires integer insert_line")
        if not isinstance(new_str, str):
            raise ValueError("editor.insert requires string new_str")
        lines = path_obj.read_text(encoding="utf-8").splitlines()
        if insert_line < 0 or insert_line > len(lines):
            raise ValueError(f"insert_line out of range: {insert_line}")
        new_lines = lines[:insert_line] + new_str.splitlines() + lines[insert_line:]
        path_obj.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        return f"inserted into {path_obj.relative_to(self.base_dir)} after line {insert_line}"


class ContainerSession:
    def __init__(self, *, workspace_dir: Path, language: str, exercise_name: str, docker_image: str):
        self.workspace_dir = workspace_dir.resolve()
        self.language = language
        self.exercise_name = exercise_name
        self.docker_image = docker_image
        self.container_name = (
            f"gepa-polyglot-{re.sub(r'[^a-zA-Z0-9_.-]+', '-', exercise_name.lower())}-{uuid.uuid4().hex[:12]}"
        )
        self.container_workdir = f"/work/{exercise_name}" if language == "cpp" else "/exercise"
        self.started = False

    def start(self) -> None:
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            self.container_name,
            "-v",
            f"{self.workspace_dir}:{self.container_workdir}",
            "-w",
            self.container_workdir,
            self.docker_image,
            "bash",
            "-lc",
            "tail -f /dev/null",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "").strip())
        self.started = True

    def exec_bash(self, command: str, *, timeout_sec: int = 30) -> str:
        if not self.started:
            raise RuntimeError("container not started")
        cmd = [
            "docker",
            "exec",
            self.container_name,
            "bash",
            "-lc",
            command,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        output = []
        if proc.stdout:
            output.append(proc.stdout.strip())
        if proc.stderr:
            output.append("STDERR:\n" + proc.stderr.strip())
        output.append(f"EXIT_CODE: {proc.returncode}")
        return _truncate("\n".join(part for part in output if part))

    def stop(self) -> None:
        if not self.started:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.started = False


def build_coding_prompt(
    *,
    policy: str,
    example: PolyglotExample,
    history: list[dict[str, Any]],
    max_turns: int,
    max_tool_calls: int,
) -> str:
    history_text = json.dumps(history[-12:], indent=2, ensure_ascii=True)
    starter_files = sorted(example.starter_code.keys())
    test_files = sorted(example.test_files.keys()) if _leak_polyglot_tests() else []
    build_files = sorted(example.build_files.keys())
    return f"""You are solving one coding task inside an isolated workspace.

Follow the coding solver policy exactly.

## Coding Solver Policy
{policy.strip()}

## Task Data
- instance_id: {example.instance_id}
- language: {example.language}
- exercise_name: {example.exercise_name}
- max_turns: {max_turns}
- max_tool_calls: {max_tool_calls}
- starter_files: {starter_files}
- test_files: {test_files}
- build_files: {build_files}
- official_test_command: {example.test_command}

## Problem Statement
{example.problem_statement}

## Available Tools
1. bash
   Input schema:
   {{
     "command": "shell command to run inside the task container workspace"
   }}

2. editor
   Input schema:
   {{
     "command": "view|create|str_replace|insert",
     "path": "relative/path/inside/workspace",
     "view_range": [start, end],        # optional for view
     "file_text": "...",                # required for create
     "old_str": "...",                  # required for str_replace
     "new_str": "...",                  # required for str_replace/insert
     "insert_line": 12                  # required for insert
   }}

## Response Format
Respond with valid JSON only, in one of these forms:

{{
  "action": "tool",
  "tool_name": "bash" | "editor",
  "tool_input": {{ ... }}
}}

or

{{
  "action": "finish",
  "summary": "brief explanation of what you changed or why you are stopping"
}}

Use at most one tool per response. Do not include markdown fences or prose outside JSON.

## Recent Interaction History
{history_text}
"""


def _leak_polyglot_tests() -> bool:
    """Whether to expose the hidden polyglot test files to the GEPA solver.

    Default False: the solver sees spec + stub only (matching KCSI and
    HyperAgents); the hidden tests are applied only at grading. Set
    ``KCSI_GEPA_LEAK_POLYGLOT_TESTS=1`` to write the tests into the solver
    workspace for reproducing earlier, test-visible baseline numbers.
    """
    val = os.environ.get("KCSI_GEPA_LEAK_POLYGLOT_TESTS", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _leak_test_output() -> bool:
    """Whether to feed the hidden test-runner stdout/stderr tails to the reflection LM.

    Default False: the reflection LM (the adaptive component) sees only the scalar
    pass/fail score plus the agent's own output -- NOT the grader's stdout/stderr
    tails, which name the hidden tests and print their assertions -- matching the
    per-task solver's information regime. Set ``KCSI_GEPA_LEAK_TEST_OUTPUT=1`` to
    feed the tails back into the reflective dataset for reproducing earlier,
    information-leaky baseline numbers. Mirrors _leak_polyglot_tests /
    _leak_arc_test_gold (leak-closed by default; env re-enables).
    """
    val = os.environ.get("KCSI_GEPA_LEAK_TEST_OUTPUT", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _redact_test_output_from_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a trace-safe result for GEPA reflection/history consumers."""
    if _leak_test_output():
        return result
    redacted = dict(result)
    redacted["test_stdout_tail"] = ""
    redacted["test_stderr_tail"] = ""
    redacted["test_result"] = ""
    redacted["test_output_withheld"] = True
    return redacted


def _information_regime_summary() -> dict[str, Any]:
    return {
        "polyglot_hidden_tests_visible_to_solver": _leak_polyglot_tests(),
        "reflection_test_output_tails_visible": _leak_test_output(),
        "default": "fair",
        "legacy_reproduction_flags": {
            "KCSI_GEPA_LEAK_POLYGLOT_TESTS": (
                "Set to 1/true/yes/on to write hidden Polyglot tests into the solver workspace."
            ),
            "KCSI_GEPA_LEAK_TEST_OUTPUT": (
                "Set to 1/true/yes/on to feed hidden test-runner stdout/stderr tails to reflection."
            ),
        },
    }


def _write_test_files(target_dir: Path, example: PolyglotExample) -> None:
    """Write the hidden test files into ``target_dir`` with the same
    per-language normalizations the grader expects. Called at grade time in
    the default (hidden-test) regime, and at workspace-prep time only when
    the leak flag is set."""
    for name, content in example.test_files.items():
        if example.language == "javascript":
            content = content.replace("xtest(", "test(").replace("xit(", "it(")
        if example.language == "java":
            content = re.sub(r'@Disabled(?:\("[^"]*"\))?\s*\n', "", content)
        _safe_write(target_dir, name, content)


def prepare_workspace(example: PolyglotExample) -> Path:
    tmpdir_root = Path(tempfile.mkdtemp(prefix="gepa-polyglot-"))
    tmpdir = tmpdir_root / example.exercise_name if example.language == "cpp" else tmpdir_root
    tmpdir.mkdir(parents=True, exist_ok=True)

    for name, content in example.build_files.items():
        _safe_write(tmpdir, name, content)
    # Default (hidden-test) regime: do NOT write the test files into the
    # solver workspace; they are added at grade time. Only write them here
    # when the leak flag is set (reproduces earlier, test-visible numbers).
    if _leak_polyglot_tests():
        _write_test_files(tmpdir, example)
    for name, content in example.starter_code.items():
        _safe_write(tmpdir, name, content)
    return tmpdir_root


def load_task_ids(task_map_path: Path) -> list[str]:
    obj = json.loads(task_map_path.read_text())
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError(f"Expected task-map JSON list[str], got {type(obj).__name__} from {task_map_path}")
    return list(obj)


def load_polyglot_examples(
    dataset_path: Path,
    *,
    task_limit: int | None = None,
    task_ids: list[str] | None = None,
) -> list[PolyglotExample]:
    data = json.loads(dataset_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset, got {type(data).__name__}")
    rows = data
    if task_ids is not None:
        by_id = {str(row["instance_id"]): row for row in data}
        missing = [task_id for task_id in task_ids if task_id not in by_id]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"Task map contains IDs missing from dataset ({len(missing)} total): {preview}")
        rows = [by_id[task_id] for task_id in task_ids]
    if task_limit is not None:
        rows = rows[:task_limit]
    examples = []
    for row in rows:
        examples.append(
            PolyglotExample(
                instance_id=str(row["instance_id"]),
                language=str(row["language"]),
                exercise_name=str(row["exercise_name"]),
                problem_statement=str(row["problem_statement"]),
                starter_code=dict(row.get("starter_code") or {}),
                test_files=dict(row.get("test_files") or {}),
                build_files=dict(row.get("build_files") or {}),
                test_command=str(row.get("test_command") or ""),
                meta_config=dict(row.get("meta_config") or {}),
            )
        )
    if not examples:
        raise ValueError(f"No examples loaded from dataset: {dataset_path}")
    return examples


def default_dataset_path() -> Path:
    candidates = [
        Path("data/polyglot_medium.json"),
        Path(__file__).resolve().parents[4] / "data" / "polyglot_medium.json",
        Path(__file__).resolve().parents[5] / "swarms" / "data" / "polyglot_medium.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def ensure_docker_image_exists(image: str) -> None:
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Required Docker image not available: {image}\n{detail}")


def run_policy_on_example(
    example: PolyglotExample,
    *,
    policy: str,
    model_id: str,
    max_llm_calls_per_task: int,
    max_turns: int,
    max_tool_calls: int,
    bash_timeout_sec: int,
    temperature: float,
    docker_image: str,
) -> dict[str, Any]:
    llm = TrackedLLM(model_id=model_id, max_llm_calls=max_llm_calls_per_task, temperature=temperature)
    workspace_root = prepare_workspace(example)
    workspace_dir = workspace_root / example.exercise_name if example.language == "cpp" else workspace_root
    editor = WorkspaceEditor(workspace_dir)
    session = ContainerSession(
        workspace_dir=workspace_dir,
        language=example.language,
        exercise_name=example.exercise_name,
        docker_image=docker_image,
    )

    history: list[dict[str, Any]] = []
    tool_calls = 0
    final_summary = ""
    final_model_response = ""
    parse_error: str | None = None
    exec_error: str | None = None
    test_result: str | None = None
    test_stdout_tail = ""
    test_stderr_tail = ""
    resolved = False
    native_score = 0.0

    try:
        _validate_test_command(example.test_command, example.language)
        session.start()
        setup_cmd = _setup_command(example.language)
        if setup_cmd:
            setup_output = session.exec_bash(setup_cmd, timeout_sec=120)
            history.append({"phase": "setup", "observation": setup_output})

        for turn in range(max_turns):
            prompt = build_coding_prompt(
                policy=policy,
                example=example,
                history=history,
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
            )
            final_model_response = llm(prompt)
            try:
                action = extract_json_object(final_model_response)
            except Exception as exc:
                parse_error = str(exc)
                history.append(
                    {
                        "phase": "model_error",
                        "turn": turn,
                        "raw_response": _truncate(final_model_response, 3000),
                        "error": parse_error,
                    }
                )
                break

            if action.get("action") == "finish":
                final_summary = str(action.get("summary") or "")
                history.append({"phase": "finish", "turn": turn, "summary": final_summary})
                break

            if action.get("action") != "tool":
                parse_error = f"Unsupported action: {action.get('action')!r}"
                history.append({"phase": "model_error", "turn": turn, "error": parse_error})
                break

            if tool_calls >= max_tool_calls:
                history.append({"phase": "limit", "turn": turn, "error": "tool budget exhausted"})
                break

            tool_name = str(action.get("tool_name") or "")
            tool_input = action.get("tool_input") if isinstance(action.get("tool_input"), dict) else {}
            try:
                if tool_name == "bash":
                    output = session.exec_bash(
                        str(tool_input.get("command") or ""),
                        timeout_sec=bash_timeout_sec,
                    )
                elif tool_name == "editor":
                    output = editor.tool(
                        str(tool_input.get("command") or ""),
                        str(tool_input.get("path") or ""),
                        view_range=tool_input.get("view_range"),
                        file_text=tool_input.get("file_text"),
                        old_str=tool_input.get("old_str"),
                        new_str=tool_input.get("new_str"),
                        insert_line=tool_input.get("insert_line"),
                    )
                else:
                    output = f"Error: Unknown tool {tool_name!r}"
            except Exception as exc:
                output = f"Error: {exc}"

            tool_calls += 1
            history.append(
                {
                    "phase": "tool",
                    "turn": turn,
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "observation": _truncate(output, 5000),
                }
            )

        # Hidden-test regime: the solver never saw the test files (they were
        # not written into its bind-mounted workspace). Write them now, just
        # before grading, so the grader can run them. (When leaking, they are
        # already present from prepare_workspace; re-writing is idempotent.)
        if not _leak_polyglot_tests():
            _write_test_files(workspace_dir, example)
        test_cmd = example.test_command
        full_cmd = f"{setup_cmd} && {test_cmd}" if setup_cmd else test_cmd
        if example.language == "cpp" and "cmake" in full_cmd:
            full_cmd = full_cmd.replace(
                "cmake -B build",
                "cmake -B build -DEXERCISM_TEST_SUITE=1 -DEXERCISM_RUN_ALL_TESTS=1",
            )

        test_output = session.exec_bash(full_cmd, timeout_sec=120)
        test_result = test_output
        resolved = "EXIT_CODE: 0" in test_output.splitlines()[-1:]
        native_score = 1.0 if resolved else 0.0
        if "STDERR:\n" in test_output:
            before, _, after = test_output.partition("STDERR:\n")
            test_stdout_tail = _truncate(before, 2000)
            test_stderr_tail = _truncate(after, 2000)
        else:
            test_stdout_tail = _truncate(test_output, 2000)
            test_stderr_tail = ""
    except Exception as exc:  # pragma: no cover - runtime dependent
        exec_error = str(exc)
    finally:
        session.stop()
        shutil.rmtree(workspace_root, ignore_errors=True)

    return {
        "instance_id": example.instance_id,
        "language": example.language,
        "exercise_name": example.exercise_name,
        "native_score": native_score,
        "resolved": resolved,
        "final_summary": final_summary,
        "raw_response": final_model_response,
        "parse_error": parse_error,
        "exec_error": exec_error,
        "history": history,
        "tool_calls": tool_calls,
        "test_stdout_tail": test_stdout_tail,
        "test_stderr_tail": test_stderr_tail,
        "test_result": _truncate(test_result or "", 4000),
        "llm_calls": llm.calls,
        "total_cost": llm.total_cost,
        "total_tokens_in": llm.total_tokens_in,
        "total_tokens_out": llm.total_tokens_out,
        "success": resolved,
    }


class PolyglotPolicyAdapter(GEPAAdapter[PolyglotExample, dict[str, Any], dict[str, Any]]):
    def __init__(
        self,
        *,
        model_id: str,
        workers: int,
        max_llm_calls_per_task: int,
        max_turns: int,
        max_tool_calls: int,
        bash_timeout_sec: int,
        temperature: float,
        max_reflection_tasks: int,
        random_seed: int,
        docker_image: str,
    ):
        self.model_id = model_id
        self.workers = workers
        self.max_llm_calls_per_task = max_llm_calls_per_task
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.bash_timeout_sec = bash_timeout_sec
        self.temperature = temperature
        self.max_reflection_tasks = max_reflection_tasks
        self.random = random.Random(random_seed)
        self.docker_image = docker_image
        self._usage_lock = threading.Lock()
        self._evaluation_calls = 0
        self._task_evaluations = 0
        self._total_task_eval_cost = 0.0
        self._total_task_eval_tokens_in = 0
        self._total_task_eval_tokens_out = 0

    def stats_snapshot(self) -> dict[str, Any]:
        with self._usage_lock:
            return {
                "evaluation_calls": self._evaluation_calls,
                "task_evaluations": self._task_evaluations,
                "total_cost": self._total_task_eval_cost,
                "total_tokens_in": self._total_task_eval_tokens_in,
                "total_tokens_out": self._total_task_eval_tokens_out,
            }

    def evaluate(
        self,
        batch: list[PolyglotExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        policy = candidate["coding_solver_policy"]
        outputs: list[dict[str, Any]] = [None] * len(batch)  # type: ignore[list-item]
        scores: list[float] = [0.0] * len(batch)
        trajectories: list[dict[str, Any] | None] = [None] * len(batch)
        objective_scores: list[dict[str, float]] = [{"test_score": 0.0} for _ in batch]

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_idx = {
                executor.submit(
                    run_policy_on_example,
                    example,
                    policy=policy,
                    model_id=self.model_id,
                    max_llm_calls_per_task=self.max_llm_calls_per_task,
                    max_turns=self.max_turns,
                    max_tool_calls=self.max_tool_calls,
                    bash_timeout_sec=self.bash_timeout_sec,
                    temperature=self.temperature,
                    docker_image=self.docker_image,
                ): idx
                for idx, example in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - runtime dependent
                    example = batch[idx]
                    result = {
                        "instance_id": example.instance_id,
                        "language": example.language,
                        "exercise_name": example.exercise_name,
                        "native_score": 0.0,
                        "resolved": False,
                        "final_summary": "",
                        "raw_response": "",
                        "parse_error": str(exc),
                        "exec_error": None,
                        "history": [],
                        "tool_calls": 0,
                        "test_stdout_tail": "",
                        "test_stderr_tail": "",
                        "test_result": "",
                        "llm_calls": [],
                        "total_cost": 0.0,
                        "total_tokens_in": 0,
                        "total_tokens_out": 0,
                        "success": False,
                    }
                outputs[idx] = {
                    "instance_id": result["instance_id"],
                    "language": result["language"],
                    "exercise_name": result["exercise_name"],
                    "resolved": result["resolved"],
                    "raw_response": result["raw_response"],
                    "total_cost": result["total_cost"],
                    "total_tokens_in": result["total_tokens_in"],
                    "total_tokens_out": result["total_tokens_out"],
                }
                scores[idx] = float(result["native_score"])
                objective_scores[idx] = {
                    "test_score": float(result["native_score"]),
                }
                if capture_traces:
                    trajectories[idx] = _redact_test_output_from_result(result)

        eval_cost = sum(float(obj.get("total_cost", 0.0)) for obj in outputs if isinstance(obj, dict))
        eval_tokens_in = sum(int(obj.get("total_tokens_in", 0)) for obj in outputs if isinstance(obj, dict))
        eval_tokens_out = sum(int(obj.get("total_tokens_out", 0)) for obj in outputs if isinstance(obj, dict))
        with self._usage_lock:
            self._evaluation_calls += 1
            self._task_evaluations += len(batch)
            self._total_task_eval_cost += eval_cost
            self._total_task_eval_tokens_in += eval_tokens_in
            self._total_task_eval_tokens_out += eval_tokens_out

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
        rows.sort(key=lambda row: (1 if row.get("success") else 0, row.get("tool_calls", 0)))
        selected = rows[: self.max_reflection_tasks]
        reflective_rows: list[dict[str, Any]] = []
        for row in selected:
            reflective_rows.append(
                {
                    "Instance ID": row.get("instance_id"),
                    "Language": row.get("language"),
                    "Exercise": row.get("exercise_name"),
                    "Outcome": "resolved" if row.get("success") else "failed",
                    "Test Score": row.get("native_score"),
                    "Tool Calls": row.get("tool_calls"),
                    "Total Cost": row.get("total_cost", 0.0),
                    "Final Summary": row.get("final_summary", ""),
                    "Parser Error": row.get("parse_error"),
                    "Execution Error": row.get("exec_error"),
                    "Test Stdout Tail": row.get("test_stdout_tail", "") if _leak_test_output() else "",
                    "Test Stderr Tail": row.get("test_stderr_tail", "") if _leak_test_output() else "",
                    "Tool History": row.get("history", [])[-8:],
                    "Model Raw Output": row.get("raw_response", "")[:4000],
                }
            )
        return {component: reflective_rows}


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


def summarize_scores(eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    solved = int(sum(1 for score in eval_batch.scores if float(score) == 1.0))
    outputs = eval_batch.outputs or []
    total_cost = sum(float(obj.get("total_cost", 0.0)) for obj in outputs if isinstance(obj, dict))
    total_tokens_in = sum(int(obj.get("total_tokens_in", 0)) for obj in outputs if isinstance(obj, dict))
    total_tokens_out = sum(int(obj.get("total_tokens_out", 0)) for obj in outputs if isinstance(obj, dict))
    return {
        "solved": solved,
        "total": len(eval_batch.scores),
        "solve_rate": solved / len(eval_batch.scores) if eval_batch.scores else 0.0,
        "total_cost": total_cost,
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
    }


def diff_stats(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_calls": int(after.get("evaluation_calls", 0)) - int(before.get("evaluation_calls", 0)),
        "task_evaluations": int(after.get("task_evaluations", 0)) - int(before.get("task_evaluations", 0)),
        "total_cost": float(after.get("total_cost", 0.0)) - float(before.get("total_cost", 0.0)),
        "total_tokens_in": int(after.get("total_tokens_in", 0)) - int(before.get("total_tokens_in", 0)),
        "total_tokens_out": int(after.get("total_tokens_out", 0)) - int(before.get("total_tokens_out", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA Polyglot train-50 coding-policy optimization baseline")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--task-map", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--reflection-model", type=str, default=None)
    parser.add_argument("--reflection-provider", type=str, default=None)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-llm-calls-per-task", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--bash-timeout-sec", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-reflection-tasks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--docker-image", type=str, default=_DEFAULT_POLYGLOT_DOCKER_IMAGE)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_path = (args.dataset or default_dataset_path()).resolve()
    model_id = infer_litellm_model(args.model, args.provider)
    reflection_model = infer_litellm_model(
        args.reflection_model or args.model, args.reflection_provider or args.provider
    )
    reflection_lm = LM(reflection_model, temperature=0.0)
    ensure_docker_image_exists(args.docker_image)

    task_ids = load_task_ids(args.task_map.resolve()) if args.task_map else None
    examples = load_polyglot_examples(dataset_path, task_limit=args.task_limit, task_ids=task_ids)
    reflection_minibatch_size = len(examples)
    run_dir = args.run_dir or (
        Path("results/internal_ddl") / f"gepa_polyglot_{model_id.split('/')[-1].replace(':', '_')}_train{len(examples)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = PolyglotPolicyAdapter(
        model_id=model_id,
        workers=args.workers,
        max_llm_calls_per_task=args.max_llm_calls_per_task,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        bash_timeout_sec=args.bash_timeout_sec,
        temperature=args.temperature,
        max_reflection_tasks=args.max_reflection_tasks,
        random_seed=args.seed,
        docker_image=args.docker_image,
    )

    baseline_eval = adapter.evaluate(examples, {"coding_solver_policy": SEED_CODING_SOLVER_POLICY}, capture_traces=True)
    baseline_summary = summarize_scores(baseline_eval)
    stats_after_baseline = adapter.stats_snapshot()
    print(
        f"[baseline] solved={baseline_summary['solved']}/{baseline_summary['total']} "
        f"solve_rate={baseline_summary['solve_rate']:.1%} total_cost=${baseline_summary['total_cost']:.4f}"
    )

    callback = IterationMetricsLogger(run_dir)
    result = optimize(
        seed_candidate={"coding_solver_policy": SEED_CODING_SOLVER_POLICY},
        trainset=examples,
        valset=None,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        run_dir=str(run_dir),
        stop_callbacks=MaxCandidateProposalsStopper(args.generations),
        perfect_score=1.0,
        skip_perfect_score=False,
        seed=args.seed,
        display_progress_bar=False,
        callbacks=[callback],
    )
    stats_after_optimize = adapter.stats_snapshot()
    optimize_task_eval_summary = diff_stats(stats_after_optimize, stats_after_baseline)
    reflection_summary = {
        "model_id": reflection_model,
        "total_cost": float(getattr(reflection_lm, "total_cost", 0.0)),
        "total_tokens_in": int(getattr(reflection_lm, "total_tokens_in", 0)),
        "total_tokens_out": int(getattr(reflection_lm, "total_tokens_out", 0)),
    }

    best_candidate = result.best_candidate
    final_eval = adapter.evaluate(examples, best_candidate, capture_traces=True)
    final_summary = summarize_scores(final_eval)
    stats_after_final = adapter.stats_snapshot()
    final_task_eval_summary = diff_stats(stats_after_final, stats_after_optimize)
    overall_summary = {
        "total_cost": float(stats_after_final["total_cost"]) + reflection_summary["total_cost"],
        "total_tokens_in": int(stats_after_final["total_tokens_in"]) + reflection_summary["total_tokens_in"],
        "total_tokens_out": int(stats_after_final["total_tokens_out"]) + reflection_summary["total_tokens_out"],
        "task_eval_cost": float(stats_after_final["total_cost"]),
        "task_eval_tokens_in": int(stats_after_final["total_tokens_in"]),
        "task_eval_tokens_out": int(stats_after_final["total_tokens_out"]),
        "reflection_cost": reflection_summary["total_cost"],
        "reflection_tokens_in": reflection_summary["total_tokens_in"],
        "reflection_tokens_out": reflection_summary["total_tokens_out"],
        "evaluation_calls": int(stats_after_final["evaluation_calls"]),
        "task_evaluations": int(stats_after_final["task_evaluations"]),
    }
    print(
        f"[final] solved={final_summary['solved']}/{final_summary['total']} "
        f"solve_rate={final_summary['solve_rate']:.1%} total_cost=${final_summary['total_cost']:.4f} "
        f"overall_cost=${overall_summary['total_cost']:.4f}"
    )

    summary = {
        "benchmark": "polyglot",
        "dataset": str(dataset_path),
        "task_map": str(args.task_map.resolve()) if args.task_map else None,
        "model_id": model_id,
        "reflection_model": reflection_model,
        "generations": args.generations,
        "workers": args.workers,
        "max_llm_calls_per_task": args.max_llm_calls_per_task,
        "max_turns": args.max_turns,
        "max_tool_calls": args.max_tool_calls,
        "bash_timeout_sec": args.bash_timeout_sec,
        "max_reflection_tasks": args.max_reflection_tasks,
        "docker_image": args.docker_image,
        "task_limit": args.task_limit,
        "baseline": baseline_summary,
        "optimize_task_eval": optimize_task_eval_summary,
        "reflection": reflection_summary,
        "final": final_summary,
        "final_task_eval": final_task_eval_summary,
        "overall": overall_summary,
        "information_regime": _information_regime_summary(),
        "best_candidate": best_candidate,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "best_coding_solver_policy.txt").write_text(best_candidate["coding_solver_policy"].strip() + "\n")


if __name__ == "__main__":
    main()
