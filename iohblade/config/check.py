"""Smoke tests for an experiment (``iohblade check``).

Before spending hours of LLM time, ``iohblade check`` verifies that every
part of an experiment works end to end, without calling an LLM:

1. **Problems**: the evaluation environment is built (or reused from the
   cache) and a known-good *reference solution* is evaluated through the
   real pipeline (subprocess, virtualenv, timeout). The reference is the
   problem's ``smoke_test_solution`` file, or else the code block in its
   example prompt. The check fails on harness errors, on errors in the
   reference solution and warns on non-finite scores or slow evaluations.
2. **Methods**: each method runs with a tiny budget and an offline LLM that
   always answers with the reference solution, on the first problem that
   passed.
3. **LLMs** (``--llm``): each configured LLM is sent one short prompt, to
   check API keys, model names and connectivity.
"""

from __future__ import annotations

import contextlib
import copy
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..llm import LLM
from ..problem import HARNESS_ERROR_PREFIX
from ..solution import Solution
from ..utils import first_class_name

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_CODE = re.compile(r"```(?:python)?\n(.*?)\n```", re.S)


@dataclass
class CheckResult:
    """Outcome of one check.

    Attributes:
        kind: ``"problem"``, ``"method"`` or ``"llm"``.
        name: Name of the component.
        status: ``"ok"``, ``"warn"``, ``"fail"`` or ``"skip"``.
        message: What was checked and what happened.
        seconds: Time the check took.
    """

    kind: str
    name: str
    status: str
    message: str
    seconds: float = 0.0

    def __str__(self) -> str:
        label = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}[self.status]
        return (
            f"[{label}] {self.kind} {self.name}: {self.message} ({self.seconds:.1f}s)"
        )


class ReplayLLM(LLM):
    """Offline LLM that always answers with the same code (for checks)."""

    def __init__(self, code: str, model: str = "smoke-test"):
        super().__init__("", model)
        self.code = code
        name = first_class_name(code) or "Solution"
        self.response = (
            f"# Description: {name}, the reference solution of the smoke test.\n"
            f"# Code:\n```python\n{code}\n```\n"
        )

    def _query(self, session, **kwargs):
        return self.response

    def get_config(self):
        return [{"model": self.model, "provider": "replay"}]


def reference_solution(problem, smoke: Path | None) -> tuple[str | None, str]:
    """The code to evaluate for ``problem`` and where it came from."""
    if smoke is not None:
        return smoke.read_text(encoding="utf-8"), f"smoke_test_solution {smoke.name}"
    match = _CODE.search(getattr(problem, "example_prompt", "") or "")
    if match:
        return match.group(1), "the example in the problem's prompt"
    return None, ""


def _score(fitness) -> float | None:
    try:
        return float(fitness)
    except (TypeError, ValueError):
        return None  # e.g. multi-objective fitness


def _compact(text: str, lines: int = 8) -> str:
    """First line plus the last ``lines`` error lines of installer output."""
    parts = [p.rstrip() for p in (text or "").strip().splitlines() if p.strip()]
    if len(parts) <= lines + 1:
        return "\n    ".join(parts)
    errors = [
        p
        for p in parts[1:]
        if "error" in p.lower() or p.lstrip().startswith(("hint", "×", "╰"))
    ]
    shown = (errors or parts[1:])[-lines:]
    return "\n    ".join(
        [parts[0], *shown, "(run `iohblade check -v` for the full output)"]
    )


def _tail(text: str, lines: int = 6) -> str:
    parts = [p for p in (text or "").strip().splitlines() if p.strip()]
    return " | ".join(parts[-lines:])


def check_problem(
    problem, smoke: Path | None, verbose: bool = False
) -> tuple[CheckResult, str | None]:
    """Build the environment and evaluate the reference solution.

    Returns:
        tuple: The result and the reference code (None if unavailable).
    """
    name = problem.name
    start = time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            problem._ensure_env()
    except Exception as e:
        return (
            CheckResult(
                "problem",
                name,
                FAIL,
                "building the evaluation environment failed: "
                + (str(e) if verbose else _compact(str(e))),
                time.monotonic() - start,
            ),
            None,
        )
    env_seconds = time.monotonic() - start
    code, origin = reference_solution(problem, smoke)
    if code is None:
        return (
            CheckResult(
                "problem",
                name,
                WARN,
                f"environment ready ({env_seconds:.1f}s), but no reference "
                "solution to evaluate: add 'smoke_test_solution' (a .py file "
                "with a working solution) to this problem in the YAML file",
                env_seconds,
            ),
            None,
        )
    evaluated = copy.deepcopy(problem)
    evaluated.logger = None
    evaluated.max_harness_errors = None  # report, do not raise
    solution = Solution(code=code, name=first_class_name(code) or "Reference")
    eval_start = time.monotonic()
    result = evaluated(solution)
    seconds = time.monotonic() - eval_start
    total = time.monotonic() - start
    error = getattr(result, "error", "") or ""
    if error.startswith(HARNESS_ERROR_PREFIX):
        return (
            CheckResult(
                "problem",
                name,
                FAIL,
                "the evaluation fails before the solution is used (a problem "
                f"with the benchmark or its environment): "
                f"{_tail(error[len(HARNESS_ERROR_PREFIX):])}",
                total,
            ),
            code,
        )
    if error:
        return (
            CheckResult(
                "problem",
                name,
                FAIL,
                f"evaluating {origin} failed: {_tail(error)}",
                total,
            ),
            code,
        )
    score = _score(result.fitness)
    timeout = getattr(problem, "eval_timeout", None)
    message = f"evaluated {origin} in {seconds:.1f}s"
    if score is not None:
        message += f", fitness {score:.6g}"
    if score is not None and not math.isfinite(score):
        feedback = _tail(getattr(result, "feedback", "") or "", 3)
        return (
            CheckResult(
                "problem",
                name,
                WARN,
                message
                + " (not finite: is the reference solution valid?"
                + (f" Feedback: {feedback}" if feedback else "")
                + ")",
                total,
            ),
            code,
        )
    if timeout and seconds > 0.5 * timeout:
        return (
            CheckResult(
                "problem",
                name,
                WARN,
                message + f" (over half of eval_timeout={timeout}s)",
                total,
            ),
            code,
        )
    return CheckResult("problem", name, OK, message, total), code


def check_method(method, problem, code: str, budget: int = 2) -> CheckResult:
    """Run ``method`` briefly with an LLM that replays ``code``."""
    start = time.monotonic()
    trial = copy.deepcopy(method)
    trial.llm = ReplayLLM(code)
    trial.budget = budget
    target = copy.deepcopy(problem)
    target.logger = None
    target.max_harness_errors = None
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            best = trial(target)
    except Exception as e:
        return CheckResult(
            "method",
            method.name,
            FAIL,
            f"running on '{problem.name}' with budget {budget} raised "
            f"{type(e).__name__}: {e}",
            time.monotonic() - start,
        )
    seconds = time.monotonic() - start
    if not hasattr(best, "fitness"):  # LLaMEA returns llamea.Solution
        return CheckResult(
            "method",
            method.name,
            WARN,
            f"ran on '{problem.name}' but returned {type(best).__name__}, "
            "not a Solution",
            seconds,
        )
    score = _score(best.fitness)
    if score is not None and not math.isfinite(score):
        return CheckResult(
            "method",
            method.name,
            WARN,
            f"ran on '{problem.name}', but its best solution has fitness "
            f"{score}: {_tail(best.error or best.feedback)}",
            seconds,
        )
    return CheckResult(
        "method",
        method.name,
        OK,
        f"ran on '{problem.name}' with budget {budget}",
        seconds,
    )


def ping_llm(name: str, llm) -> CheckResult:
    """Send one short prompt to an LLM."""
    start = time.monotonic()
    try:
        answer = llm.query(
            [{"role": "user", "content": "Reply with the single word OK."}]
        )
    except Exception as e:
        return CheckResult(
            "llm",
            name,
            FAIL,
            f"query failed: {type(e).__name__}: {e}",
            time.monotonic() - start,
        )
    seconds = time.monotonic() - start
    text = str(answer or "").strip().replace("\n", " ")
    if not text:
        return CheckResult("llm", name, WARN, "returned an empty answer", seconds)
    return CheckResult("llm", name, OK, f"answered {text[:60]!r}", seconds)


def run_checks(
    plan,
    *,
    methods: bool = True,
    llms: bool = False,
    report: Callable[[CheckResult], None] | None = None,
    verbose: bool = False,
) -> list[CheckResult]:
    """Run all checks of an :class:`~iohblade.config.ExperimentPlan`.

    Args:
        plan: The plan to check.
        methods: Also run every method briefly.
        llms: Also send one prompt to every configured LLM.
        report: Called with each result as soon as it is known.
        verbose: Show full installer output in environment errors.

    Returns:
        list[CheckResult]: All results, in the order they were run.
    """
    results: list[CheckResult] = []

    def add(result):
        results.append(result)
        if report is not None:
            report(result)

    reference = None  # (problem, code) for the method checks
    smoke_tests = plan.smoke_tests or [None] * len(plan.problems)
    try:
        for problem, smoke in zip(plan.problems, smoke_tests):
            result, code = check_problem(problem, smoke, verbose)
            add(result)
            if reference is None and result.status in (OK, WARN) and code:
                reference = (problem, code)
        if methods:
            for method in plan.methods:
                if reference is None:
                    add(
                        CheckResult(
                            "method",
                            method.name,
                            SKIP,
                            "no problem with a working reference solution to run on",
                        )
                    )
                    continue
                add(check_method(method, reference[0], reference[1]))
        if llms:
            for name, llm in plan.llms.items():
                add(ping_llm(name, llm))
    finally:
        for problem in plan.problems:
            problem.cleanup()
    return results
