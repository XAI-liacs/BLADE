import sys
import textwrap

import numpy as np
import pytest

from iohblade.problem import (
    HARNESS_ERROR_PREFIX,
    _build_eval_script,
    _eval_pythonpath,
)
from iohblade.solution import Solution
from iohblade.utils import HarnessError

from .harness_problems import CandidateBugProblem, GoodProblem, HarnessBugProblem


@pytest.fixture(scope="module")
def shared_env():
    """Build one evaluation virtualenv and reuse it for all tests here."""
    problem = GoodProblem()
    problem._ensure_env()
    yield problem._env_path, problem._python_bin
    problem.cleanup()


def _use_env(problem, shared_env):
    problem._env_path, problem._python_bin = shared_env
    problem.eval_timeout = 120
    return problem


def _solution(code="def run():\n    raise ValueError('bad candidate')\n"):
    return Solution(code=code, name="Candidate")


def test_good_problem_scores(shared_env):
    problem = _use_env(GoodProblem(), shared_env)
    sol = problem(_solution("x = 1"))
    assert sol.fitness == 5.0, sol.feedback


def test_candidate_error_is_not_a_harness_error(shared_env):
    problem = _use_env(CandidateBugProblem(), shared_env)
    for _ in range(problem.max_harness_errors + 1):
        sol = problem(_solution())
        assert sol.fitness == -np.inf
        assert "bad candidate" in sol.error
        assert not sol.error.startswith(HARNESS_ERROR_PREFIX)
    assert problem._consecutive_harness_errors == 0


def test_harness_error_aborts_after_limit(shared_env):
    problem = _use_env(HarnessBugProblem(), shared_env)
    problem.max_harness_errors = 2
    sol = problem(_solution())
    assert sol.fitness == -np.inf
    assert sol.error.startswith(HARNESS_ERROR_PREFIX)
    assert "bug in the evaluator" in sol.error
    with pytest.raises(HarnessError, match="2 consecutive evaluations"):
        problem(_solution())


def test_harness_error_is_not_swallowed_by_except_exception(shared_env):
    problem = _use_env(HarnessBugProblem(), shared_env)
    problem.max_harness_errors = 1
    with pytest.raises(HarnessError):
        try:
            problem(_solution())
        except Exception:  # what search methods typically do
            pass


def test_harness_limit_can_be_disabled(shared_env):
    problem = _use_env(HarnessBugProblem(), shared_env)
    problem.max_harness_errors = None
    for _ in range(4):
        sol = problem(_solution())
        assert sol.error.startswith(HARNESS_ERROR_PREFIX)


def test_failing_import_block_is_a_harness_error(shared_env):
    problem = _use_env(GoodProblem(), shared_env)
    problem.imports = "import numpy as np\nimport package_that_does_not_exist\n"
    sol = problem(_solution("x = 1"))
    assert sol.error.startswith(HARNESS_ERROR_PREFIX)
    assert "package_that_does_not_exist" in sol.error


def test_success_resets_harness_counter(shared_env):
    problem = _use_env(GoodProblem(), shared_env)
    problem._consecutive_harness_errors = 2
    problem(_solution("x = 1"))
    assert problem._consecutive_harness_errors == 0


def test_env_setup_failure_is_a_harness_error(monkeypatch):
    problem = GoodProblem()
    problem.max_harness_errors = 2

    def broken_env():
        raise RuntimeError("Failed to install the dependencies ['nope']")

    monkeypatch.setattr(problem, "_ensure_env", broken_env)
    sol = problem(_solution("x = 1"))
    assert sol.error.startswith(HARNESS_ERROR_PREFIX)
    with pytest.raises(HarnessError):
        problem(_solution("x = 1"))


def test_problem_from_external_module_needs_extra_pythonpath(
    shared_env, tmp_path, monkeypatch
):
    module_dir = tmp_path / "user_code"
    module_dir.mkdir()
    (module_dir / "blade_user_problem_mod.py").write_text(textwrap.dedent("""
            from typing import Any
            from iohblade.problem import Problem


            class UserProblem(Problem):
                def evaluate(self, solution):
                    solution.set_scores(42.0)
                    return solution

                def test(self, solution):
                    return solution

                def to_dict(self):
                    return {}

                def get_config(self) -> dict[str, Any]:
                    return {}
            """))
    monkeypatch.syspath_prepend(str(module_dir))
    import blade_user_problem_mod

    try:
        problem = _use_env(blade_user_problem_mod.UserProblem(), shared_env)
        sol = problem(_solution("x = 1"))
        # The venv cannot import the module: reported as a harness error.
        assert sol.error.startswith(HARNESS_ERROR_PREFIX), sol.error
        assert "blade_user_problem_mod" in sol.error

        problem.extra_pythonpath = [str(module_dir)]
        sol = problem(_solution("x = 1"))
        assert sol.fitness == 42.0, sol.feedback
    finally:
        sys.modules.pop("blade_user_problem_mod", None)


def test_eval_pythonpath_order(tmp_path):
    problem = GoodProblem()
    problem.extra_pythonpath = [str(tmp_path), str(tmp_path)]
    parts = _eval_pythonpath(problem, "existing").split(__import__("os").pathsep)
    assert parts[1] == str(tmp_path.resolve())
    assert parts[-1] == "existing"
    assert len(parts) == 3  # duplicates removed


def test_eval_pythonpath_without_attribute():
    class Legacy:
        pass  # e.g. unpickled from an older BLADE version

    assert _eval_pythonpath(Legacy(), "")


def test_future_imports_stay_first():
    script = _build_eval_script(
        "from __future__ import annotations\nimport numpy as np",
        "p.pkl",
        "s.pkl",
        "r.pkl",
        "m.txt",
    )
    assert script.startswith("from __future__ import annotations")
    compile(script, "script", "exec")
