"""Problems used by test_harness_errors.py.

They live in their own module without a pytest import, because the
evaluation subprocess unpickles them in a virtualenv without pytest.
"""

from typing import Any

from iohblade.problem import Problem


class _BaseTestProblem(Problem):
    def test(self, solution):
        return solution

    def to_dict(self):
        return {}

    def get_config(self) -> dict[str, Any]:
        return {}


class HarnessBugProblem(_BaseTestProblem):
    """Fails before using the generated code (a bug in the problem)."""

    def evaluate(self, solution):
        raise RuntimeError("bug in the evaluator")


class CandidateBugProblem(_BaseTestProblem):
    """Runs the generated code, which raises."""

    def evaluate(self, solution):
        namespace = {}
        exec(solution.code, namespace)
        namespace["run"]()
        solution.set_scores(1.0)
        return solution


class GoodProblem(_BaseTestProblem):
    """Scores the length of the generated code."""

    def evaluate(self, solution):
        solution.set_scores(float(len(solution.code)))
        return solution
