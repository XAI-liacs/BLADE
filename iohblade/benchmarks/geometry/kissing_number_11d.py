import inspect
import math
import random
import textwrap
from typing import Any

import numpy as np

from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import clean_local_namespace, prepare_namespace
from iohblade.problem import Problem
from iohblade.solution import Solution


class KissingNumber11D(Problem):
    """
    Kissing number lower bound in 11D via Lemma 1 (AlphaEvolve Appendix B.11).

    Candidate returns an array C with shape (m, 11) of non-zero points.
    Feasible iff   min_{x != y} ||x - y|| >= max_x ||x||   and   0 \notin C.
    Score (to maximize) = m = |C|.
    Best Known: 593.

    Notes:
    - The AlphaEvolve lemma does not require integrality. It only needs the inequality
      min_{x != y} ||x - y|| >= max_x ||x|| and 0 \notin C. We therefore do not enforce
      integer coordinates.
    """

    def __init__(
        self,
        tolerance: float = 0.0,
        best_known: int = 593,
        best_solution: list[list[int]] | None = None,
        eval_timeout: int = 600,
    ):
        super().__init__(name="kissing_number_11d", eval_timeout=eval_timeout)
        self.dim = 11
        self.tolerance = float(tolerance)
        self.best_solution = best_solution
        self.best_known = best_known
        log_instantiated(
            f"Instantiated Kissing Number {self.dim}D, and best solution: {self.best_known}."
        )
        self.dependencies += ["scipy"]
        self.minimisation = False

        self.task_prompt = (
            """
Write a python class with function `__call__`, that generate a solution for the """
            + f"{self.dim}-D Kissing Number problem."
            + """
- The `__call__` method must return n points as array of """
            + f"{self.dim} dimensional integer tuples."
            + r"""
- The solution is scored as n = |(C\subset\mathbb{Z}^{11}\setminus{0})| where:
    - (\min_{x\ne y}|x-y|\ge \max_x|x|)
- The optimisation goal is to maximise the score.
"""
        )
        # - The environment only provides access to the libraries:
        #     - {"\n    - ".join(allowed)}
        if self.best_solution is None:
            example_code = f"""import itertools


class KissingNumberSolver:
    def __init__(self):
        self.dim = {self.dim}

    def __call__(self):
        # All integer vectors with exactly two non-zero entries equal to +-1.
        # Each has norm sqrt(2) and any two differ by at least sqrt(2), so
        # min ||x - y|| >= max ||x|| holds. Maximise the number of points.
        points = []
        for i, j in itertools.combinations(range(self.dim), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    p = [0] * self.dim
                    p[i] = si
                    p[j] = sj
                    points.append(tuple(p))
        return points
"""
        else:
            example_code = """class KissingNumberSolver:
    def __init__(self, best_known_configuration: list[list[int]] | None):
        # Accepts a best known configuration (if available) for the problem,
        # as an initial configuration, which can then be optimised further.
        self.best_known_configuration = best_known_configuration

    def __call__(self):
        # Baseline: return the best known configuration unchanged.
        return [tuple(p) for p in self.best_known_configuration]
"""

        self.example_prompt = (
            "\nMust follow the following template for code:\n"
            "# Description: A short one line description of technique used.\n"
            "# Code:\n"
            f"```python\n{example_code}```\n"
        )

        self.format_prompt = textwrap.dedent("""

Give an excellent and novel algorithm to solve this task and also give it a
one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```

""")
        self.minimisation = False

    @staticmethod
    def _pairwise_d2(P: np.ndarray) -> np.ndarray:
        """Squared distances with +inf on diagonal."""
        G = P @ P.T
        n2 = np.sum(P * P, axis=1, keepdims=True)
        D2 = n2 + n2.T - 2.0 * G
        np.fill_diagonal(D2, np.inf)
        D2[D2 < 0] = 0.0
        return D2

    def evaluate(self, solution: Solution, explogger=None) -> Solution:
        code = solution.code
        name = solution.name if solution.name else "KissingNumber11D"

        try:
            local_ns = {}
            safe_globals = prepare_namespace(code, allowed=self.dependencies)
            compiled_code = compile(code, name, "exec")

            exec(compiled_code, safe_globals, local_ns)

            cls = local_ns[name]
            if self.best_solution is not None:
                C = np.array(cls(self.best_solution)(), dtype=float)
            else:
                C = np.array(cls()(), dtype=float)

            if C.ndim != 2 or C.shape[1] != self.dim:
                raise ValueError(f"expected shape (m, {self.dim})")
            if not np.isfinite(C).all():
                raise ValueError("non-finite coordinates")
            try:
                norms2 = np.sum(C * C, axis=1)
            except Exception as e:
                raise ValueError(f"Possibly jagged series: {c}. Got error {e}")
            if np.any(norms2 <= self.tolerance):
                raise ValueError("zero vector present")

            D2 = self._pairwise_d2(C)
            d2_min = float(np.min(D2))
            r2_max = float(np.max(norms2))
            if d2_min + 1e-15 < r2_max - self.tolerance:
                raise ValueError("lemma condition violated")

            m = int(C.shape[0])
            solution.set_scores(float(m), f"|C|={m}")
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", e)
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

    def get_config(self) -> dict[str, Any]:
        from iohblade.tags import (
            Benchmark,
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            StructureTag,
            VariableType,
        )

        tags: list[Any] = [
            PrimaryCategories.CO,
            StructureTag.PACKING,
            ObjectiveType.SINGLE_OBJECTIVE,
            NoiseType.NOISELESS,
        ]
        tags.extend([Benchmark.KISSING_NUMBER_11D, VariableType.DISCRETE])

        extra_config = {
            "dimensions": self.dim,
            "tolerance": self.tolerance,
            "dependencies": self.dependencies,
        }

        evaluator = "\n\n".join(
            [
                inspect.getsource(self._pairwise_d2),
                inspect.getsource(self.evaluate),
            ]
        )

        config = {
            "tags": tags,
            "name": "Kissing Number",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator,
            "config": extra_config,
        }
        return config


if __name__ == "__main__":
    kiss = KissingNumber11D()
    for key, value in kiss.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
