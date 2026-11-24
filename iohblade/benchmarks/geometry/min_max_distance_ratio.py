import inspect
import textwrap
from typing import Any

import numpy as np

from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import clean_local_namespace, prepare_namespace
from iohblade.problem import Problem


class MinMaxMinDistanceRatio(Problem):
    """
    Minimize (max_pairwise_distance / min_pairwise_distance)^2 for n points in R^d.
    We operate on squared distances to match the paper's reporting; score = ratio^2.
    Best Known Score is: 12.889266112 for 16 points in 2D and 4.165849767 for 14 points in 3D.
    """

    def __init__(
        self,
        n_points: int,
        dim: int,
        best_known: float | None = None,
        tolerance: float = 1e-12,
        best_solution: list[tuple] | None = None,
        eval_timeout: int = 600,
    ):
        super().__init__(
            name=f"min_max_min_distance_ratio-{dim}d", eval_timeout=eval_timeout
        )
        self.n_points = int(n_points)
        self.dim = int(dim)
        self.tolerance = float(tolerance)
        self.best_solution = best_solution

        # Guard against a bad "best_solution": drop it (on self, which is
        # what the prompts and evaluate() use) if its shape does not match.
        if self.best_solution is not None:
            if len(self.best_solution) != self.n_points:
                self.best_solution = None

        if self.best_solution is not None:
            for point in self.best_solution:
                if len(point) != self.dim:
                    self.best_solution = None
                    break

        if self.n_points < 2:
            raise ValueError("n_points must be >= 2")

        if self.dim < 1:
            raise ValueError("dim must be >= 1")

        self.best_known = best_known
        log_instantiated(
            f"Instantiated Min / Max distance ratio problem in {self.dim} dimensions, and best solution: {self.best_known}."
        )
        self.minimisation = True
        self.dependencies += ["scipy", "scikit-learn"]

        # Prompt declarations.....

        formula = "minimise [max_{i < j} d(i, j) / min{i < j} d(i, j)]^2"

        self.task_prompt = textwrap.dedent(f"""
* Write a python class with `__call__` method that:
    * Returns a set of {self.n_points} points, in {self.dim}-D hypervolume.
    * The Optimisation goal is to {formula}.
    * The tolerance for the point overlap is set to {self.tolerance}
""")
        if self.best_solution is None:
            example_code = """import numpy as np


class MinMaxDistanceSolver:
    def __init__(self, n_points: int, dimensions: int):
        self.n_points = n_points
        self.dimensions = dimensions

    def __call__(self) -> list[tuple[float, ...]]:
        # Uniform random points in the unit hypercube (distinct points,
        # so the minimum pairwise distance is non-zero).
        rng = np.random.default_rng(0)
        points = rng.random((self.n_points, self.dimensions))
        return [tuple(p) for p in points]
"""
        else:
            example_code = """import numpy as np


class MinMaxDistanceSolver:
    def __init__(
        self,
        n_points: int,
        dimensions: int,
        best_known_configuration: list[tuple[float, ...]] | None,
    ):
        # Accepts a best known configuration (if available) for the problem,
        # as an initial configuration, which can then be optimised further.
        self.n_points = n_points
        self.dimensions = dimensions
        self.best_known_configuration = best_known_configuration

    def __call__(self) -> list[tuple[float, ...]]:
        # Baseline: return the best known configuration unchanged.
        points = np.asarray(self.best_known_configuration, dtype=float)
        return [tuple(p) for p in points]
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

    @staticmethod
    def _pairwise_d2(P: np.ndarray) -> np.ndarray:
        """Squared distances of all pairs i < j (condensed form).

        Computed from coordinate differences, so precision does not depend on
        how far the points are from the origin (unlike the Gram-matrix form).
        """
        from scipy.spatial.distance import pdist

        return pdist(P, metric="sqeuclidean")

    def evaluate(self, solution, explogger=None):
        code = solution.code
        name = solution.name if solution.name else "MinMaxDistanceSolver"
        try:
            local_ns = {}
            safe = prepare_namespace(code, self.dependencies)
            compiled_code = compile(code, name, "exec")
            exec(compiled_code, safe, local_ns)

            cls = local_ns[name]
            if self.best_solution is not None:
                P = cls(self.n_points, self.dim, self.best_solution)()
            else:
                P = cls(self.n_points, self.dim)()
        except Exception as e:
            solution.set_scores(float("inf"), f"exec-error {e}", e)
            return solution

        try:
            P = np.asarray(P, dtype=np.float64)
            if P.ndim != 2 or P.shape != (self.n_points, self.dim):
                raise ValueError(f"expected shape (n={self.n_points}, d={self.dim})")
            if not np.isfinite(P).all():
                raise ValueError("non-finite coordinates")

            D2 = self._pairwise_d2(P)
            d2_min = float(np.min(D2))
            if not np.isfinite(d2_min) or d2_min <= self.tolerance:
                raise ValueError("minimum distance too small or zero")
            d2_max = float(np.max(D2))
            ratio_sq = d2_max / d2_min
            score = float(ratio_sq)
            if not np.isfinite(score):
                raise ValueError(f"non-finite ratio {score}")
            msg = f"ratio_sq={ratio_sq:.12g}."
            if self.best_known is not None:
                msg += f" Best known score is {self.best_known}."
            solution.set_scores(score, msg)
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", e)
        return solution

    def test(self, solution):
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
        tags.extend([Benchmark.MIN_MAX_DISTANCE_RATIO, VariableType.CONTINUOUS])

        extra_config = {
            "dimensions": self.dim,
            "n_points": self.n_points,
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
            "name": "Min Max Distance Ratio",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator,
            "config": extra_config,
        }
        return config


if __name__ == "__main__":
    mmd = MinMaxMinDistanceRatio(n_points=10, dim=2, best_known=0)
    for key, value in mmd.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
