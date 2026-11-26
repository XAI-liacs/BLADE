import inspect
import textwrap
from typing import Any, Optional

import numpy as np

from iohblade.benchmarks.geometry.geometry_base_class import GeometryBase
from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import clean_local_namespace, prepare_namespace
from iohblade.problem import Problem


class HeilbronnTriangle(GeometryBase, Problem):
    """
    Heilbronn on a unit-area triangle (Appendix B.9).
    Candidate may return:
      - points: ndarray (n,2) interpreted inside a default unit-area triangle, or
      - (triangle, points) with triangle shape (3,2), which we rescale to area 1, or
      - {'triangle': tri, 'points': pts}.
    Score = minimum triangle area among the n points (maximize).
    Best Known = 0.0365 for n = 11
    """

    def __init__(
        self,
        n_points: int,
        best_known: Optional[float] = None,
        tolerance: float = 1e-12,
        best_solution: list[tuple[float, float]] | None = None,
        triangle_best_solution: list[tuple[float, float]] | None = None,
        eval_timeout: int = 600,
    ):
        GeometryBase.__init__(
            self,
            task_name=f"heilbronn_triangle-n{n_points}",
            n_points=n_points,
            tolerance=tolerance,
            best_known=best_known if best_known is not None else float("-inf"),
        )
        Problem.__init__(
            self, name=f"heilbronn_triangle-n{n_points}", eval_timeout=eval_timeout
        )

        if best_solution:
            if triangle_best_solution is None:
                self.triangle_best_solution = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
            else:
                self.triangle_best_solution = triangle_best_solution
        else:
            self.triangle_best_solution = None
        self.best_solution = best_solution

        log_instantiated(
            f"Instantiated Heibronn Triangle Problem with number of points: {self.n_points}, and best solution: {self.best_known}."
        )

        self.task_prompt = textwrap.dedent("""
Write a python class with function `__call__`, that generate a solution for Heilbronn on a unit area triangle.
- The `__call__` method may return:
  - (None, points) where points is ndarray (n,2) interpreted inside a default unit-area triangle, or
  - (triangle, points): with triangle shape (3,2), both of which we rescale similarly as to have area of triangle = 1.
    - Upon scaling points must lie inside the tringle, within the given tolerance.
- The optimisation goal is to maximise the area of the smallest triangle, formed by picking 3 of the n points.
""")
        self.task_prompt += (
            f"- The tolerence of the solution is set to {self.tolerance}"
        )

        if self.best_solution is None:
            example_code = """import numpy as np


class HeilbronnTriangleSolver:
    def __init__(self, n_points: int):
        self.n_points = n_points

    def __call__(self):
        # Uniform random points inside the default unit-area triangle
        # (0, 0), (1, 0), (0, 2).
        rng = np.random.default_rng(0)
        u = rng.random((self.n_points, 2))
        flip = u.sum(axis=1) > 1.0
        u[flip] = 1.0 - u[flip]
        points = np.column_stack([u[:, 0], 2.0 * u[:, 1]])
        # Option 1: (None, points) uses the default triangle.
        # Option 2: return (triangle, points) with triangle of shape (3, 2).
        return (None, points)
"""
        else:
            example_code = """import numpy as np


class HeilbronnTriangleSolver:
    def __init__(
        self,
        n_points: int,
        best_known_configuration: list[tuple[float, float]] | None,
        in_triangle: list[tuple[float, float]],
    ):
        # Accepts a best known configuration (if available) for the problem,
        # as an initial configuration, which can then be optimised further.
        self.n_points = n_points
        self.best_known_configuration = best_known_configuration
        self.in_triangle = in_triangle

    def __call__(self):
        # Baseline: return the best known configuration unchanged, as
        # (triangle, points) with triangle of shape (3, 2).
        triangle = np.asarray(self.in_triangle, dtype=float)
        points = np.asarray(self.best_known_configuration, dtype=float)
        return (triangle, points)
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
        self.dependencies += ["scipy", "shapely"]

    def evaluate(self, solution, explogger=None):
        code = solution.code
        name = solution.name
        try:
            safe = prepare_namespace(code, self.dependencies)
            local_ns = {}
            compiled_code = compile(code, filename=name, mode="exec")

            exec(compiled_code, safe, local_ns)
            cls = local_ns[solution.name]
            if self.best_solution is None:
                triangle, points = cls(self.n_points)()
            else:
                triangle, points = cls(
                    self.n_points,
                    best_known_configuration=self.best_solution,
                    in_triangle=self.triangle_best_solution,
                )()
        except Exception as e:
            # tb = e.__traceback__
            solution.set_scores(
                float("-inf"),
                f"exec-error {e}",
                e,
            )
            return solution

        try:
            if triangle is not None:
                T, P = self._parse_candidate((triangle, points))
            else:
                T, P = self._parse_candidate(points)
            T = self.to_np_points(T, expected_n=3)
            P = self.to_np_points(P, expected_n=self.n_points)
            # Rescale the triangle and the points by the same factor, so the
            # triangle has area 1 and the configuration keeps its shape.
            area = abs(self.polygon_area(T))
            if not area > 0:
                raise ValueError("triangle area is zero")
            scale = 1.0 / np.sqrt(area)
            T = T * scale
            P = P * scale

            a, b, c = T[0], T[1], T[2]
            for i, p in enumerate(P):
                if not self.point_in_triangle(p, a, b, c, tol=self.tolerance):
                    raise ValueError(f"point indexed {i}--{p}--outside triangle")

            min_area = self.min_triangle_area(P, tol=self.tolerance)
            score = float(min_area)  # maximize
            if not np.isfinite(score):
                raise ValueError(f"non-finite minimum area {score}")
            solution.set_scores(
                score,
                f"Area of Smallest Triangle={min_area:.6g}"
                + self.best_known_feedback(self.best_known),
            )
        except Exception as e:
            solution.set_scores(
                float("-inf"),
                f"calc-error, for values returned by candidate: Triangle {triangle}, points: {points}",
                e,
            )
        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

    def get_config(self) -> dict[str, Any]:
        from iohblade.tags import Benchmark, VariableType

        extra_config = {
            "n_points": self.n_points,
            "tolerance": self.tolerance,
            "dependencies": self.dependencies,
        }

        evaluator = "\n\n".join(
            [
                inspect.getsource(self._parse_candidate),
                inspect.getsource(self.polygon_area),
                inspect.getsource(self.to_np_points),
                inspect.getsource(self.point_in_triangle),
                inspect.getsource(self.min_triangle_area),
                inspect.getsource(self.evaluate),
            ]
        )

        self.tags.extend([Benchmark.HEILBRONN_TRIANGLE, VariableType.CONTINUOUS])

        config = {
            "tags": self.tags,
            "name": "Heilbronn Triangle",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator,
            "config": extra_config,
        }
        return config


if __name__ == "__main__":
    hbt = HeilbronnTriangle(n_points=10, best_known=1.11)
    for key, value in hbt.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
