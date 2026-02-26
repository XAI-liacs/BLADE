import math
from typing import Optional
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace


class SphericalCode(Problem):
    """
    Sherical Code is a `maximisation` problem where 30 distinct points are projected on to a
    unit sphere, and the optimisation goal is to maximise the minimum pairwise angle.
    Best Known: n=30: 0.6736467551690225
    """

    def __init__(
        self,
        n_points: int = 30,
        best_known: Optional[float] = 0.6736467551690225,
        tolerance: float = 1e-12,
        logger=None,
    ):
        self.n_points = n_points
        self.best_known = best_known
        self.tolerance = tolerance
        self.minimisation = False
        self.logger = logger
        Problem.__init__(
            self,
            logger=logger,
            name=f"spherical-code-n30",
            dependencies=["scipy", "shapely"],
        )
        ## Set prompts:
        self.task_prompt = (
            f"""
Write a python class with function `__call__`, that generate a solution for Spherical Code Problem on a unit sphere.
- The class must initialise with 2 positional parameters:
    1. n_points: Number of 3-D points that __call__ returns.
    2. tolerance: The minimum allowed distance between any 2 points on sphere.
- The `__call__` method must return:
    - `points : list[tuple[float, float, float]]`: A list of {self.n_points} 3-D points as solution to the problem.
- The optimisation goal is to maximize the minimum pairwise angle given by:

"""
            + "\\[\\theta_{\\min} = \\min_{i < j} \\cos^{-1}(\\braket{p_i, p_j})\\]"
        )

        self.example_prompt = f"""
An example response can be
---
# Descripition:
A random selection algorithm for Spherical Code Solver.
# Code:
```python
import random

class SphericalCodeSolver:
    def __init__(self, n_points, tolerance):
        self.n_points = n_points
        self.tolerance = tolerance

    def __call__(self) -> list[tuple[float, float, float]]:
        points = []
        for _ in range(self.n_points):
            points.append(
                (random.random(), randm.random(), random.random())
            )
        return points
```
"""
        self.format_prompt = """
Give an excellent and novel algorithm to solve this task and also give it a
one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

    def _check_dimension(self, points: list[tuple[float, float, float]]):
        if len(points) != self.n_points:
            raise ValueError(f"Expeceted {self.n_points} points, got {len(points)}.")
        for point in points:
            if len(point) != 3:
                raise ValueError(
                    f"Expected each point to be 3-D, got {point}, which is {len(point)}D."
                )

    def _get_angle(
        self, point1: tuple[float, float, float], point2: tuple[float, float, float]
    ) -> float:
        dot_product = 0
        for i in range(3):
            dot_product += point1[i] * point2[i]
            dot_product = max(-1.0, min(1.0, dot_product))
        try:
            return math.acos(dot_product)
        except:
            raise ValueError(f"Got domain error for point {point1} . {point2}.")

    def _get_min_angle(self, points: list[tuple[float, float, float]]):
        min_angle = float("inf")
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                theta = self._get_angle(points[i], points[j])
                min_angle = min(min_angle, theta)
        return min_angle

    def _get_unit_vetor(
        self, point: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        magnitude = 0
        for i in range(3):
            magnitude += point[i] ** 2
        magnitude = magnitude**0.5
        return (
            point[0] / magnitude,
            point[1] / magnitude,
            point[2] / magnitude,
        )

    def _check_tolerance(self, points: list[tuple[float, float, float]]):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = (
                    (points[i][0] - points[j][0]) ** 2
                    + (points[i][1] - points[j][1]) ** 2
                    + (points[i][2] - points[j][2]) ** 2
                ) ** 0.5
                if distance < self.tolerance:
                    raise ValueError(
                        f"Point {points[i]} and {points[j]} are closer than allowed tolerance on sphere."
                    )

    def evaluate(self, solution: Solution, logger=None):
        name = solution.name if solution.name else "SphericalCode"
        code = solution.code

        try:
            if code:
                local_ns = {}
                global_ns = prepare_namespace(code, allowed=self.dependencies)
                compiled_code = compile(code, name, "exec")

                exec(compiled_code, global_ns, local_ns)
                cls = local_ns[name]
                points = cls(self.n_points, self.tolerance)()

                self._check_dimension(points)
                points = [self._get_unit_vetor(point) for point in points]
                self._check_tolerance(points)
                min_angle = self._get_min_angle(points)
                solution = solution.set_scores(
                    min_angle,
                    f"Got min angle {min_angle}, best known is {self.best_known}.",
                )
            else:
                raise NotImplementedError(
                    "Code not extractable, make sure code is encased in code block: ```."
                )
        except Exception as e:
            solution = solution.set_scores(
                -float("inf"), feedback=f"Got an error: {e}", error=e
            )
        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    sc = SphericalCode()
    print(sc.get_prompt())
