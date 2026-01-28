from typing import Optional

from iohblade.benchmarks.geometry.geometry_base_class import GeometryBase
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace
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
        best_known: Optional[float],
        tolerance: float = 1e-12,
        best_solution: list[tuple[float, float]] | None = None,
        triangle_best_solution: list[tuple[float, float]] | None = None,
    ):
        GeometryBase.__init__(
            self,
            task_name=f"heilbronn_triangle-n{n_points}",
            n_points=n_points,
            tolerance=tolerance,
            best_known=best_known if best_known is not None else float("-inf"),
        )
        Problem.__init__(self, name=f"heilbronn_triangle-n{n_points}")

        if best_solution:
            if triangle_best_solution is None:
                self.triangle_best_solution = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
            else:
                self.triangle_best_solution = triangle_best_solution
        else:
            self.triangle_best_solution = None
        self.best_solution = best_solution

        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Heibronn Triangle Problem with number of points: {self.n_points}, and best solution: {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
        )

        self.task_prompt = """
Write a python class with function `__call__`, that generate a solution for Heilbronn on a unit area triangle.
- The `__call__` method may return:
  - (None, points) where points is ndarray (n,2) interpreted inside a default unit-area triangle, or
  - (triangle, points): with triangle shape (3,2), both of which we rescale similarly as to have area of triangle = 1.
    - Upon scaling points must lie inside the tringle, within the given tolerance.
- The optimisation goal is to maximise the area of the smallest triangle, formed by picking 3 of the n points.
"""
        self.task_prompt += (
            f"- The tolerence of the solution is set to {self.tolerance}"
        )

        best_known_initialiser = """
    def __init__(self, n_points : int):
        pass
"""
        if self.best_solution is not None:
            best_known_initialiser = """
    def __init__(self, n_points: int, best_known_configuration: list[tuple[float, float]] | None, in_triangle: list[tuple[float, float]]):
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        optimised for better results.
        pass
"""

        call_format = f"""
def __call__(self):
    # Option 1: points only
    return (None, np.zeros(({self.n_points}, 2)))

    # Option 2: (triangle, points)
    return (np.zeros((3, 2)), np.zeros(({self.n_points}, 2)))
"""

        self.example_prompt = f"""
Must follow the following template for code:
Description: A short one line description of technique used.

```python

class HeilbronnTriangle-n{self.n_points}:
    
    {best_known_initialiser}

    {call_format}

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
        self.minimisation = False
        self.dependencies += ["scipy", "shapely"]

    def evaluate(self, solution, explogger=None):
        code = solution.code
        try:
            safe = prepare_namespace(code, self.dependencies)
            local_ns = {}
            exec(code, safe, local_ns)
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
                "exec-failed",
            )
            return solution

        try:
            if triangle:
                T, P = self._parse_candidate((triangle, points))
            else:
                T, P = self._parse_candidate(points)
            T = self._ensure_unit_area(self.to_np_points(T, expected_n=3))
            P = self.to_np_points(P, expected_n=self.n_points)

            a, b, c = T[0], T[1], T[2]
            for i, p in enumerate(P):
                if not self.point_in_triangle(p, a, b, c, tol=self.tolerance):
                    raise ValueError(f"point indexed {i}--{p}--outside triangle")

            min_area = self.min_triangle_area(P, tol=self.tolerance)
            score = float(min_area)  # maximize
            solution.set_scores(
                score,
                f"Area of Smallest Triangle={min_area:.6g}, best known={self.best_known}",
            )
        except Exception as e:
            solution.set_scores(
                float("-inf"),
                f"calc-error {e}.",
                f"Values Returned: Triangle {triangle}, points: {points}",
            )
        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    hbt = HeilbronnTriangle(n_points=10, best_known=1.11)
    print(hbt.get_prompt())
