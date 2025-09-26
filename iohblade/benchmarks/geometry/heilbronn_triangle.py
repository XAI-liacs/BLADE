# import traceback

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
    """

    def __init__(self, n_points: int, tolerance: float = 1e-12):
        GeometryBase.__init__(
            self,
            task_name=f"heilbronn_triangle-n{n_points}",
            n_points=n_points,
            tolerance=tolerance,
        )
        Problem.__init__(self, name=f"heilbronn_triangle-n{n_points}")

        self.task_prompt = """
Write a python class with function `__call__`, that generate a solution for Heilbronn on a unit area triangle.
- The `__call__` method may return:
  - points (`pts`): ndarray (n,2) interpreted inside a default unit-area triangle, or
  - (triangle, points) (`(tri, pts)`): with triangle shape (3,2), which we rescale to area 1, and `pts` from above, or
  - A dictionary of {'triangle': tri, 'points': pts}.
- The solution is scored as minimum triangle area formed by picking 3 of the n points.
- The optimisation goal is to maximise the score.
"""
        self.task_prompt += (
            f"- The tolerence of the solution is set to {self.tolerance}"
        )

        call_format = f"""
def __call__(self):
    # The return value must be one of the following (ndarray-based) formats:

    # Option 1: points only
    return np.zeros(({self.n_points}, 2))

    # Option 2: (triangle, points)
    return (np.zeros((3, 2)), np.zeros(({self.n_points}, 2)))
    # Option 3: dictionary with triangle and points
    return """
        call_format += (
            """{
        'triangle': np.zeros((3, 2)),
        'points': np.zeros(("""
            + str(self.n_points)
            + """, 2))
    }
"""
        )
        self.example_prompt = f"""
Must follow the following template for code:
Description: A short one line description of technique used.
```
class HeilbronnTriangle-n{self.n_points}:
    def __init__(self, n_points : int):
        pass
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
        safe = prepare_namespace(code, self.dependencies)
        try:
            local_ns = {}
            exec(code, safe, local_ns)
            local_ns = clean_local_namespace(local_ns, safe)
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            result = cls(self.n_points)()
        except Exception as e:
            # tb = e.__traceback__
            solution.set_scores(
                float("-inf"),
                f"exec-error {e}",
                "exec-failed",
            )
            return solution

        try:
            T, P = self._parse_candidate(result)
            T = self._ensure_unit_area(self.to_np_points(T, expected_n=3))
            P = self.to_np_points(P, expected_n=self.n_points)

            a, b, c = T[0], T[1], T[2]
            for i, p in enumerate(P):
                if not self.point_in_triangle(p, a, b, c, tol=self.tolerance):
                    raise ValueError(f"point indexed {i}--{p}--outside triangle")

            min_area = self.min_triangle_area(P, tol=self.tolerance)
            score = float(min_area)  # maximize
            solution.set_scores(score, f"Area of Smallest Triangle={min_area:.6g}")
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    hbt = HeilbronnTriangle(n_points=10)
    print(hbt.get_prompt())
