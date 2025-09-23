import math

from iohblade.benchmarks.geometry.geometry_base_class import GeometryBase
from iohblade.misc.prepare_namespace import prepare_namespace, _add_builtins_into, clean_local_namespace
from iohblade.problem import Problem

class HeilbronnConvexRegion(GeometryBase, Problem):
    """
    Heilbronn on a unit-area convex region (Appendix B.10).
    Input: n points. We use their convex hull as the region and rescale to area 1.
    Score = minimum triangle area after rescaling (maximize).
    """

    def __init__(self, n_points: int, tolerance: float = 1e-12):
        GeometryBase.__init__(self, task_name=f"heilbronn_convex_region-n{n_points}", n_points=n_points, tolerance=tolerance)
        Problem.__init__(self, name=f"heilbronn_convex_region-n{n_points}")

        self.dependencies += ["scipy", "shapely"]
        allowed = self.dependencies.copy()
        _add_builtins_into(allowed)

        self.task_prompt = f"""
Write a python class with function `__call__`, that generate a solution for the Heilbronn on a unit-area convex region
- The `__call__` method must return n points of type ndarray (n,2).
    - We use their convex hull as the region and rescale to area of 1 sq unit.
- The solution is scored on the area of smallest triangle formed by picking 3 of the n points, after rescaling.
- The optimisation goal is to maximise the score.
- The environment only provides access to the libraries:
    - {"\n    - ".join(allowed)}
"""
        self.task_prompt += f"- The tolerence of the solution is set to {self.tolerance}"

        self.example_prompt = f"""
Must follow the following template for code:
Description: A short one line description of technique used.
```
class HeilbronnConvexRegion-n{self.n_points}:
    def __init__(self, n_points : int):
        pass
    def __call__(self):
        return np.zeros(({self.n_points}, 2))

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

    def evaluate(self, solution, explogger=None):
        code = solution.code
        safe = prepare_namespace(code, self.dependencies)
        try:
            local_ns = {}
            exec(code, safe, local_ns)
            local_ns = clean_local_namespace(local_ns, safe)
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            result = cls(self.n_points)()
            print(result)
            P = self.to_np_points(result)
        except Exception as e:
            # tb = e.__traceback__
            solution.set_scores(float("-inf"), f"exec-error \n{e}", "exec-failed")
            return solution

        try:
            if P.ndim != 2 or P.shape != (self.n_points, 2):
                raise ValueError(f"points must be shape (n={self.n_points}, 2)")

            H = self.convex_hull(P)
            A = abs(self.polygon_area(H))
            if A <= self.tolerance:
                raise ValueError("degenerate: convex hull area ≈ 0")

            s = 1.0 / math.sqrt(A)  # scale so area(hull) == 1
            P1 = P * s

            min_area = self.min_triangle_area(P1, tol=self.tolerance)
            score = float(min_area)
            solution.set_scores(score, f"min_triangle_area={min_area:.6g}")
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

if __name__ == "__main__":
    hbc = HeilbronnConvexRegion(n_points=10)
    print(hbc.get_prompt())
