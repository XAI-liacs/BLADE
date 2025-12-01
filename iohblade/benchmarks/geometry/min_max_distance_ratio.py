import numpy as np
from iohblade.problem import Problem
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace


class MinMaxMinDistanceRatio(Problem):
    """
    Minimize (max_pairwise_distance / min_pairwise_distance)^2 for n points in R^d.
    We operate on squared distances to match the paper's reporting; score = ratio^2.
    Best Known Score is: 12.889266 for 16 points in 2D and 4.1658 in 14 points 3D.
    """

    def __init__(
        self,
        n_points: int,
        dim: int,
        best_known: float | None,
        tolerance: float = 1e-12,
        best_solution: list[tuple] | None = None,
    ):
        super().__init__(name=f"min_max_min_distance_ratio-{dim}d")
        self.n_points = int(n_points)
        self.dim = int(dim)
        self.tolerance = float(tolerance)
        self.best_solution = best_solution

        # Guard against bad "best_solution."
        if best_solution:
            if len(best_solution) != self.n_points:
                best_solution = None

        if best_solution:
            for point in best_solution:
                if len(point) != dim:
                    self.best_solution = None
                    break

        if self.n_points < 2:
            raise ValueError("n_points must be >= 2")

        if self.dim < 1:
            raise ValueError("dim must be >= 1")

        self.best_known = best_known
        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Min / Max distance ratio problem in {self.dim} dimensions, and best solution: {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
        )
        self.minimisation = True
        self.dependencies += ["scipy", "scikit-learn"]

        # Prompt declarations.....

        formula = "minimise [max_{i < j} d(i, j) / min{i < j} d(i, j)]^2"

        self.task_prompt = f"""
* Write a python class with `__call__` method that:
    * Returns a set of {self.n_points} points, in {self.dim}-D hypervolume.
    * The Optimisation goal is to {formula}.
    * The tolerance for the point overlap is set to {self.tolerance}
"""
        best_known_initialiser = """
    def __init__(self, n_points : int, dimensions : int):
        pass"""
        if self.best_solution is not None:
            best_known_initialiser = """
    def __init__(self, n_points: int, dimensions: int, best_known_configuration: list[float] | None):
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        optimised for better results.
        pass
"""

        self.example_prompt = f"""
Must follow the following template for code:
Description: A short one line description of technique used.

```python
class MinMaxDistanceSolver:
    {best_known_initialiser}

    def __call__(self) -> list[tuple[float, float]]:
        return [(0.0,) * {self.dim}] * {self.n_points}
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

    @staticmethod
    def _pairwise_d2(P: np.ndarray) -> np.ndarray:
        G = P @ P.T
        diag = np.diag(G)
        D2 = diag[:, None] + diag[None, :] - 2.0 * G
        np.fill_diagonal(D2, np.inf)
        return D2

    def evaluate(self, solution, explogger=None):
        code = solution.code
        try:
            local_ns = {}
            safe = prepare_namespace(code, self.dependencies)
            exec(code, safe, local_ns)
            local_ns = clean_local_namespace(local_ns, safe)
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            try:
                P = cls(self.n_points, self.dim, self.best_solution)()
            except:
                P = cls(self.n_points, self.dim)()
        except Exception as e:
            solution.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
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
            d2_max = float(np.max(D2[np.isfinite(D2)]))
            ratio_sq = d2_max / d2_min
            score = float(ratio_sq)
            msg = f"ratio_sq={ratio_sq:.12g}."
            if self.best_known is not None:
                msg += f" Best known score is {self.best_known}."
            solution.set_scores(score, msg)
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    mmd = MinMaxMinDistanceRatio(n_points=10, dim=2, best_known=0)
    print(mmd.get_prompt())
