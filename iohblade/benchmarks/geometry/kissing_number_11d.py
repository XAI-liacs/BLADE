from __future__ import annotations
import math, random
import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace


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
    ):
        super().__init__(name="kissing_number_11d")
        self.dim = 11
        self.tolerance = float(tolerance)
        self.best_solution = best_solution
        self.best_known = best_known
        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Kissing Number {self.dim}D, and best solution: {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
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
        best_known_initialiser = """
    def__init__(self):
        pass
"""
        if self.best_solution is not None:
            best_known_initialiser = """
    def __init__(self, best_known_configuration: list[float] | None):
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        optimised for better results.
        pass
"""

        self.example_prompt = f"""
Must follow the following template for code:
Description: A short one line description of technique used.
```
class KissingNumber-{self.dim}d:

    {best_known_initialiser}

    def __call__(self):
        return np.zeros((n, {self.dim}))        #Maximise n.

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
        safe_globals = prepare_namespace(code, allowed=self.dependencies)
        try:
            local_ns = {}
            exec(code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)

            cls = next(v for v in local_ns.values() if isinstance(v, type))
            try:
                C = np.array(cls(self.best_solution)(), dtype=float)
            except:
                C = np.array(cls()(), dtype=float)

            if C.ndim != 2 or C.shape[1] != self.dim:
                raise ValueError(f"expected shape (m, {self.dim})")
            if not np.isfinite(C).all():
                raise ValueError("non-finite coordinates")

            norms2 = np.sum(C * C, axis=1)
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
            solution.set_scores(float("-inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    kiss = KissingNumber11D()
    print(kiss.get_prompt())
