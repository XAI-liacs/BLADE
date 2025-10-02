import builtins
import math
import random
import importlib
import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace


class ErdosMinOverlap(Problem):
    """
    The continuous Erdős minimum-overlap problem seeks to find measurable functions f,g: [-1,1] → [0,1] that satisfy:
        * Complementarity: f(x) + g(x) = 1 for all x ∈ [-1,1]
        * Unit mass: ∫_{-1}^1 f(x)dx = ∫_{-1}^1 g(x)dx = 1
        * Bounds: f(x), g(x) ∈ [0,1] for all x ∈ [-1,1]
    Optimisation;
        * Minimise superposition of x in [-2, 2], over f(t)g(x+t).
        * Alpha evolve: C_5 ≤ 0.380924 (previous best: 0.380927).
    """

    def __init__(
        self,
        task_name="erdos_min_overlap",
        n_bins: int = 800,
        tolerance=1e-6,
        best_known=0.380924,
        best_solution: list[float] | None = None,
    ):
        super().__init__(name=task_name)
        self.task_name = task_name
        self.n_bins = n_bins
        self.tolerance = tolerance
        self.best_known = best_known
        self.best_solution = best_solution
        print(
            f"""
-------------------------------------------------------------------
Instantiated Erdös Min Overlap Problem, best known {self.best_known}.
-------------------------------------------------------------------
"""
        )

        self.minimisation = True
        self.dependencies += ["scipy"]

        self.task_prompt = """
* Write a Python class whose `__call__` returns a list f of length N with values in range [0,1], such that:
    * It has domain in range [-1, 1], over `N` equal bins with dx = 2/N.
    * g is a point wise complement hence g = 1 - f, giving us f + g = 1. Both lie in range [0,1].
    * It follows integral constraints: ∫_{-1}^{1} f = 1 (hence ∫ g = 1).
        * The runner checks dx*sum(f)≈1.
    * Overlap functional uses zero-extension of g outside [-1,1].
    * Optimize the objective:
        * minimize  sup_{x ∈ [-2,2]} ∫_{-1}^{1} f(t) · g(x+t) dt,  with g = 1 - f
    * Do not use scipy's interp1d, it is no longer supported.
"""

        self.task_prompt += f"""
    * Use N = {self.n_bins}.
    * The tolerance of f + g similar to 1 is set to {self.tolerance}
        """

        best_known_initialiser = ""
        if best_solution is not None:
            best_known_initialiser = """
    def __init__(self, best_known_configuration: list[float] | None):
        # Accepts a best known configuration (if available) for the problem, as a initial configuration, which is then 
        optimised for better results.
"""
        self.example_prompt = f"""

An example template of such program is given by:
```python
class ErdosCandidate:

    {best_known_initialiser}

    def __call__(self):
        return [0,0]*{self.n_bins}
```

"""

        self.format_prompt = """

Give an excellent and novel algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```

"""
        if self.n_bins <= 0:
            raise ValueError("Expected n-bins to be positive number.")

        self.dx = 2 / self.n_bins

    def problem_description(self) -> str:
        return f"{self.task_name} | N={self.n_bins} bins on [-1,1], dx={self.dx:g}"

    def _sup_overlap(self, f: np.ndarray, g: np.ndarray) -> float:
        N = f.size
        dx = self.dx

        max_val = 0.0

        for s in range(-(N - 1), N):
            if s >= 0:
                a = f[: N - s]
                b = g[s:]
            else:
                a = f[-s:]
                b = g[: N + s]
            v = dx * float(np.dot(a, b))
            max_val = max(max_val, v)
        return max_val

    def evaluate(self, solution: Solution, explogger=None):
        local_ns = {}
        code = solution.code

        try:
            safe_globals = prepare_namespace(code, self.dependencies)

            exec(code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)
            cls = next(v for v in local_ns.values() if isinstance(v, type))

            try:
                f = np.asarray(
                    cls(best_known_configuration=self.best_solution)(), dtype=np.float64
                )
            except:
                f = np.asarray(cls()(), dtype=np.float64)
        except Exception as e:
            solution.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            if f.ndim != 1 or f.size != self.n_bins:
                raise ValueError(f"f must be length {self.n_bins}")
            # bounds f ∈ [0,1] within tolerance
            if np.any(f < -self.tolerance) or np.any(f > 1.0 + self.tolerance):
                raise ValueError("entries of f must lie in [0,1]")

            dx = self.dx
            I_f = dx * float(np.sum(f))
            if abs(I_f - 1.0) > self.tolerance:
                raise ValueError(f"integral ∫f must be 1 (got {I_f:.6g})")

            g = 1.0 - f  # f+g=1 pointwise

            score = self._sup_overlap(f, g)  # minimize
            msg = f"Score = {score:.6g}; with configuration: N={self.n_bins}, dx={dx:.6g}, If={I_f:.6g}.\n\t Best known score = {self.best_known}"
            solution.set_scores(score, msg)
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", "calc-failed")

        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__
