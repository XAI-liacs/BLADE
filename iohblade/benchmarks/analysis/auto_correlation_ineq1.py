import scipy
import numpy as np
import math, random

from scipy.optimize import minimize

from iohblade.problem import Problem
from iohblade.solution import Solution

if __name__ == "__main__":          #Weird $PYTHONPATH conflict.
    from auto_correlation_base_spec import AutoCorrBaseSpec
else:
    from .auto_correlation_base_spec import AutoCorrBaseSpec

class AutoCorrIneq1(AutoCorrBaseSpec, Problem):
    r"""
    Auto Correlation Inequality 1:
        Takes 0 arugements, instantiates evaluator and base class with appropritate
        functionality.
        Optimisation:
            \[\min \max_t frac{(f*f)(t)}{(\int f)^2}\]
        Best known auto-correlation 1 score is C₁ <= 1.5053 (prev 1.5098).
    """
    def __init__(self):
        AutoCorrBaseSpec.__init__(self, task_name="auto_corr_ineq_1", n_bins=600)
        Problem.__init__(self, name=self.task_name)
        self.task_prompt = self.make_task_prompt("minimize  max_t (f*f)(t) / (∫ f)^2")
        self.example_prompt = self.make_example_prompt("AutoCorrCandidate")
        self.format_prompt = self.make_format_prompt()
        self.dependencies += ["scipy"]      #Allow scipy to be accessed in the isolate environment.

    def evaluate(self, solution:Solution) -> Solution:
        code = solution.code

        local_parameters = {}
        global_parameters = {
            "math": math,
            "random": random,
            "np": np,
            "scipy": scipy,
            "minimize": minimize
        }

        try:
            exec(code, global_parameters, local_parameters)
            cls = next(v for v in local_parameters.values() if isinstance(v, type))
            f = np.asarray(cls()(), dtype=np.float64)
        except Exception as e:
            print("\t Exception in `auto_correlation_ineq1.py`, " + e.__repr__())
            solution.set_scores(float("inf"), f"exec-error {e}", "exec-failed"); return solution

        try:
            if f.ndim != 1 or f.size != self.n_bins:
                raise ValueError(f"f must be 1D with length N={self.n_bins}")
            if self.require_non_negative and np.any(f < 0):
                raise ValueError("C1 requires f ≥ 0")

            dx = self.dx
            g  = dx * np.convolve(f, f, mode="full")
            I  = dx * float(np.sum(f))
            if I <= 0:
                raise ValueError("Integral ∫f must be > 0 for C1")

            score = float(np.max(g) / (I * I))   # minimize
            solution.set_scores(score, f"C1 ratio = {score:.6g}")
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

if __name__ == "__main__":
    ac1 = AutoCorrIneq1()
    print(ac1.task_prompt)
    print(ac1.eval_timeout)
    print(ac1._env_path)
