import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution

from iohblade.benchmarks.analysis.auto_correlation_base_spec import AutoCorrBaseSpec


class AutoCorrIneq2(AutoCorrBaseSpec, Problem):
    r"""
    Auto Correlation Inequality 1:
        Takes 0 arugements, instantiates evaluator and base class with appropritate
        functionality.
        Optimisation:
            \[\min -(||f*f||_2^2 / (||f*f||_1 • ||f*f||_\infty))\]
        Best known auto-correlation 1 score by alpha evolve: is C_2 >= 0.8962 (prev 0.8892).
    """

    def __init__(
        self, best_known: float = 0.8962, best_solution: list[float] | None = None
    ):
        AutoCorrBaseSpec.__init__(
            self,
            task_name="auto_corr_ineq_2",
            n_bins=50,
            best_known=best_known,
            best_solution=best_solution,
        )
        Problem.__init__(self, name=self.task_name)

        self.task_prompt = self.make_task_prompt(
            "minimize  -( ||f*f||_2^2 / ( ||f*f||_1 · ||f*f||_∞ ) )"
        )
        self.example_prompt = self.make_example_prompt("AutoCorreCandidate_2")
        self.format_prompt = self.make_format_prompt()

        self.dependencies += ["scipy"]

        self.minimisation = False

    def evaluate(self, solution: Solution) -> Solution:
        code = solution.code

        try:
            f, err = self._get_time_series(code)
            if err is not None:
                raise err
        except Exception as e:
            print("\t Exception in `auto_correlation_ineq2.py`, " + e.__repr__())
            solution.set_scores(float("-inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            if f.ndim != 1 or f.size == 0:
                raise ValueError("f must be a non-empty 1D array")
            if self.require_non_negative and np.any(f < 0):
                raise ValueError("C2 requires f ≥ 0")

            dx = self.dx
            g = dx * np.convolve(f, f, mode="full")

            L1 = dx * float(np.sum(np.abs(g)))  # ∫|g|
            L2sq = dx * float(np.sum(g * g))  # ∫g^2
            Linf = float(np.max(g))  # max_t g(t)
            den = L1 * Linf
            if den == 0.0:
                raise ValueError("Denominator zero in C2 ratio")

            score = L2sq / den  # maximize in paper
            solution.set_scores(
                score, f"C2 ratio = {score:.6g}, best known = {self.best_known:.6g}"
            )
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__
