import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution

from iohblade.benchmarks.analysis.auto_correlation_base_spec import AutoCorrBaseSpec


class AutoCorrIneq3(AutoCorrBaseSpec, Problem):
    r"""
    Auto Correlation Inequality 1:
        Takes 0 arugements, instantiates evaluator and base class with appropritate
        functionality.
        Optimisation:
            \[\max_t |||f*f||(t)| / (∫f)^2 \]
        Best known auto-correlation 1 score by alpha evolve: is C_3 <= 1.4557 (prev 1.4581).
    """

    def __init__(
        self, best_known: float = 1.4557, best_solution: list[float] | None = None
    ):
        AutoCorrBaseSpec.__init__(
            self,
            task_name="auto_corr_ineq_3",
            n_bins=400,
            best_known=best_known,
            best_solution=best_solution,
        )
        Problem.__init__(self, name=self.task_name)

        self.task_prompt = self.make_task_prompt("minimize  max_t |(f*f)(t)| / (∫ f)^2")
        self.example_prompt = self.make_example_prompt("AutoCorreCandidate_2")
        self.format_prompt = self.make_format_prompt()

        self.dependencies += ["scipy"]

        self.minimisation = True

    def evaluate(self, solution: Solution) -> Solution:
        code = solution.code

        try:
            f, err = self._get_time_series(code)
            if err is not None:
                raise err
        except Exception as e:
            print("\t Exception in `auto_correlation_ineq3.py`, " + e.__repr__())
            solution.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            if f.ndim != 1 or f.size == 0:
                raise ValueError("f must be a non-empty 1D array")

            dx = self.dx
            g = dx * np.convolve(f, f, mode="full")
            I = dx * float(np.sum(f))
            if I == 0.0:
                raise ValueError("Integral ∫f must be nonzero for C3")

            score = float(np.max(np.abs(g)) / (I * I))  # minimize
            solution.set_scores(
                score, f"C3 ratio = {score:.6g}, best known = {self.best_known:.6g}"
            )
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__
