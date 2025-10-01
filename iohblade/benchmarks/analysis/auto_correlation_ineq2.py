import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution

from iohblade.benchmarks.analysis.auto_correlation_base_spec import AutoCorrBaseSpec


best_solution = [0.0022217502753395443, 0.798058737836952, 0.4369294656327977, 1.1704958412868685, 1.3413665690827143, 1.5342366222696133, 1.7690742844401723, 1.9329450122360183, 2.2225113878900893, 1.9363966992163675, 2.0382191032475467, 2.2010898310433933, 2.0229605588392388, 2.029541518023742, 2.2636974412575626, 1.9622346498507677, 2.0781053776466134, 2.9856571697702514, 3.4418422600649374, 3.3477129878607825, 3.253250196453988, 3.420135507780267, 3.2509579118114464, 3.2308578066681575, 3.4707132763246245, 2.6462657430572087, 0.9614362498214617, 0, 0.0008733532713782356, 0.00041056186458359313, 0.00029587319086208687, 5.039012949497012e-06, 0, 0.5858888998745988, 6.741440691998236, 7.934548956206666e-06, 0.00013382382526231794, 4.551621108101551e-06, 0.0008898629473865954, 1.083008496291632e-05, 0.0006121618352774956, 0.0011493704284828532, 7.157034681754761, 9.111886252846807, 3.3127569806426527, 8.556232703271356e-06, 0.00017950056213609822, 2.7122354902710758e-06, 1.4036462843158317e-05, 1.1451768709981007e-05]


class AutoCorrIneq2(AutoCorrBaseSpec, Problem):
    r"""
    Auto Correlation Inequality 1:
        Takes 0 arugements, instantiates evaluator and base class with appropritate
        functionality.
        Optimisation:
            \[\min -(||f*f||_2^2 / (||f*f||_1 • ||f*f||_\infty))\]
        Best known auto-correlation 1 score by alpha evolve: is C_2 >= 0.8962 (prev 0.8892).
    """

    def __init__(self, best_known: float = 0.8962):
        AutoCorrBaseSpec.__init__(
            self, task_name="auto_corr_ineq_2", n_bins=50, best_known=best_known, best_solution=best_solution
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
