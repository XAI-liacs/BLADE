import inspect
from typing import Any

import numpy as np

from iohblade.benchmarks.analysis.auto_correlation_base_spec import AutoCorrBaseSpec
from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.tags import Benchmark, PrimaryCategories, StructureTag


class AutoCorrIneq2(AutoCorrBaseSpec, Problem):
    r"""
    Auto Correlation Inequality 1:
        Takes 0 arugements, instantiates evaluator and base class with appropritate
        functionality.
        Optimisation:
            \[\min -(||f*f||_2^2 / (||f*f||_1 • ||f*f||_\infty))\]
        Best known auto-correlation 2 score by alpha evolve: is C_2 >= 0.8962 (prev 0.8892).
    """

    def __init__(
        self,
        best_known: float = 0.8962,
        best_solution: list[float] | None = None,
        eval_timeout: int = 600,
    ):
        AutoCorrBaseSpec.__init__(
            self,
            task_name="auto_corr_ineq_2",
            n_bins=50,
            best_known=best_known,
            best_solution=best_solution,
        )
        Problem.__init__(self, name=self.task_name, eval_timeout=eval_timeout)

        self.task_prompt = self.make_task_prompt(
            "minimize  -( ||f*f||_2^2 / ( ||f*f||_1 · ||f*f||_∞ ) )"
        )
        self.example_prompt = self.make_example_prompt("SignalCandidate")
        self.format_prompt = self.make_format_prompt()

        self.dependencies += ["scipy"]

        self.minimisation = False

    def evaluate(self, solution: Solution) -> Solution:
        code = solution.code

        try:
            f, err = self._get_time_series(code, name=solution.name)
            if err is not None:
                raise err
        except Exception as e:
            solution = solution.set_scores(float("-inf"), f"exec-error {e}", e)
            return solution

        try:
            if f.ndim != 1 or f.size == 0:
                raise ValueError("f must be a non-empty 1D array")
            if not np.all(np.isfinite(f)):
                raise ValueError("f must contain only finite values (no nan/inf)")
            if self.require_non_negative and np.any(f < 0):
                raise ValueError("C2 requires f ≥ 0")

            dx = self.dx
            # f is piecewise constant on bins of width dx, so g = f*f is
            # piecewise linear; g holds its values at the knots, and it is
            # zero at the two outer knots. Integrate it exactly.
            g = dx * np.convolve(f, f, mode="full")
            gp = np.concatenate(([0.0], g, [0.0]))
            g0, g1 = gp[:-1], gp[1:]

            L1 = dx * float(np.sum(np.abs(g)))  # ∫|g| (exact as g ≥ 0)
            L2sq = dx / 3.0 * float(np.sum(g0 * g0 + g0 * g1 + g1 * g1))  # ∫g^2
            Linf = float(np.max(g))  # max_t g(t), attained at a knot
            den = L1 * Linf
            if den == 0.0:
                raise ValueError("Denominator zero in ratio")

            score = L2sq / den  # maximize in paper
            if not np.isfinite(score):
                raise ValueError(f"non-finite ratio {score}")
            solution = solution.set_scores(
                score,
                f"ratio = {score:.6g}" + self.best_known_feedback(self.best_known),
            )
        except Exception as e:
            solution = solution.set_scores(float("-inf"), f"calc-error {e}", e)
        return solution

    def get_config(self) -> dict[str, Any]:
        return {
            "tags": [
                PrimaryCategories.BBO,
                Benchmark.ANALYSIS,
                Benchmark.TRENDS,
                StructureTag.TIME_SERIES,
            ],
            "name": "Auto-Correlation 2",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": inspect.getsource(self.evaluate),
            "config": {
                "dependencies": self.dependencies,
                "imports": self.imports,
                "n_bins": self.n_bins,
            },
        }

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    ac2 = AutoCorrIneq2()
    config = ac2.get_config()
    for key, value in config.items():
        print(f"------------------------------{key}------------------------------")
        print(value)
    import hashlib
    import json

    json_data = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config["hash"] = hashlib.sha256(json_data).hexdigest()
    with open("problem.json", "w+") as f:
        f.write(json.dumps(config, indent=4))
