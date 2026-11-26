import inspect
import math
from typing import Any

import numpy as np

from iohblade.benchmarks.number_theory.number_theory_base import NumberTheoryBase
from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import clean_local_namespace, prepare_namespace
from iohblade.problem import Problem
from iohblade.solution import Solution

# Limits on U. The AlphaEvolve construction stored as the seed (notebook
# B.6, ``solution_1``) has |U| = 2003 and max(U) = 1,040,200, so the limits
# are |U| <= 2048 (the next power of two, a little headroom above 2003) and
# max(U) <= 2**21 = 2,097,152 (just above 2 * max(seed), so a candidate can
# also stretch the seed). Evaluation cost grows with max(U) (FFT of length
# ~4 * max(U)), which is why the value limit is not larger.
DEFAULT_MAX_SET_SIZE = 2048
MAX_VALUE = 2**21


class SumDifference(NumberTheoryBase, Problem):
    """
    Sums vs differences implemented via the single-set U evaluator (Appendix B.6, eq. (3)).
    Score:
        c(U) = 1 + log(|U−U|/|U+U|) / log(2*max(U)+1)
    with constraints U ⊂ ℤ≥0, 0 ∈ U, |U| ≤ max_set_size, max(U) ≤ max_value.

    The default best score 1.147989 is what the evaluator gives the stored
    seed (AlphaEvolve notebook B.6 ``solution_1``, |U| = 2003). The paper's
    better bound 1.1584 comes from ``solution_2`` with 54,265 elements, which
    does not fit within these limits.
    """

    def __init__(
        self,
        max_set_size: int = DEFAULT_MAX_SET_SIZE,
        best_known: float = 1.147989,
        best_solution: list[int] | None = None,
        max_value: int = MAX_VALUE,
        eval_timeout: int = 600,
        best_score: float | None = None,
    ):
        """
        Args:
            max_set_size: Maximum size of the set U.
            best_known: Best known score c(U) (reported with the results).
            best_solution: A known set U, shown as the example when given.
            max_value: Maximum element of U.
            eval_timeout: Seconds before an evaluation is stopped.
            best_score: Old name of ``best_known``, kept for compatibility.
        """
        task_name = "sums_vs_differences"
        self.max_set_size = int(max_set_size)
        self.max_value = int(max_value)
        if best_score is not None:
            best_known = best_score
        self.best_known = best_known
        self.best_score = best_known  # old attribute name

        NumberTheoryBase.__init__(self, task_name, best_solution)
        Problem.__init__(self, name=task_name, eval_timeout=eval_timeout)

        log_instantiated(
            f"Instantiated Sums vs Difference benchmark with best known solution {self.best_score}."
        )

        # Prompting strings used by the outer orchestration, if any.
        self.task_prompt = self.make_task_prompt(
            "construct a finite set U ⊂ ℤ≥0 with 0 ∈ U that maximizes "
            "c(U) = 1 + log(|U−U|/|U+U|) / log(2*max(U)+1)",
            max_value=self.max_value,
        )
        self.example_prompt = self.make_example_prompt(
            "SumDiffCandidate", max_value=self.max_value
        )
        self.format_prompt = self.make_format_prompt()

        self.minimisation = False
        self.dependencies += []

    def evaluate(self, solution: Solution, explogger=None) -> Solution:
        """
        Score a candidate set U via the Gyarmati–Hennecart–Ruzsa bound:
          c(U) = 1 + log(|U−U|/|U+U|) / log(2*max(U)+1),
        with required constraints: U ⊂ ℤ≥0, 0 ∈ U, |U| ≤ max_set_size and
        max(U) ≤ max_value.
        """

        code = solution.code
        name = solution.name if solution.name else "SumDiffCandidate"

        try:
            safe_globals = prepare_namespace(code, self.dependencies)
            local_ns = {}

            compiled_code = compile(code, name, "exec")
            exec(compiled_code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)

            cls = local_ns[name]
            U = []
            if self.best_solution is not None:
                U = cls(
                    self.max_set_size, best_known_configuration=self.best_solution
                )()
            else:
                U = cls(self.max_set_size)()
        except Exception as e:
            solution.set_scores(-float("inf"), f"exec-error {e}", e)
            return solution

        try:
            # Normalize and validate U
            U = sorted({int(x) for x in U})
            ok, msg = self._validate_U(U)
            if not ok:
                solution.set_scores(-float("inf"), f"invalid-U: {msg}", ValueError(msg))
                return solution

            M = U[-1]

            # Indicator vector on [0..M]
            a = np.zeros(M + 1, dtype=np.float64)
            a[np.array(U, dtype=np.int64)] = 1.0

            # FFT length
            need = 2 * M + 1
            L = 1
            while L < need:
                L <<= 1

            FA = np.fft.rfft(a, L)

            # |U+U| support from conv(a,a)
            conv_sum = np.fft.irfft(FA * FA, L)[:need]
            sum_sz = int((np.round(conv_sum) > 0).sum())  # robust to fp noise

            # |U−U| support from conv(a, a[::-1])  (cross-correlation support)
            FR = np.fft.rfft(a[::-1], L)
            conv_diff = np.fft.irfft(FA * FR, L)[:need]
            diff_sz = int((np.round(conv_diff) > 0).sum())

            if sum_sz <= 0 or diff_sz <= 0:
                solution.set_scores(
                    -float("inf"),
                    "degenerate U",
                    ValueError(f"U degenerated: {sum_sz}, {diff_sz}"),
                )
                return solution

            denom = math.log(2 * M + 1)
            if denom <= 0:
                solution.set_scores(
                    -float("inf"),
                    "log(2*max(U)+1) <= 0",
                    ValueError(f"Got log(2*max(U) + 1)={denom}"),
                )
                return solution

            c = 1.0 + math.log(diff_sz / sum_sz) / denom
            solution.set_scores(
                c,
                f"c(U) = {c:.6f}; |U-U|={diff_sz}, |U+U|={sum_sz}, max(U)={M}, "
                f"|U|={len(U)}"
                + self.best_known_feedback(self.best_score, "best known score"),
            )

        except Exception as e:
            solution.set_scores(-float("inf"), f"calc-error {e}", e)
        return solution

    # --- helpers aligned to single-set U ---

    def _validate_U(self, U):
        if not U:
            return False, "U empty"
        if U[0] != 0:
            return False, "0 must be in U"
        if any(x < 0 for x in U):
            return False, "U must be non-negative"
        if len(U) > self.max_set_size:
            return False, f"|U|>{self.max_set_size}"
        if U[-1] > self.max_value:
            return False, f"max(U)={U[-1]} exceeds {self.max_value}"
        return True, ""

    def _compute_support_stats(self, U):
        """Optional: trivial support bounds for logging/debug."""
        M = max(U)
        return {
            "support_sum_max": 2 * M + 1,  # 0..2M
            "support_diff_max": 2 * M + 1,  # -M..M
            "U_size": len(U),
            "M": M,
        }

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

    def get_config(self) -> dict[str, Any]:
        from iohblade.tags import (
            Benchmark,
            NoiseType,
            ObjectiveType,
            PrimaryCategories,
            StructureTag,
            VariableType,
        )

        tags: list[Any] = [
            PrimaryCategories.CO,
            Benchmark.SUMS_VS_DIFFERENCES,
            VariableType.DISCRETE,
            NoiseType.NOISELESS,
            ObjectiveType.SINGLE_OBJECTIVE,
            StructureTag.MATHS,
        ]
        evaluator = "\n\n".join(
            [
                inspect.getsource(self._compute_support_stats),
                inspect.getsource(self._validate_U),
                inspect.getsource(self.evaluate),
            ]
        )

        config = {
            "tags": tags,
            "name": "Sums vs Differences",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator,
            "config": {
                "dependencies": self.dependencies,
                "imports": self.imports,
                "max_set_size": self.max_set_size,
                "max_value": self.max_value,
            },
        }

        return config


if __name__ == "__main__":
    sd = SumDifference()
    for key, value in sd.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
