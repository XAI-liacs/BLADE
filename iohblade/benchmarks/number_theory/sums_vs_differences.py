import math
import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution

from iohblade.benchmarks.number_theory.number_theory_base import NumberTheoryBase
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace


class SumDifference(NumberTheoryBase, Problem):
    """
    Sums vs differences implemented via the single-set U evaluator (Appendix B.6, eq. (3)).
    Score:
        c(U) = 1 + log(|U−U|/|U+U|) / log(2*max(U)+1)
    with constraints U ⊂ ℤ≥0, 0 ∈ U, |U| ≤ max_set_size.
    """

    def __init__(
        self,
        max_set_size: int = 1000,
        best_score=1.1584,
        best_solution: list[int] | None = None,
    ):
        task_name = "sums_vs_differences"
        self.max_set_size = int(max_set_size)
        self.best_score = best_score

        NumberTheoryBase.__init__(self, task_name, best_solution)
        Problem.__init__(self, task_name)

        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Sums vs Difference benchmark with best known solution {self.best_score}.
--------------------------------------------------------------------------------------------------------------------
"""
        )

        # Prompting strings used by the outer orchestration, if any.
        self.task_prompt = self.make_task_prompt(
            "construct a finite set U ⊂ ℤ≥0 with 0 ∈ U that maximizes "
            "c(U) = 1 + log(|U−U|/|U+U|) / log(2*max(U)+1)"
        )
        self.example_prompt = self.make_example_prompt("SumDiffCandidate")
        self.format_prompt = self.make_format_prompt()

        self.minimisation = False
        self.dependencies += []

    def evaluate(self, solution: Solution, explogger=None) -> Solution:
        """
        Score a candidate set U via the Gyarmati–Hennecart–Ruzsa bound:
          c(U) = 1 + log(|U−U|/|U+U|) / log(2*max(U)+1),
        with required constraints: U ⊂ ℤ≥0, 0 ∈ U, |U| ≤ max_set_size.
        """

        code = solution.code

        try:
            safe_globals = prepare_namespace(code, self.dependencies)
            local_ns = {}
            exec(code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)

            cls = next(v for v in local_ns.values() if isinstance(v, type))
            U = []
            try:
                U = cls(
                    self.max_set_size, best_known_configuration=self.best_solution
                )()
            except:
                U = cls(self.max_set_size)()
        except Exception as e:
            solution.set_scores(-float("inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            # Normalize and validate U
            U = sorted({int(x) for x in U})
            ok, msg = self._validate_U(U)
            if not ok:
                solution.set_scores(-float("inf"), msg, "invalid-U")
                return solution

            M = U[-1]
            if M > 1_000_000:
                solution.set_scores(
                    -float("inf"), f"max(U) too large: {M}", "range-exceeded"
                )
                return solution

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
                solution.set_scores(-float("inf"), "degenerate U", "degenerate")
                return solution

            denom = math.log(2 * M + 1)
            if denom <= 0:
                solution.set_scores(-float("inf"), "log(2*max(U)+1) <= 0", "degenerate")
                return solution

            c = 1.0 + math.log(diff_sz / sum_sz) / denom
            solution.set_scores(
                c,
                f"C6 ≥ {c:.6f}; |U-U|={diff_sz}, |U+U|={sum_sz}, max(U)={M}, |U|={len(U)}, best known score {self.best_score}",
            )

        except Exception as e:
            solution.set_scores(-float("inf"), f"calc-error {e}", "calc-failed")
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
        max_abs_value = 10**6
        if U[-1] > max_abs_value:
            return False, f"max(U)={U[-1]} exceeds {max_abs_value}"
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


if __name__ == "__main__":
    sd = SumDifference()
    print(sd.get_prompt())
