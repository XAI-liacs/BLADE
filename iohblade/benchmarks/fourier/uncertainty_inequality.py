import inspect
import math
from typing import Any

import numpy as np
from numpy.polynomial.hermite import herm2poly, hermval
from numpy.polynomial.polynomial import polyval

from iohblade.benchmarks.fourier.fourier_base import FourierBase
from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import prepare_namespace
from iohblade.problem import Problem
from iohblade.solution import Solution


class UncertaintyInequality(FourierBase, Problem):
    """
    Minimize r_max^2 / (2*pi) per Appendix B.4.

    The candidate gives K = n_terms coefficients for H_0, H_4, ..., and the
    evaluator appends the H_{4K} coefficient that makes P(0) = 0, as in the
    AlphaEvolve notebook (B.4). r_max is the largest positive sign change of
    P(x) / x^2 (after flipping P if its leading coefficient is negative).

    The default best known 0.3521 is the AlphaEvolve bound (the stored seed
    scores 0.3520991). The previous value 0.3216 belongs to a different
    construction (Laguerre-based), not to this formulation.
    """

    def __init__(
        self,
        n_terms: int = 3,
        best_known: float = 0.3521,
        best_solution: list[float] | None = None,
        eval_timeout: int = 600,
    ):
        FourierBase.__init__(
            self,
            task_name="fourier_uncertainty_C4",
            n_terms=n_terms,
            best_known_configuration=best_solution,
        )
        Problem.__init__(self, name="fourier_uncertainty_C4", eval_timeout=eval_timeout)
        self.task_prompt = self.make_task_prompt("minimize  r_max^2 / (2*pi)")
        self.example_prompt = self.make_example_prompt("FourierCandidate")
        self.format_prompt = self.make_format_prompt()

        self.best_known = best_known
        log_instantiated(
            f"Instantiated Fourier Uncertainty Inequality problem with number of terms = {self.n_terms}, best known {self.best_known}."
        )

        self.dependencies += ["scipy"]
        self.minimisation = True

    # ---- helpers -------------------------------------------------------------

    def _build_hcoef(self, c: np.ndarray) -> np.ndarray:
        """Map c[k] to Hermite series coefficients a[n] with n in {0,4,8,...}."""
        deg = 4 * (c.size - 1)
        hcoef = np.zeros(deg + 1, dtype=np.float64)
        for k, ck in enumerate(c):
            hcoef[4 * k] = ck
        return hcoef

    def _complete_coefficients(self, c: np.ndarray) -> np.ndarray:
        """Append the coefficient of H_{4K} that makes P(0) = 0.

        With K free coefficients c[0..K-1] for H_0, H_4, ..., H_{4(K-1)},
        the last coefficient is c_K = -sum_k c[k] H_{4k}(0) / H_{4K}(0).
        Physicists' Hermite values are H_{2m}(0) = (-1)^m (2m)! / m!, so
        H_0(0) = 1, H_4(0) = 12, H_8(0) = 1680 and H_12(0) = 665280; for the
        default K = 3 this is c_3 = -(c0 + 12 c1 + 1680 c2) / 665280
        (AlphaEvolve notebook, section B.4).
        """
        K = c.size
        h0 = [
            math.factorial(4 * k) / math.factorial(2 * k) for k in range(K + 1)
        ]  # H_{4k}(0); (-1)^(2k) = 1
        last = -float(np.dot(c, h0[:K])) / h0[K]
        return np.append(c, last)

    def _P(self, x: np.ndarray, hcoef: np.ndarray) -> np.ndarray:
        return hermval(x, hcoef)

    def _P_at(self, x: float, hcoef: np.ndarray) -> float:
        """P at a single point. (float() of a 1-element array is an error in
        recent NumPy versions, which the evaluation environment installs.)"""
        return float(hermval(float(x), hcoef))

    def _largest_sign_change(self, hcoef: np.ndarray) -> float:
        """Largest positive x at which P(x) / x^2 changes sign.

        P is even with P(0) = 0, so P(x) / x^2 is a polynomial Q. Its real
        positive roots are found with numpy; the largest one at which Q
        changes sign is refined by bisection and returned.
        """
        power = herm2poly(hcoef)  # P in the monomial basis, low -> high
        scale = float(np.max(np.abs(power)))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("P is identically zero.")
        q = power[2:] / scale  # Q(x) = P(x) / x^2 (P(0) = P'(0) = 0)
        q = np.trim_zeros(np.where(np.abs(q) < 1e-14, 0.0, q), "b")
        if q.size < 2:
            raise ValueError("P(x) / x^2 is constant; it has no sign change.")
        if q[-1] < 0:
            # Leading coefficient negative: flip P (preserves the roots).
            q = -q

        def Q(x: float) -> float:
            return float(polyval(x, q))

        roots = np.roots(q[::-1])
        real = np.sort(roots.real[np.abs(roots.imag) <= 1e-7 * (1 + np.abs(roots))])
        for r in real[::-1]:
            if r <= 0:
                break
            delta = 1e-6 * max(r, 1.0)
            a, b = r - delta, r + delta
            fa, fb = Q(a), Q(b)
            if fa < 0.0 < fb:
                # Bisection: Q(a) < 0 <= Q(b).
                for _ in range(100):
                    m = 0.5 * (a + b)
                    if Q(m) >= 0.0:
                        b = m
                    else:
                        a = m
                    if b - a < 1e-14 * max(r, 1.0):
                        break
                return 0.5 * (a + b)
        raise ValueError("P(x) / x^2 has no positive sign change.")

    # ---- evaluation ----------------------------------------------------------

    def evaluate(self, solution: Solution, explogger=None):
        code = solution.code
        name = solution.name
        # 1) execute candidate
        try:
            local_ns = {}
            safe_globals = prepare_namespace(code, self.dependencies)

            compiled_code = compile(code, filename=name, mode="exec")
            exec(compiled_code, safe_globals, local_ns)
            cls = local_ns[name]

            if self.best_known_configuration is not None:
                c = np.asanyarray(
                    cls(self.n_terms, self.best_known_configuration)(), dtype=np.float64
                )
            else:
                c = np.asarray(cls(self.n_terms)(), dtype=np.float64)

        except Exception as e:
            solution.set_scores(float("inf"), f"exec-error {e}", e)
            return solution

        # 2) validate and score (AlphaEvolve notebook, section B.4)
        try:
            if c.ndim != 1 or c.size != self.n_terms:
                raise ValueError(
                    f"Expected {self.n_terms} coefficients for H_0,H_4,..."
                )
            if not np.all(np.isfinite(c)):
                raise ValueError("Coefficients must be finite.")

            # The evaluator adds the H_{4K} term that makes P(0) = 0.
            full = self._complete_coefficients(c)
            hcoef = self._build_hcoef(full)

            # If the leading coefficient is negative, P is flipped; the
            # score is the largest sign change of P(x) / x^2.
            r = self._largest_sign_change(hcoef)

            score = float(r * r / (2.0 * math.pi))  # paper’s bound
            solution.set_scores(
                score,
                f"Score = {score:.9g}; r_max={r:.6g}; added H_{4 * self.n_terms} "
                f"coefficient {full[-1]:.6g}"
                + self.best_known_feedback(self.best_known, "best known score"),
            )
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", e)
        return solution

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

        evaluator_string = (
            inspect.getsource(self._build_hcoef)
            + "\n"
            + inspect.getsource(self._complete_coefficients)
            + "\n"
            + inspect.getsource(self._largest_sign_change)
            + "\n"
            + inspect.getsource(self.evaluate)
        )

        tags: list[Any] = [PrimaryCategories.OTHER]
        tags.append(Benchmark.FOURIER_UNCERTAINTY)
        tags.append(NoiseType.NOISELESS)
        tags.append(ObjectiveType.SINGLE_OBJECTIVE)
        tags.append(VariableType.CONTINUOUS)
        tags.append(StructureTag.SIGNAL)

        config = {
            "tags": tags,
            "name": "Fourier Uncertainty Inequality",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator_string,
            "config": {
                "n_terms": self.n_terms,
                "tolerance": self.tolerance,
                "x_max": self.x_max,
                "grid_step": self.grid_step,
                "check_points": self.check_points,
            },
        }
        return config


if __name__ == "__main__":
    uncertain_ineq = UncertaintyInequality()
    for key, value in uncertain_ineq.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
