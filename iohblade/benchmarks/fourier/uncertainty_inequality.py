import math

import numpy as np
from numpy.polynomial.hermite import hermval

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace

from iohblade.benchmarks.fourier.fourier_base import FourierBase


class UncertaintyInequality(FourierBase, Problem):
    """
    Minimize r_max^2 / (2*pi) per Appendix B.4.
    r_max = largest positive root beyond which P(x) >= 0.
    """

    def __init__(
        self,
        n_terms: int = 3,
        best_known: float = 0.3216,
        best_solution: list[float] | None = None,
    ):
        FourierBase.__init__(
            self,
            task_name="fourier_uncertainty_C4",
            n_terms=n_terms,
            best_known_configuration=best_solution,
        )
        Problem.__init__(self, name="fourier_uncertainty_C4")
        self.task_prompt = self.make_task_prompt("minimize  r_max^2 / (2*pi)")
        self.example_prompt = self.make_example_prompt("FourierCandidate")
        self.format_prompt = self.make_format_prompt()

        self.best_known = best_known
        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Fourier Uncertainty Inequality problem with number of terms = {self.n_terms}, best known {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
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

    def _P(self, x: np.ndarray, hcoef: np.ndarray) -> np.ndarray:
        return hermval(x, hcoef)

    def _largest_positive_root(self, hcoef: np.ndarray) -> float:
        """
        Scan [0, x_max] to find last sign change then refine with bisection.
        Requires P(0) < 0 and P is eventually >= 0.
        """
        eps = 1e-12
        x_prev = 0.0
        y_prev = float(self._P(np.array([eps]), hcoef))  # avoid exact zero at 0
        bracket = None
        x = self.grid_step
        while x <= self.x_max + 1e-12:
            y = float(self._P(np.array([x]), hcoef))
            if y_prev <= 0.0 and y >= 0.0:
                bracket = (x_prev, x)
            x_prev, y_prev = x, y
            x += self.grid_step

        if bracket is None:
            raise ValueError(
                "No positive root found in [0, x_max]. Increase x_max or adjust coefficients."
            )

        a, b = bracket
        fa = float(self._P(np.array([a]), hcoef))
        fb = float(self._P(np.array([b]), hcoef))
        if fa > 0 or fb < 0:
            raise ValueError("Invalid bracket for bisection.")

        # Bisection
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = float(self._P(np.array([m]), hcoef))
            if fm >= 0.0:
                b, fb = m, fm
            else:
                a, fa = m, fm
            if b - a < 1e-10:
                break
        return 0.5 * (a + b)

    def _check_tail_nonnegative(self, hcoef: np.ndarray, r: float) -> None:
        xs = np.linspace(
            max(r + 5 * self.grid_step, r + 1e-6), self.x_max, self.check_points
        )
        if xs.size == 0:
            return
        vals = self._P(xs, hcoef)
        if np.min(vals) < -1e-9:
            raise ValueError(
                "P(x) becomes negative beyond r_max; tail nonnegativity violated."
            )

    # ---- evaluation ----------------------------------------------------------

    def evaluate(self, solution: Solution, explogger=None):
        code = solution.code

        # 1) execute candidate
        try:
            safe_globals = prepare_namespace(code, self.dependencies)
            local_ns = {}
            exec(code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            try:
                c = np.asanyarray(
                    cls(self.best_known_configuration)(), dtype=np.float64
                )
            except:
                c = np.asarray(cls(self.n_terms)(), dtype=np.float64)

        except Exception as e:
            solution.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
            return solution

        # 2) validate and score
        try:
            if c.ndim != 1 or c.size != self.n_terms:
                raise ValueError(
                    f"Expected {self.n_terms} coefficients for H_0,H_4,..."
                )

            # scale invariance: normalize so that leading coeff = 1 (if nonzero)
            if not np.isfinite(c[-1]) or abs(c[-1]) < self.tolerance:
                raise ValueError("Leading coefficient must be nonzero.")
            if c[-1] < 0:
                # flip sign to make leading coefficient positive; preserves score
                c = -c

            hcoef = self._build_hcoef(c)

            p0 = float(self._P(np.array([0.0]), hcoef))
            if not np.isfinite(p0) or p0 >= 0.0:
                raise ValueError("Constraint P(0) < 0 failed.")

            # ensure eventual positivity at far right
            p_far = float(self._P(np.array([self.x_max]), hcoef))
            if p_far < 0.0:
                raise ValueError("P(x_max) < 0; not positive for large |x|.")

            r = self._largest_positive_root(hcoef)
            self._check_tail_nonnegative(hcoef, r)

            score = float(r * r / (2.0 * math.pi))  # paper’s bound
            solution.set_scores(
                score,
                f"Score = {score:.9g}; r_max={r:.6g}; best known score = {self.best_known}",
            )
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", "calc-failed")

        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    uncertain_ineq = UncertaintyInequality()
    print(uncertain_ineq.get_prompt())
