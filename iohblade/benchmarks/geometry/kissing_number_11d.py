from __future__ import annotations
import math, random
import numpy as np

from iohblade.problem import Problem
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace

class KissingNumber11D(Problem):
    """
    Kissing number lower bound in 11D via Lemma 1 (AlphaEvolve Appendix B.11).

    Candidate returns an array C with shape (m, 11) of non-zero points.
    Feasible iff   min_{x != y} ||x - y|| >= max_x ||x||   and   0 \notin C.
    Score (to maximize) = m = |C|.

    Notes:
    - The AlphaEvolve lemma does not require integrality. It only needs the inequality
      min_{x != y} ||x - y|| >= max_x ||x|| and 0 \notin C. We therefore do not enforce
      integer coordinates.
    """

    def __init__(self, tolerance: float = 0.0):
        super().__init__("kissing_number_11d")
        self.dim = 11
        self.tolerance = float(tolerance)


    @staticmethod
    def _pairwise_d2(P: np.ndarray) -> np.ndarray:
        """Squared distances with +inf on diagonal."""
        G = P @ P.T
        n2 = np.sum(P * P, axis=1, keepdims=True)
        D2 = n2 + n2.T - 2.0 * G
        np.fill_diagonal(D2, np.inf)
        D2[D2 < 0] = 0.0
        return D2

    def evaluate(self, solution, explogger=None):
        code = solution.code
        safe_globals = prepare_namespace(code, allowed=self.dependencies)
        try:
            local_ns = {}
            exec(code, safe_globals, local_ns)
            local_ns = clean_local_namespace(local_ns, safe_globals)
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            C = np.array(cls()(), dtype=float)

            if C.ndim != 2 or C.shape[1] != self.dim:
                raise ValueError(f"expected shape (m, {self.dim})")
            if not np.isfinite(C).all():
                raise ValueError("non-finite coordinates")

            norms2 = np.sum(C * C, axis=1)
            if np.any(norms2 <= self.tolerance):
                raise ValueError("zero vector present")

            D2 = self._pairwise_d2(C)
            d2_min = float(np.min(D2))
            r2_max = float(np.max(norms2))
            if d2_min + 1e-15 < r2_max - self.tolerance:
                raise ValueError("lemma condition violated")

            m = int(C.shape[0])
            solution.set_scores(float(m), f"|C|={m}")
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", "calc-failed")
        return solution
