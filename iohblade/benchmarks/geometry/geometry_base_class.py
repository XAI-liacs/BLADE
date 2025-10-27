import numpy as np
import math

from typing import Optional, Any


class GeometryBase:
    """
    Base helpers for 2D point-set geometry benchmarks.

    Candidates usually return an array of shape (n,2) with planar points.
    Utilities provided here implement the paper’s unit-area normalizations
    and hard feasibility checks (Appendix B.9–B.10).
    """

    def __init__(
        self,
        task_name: str,
        best_known: float,
        n_points: Optional[int] = None,
        tolerance: float = 1e-12,
    ):
        self.task_name = task_name
        self.n_points = None if n_points is None else int(n_points)
        self.tolerance = tolerance

        self.best_known = best_known

    @staticmethod
    def to_np_points(
        obj: Any, expected_n: Optional[int] = None
    ) -> np.ndarray:  # throws error.
        P = np.asarray(obj, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError("expected array with shape (n,2)")
        if expected_n is not None and P.shape[0] != expected_n:
            raise ValueError(f"expected {expected_n} points")
        if not np.isfinite(P).all():
            raise ValueError("non-finite coordinates")
        return P

    # -----------------------------Geometric Primitives----------------------------------#
    @staticmethod
    def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return abs(
            0.5 * float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        )

    @staticmethod
    def polygon_area(poly: np.ndarray) -> float:
        P = GeometryBase.to_np_points(poly)
        if P.shape[0] < 3:
            return 0.0
        x = P[:, 0]
        y = P[:, 1]
        s = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return 0.5 * s

    @staticmethod
    def convex_hull(P: np.ndarray) -> np.ndarray:
        P = GeometryBase.to_np_points(P)
        # Andrew’s monotone chain
        P2 = np.unique(P[np.lexsort((P[:, 1], P[:, 0]))], axis=0)
        if P2.shape[0] <= 1:
            return P2.copy()

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in P2:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        upper = []
        for p in reversed(P2):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
        return hull

    @staticmethod
    def point_in_triangle(
        p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 0.0
    ) -> bool:
        # Barycentric with inclusive boundary
        v0 = b - a
        v1 = c - a
        v2 = p - a
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(den) <= tol:
            return False
        u = (v2[0] * v1[1] - v2[1] * v1[0]) / den
        v = (v0[0] * v2[1] - v0[1] * v2[0]) / den
        return (u >= -tol) and (v >= -tol) and (u + v <= 1.0 + tol)

    def min_triangle_area(self, P: np.ndarray, tol: float | None = None) -> float:
        P = GeometryBase.to_np_points(P)
        t = self.tolerance if tol is None else float(tol)
        n = P.shape[0]
        if n < 3:
            return 0.0
        best = float("inf")
        for i in range(n - 2):
            a = P[i]
            for j in range(i + 1, n - 1):
                b = P[j]
                for k in range(j + 1, n):
                    c = P[k]
                    A = GeometryBase.triangle_area(a, b, c)
                    if A < best:
                        best = A
                        if best <= t:
                            return 0.0
        return float(best)

    # ---------- unit-area helpers ----------
    @staticmethod
    def _default_unit_triangle() -> np.ndarray:
        # Right triangle of area 1: base=1, height=2
        return np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    @staticmethod
    def _ensure_unit_area(T: np.ndarray) -> np.ndarray:
        T = GeometryBase.to_np_points(T, expected_n=3)
        A = abs(GeometryBase.polygon_area(T))
        if A <= 0:
            raise ValueError("triangle area is zero")
        s = 1.0 / math.sqrt(abs(A))
        return T * s

    def _parse_candidate(self, result: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Accept either:
          - points only -> use default unit-area triangle
          - (triangle, points)
          - {'triangle': tri, 'points': pts}
        """
        if isinstance(result, dict):
            tri = result.get("triangle", None)
            pts = result.get("points", None)
            if tri is None or pts is None:
                raise ValueError("dict must contain 'triangle' and 'points'")
            return self.to_np_points(tri, expected_n=3), self.to_np_points(
                pts, expected_n=self.n_points
            )
        if isinstance(result, (tuple, list)) and len(result) == 2:
            tri, pts = result
            return self.to_np_points(tri, expected_n=3), self.to_np_points(
                pts, expected_n=self.n_points
            )
        # assume points only
        pts = self.to_np_points(result, expected_n=self.n_points)
        return self._default_unit_triangle(), pts

    # ---------- prompt helper ----------
    def make_task_prompt(self, headline: str) -> str:
        return f"""
Task: {headline}
Constraints: use hard feasibility checks. Optimisation objective is to maximise.
Output: return numpy array of shape (n,2), or a dict with keys 'triangle' and 'points'."""
