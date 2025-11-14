from typing import Any
import math
import numpy as np
from typing import Tuple

from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace
from iohblade.problem import Problem
from iohblade.solution import Solution
from .packing_base import PackingBase


class HexagonPacking(PackingBase, Problem):
    """
    Pack n disjoint unit regular hexagons (side length = 1) inside a regular outer hexagon.
    Candidate returns an array with shape (n,3): rows [x, y, theta] for each inner hex.
    Score (to minimise) = outer_side_length required to contain all inner hexes.
    Boundary contact allowed; interiors must be disjoint.
    """

    def __init__(
        self,
        n_hex: int,
        best_known: float,
        tolerance: float = 1e-12,
        best_solution: list[Any] | None = None,
    ):
        task_name = f"hexagon_packing-n{n_hex}"
        self.best_known = best_known
        PackingBase.__init__(self, name=task_name, best_solution=best_solution)
        Problem.__init__(self, name=task_name)
        self.n_hex = int(n_hex)
        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Hexagon Packing Problem with number of hexagons: {self.n_hex}, and best solution: {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
        )
        self.tolerance = float(tolerance)
        self.minimisation = True
        self.dependencies += ["scipy"]

        self.task_prompt = self.make_hexagon_task_prompt(tolerance=self.tolerance)
        self.example_prompt = self.make_hexagon_example_prompt(
            f"HexagonPacking-n{n_hex}", self.n_hex
        )
        self.format_prompt = self.make_format_prompt()

    # ---------- geometry ----------
    def _unit_hex_vertices(
        self, center: Tuple[float, float], theta: float
    ) -> np.ndarray:
        cx, cy = float(center[0]), float(center[1])
        ang = float(theta)
        verts = []
        # Regular hexagon with side length = circumradius = 1
        for k in range(6):
            a = ang + k * (math.pi / 3.0)
            verts.append([cx + math.cos(a), cy + math.sin(a)])
        return np.asarray(verts, dtype=np.float64)

    def _projections_ranges(self, V: np.ndarray):
        u0 = np.array([1.0, 0.0])
        u1 = np.array([0.5, math.sqrt(3) / 2.0])
        u2 = np.array([-0.5, math.sqrt(3) / 2.0])
        ranges = []
        for u in (u0, u1, u2):
            p = V @ u
            ranges.append(float(p.max() - p.min()))
        return tuple(ranges)

    def _outer_side_from_vertices(self, V: np.ndarray) -> float:
        # Minimal apothem a* = max_i range_i / 2. Side length L = 2 a* / sqrt(3).
        r0, r1, r2 = self._projections_ranges(V)
        a = 0.5 * max(r0, r1, r2)
        return 2.0 * a / math.sqrt(3.0)

    def _intervals_overlap_strict(
        self, a_min, a_max, b_min, b_max, tolerance: float
    ) -> bool:
        # strict interior overlap; touching allowed
        return not (a_max <= b_min + tolerance or b_max <= a_min + tolerance)

    def _overlap_strict(
        self, poly1: np.ndarray, poly2: np.ndarray, tolerance: float
    ) -> bool:
        # SAT for convex polygons; treat boundary contact as non-overlap
        def edges(poly):
            return np.roll(poly, -1, axis=0) - poly

        def axes(poly):
            E = edges(poly)
            N = np.stack([np.array([-e[1], e[0]]) for e in E], axis=0)
            L = np.linalg.norm(N, axis=1, keepdims=True)
            mask = L[:, 0] > 0
            N[mask] /= L[mask]
            return N

        for N in (axes(poly1), axes(poly2)):
            for n in N:
                p1 = poly1 @ n
                p2 = poly2 @ n
                if not self._intervals_overlap_strict(
                    p1.min(), p1.max(), p2.min(), p2.max(), tolerance
                ):
                    return False
        return True  # interior overlap

    # ---------- evaluation ----------
    def evaluate(self, solution, explogger=None):
        code = solution.code

        try:
            safe = prepare_namespace(code, self.dependencies)
            local_ns = {}
            exec(code, safe, local_ns)
            local_ns = clean_local_namespace(local_ns, safe)

            cls = next(v for v in local_ns.values() if isinstance(v, type))
            try:
                arr = cls(self.n_hex, best_known_configuration=self.best_known)()
            except:
                arr = cls(self.n_hex)()
        except Exception as e:
            solution.set_scores(float("inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            A = np.asarray(arr, dtype=np.float64)
            if A.ndim != 2 or A.shape != (self.n_hex, 3):
                raise ValueError(f"expected shape (n={self.n_hex}, 3)")
            if not np.isfinite(A).all():
                raise ValueError("non-finite values")

            polys = []
            for i in range(self.n_hex):
                x, y, th = A[i]
                polys.append(self._unit_hex_vertices((x, y), th))

            # Interiors must be disjoint
            for i in range(self.n_hex):
                for j in range(i + 1, self.n_hex):
                    if self._overlap_strict(polys[i], polys[j], self.tolerance):
                        raise ValueError(f"hexagons {i} and {j} overlap")

            V = np.vstack(polys)
            side = self._outer_side_from_vertices(V)
            score = float(side)
            solution.set_scores(
                score,
                f"outer_side_length={side:.6g}, best known side length={self.best_known}",
            )
        except Exception as e:
            solution.set_scores(float("inf"), f"calc-error {e}", "calc-failed")
        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    hex = HexagonPacking(11, 1.167)
    print(hex.get_prompt())
