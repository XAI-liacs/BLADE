import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution
from .packing_base import PackingBase
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace


class RectanglePacking(PackingBase, Problem):
    """Appendix B.13: Pack n disjoint circles in a rectangle of perimeter P=4 to maximize the sum of radii.

    Candidate output:
      - Prefer: (circles, width, height) where circles is (n,3) and 2*(width+height)=P.
      - Fallback: circles only ⇒ evaluated in a square with width=height=P/4.

      - Score: Sum of radii of the internal circles.
      -Best Score: 2.3658
    """

    def __init__(
        self,
        n_circles: int = 21,
        perimeter: float = 4.0,
        tolerance: float = 1e-12,
        best_known=2.3658,
        best_solution: list[tuple[float, float, float]] | None = None,
    ):
        self.n_circles = int(n_circles)
        self.perimeter = float(perimeter)
        self.tolerance = float(tolerance)
        self.best_known = float(best_known)
        self.best_solution = best_solution

        task_name = f"rectangle_packing_n{self.n_circles}_perim{self.perimeter:g}"
        PackingBase.__init__(self, task_name)
        Problem.__init__(self, name=task_name)

        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Rectangle packing problem with rectangle perimeter = {self.perimeter}, and best solution: {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
        )

        headline = (
            f"Pack n disjoint circles inside a rectangle of perimeter {self.perimeter}."
        )
        contract = "Return either (U, width, height) with U shape (n,3), or just U. If only U is returned, width=height=perimeter/4 is assumed."
        objective = (
            "Maximize the sum of radii ∑_i r_i subject to 2*(width+height)=perimeter."
        )
        self.task_prompt = self.make_task_prompt(headline, contract, objective)
        self.example_prompt = self.make_example_prompt(
            "RectangleCandidate",
            body_hint="""
    import numpy as np
    n = getattr(self, 'n_circles', 8)
    P = getattr(self, 'perimeter', 4.0)
    w = h = P/4.0  # square fallback
    g = int(np.ceil(np.sqrt(n)))
    r = min(w,h)/(2*(g+1))
    U=[]
    for i in range(n):
    row, col = divmod(i, g)
    x = (col+1)/(g+1)*w
    y = (row+1)/(g+1)*h
    U.append([x, y, r])
    return np.array(U, dtype=float), w, h
""",
            n_circles=self.n_circles,
        )
        self.format_prompt = self.make_format_prompt()
        self.dependencies += ["scipy"]
        self.minimisation = False

    def evaluate(self, solution: Solution, explogger=None):
        code = solution.code
        try:
            safe = prepare_namespace(code, self.dependencies)
            local_ns = {}
            exec(code, safe, local_ns)
            local_ns = clean_local_namespace(local_ns, safe)

            cls = next(v for v in local_ns.values() if isinstance(v, type))
            try:
                result = cls(self.n_circles, self.best_solution)()
            except:
                result = cls(self.n_circles)()

            if isinstance(result, tuple) and len(result) == 3:
                U, width, height = result
                width = float(width)
                height = float(height)
            else:
                U = result
                width = height = self.perimeter / 4.0
        except Exception as e:
            solution.set_scores(float("-inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            # perimeter equality
            if width <= 0 or height <= 0:
                solution.set_scores(
                    float("-inf"),
                    f"non-positive rectangle dimensions {width}×{height}",
                    "invalid-dimensions",
                )
                return solution
            if abs(2 * (width + height) - self.perimeter) > self.tolerance:
                solution.set_scores(
                    float("-inf"),
                    f"perimeter mismatch: 2*(w+h)={2*(width+height):.12f}",
                    "perimeter-mismatch",
                )
                return solution

            U = np.asarray(U, dtype=float)
            if U.shape != (self.n_circles, 3):
                solution.set_scores(
                    float("-inf"),
                    f"expected ({self.n_circles},3), got {U.shape}",
                    "format-error",
                )
                return solution

            if np.any(U[:, 2] <= 0):
                idx = int(np.where(U[:, 2] <= 0)[0][0])
                solution.set_scores(
                    float("-inf"),
                    f"non-positive radius at index {idx}",
                    "invalid-radius",
                )
                return solution

            # containment
            x, y, r = U[:, 0], U[:, 1], U[:, 2]
            if (
                np.any(x - r < -self.tolerance)
                or np.any(x + r > width + self.tolerance)
                or np.any(y - r < -self.tolerance)
                or np.any(y + r > height + self.tolerance)
            ):
                solution.set_scores(
                    float("-inf"), "circle outside rectangle", "out-of-bounds"
                )
                return solution

            # disjointness
            for i in range(self.n_circles):
                for j in range(i + 1, self.n_circles):
                    dx = U[i, 0] - U[j, 0]
                    dy = U[i, 1] - U[j, 1]
                    if dx * dx + dy * dy < (U[i, 2] + U[j, 2] - self.tolerance) ** 2:
                        solution.set_scores(
                            float("-inf"), f"overlap between {i} and {j}", "overlap"
                        )
                        return solution

            score = float(np.sum(U[:, 2]))
            solution.set_scores(
                score,
                f"sum_of_radii={score:.6f}; dims={width:.6f}×{height:.6f}; n={self.n_circles}, best known={self.best_known}",
            )
            return solution
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", "calc-failed")
            return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__


if __name__ == "__main__":
    rect_packing = RectanglePacking()
    print(rect_packing.get_prompt())
