import inspect
from typing import Any

import numpy as np

from iohblade.benchmarks.packing.packing_base import PackingBase
from iohblade.benchmarks.suite_utils import log_instantiated
from iohblade.misc.prepare_namespace import prepare_namespace
from iohblade.problem import Problem
from iohblade.solution import Solution


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
        eval_timeout: int = 600,
    ):
        self.n_circles = int(n_circles)
        self.perimeter = float(perimeter)
        self.tolerance = float(tolerance)
        self.best_known = float(best_known)
        self.best_solution = best_solution

        task_name = f"rectangle_packing_n{self.n_circles}_perim{self.perimeter:g}"
        PackingBase.__init__(self, task_name, best_solution=best_solution)
        Problem.__init__(self, name=task_name, eval_timeout=eval_timeout)

        log_instantiated(
            f"Instantiated Rectangle packing problem with rectangle perimeter = {self.perimeter}, and best solution: {self.best_known}."
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
            best_body_hint="""
    import numpy as np
    # Baseline: return the best known configuration unchanged, with the
    # rectangle it was packed in: width from the circles, height from the
    # perimeter 2 * (width + height) = P.
    U = np.array(self.best_known_configuration, dtype=float)
    P = getattr(self, 'perimeter', 4.0)
    w = float(np.max(U[:, 0] + U[:, 2]))
    return U, w, P / 2.0 - w
""",
            n_circles=self.n_circles,
        )
        self.format_prompt = self.make_format_prompt()
        self.dependencies += ["scipy"]
        self.minimisation = False

    def evaluate(self, solution: Solution, explogger=None):
        code = solution.code
        name = solution.name if solution.name else "RectanglePackingSolver"
        try:
            safe = prepare_namespace(code, self.dependencies)
            local_ns = {}
            compiled_code = compile(code, name, "exec")
            exec(compiled_code, safe, local_ns)

            cls = local_ns[name]
            if self.best_solution is not None:
                result = cls(self.n_circles, self.best_solution)()
            else:
                result = cls(self.n_circles)()

            if isinstance(result, tuple) and len(result) == 3:
                U, width, height = result
                width = float(width)
                height = float(height)
            else:
                U = result
                width = height = self.perimeter / 4.0
        except Exception as e:
            solution.set_scores(float("-inf"), f"exec-error {e}", e)
            return solution

        try:
            if not (np.isfinite(width) and np.isfinite(height)):
                raise ValueError(f"non-finite rectangle dimensions {width} x {height}")
            # perimeter equality
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"non-positive rectangle dimensions {width} x {height}"
                )
            if abs(2 * (width + height) - self.perimeter) > self.tolerance:
                raise ValueError(
                    f"perimeter mismatch {2*(width+height):.12f}, expected {self.perimeter:.12f}"
                )

            U = np.asarray(U, dtype=float)
            if U.shape != (self.n_circles, 3):
                raise ValueError(f"expected ({self.n_circles},3), got {U.shape}")
            if not np.isfinite(U).all():
                idx = int(np.where(~np.isfinite(U).all(axis=1))[0][0])
                raise ValueError(
                    f"non-finite value (nan or inf) in circle {idx}: {U[idx]}"
                )

            if np.any(U[:, 2] <= 0):
                idx = int(np.where(U[:, 2] <= 0)[0][0])
                raise ValueError(
                    f"non-positive radius for circle {idx}: r = {U[idx, 2]}"
                )
            # containment
            x, y, r = U[:, 0], U[:, 1], U[:, 2]
            if (
                np.any(x - r < -self.tolerance)
                or np.any(x + r > width + self.tolerance)
                or np.any(y - r < -self.tolerance)
                or np.any(y + r > height + self.tolerance)
            ):
                raise ValueError("circle(s) exceed rectangle boundary.")

            # disjointness
            for i in range(self.n_circles):
                for j in range(i + 1, self.n_circles):
                    dx = U[i, 0] - U[j, 0]
                    dy = U[i, 1] - U[j, 1]
                    if dx * dx + dy * dy < (U[i, 2] + U[j, 2] - self.tolerance) ** 2:
                        raise ValueError(f"circles {i} and {j} overlap.")

            score = float(np.sum(U[:, 2]))
            if not np.isfinite(score):
                raise ValueError(f"non-finite sum of radii {score}")
            solution.set_scores(
                score,
                f"sum_of_radii={score:.6f}; dims={width:.6f}×{height:.6f}; "
                f"n={self.n_circles}" + self.best_known_feedback(self.best_known),
            )
            return solution
        except Exception as e:
            solution.set_scores(float("-inf"), f"calc-error {e}", e)
            return solution

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return self.__dict__

    def get_config(self) -> dict[str, Any]:
        from iohblade.tags import Benchmark, StructureTag, VariableType

        evaluator = inspect.getsource(self.evaluate)

        self.tags.extend(
            [
                Benchmark.RECTANGLE_PACKING,
                VariableType.DISCRETE,
                StructureTag.PACKING,
                StructureTag.GEOMETRIC,
            ]
        )
        config = {
            "tags": self.tags,
            "name": "Rectangle Packing",
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "evaluator": evaluator,
            "config": {
                "n_circles": self.n_circles,
                "perimeter": self.perimeter,
                "tolerance": self.tolerance,
                "depencencies": self.dependencies,
                "imports": self.imports,
            },
        }
        return config


if __name__ == "__main__":
    rect_packing = RectanglePacking()
    for key, value in rect_packing.get_config().items():
        print(f"------------------------------{key}------------------------------")
        print(value)
