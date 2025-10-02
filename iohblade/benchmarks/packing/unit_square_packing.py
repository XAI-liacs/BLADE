import numpy as np

from iohblade import Solution
from iohblade.problem import Problem

from .packing_base import PackingBase
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace


class UnitSquarePacking(PackingBase, Problem):
    """Appendix B.12: Pack n disjoint circles in the unit square [0,1]×[0,1] to maximize the sum of radii."""

    def __init__(
        self,
        n_circles: int,
        best_known: float,
        tolerance: float = 1e-12,
        best_solution: list[tuple[float, float, float]] | None = None,
    ):
        self.n_circles = int(n_circles)
        self.tolerance = float(tolerance)
        self.best_known = float(best_known)
        self.best_solution = best_solution

        task_name = f"unit_square_packing_n{self.n_circles}"
        PackingBase.__init__(self, task_name)
        Problem.__init__(self, name=task_name)

        print(
            f"""
--------------------------------------------------------------------------------------------------------------------
Instantiated Unit Square Packing problem, with {self.n_circles} circles, best known score {self.best_known}.
--------------------------------------------------------------------------------------------------------------------
"""
        )

        headline = "Packing n disjoint circles inside the unit square [0,1]×[0,1]."
        contract = "Return a numpy array U with shape (n,3), U[i]=[x_i, y_i, r_i]."
        objective = "Maximize the sum of radii ∑_i r_i."
        self.task_prompt = self.make_task_prompt(headline, contract, objective)
        self.example_prompt = self.make_example_prompt(
            "UnitSquareCandidate", n_circles=self.n_circles
        )
        self.format_prompt = self.make_format_prompt()
        self.minimisation = False
        self.dependencies += ["scipy"]

    # ---------- evaluation ----------
    def evaluate(self, solution: Solution, explogger=None):
        code = solution.code
        safe = prepare_namespace(code, self.dependencies)
        try:
            local_ns = {}

            exec(code, safe, local_ns)
            local_ns = clean_local_namespace(local_ns, safe)
            cls = next(v for v in local_ns.values() if isinstance(v, type))
            circles = cls(self.n_circles)()
        except Exception as e:
            solution.set_scores(float("-inf"), f"exec-error {e}", "exec-failed")
            return solution

        try:
            U = np.asarray(circles, dtype=float)
            if U.shape != (self.n_circles, 3):
                solution.set_scores(
                    float("-inf"),
                    f"expected ({self.n_circles},3), got {U.shape}",
                    "format-error",
                )
                return solution

            # radii positive
            if np.any(U[:, 2] <= 0):
                idx = int(np.where(U[:, 2] <= 0)[0][0])
                solution.set_scores(
                    float("-inf"),
                    f"non-positive radius at index {idx}",
                    "invalid-radius",
                )
                return solution

            # containment in unit square
            x, y, r = U[:, 0], U[:, 1], U[:, 2]
            if (
                np.any(x - r < -self.tolerance)
                or np.any(x + r > 1 + self.tolerance)
                or np.any(y - r < -self.tolerance)
                or np.any(y + r > 1 + self.tolerance)
            ):
                solution.set_scores(
                    float("-inf"), "circle outside the unit square", "out-of-bounds"
                )
                return solution

            # pairwise disjoint
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
                f"sum_of_radii={score:.6f}; n={self.n_circles}, best known={self.best_known}",
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
    u26 = UnitSquarePacking()
    print(u26.get_prompt())
    dictionary = u26.to_dict()
    for key in dictionary:
        print(key, dictionary[key], sep="\t")
