"""A small custom benchmark: design an optimiser for the sphere function.

Used by ``examples/configs/custom_problem.yml``. The LLM writes a class
``__init__(self, budget, dim)`` / ``__call__(self, func)`` that minimises
``func`` within ``budget`` evaluations; the fitness is the negated best
value found, averaged over a few random shifts (higher is better).
"""

from typing import Any

import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.utils import first_class_name


class SphereProblem(Problem):
    """Minimise a shifted sphere function in ``dim`` dimensions."""

    def __init__(
        self, dim: int = 5, budget: int = 500, instances: int = 3, name="Sphere"
    ):
        super().__init__(name=name, eval_timeout=120)
        self.dim = dim
        self.budget = budget
        self.instances = instances
        self.minimisation = False
        self.task_prompt = (
            f"Write a Python class that minimises a black-box function of {dim} "
            f"variables in [-5, 5] using at most {budget} function evaluations.\n"
        )
        self.example_prompt = (
            "class RandomSearch:\n"
            "    def __init__(self, budget, dim):\n"
            "        self.budget, self.dim = budget, dim\n\n"
            "    def __call__(self, func):\n"
            "        best_f, best_x = float('inf'), None\n"
            "        for _ in range(self.budget):\n"
            "            x = np.random.uniform(-5, 5, self.dim)\n"
            "            f = func(x)\n"
            "            if f < best_f:\n"
            "                best_f, best_x = f, x\n"
            "        return best_f, best_x\n"
        )
        self.format_prompt = (
            "Give the full class in one ```python code block. numpy is "
            "available as np.\n"
        )

    def evaluate(self, solution: Solution) -> Solution:
        namespace = {"np": np}
        exec(solution.code, namespace)
        algorithm_class = namespace[first_class_name(solution.code)]
        scores = []
        for seed in range(self.instances):
            shift = np.random.default_rng(seed).uniform(-3, 3, self.dim)
            calls = 0

            def sphere(x):
                nonlocal calls
                calls += 1
                if calls > self.budget:
                    raise RuntimeError("Evaluation budget exceeded.")
                return float(np.sum((np.asarray(x) - shift) ** 2))

            best_f, _ = algorithm_class(budget=self.budget, dim=self.dim)(sphere)
            scores.append(-best_f)
        fitness = float(np.mean(scores))
        solution.set_scores(fitness, f"Mean negated best value: {fitness:.4g}.")
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self) -> dict[str, Any]:
        return {"dim": self.dim, "budget": self.budget, "instances": self.instances}

    def get_config(self) -> dict[str, Any]:
        return {
            "tags": [],
            "name": self.name,
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "config": self.to_dict(),
        }
