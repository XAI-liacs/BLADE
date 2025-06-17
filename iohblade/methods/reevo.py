from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution

try:
    from reevo import ReEvo as ReEvoAlgorithm
    from reevo.utils.llm_client.base import BaseClient
except Exception:  # pragma: no cover - optional dependency
    ReEvoAlgorithm = None
    BaseClient = object  # type: ignore


class _BladeReEvoClient(BaseClient):
    """Adapter that exposes the interface expected by ReEvo."""

    def __init__(self, llm: LLM, temperature: float = 1.0) -> None:
        super().__init__(model=llm.model, temperature=temperature)
        self.llm = llm

    def _chat_completion_api(
        self, messages: list[dict], temperature: float, n: int = 1
    ):
        responses = []
        for _ in range(n):
            content = self.llm.query(messages)
            responses.append(SimpleNamespace(message=SimpleNamespace(content=content)))
        return responses


class ReEvo(Method):
    """Wrapper for the ReEvo baseline."""

    def __init__(self, llm: LLM, budget: int, name: str = "ReEvo", **kwargs: Any):
        super().__init__(llm, budget, name)
        self.kwargs = kwargs

    def _eval_population(self, reevo: Any, population: list[dict], problem: Problem):
        for individual in population:
            reevo.function_evals += 1
            if individual.get("code") is None:
                individual["exec_success"] = False
                individual["obj"] = float("inf")
                continue
            solution = Solution(code=individual["code"])
            solution = problem(solution)
            if reevo.obj_type == "min":
                individual["obj"] = solution.fitness
            else:
                individual["obj"] = -solution.fitness
            individual["exec_success"] = True
        return population

    def __call__(self, problem: Problem):
        if ReEvoAlgorithm is None:
            raise ImportError("reevo package is not installed")

        from omegaconf import OmegaConf

        cfg_dict = {
            "max_fe": self.budget,
            "pop_size": self.kwargs.get("pop_size", 10),
            "init_pop_size": self.kwargs.get("init_pop_size", 30),
            "mutation_rate": self.kwargs.get("mutation_rate", 0.5),
            "timeout": self.kwargs.get("timeout", 20),
            "problem": {
                "problem_name": "blade_problem",
                "description": problem.task_prompt,
                "problem_size": getattr(problem, "dim", 1),
                "func_name": "algorithm",
                "obj_type": "max",
                "problem_type": "black_box",
            },
        }
        cfg = OmegaConf.create(cfg_dict)
        client = _BladeReEvoClient(self.llm)
        reevo = ReEvoAlgorithm(cfg, root_dir=".", generator_llm=client)
        # Override evaluation to use BLADE problems
        reevo.evaluate_population = lambda pop: self._eval_population(
            reevo, pop, problem
        )
        code, _ = reevo.evolve()
        name = "OptimizationAlgorithm"
        return Solution(code=code, name=name)

    def to_dict(self):
        return {
            "method_name": self.name if self.name is not None else "ReEvo",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }
