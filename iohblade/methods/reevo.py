from __future__ import annotations

import logging
import traceback
from types import SimpleNamespace
from typing import Any

from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import class_info, first_class_name

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
            responses.append({"message": {"role": "assistant", "content": content}})
        return responses

    def multi_chat_completion(
        self,
        messages_list,
        n=1,
        temperature=None,
    ):
        """Sequentially generate responses for many independent conversations.

        Parameters
        ----------
        messages_list : list[list[dict]] | list[dict]
            Either *one* conversation (list[dict]) or a list of conversations.
        n : int, default=1
            Number of completions **per conversation**.  For multiple
            conversations `n` must remain 1.
        temperature : float | None, default=None
            Sampling temperature.

        Returns
        -------
        list[str]
            The content field of each returned message, flattened.
        """
        # Normalise input ---------------------------------------------------
        if not isinstance(messages_list, list):
            raise TypeError("messages_list must be a list")
        if messages_list and not isinstance(messages_list[0], list):
            # Single conversation – wrap it so downstream code can iterate.
            messages_list = [messages_list]  # type: ignore[assignment]

        if len(messages_list) > 1 and n != 1:
            raise ValueError(
                "Currently, only n=1 is supported for multi‑chat completion."
            )

        # ------------------------------------------------------------------
        # Sequential execution (no ThreadPoolExecutor) ----------------------
        # ------------------------------------------------------------------
        contents = []
        for msgs in messages_list:
            for choice in self.chat_completion(
                n=n, messages=msgs, temperature=temperature
            ):
                contents.append(choice["message"]["content"])
        return contents


class ReEvo(Method):
    """Wrapper for the ReEvo baseline."""

    def __init__(self, llm: LLM, budget: int, name: str = "ReEvo", **kwargs: Any):
        super().__init__(llm, budget, name)
        self.kwargs = kwargs

    def _eval_population(self, reevo: Any, population: list[dict], problem: Problem):
        for response_id in range(len(population)):
            individual = population[response_id]
            reevo.function_evals += 1
            if individual.get("code") is None:
                individual["exec_success"] = False
                individual["obj"] = float("inf")
                continue
            solution = Solution(
                code=individual["code"],
                name=first_class_name(individual["code"]) or "AlgorithmName",
                description=class_info(individual["code"])[1]
                or "No description provided.",
            )
            solution = problem(solution)

            if solution.error != "":
                # If the solution has an error, we mark it as invalid.
                individual["exec_success"] = False
                individual["obj"] = -solution.fitness
                population[response_id] = reevo.mark_invalid_individual(
                    individual, solution.error
                )
                continue
            # Re-Evo always minimizes. (while BLADE problems are maximization)
            individual["obj"] = -solution.fitness

            individual["exec_success"] = True
            population[response_id] = individual
        return population

    def __call__(self, problem: Problem):
        if ReEvoAlgorithm is None:
            raise ImportError(
                "reevo package is not installed, please install it using `poetry install --with methods`."
            )

        from omegaconf import OmegaConf

        cfg_dict = {
            "max_fe": self.budget,
            "pop_size": self.kwargs.get("pop_size", 10),
            "init_pop_size": self.kwargs.get("init_pop_size", 20),
            "mutation_rate": self.kwargs.get("mutation_rate", 0.5),
            "timeout": self.kwargs.get("timeout", 20),
            "problem": {
                "problem_name": problem.name,
                "description": problem.task_prompt,
                "problem_size": getattr(problem, "dim", 1),
                "func_name": "AlgorithmName",
                "seed_func": problem.example_prompt,
                "func_signature": f"{problem.func_name}(self)",
                "obj_type": "max",
                "problem_type": "blade",
                "func_desc": "",
                "external_knowledge": "",
            },
        }
        cfg = OmegaConf.create(cfg_dict)
        client = _BladeReEvoClient(self.llm)
        reevo = ReEvoAlgorithm(
            cfg, root_dir=self.kwargs.get("output_path", "./"), generator_llm=client
        )
        # Override evaluation to use BLADE problems
        reevo.evaluate_population = lambda pop: self._eval_population(
            reevo, pop, problem
        )
        reevo.init_population()
        code, _ = reevo.evolve()
        name = first_class_name(code) or "AlgorithmName"
        sol = Solution(code=code, name=name)
        sol.set_scores(-reevo.best_obj_overall, "", "")
        return sol

    def to_dict(self):
        return {
            "method_name": self.name if self.name is not None else "ReEvo",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }
