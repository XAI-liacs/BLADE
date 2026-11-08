from __future__ import annotations

import logging
import re
import textwrap
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


_CODE_BLOCK_RE = re.compile(
    r"^\s*```(?:python)?\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)


def _normalize_code_snippet(text: str) -> str:
    """Take the code out of a prompt or response and remove its extra indentation."""
    if not text:
        return ""
    normalized = textwrap.dedent(text).strip()
    match = _CODE_BLOCK_RE.search(normalized)
    code = match.group(1) if match else normalized
    return textwrap.dedent(code).strip()


def _seed_payload(problem: Problem) -> tuple[str, str]:
    """Return the seed code inside a python code block, and its class name."""
    seed_code = _normalize_code_snippet(getattr(problem, "example_prompt", ""))
    if not seed_code:
        raise ValueError(
            "ReEvo requires an executable example_prompt to initialize its seed "
            "individual, but none could be extracted."
        )
    # Stop early if the example code in the prompt is not valid Python.
    compile(seed_code, "<reevo-seed>", "exec")
    class_name = first_class_name(seed_code) or "AlgorithmName"
    return f"```python\n{seed_code}\n```", class_name


def _func_signature(problem: Problem) -> str:
    func_name = getattr(problem, "func_name", "__call__")
    func_inputs = list(getattr(problem, "func_inputs", []) or [])
    if func_inputs:
        return f"{func_name}(self, {', '.join(func_inputs)})"
    return f"{func_name}(self)"


def _func_desc(problem: Problem, class_name: str) -> str:
    init_inputs = ", ".join(getattr(problem, "init_inputs", []) or [])
    call_inputs = ", ".join(getattr(problem, "func_inputs", []) or [])
    parts = [
        f"Implement a single Python class called `{class_name}`.",
    ]
    if init_inputs:
        parts.append(
            f"The constructor signature should be `__init__(self, {init_inputs})`."
        )
    if call_inputs:
        parts.append(
            f"The main inference signature should be `{getattr(problem, 'func_name', '__call__')}(self, {call_inputs})`."
        )
    return " ".join(parts)


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
        minimisation = getattr(problem, "minimisation", False)
        for response_id in range(len(population)):
            individual = population[response_id]
            reevo.function_evals += 1
            raw_code = individual.get("code")
            if raw_code is None:
                individual["exec_success"] = False
                individual["obj"] = ((-1) ** int(not minimisation)) * float("inf")
                continue
            individual["code"] = _normalize_code_snippet(raw_code)
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
                individual["obj"] = float("inf")
                population[response_id] = reevo.mark_invalid_individual(
                    individual, solution.error
                )
                continue
            # Re-Evo always minimizes. (while BLADE problems are maximization)
            individual["obj"] = ((-1) ** int(not minimisation)) * solution.fitness

            individual["exec_success"] = True
            population[response_id] = individual
        return population

    def __call__(self, problem: Problem):
        if ReEvoAlgorithm is None:
            raise ImportError(
                "reevo package is not installed, please install it using `poetry install --with methods`."
            )

        from omegaconf import OmegaConf

        seed_func, seed_class_name = _seed_payload(problem)
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
                "func_name": seed_class_name,
                "seed_func": seed_func,
                "func_signature": _func_signature(problem),
                "obj_type": "max",
                "problem_type": "blade",
                "func_desc": _func_desc(problem, seed_class_name),
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
        code = _normalize_code_snippet(code)
        name = first_class_name(code) or "AlgorithmName"
        sol = Solution(code=code, name=name)
        sol.set_scores(abs(reevo.best_obj_overall), "", "")
        return sol

    def to_dict(self):
        return {
            "method_name": self.name if self.name is not None else "ReEvo",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }

    def get_config(self) -> dict[str, Any]:
        config = self.kwargs.copy()
        config["budget"] = self.budget
        return {
            "name": self.name,
            "source": "https://github.com/nikivanstein/reevo",
            "config": config,
        }
