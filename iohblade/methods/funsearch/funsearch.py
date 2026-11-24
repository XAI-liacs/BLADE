"""FunSearch as a BLADE method.

Based on Romera-Paredes et al., "Mathematical discoveries from program search
with large language models", Nature 625, 468-475 (2024), its reference code
(https://github.com/google-deepmind/funsearch, Apache-2.0) and the LLM4AD
implementation (https://github.com/Optima-CityU/llm4ad, BSD-2-Clause; Liu et
al., "LLM4AD: A Platform for Algorithm Design with Large Language Model",
arXiv:2412.17287).
"""

from __future__ import annotations

from typing import Any

from ...fitness import Fitness, is_better, to_minimised
from ...llm import LLM
from ...method import Method
from ...problem import Problem
from ...solution import Solution
from ..prompt_utils import task_context
from .programs_database import ProgramsDatabase


class FunSearch(Method):
    """FunSearch: island-based evolution with best-shot prompting.

    The programs database holds ``num_islands`` islands. On each island,
    programs are clustered by score. A prompt shows ``programs_per_prompt``
    programs of one island, sampled by a softmax over cluster scores and
    ordered from worst to best, and asks for an improved next version.
    The ``samples_per_prompt`` new programs are evaluated and added to that
    island. Every ``reset_period`` evaluations the weaker half of the islands
    is restarted from the best program of a stronger island.

    Differences with the reference implementation:

    * a program is a complete solution (for example a class), not a single
      function in a fixed skeleton; the problem's task and format prompts
      frame the versions;
    * the first program is sampled from the problem prompt (the reference
      starts from a hand-written template), and is added to every island;
    * islands are reset after a number of evaluations instead of a
      wall-clock period, so runs are reproducible with the experiment seed;
    * samples are drawn and evaluated one at a time (no parallel samplers).

    Exactly ``budget`` solutions are evaluated. Failed evaluations count
    towards the budget but are not added to the database, and are never
    returned as the best solution unless no evaluation succeeded.
    Only single-objective (scalar) fitness is supported.

    Args:
        llm: The LLM used to generate programs.
        budget: Number of evaluations.
        name: Name of the method.
        num_islands: Number of islands.
        programs_per_prompt: Number of programs shown in a prompt (k).
        samples_per_prompt: Number of programs sampled from each prompt.
        reset_period: Evaluations between island resets. ``None`` uses
            ``budget // 5`` (four resets per run); ``0`` disables resets.
            There is no reset after the last evaluation.
        temperature_init: Initial temperature of the cluster sampling.
        temperature_period: Period, in programs per island, of the linear
            decay of the cluster sampling temperature.
    """

    def __init__(
        self,
        llm: LLM,
        budget: int,
        name: str = "FunSearch",
        num_islands: int = 10,
        programs_per_prompt: int = 2,
        samples_per_prompt: int = 4,
        reset_period: int | None = None,
        temperature_init: float = 0.1,
        temperature_period: int = 30_000,
    ):
        super().__init__(llm, budget, name)
        _check_int("budget", budget, 0)
        _check_int("num_islands", num_islands, 1)
        _check_int("programs_per_prompt", programs_per_prompt, 1)
        _check_int("samples_per_prompt", samples_per_prompt, 1)
        if reset_period is not None:
            _check_int("reset_period", reset_period, 0)
        _check_int("temperature_period", temperature_period, 1)
        if not isinstance(temperature_init, (int, float)) or temperature_init <= 0:
            raise ValueError(
                f"temperature_init must be a number > 0 (got {temperature_init!r})."
            )
        self.num_islands = num_islands
        self.programs_per_prompt = programs_per_prompt
        self.samples_per_prompt = samples_per_prompt
        self.reset_period = reset_period
        self.temperature_init = float(temperature_init)
        self.temperature_period = temperature_period
        self.database: ProgramsDatabase | None = None
        self.minimisation = False

    # region Run.
    def __call__(self, problem: Problem) -> Solution | None:
        """Run FunSearch on ``problem``.

        Returns:
            Solution: The best solution found (None when ``budget`` is 0).
        """
        self.minimisation = bool(getattr(problem, "minimisation", False))
        self.database = ProgramsDatabase(
            num_islands=self.num_islands,
            programs_per_prompt=self.programs_per_prompt,
            temperature_init=self.temperature_init,
            temperature_period=self.temperature_period,
        )
        reset_period = self._reset_period()
        next_reset = reset_period
        self._evaluations = 0
        self._best: Solution | None = None

        # The first programs come from the problem prompt; they are added to
        # every island. Retry until one succeeds (or the budget is spent).
        while self._evaluations < self.budget and self.database.empty:
            solution = self._evaluate(problem, problem.get_prompt(), [], 0)
            score = self._score(solution)
            if score is not None:
                self.database.register(solution, None, score)

        generation = 1
        while self._evaluations < self.budget:
            prompt = self.database.get_prompt()
            text = self._prompt(problem, prompt.programs)
            parent_ids = [program.id for program in prompt.programs]
            for _ in range(
                min(self.samples_per_prompt, self.budget - self._evaluations)
            ):
                solution = self._evaluate(problem, text, parent_ids, generation)
                solution.metadata["island"] = prompt.island_id
                score = self._score(solution)
                if score is not None:
                    self.database.register(solution, prompt.island_id, score)
            if reset_period and next_reset <= self._evaluations < self.budget:
                self.database.reset_islands()
                next_reset = (self._evaluations // reset_period + 1) * reset_period
            generation += 1
        return self._best

    def _reset_period(self) -> int:
        if self.reset_period is not None:
            return self.reset_period
        return self.budget // 5

    def _evaluate(
        self, problem: Problem, prompt: str, parent_ids: list[str], generation: int
    ) -> Solution:
        solution = self.llm.sample_solution(
            [{"role": "user", "content": prompt}], parent_ids=parent_ids
        )
        solution.generation = generation
        solution = problem(solution)
        self._evaluations += 1
        if self._best is None or is_better(
            solution.fitness, self._best.fitness, self.minimisation
        ):
            self._best = solution
        return solution

    def _score(self, solution: Solution) -> float | None:
        """The score FunSearch maximises, or None for a failed evaluation."""
        if isinstance(solution.fitness, Fitness):
            raise ValueError(
                "FunSearch supports single-objective problems only; "
                f"'{solution.name}' got a multi-objective fitness "
                f"({solution.fitness})."
            )
        objective = to_minimised(solution.fitness, self.minimisation)
        if objective == float("inf"):
            return None
        return -objective

    def _prompt(self, problem: Problem, programs: list[Solution]) -> str:
        """A best-shot prompt: the programs as versions, worst first."""
        k = len(programs)
        # The task with the problem's extra context, but without its example
        # (the programs below take its place) and format (added at the end).
        parts = [task_context(problem)]
        parts.append(
            f"Below are {k} version(s) of an algorithm for this task, ordered "
            "from worst to best: each version is an improved version of the "
            "previous one."
            if k > 1
            else "Below is version 0 of an algorithm for this task."
        )
        for i, program in enumerate(programs):
            header = f"# Version {i}"
            if i > 0:
                header += f" (improved version of version {i - 1})"
            parts.append(f"{header}\n```python\n{program.code.strip()}\n```")
        parts.append(
            f"Write version {k}: an improved version of version {k - 1}. "
            "Give the complete code."
        )
        parts.append(problem.format_prompt.strip())
        return "\n\n".join(part for part in parts if part)

    # endregion

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.name if self.name is not None else "FunSearch",
            "budget": self.budget,
            **self._params(),
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "FunSearch",
            "source": "https://github.com/XAI-liacs/BLADE/blob/main/iohblade/methods/funsearch",
            "config": self._params(),
        }

    def _params(self) -> dict[str, Any]:
        return {
            "num_islands": self.num_islands,
            "programs_per_prompt": self.programs_per_prompt,
            "samples_per_prompt": self.samples_per_prompt,
            "reset_period": self.reset_period,
            "temperature_init": self.temperature_init,
            "temperature_period": self.temperature_period,
        }


def _check_int(name: str, value: Any, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value!r}).")


# Old name of the class (the stub before #216), kept for existing scripts.
funsearch = FunSearch
