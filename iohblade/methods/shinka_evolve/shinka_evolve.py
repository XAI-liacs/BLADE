"""ShinkaEvolve as a BLADE method.

Based on Lange et al., "ShinkaEvolve: Towards Open-Ended and Sample-Efficient
Program Evolution" (arXiv:2509.19349, 2025) and its code
(https://github.com/SakanaAI/ShinkaEvolve, version 0.0.8, Apache-2.0).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ...fitness import Fitness, is_better, to_minimised
from ...llm import LLM, Multi_LLM
from ...method import Method
from ...problem import Problem
from ...solution import Solution
from ..prompt_utils import task_context
from . import prompts
from .bandit import AsymmetricUCB
from .database import PARENT_SELECTIONS, Program, ProgramDatabase, Sample
from .novelty import Embedding, cosine_similarity, resolve_embedding
from .patches import PatchError, apply_diff, extract_code, extract_tag

PATCH_TYPES = ("diff", "full", "cross")


class ShinkaEvolve(Method):
    """ShinkaEvolve: sample-efficient program evolution.

    Each generation evaluates exactly one program:

    1. **LLM choice.** When the LLM is a ``Multi_LLM`` (``provider: multi``),
       an asymmetric UCB bandit chooses one of its LLMs; the reward is a
       child's improvement over its parent and the initial program.
    2. **Parent and inspirations.** An island is drawn uniformly; a parent
       is drawn from its archive programs (``parent_selection``), together
       with ``num_archive_inspirations`` + ``num_top_k_inspirations``
       other programs that are shown in the prompt, worst first.
    3. **Patch.** A diff (SEARCH/REPLACE edits), a full rewrite (one of
       five prompt variants) or a crossover with an inspiration, drawn with
       ``patch_type_probs``. On an island without a correct program, a
       failed program is fixed instead. A reply that cannot be applied is
       retried in the same conversation (``max_patch_attempts``), then
       with a new parent (``max_patch_resamples``).
    4. **Novelty rejection** (optional). A candidate whose code embedding
       is more similar than ``novelty_threshold`` to a program on the island
       is rejected, unless an LLM judge finds it meaningfully different;
       a new candidate is then proposed (``max_novelty_attempts``).
    5. **Meta recommendations** (optional). Every ``meta_interval``
       evaluations the LLM summarises the new programs, updates its global
       insights and writes ``meta_max_recommendations`` recommendations;
       one of them is added to later diff and full-rewrite prompts.

    The first program is sampled from the problem prompt and copied to
    every island; children stay on their parent's island, and programs
    migrate between islands when ``migration_rate`` > 0.

    Differences with ShinkaEvolve, which are needed for BLADE's contracts:

    * **Budget.** ShinkaEvolve counts generations and loses a generation
      when no patch applies or all candidates are rejected as not novel.
      Here every generation ends with one evaluation: if no patch applies,
      the code of the last reply is evaluated (and fails if it has none);
      if every candidate is rejected, the last one is evaluated. Rejected
      candidates are not evaluated, but their LLM calls are logged.
    * **Programs** are complete solutions without EVOLVE-BLOCK markers;
      the problem's task prompt is part of every prompt, and fitness is
      shown with the problem's direction (``problem.minimisation``).
    * **Embeddings** default to a local character n-gram embedding instead
      of an embedding API; pass a function from code to a vector to use
      another. Novelty rejection and meta recommendations are off by
      default, as in ShinkaEvolve's packaged defaults.
    * **The bandit** is the linear reward variant without API-cost
      awareness; there is no random temperature per query.
    * Proposals and evaluations run one at a time (no asynchronous jobs).

    Failed evaluations never become the best solution unless no evaluation
    succeeded. Only single-objective (scalar) fitness is supported.

    Args:
        llm: The LLM, or a ``Multi_LLM`` to let the bandit choose LLMs.
        budget: Number of evaluations.
        name: Name of the method.
        num_islands: Number of islands.
        archive_size: Maximum number of programs in the archive.
        num_archive_inspirations: Archive programs shown in a prompt.
        num_top_k_inspirations: Best programs shown in a prompt.
        elite_selection_ratio: Share of archive inspirations taken from the
            island's best programs.
        parent_selection: ``weighted`` (sigmoid of the score relative to
            the median, divided by 1 + number of children), ``power_law``
            or ``beam_search``.
        parent_selection_lambda: Sharpness of the weighted selection.
        exploitation_alpha: Exponent of the power-law selection.
        num_beams: Children of a beam-search parent.
        migration_interval: Generations between migrations.
        migration_rate: Share of an island's programs that migrate.
        patch_type_probs: Probabilities of ``diff``, ``full`` and
            ``cross`` patches.
        max_patch_attempts: LLM replies per parent before resampling.
        max_patch_resamples: Parents tried per generation.
        use_text_feedback: Show the evaluation feedback in prompts.
        novelty_rejection: Reject candidates that are too similar.
        novelty_threshold: Cosine similarity above which a candidate is
            checked (or rejected without a judge).
        max_novelty_attempts: Candidates per generation.
        novelty_judge: Ask the LLM whether a similar candidate is novel.
        embedding: ``"ngram"`` or a function from code to a vector.
        meta_interval: Evaluations between meta recommendation updates
            (``None``: off).
        meta_max_recommendations: Number of recommendations.
        task_sys_msg: The system message (Shinka's default if None).
        bandit_c: Exploration coefficient of the bandit.
        bandit_epsilon: Exploration probability of the bandit.
        bandit_decay: Decay of the bandit statistics.
    """

    def __init__(
        self,
        llm: LLM,
        budget: int,
        name: str = "ShinkaEvolve",
        num_islands: int = 2,
        archive_size: int = 40,
        num_archive_inspirations: int = 1,
        num_top_k_inspirations: int = 1,
        elite_selection_ratio: float = 0.3,
        parent_selection: str = "weighted",
        parent_selection_lambda: float = 10.0,
        exploitation_alpha: float = 1.0,
        num_beams: int = 5,
        migration_interval: int = 10,
        migration_rate: float = 0.0,
        patch_type_probs: dict[str, float] | None = None,
        max_patch_attempts: int = 1,
        max_patch_resamples: int = 3,
        use_text_feedback: bool = False,
        novelty_rejection: bool = False,
        novelty_threshold: float = 0.99,
        max_novelty_attempts: int = 3,
        novelty_judge: bool = True,
        embedding: str | Embedding = "ngram",
        meta_interval: int | None = None,
        meta_max_recommendations: int = 5,
        task_sys_msg: str | None = None,
        bandit_c: float = 1.0,
        bandit_epsilon: float = 0.2,
        bandit_decay: float = 0.95,
    ):
        super().__init__(llm, budget, name)
        for key, value, minimum in (
            ("budget", budget, 0),
            ("num_islands", num_islands, 1),
            ("archive_size", archive_size, 1),
            ("num_archive_inspirations", num_archive_inspirations, 0),
            ("num_top_k_inspirations", num_top_k_inspirations, 0),
            ("num_beams", num_beams, 1),
            ("migration_interval", migration_interval, 1),
            ("max_patch_attempts", max_patch_attempts, 1),
            ("max_patch_resamples", max_patch_resamples, 1),
            ("max_novelty_attempts", max_novelty_attempts, 1),
            ("meta_max_recommendations", meta_max_recommendations, 1),
        ):
            _check_int(key, value, minimum)
        if meta_interval is not None:
            _check_int("meta_interval", meta_interval, 1)
        for key, value, low, high in (
            ("elite_selection_ratio", elite_selection_ratio, 0.0, 1.0),
            ("migration_rate", migration_rate, 0.0, 1.0),
            ("novelty_threshold", novelty_threshold, -1.0, 1.0),
            ("bandit_epsilon", bandit_epsilon, 0.0, 1.0),
            ("bandit_decay", bandit_decay, 1e-12, 1.0),
        ):
            _check_number(key, value, low, high)
        _check_number("parent_selection_lambda", parent_selection_lambda, 1e-12)
        _check_number("exploitation_alpha", exploitation_alpha, 0.0)
        _check_number("bandit_c", bandit_c, 0.0)
        if parent_selection not in PARENT_SELECTIONS:
            raise ValueError(
                f"parent_selection must be one of {', '.join(PARENT_SELECTIONS)} "
                f"(got {parent_selection!r})."
            )
        self.patch_type_probs = _patch_type_probs(patch_type_probs)
        self.embedding = embedding
        self._embed = resolve_embedding(embedding)
        self.params = {
            "num_islands": num_islands,
            "archive_size": archive_size,
            "num_archive_inspirations": num_archive_inspirations,
            "num_top_k_inspirations": num_top_k_inspirations,
            "elite_selection_ratio": elite_selection_ratio,
            "parent_selection": parent_selection,
            "parent_selection_lambda": parent_selection_lambda,
            "exploitation_alpha": exploitation_alpha,
            "num_beams": num_beams,
            "migration_interval": migration_interval,
            "migration_rate": migration_rate,
            "patch_type_probs": dict(self.patch_type_probs),
            "max_patch_attempts": max_patch_attempts,
            "max_patch_resamples": max_patch_resamples,
            "use_text_feedback": use_text_feedback,
            "novelty_rejection": novelty_rejection,
            "novelty_threshold": novelty_threshold,
            "max_novelty_attempts": max_novelty_attempts,
            "novelty_judge": novelty_judge,
            "embedding": embedding if isinstance(embedding, str) else "custom",
            "meta_interval": meta_interval,
            "meta_max_recommendations": meta_max_recommendations,
            "task_sys_msg": task_sys_msg,
            "bandit_c": bandit_c,
            "bandit_epsilon": bandit_epsilon,
            "bandit_decay": bandit_decay,
        }
        self.task_sys_msg = task_sys_msg or prompts.DEFAULT_TASK_SYS_MSG
        self.database: ProgramDatabase | None = None
        self.bandit: AsymmetricUCB | None = None
        self.recommendations: list[str] = []
        self.insights = ""
        self.minimisation = False

    def __getattr__(self, key):  # parameters as attributes
        params = self.__dict__.get("params", {})
        if key in params:
            return params[key]
        raise AttributeError(key)

    # region Run.
    def __call__(self, problem: Problem) -> Solution | None:
        """Run ShinkaEvolve on ``problem``.

        Returns:
            Solution: The best solution found (None when ``budget`` is 0).
        """
        self.minimisation = bool(getattr(problem, "minimisation", False))
        p = self.params
        self.database = ProgramDatabase(
            num_islands=p["num_islands"],
            archive_size=p["archive_size"],
            num_archive_inspirations=p["num_archive_inspirations"],
            num_top_k_inspirations=p["num_top_k_inspirations"],
            elite_selection_ratio=p["elite_selection_ratio"],
            parent_selection=p["parent_selection"],
            parent_selection_lambda=p["parent_selection_lambda"],
            exploitation_alpha=p["exploitation_alpha"],
            num_beams=p["num_beams"],
            migration_interval=p["migration_interval"],
            migration_rate=p["migration_rate"],
        )
        self.arms = (
            list(self.llm.llms) if isinstance(self.llm, Multi_LLM) else [self.llm]
        )
        self.bandit = AsymmetricUCB(
            len(self.arms), p["bandit_c"], p["bandit_epsilon"], p["bandit_decay"]
        )
        self.recommendations, self.insights = [], ""
        self._since_meta: list[Program] = []
        self._evaluations = 0
        self._best: Solution | None = None
        if self.budget == 0:
            return None

        # Generation 0: the initial program, from the problem prompt.
        solution = self.llm.sample_solution(
            [{"role": "user", "content": problem.get_prompt()}]
        )
        solution.generation = 0
        solution.operator = "init"
        score = self._evaluate(problem, solution)
        initial = self.database.add_initial(
            solution, score, self._embedding(solution.code)
        )
        self.bandit.set_baseline(score)
        self._after_evaluation(initial)

        generation = 1
        while self._evaluations < self.budget:
            arm = self.bandit.select()
            llm = self.arms[arm]
            solution, sample = self._propose(problem, llm, generation)
            score = self._evaluate(problem, solution)
            if score is not None and self.database.best() is None:
                self.bandit.set_baseline(score)
            program = self.database.add(
                solution,
                score,
                sample.parent,
                generation,
                self._embedding(solution.code),
            )
            solution.metadata["island"] = program.island
            self.bandit.update(arm, score, sample.parent.score)
            self.database.migrate(generation)
            self._after_evaluation(program)
            generation += 1
        return self._best

    def _evaluate(self, problem: Problem, solution: Solution) -> float | None:
        """Evaluate ``solution``; return its (maximised) score, or None."""
        solution = problem(solution)
        self._evaluations += 1
        if self._best is None or is_better(
            solution.fitness, self._best.fitness, self.minimisation
        ):
            self._best = solution
        if isinstance(solution.fitness, Fitness):
            raise ValueError(
                "ShinkaEvolve supports single-objective problems only; "
                f"'{solution.name}' got a multi-objective fitness "
                f"({solution.fitness})."
            )
        objective = to_minimised(solution.fitness, self.minimisation)
        return None if objective == float("inf") else -objective

    def _embedding(self, code: str):
        if not self.params["novelty_rejection"]:
            return None
        return np.asarray(self._embed(code), dtype=float)

    # endregion

    # region Proposals.
    def _propose(
        self, problem: Problem, llm: LLM, generation: int
    ) -> tuple[Solution, Sample]:
        """Propose the program of a generation (not yet evaluated)."""
        p = self.params
        rejections = 0
        for attempt in range(p["max_novelty_attempts"]):
            solution, sample = self._patch(problem, llm)
            solution.generation = generation
            if attempt == p["max_novelty_attempts"] - 1 or self._is_novel(
                solution, sample
            ):
                break
            rejections += 1
        solution.metadata["novelty_rejections"] = rejections
        solution.metadata["llm"] = getattr(llm, "model", "")
        return solution, sample

    def _patch(self, problem: Problem, llm: LLM) -> tuple[Solution, Sample]:
        """Sample a parent and turn an LLM reply into a new program."""
        p = self.params
        reply, error = "", None
        for _ in range(p["max_patch_resamples"]):
            sample = self.database.sample()
            patch_type = self._patch_type(sample)
            prompt, partner, variant = self._prompt(problem, sample, patch_type)
            messages = [{"role": "user", "content": prompt}]
            for _ in range(p["max_patch_attempts"]):
                reply = llm.query(messages)
                try:
                    if patch_type == "diff":
                        code, _ = apply_diff(sample.parent.solution.code, reply)
                    else:
                        code = extract_code(reply)
                except PatchError as e:
                    error = str(e)
                    messages += [
                        {"role": "assistant", "content": reply},
                        {
                            "role": "user",
                            "content": prompts.PATCH_RETRY_MSG.format(error=error),
                        },
                    ]
                    continue
                parents = [sample.parent.id] + ([partner.id] if partner else [])
                solution = self._solution(llm, code, reply, parents, patch_type)
                solution.metadata["full_variant"] = variant
                return solution, sample
        # No reply could be applied: evaluate the code of the last reply (a
        # program that raises the extraction error when it has none).
        code = llm.extract_algorithm_code(reply)
        solution = self._solution(llm, code, reply, [sample.parent.id], patch_type)
        solution.metadata["patch_error"] = error
        return solution, sample

    def _solution(self, llm, code, reply, parent_ids, patch_type) -> Solution:
        name = llm.extract_classname(code) or extract_tag(reply, "NAME")
        return Solution(
            code=code,
            name=name,
            description=extract_tag(reply, "DESCRIPTION"),
            parent_ids=parent_ids,
            operator=patch_type,
        )

    def _patch_type(self, sample: Sample) -> str:
        if sample.fix:
            return "fix"
        probs = dict(self.patch_type_probs)
        if not sample.inspirations:
            probs.pop("cross", None)
        if sum(probs.values()) <= 0:
            return "full"
        types = list(probs)
        weights = np.array([probs[t] for t in types], dtype=float)
        return types[np.random.choice(len(types), p=weights / weights.sum())]

    def _is_novel(self, solution: Solution, sample: Sample) -> bool:
        """Shinka's novelty rejection sampling, applied once every island
        has a correct program."""
        p = self.params
        if not p["novelty_rejection"] or not self.database.initialised:
            return True
        embedding = np.asarray(self._embed(solution.code), dtype=float)
        others = [
            program
            for program in self.database.island_programs(sample.parent.island)
            if program.embedding is not None
        ]
        if not others:
            return True
        similarities = [cosine_similarity(embedding, o.embedding) for o in others]
        best = int(np.argmax(similarities))
        if similarities[best] <= p["novelty_threshold"]:
            return True
        if not p["novelty_judge"]:
            return False
        reply = self.llm.query(
            [
                {
                    "role": "user",
                    "content": prompts.NOVELTY_SYSTEM_MSG
                    + "\n\n"
                    + prompts.NOVELTY_USER_MSG.format(
                        existing_code=others[best].solution.code,
                        proposed_code=solution.code,
                    ),
                }
            ]
        )
        return reply.strip().lstrip("*# ").upper().startswith("NOVEL")

    # endregion

    # region Prompts.
    def _fitness_text(self, program: Program) -> str:
        fitness = program.solution.fitness
        text = f"Fitness ({prompts.direction_text(self.minimisation)}): {fitness}"
        feedback = program.solution.feedback
        if self.params["use_text_feedback"] and feedback and feedback.strip():
            text += (
                "\n\nHere is additional text feedback about the current program:"
                f"\n\n{feedback.strip()}"
            )
        return text

    def _prompt(
        self, problem: Problem, sample: Sample, patch_type: str
    ) -> tuple[str, Program | None, str | None]:
        """The prompt of a patch; also the crossover partner and the full
        rewrite variant, if any."""
        parent = sample.parent
        system = f"{self.task_sys_msg}\n\n# Problem\n\n{task_context(problem)}"
        if self.recommendations and patch_type in ("diff", "full"):
            recommendation = self.recommendations[
                np.random.randint(len(self.recommendations))
            ]
            system += f"\n\n# Potential Recommendations\n\n{recommendation}"
        partner = variant = None
        if patch_type == "fix":
            error = parent.solution.error or parent.solution.feedback
            body = prompts.FIX_ITER_MSG.format(
                code=parent.solution.code.strip(),
                error=error.strip() or "No error output captured.",
            )
            return f"{system}\n\n{prompts.FIX_SYS_FORMAT}\n\n{body}", None, None
        if patch_type == "diff":
            fmt = prompts.DIFF_SYS_FORMAT
            task = prompts.DIFF_TASK.format(goal=prompts.goal_text(self.minimisation))
        elif patch_type == "full":
            variants = list(prompts.FULL_SYS_FORMATS)
            variant = variants[np.random.randint(len(variants))]
            fmt, task = prompts.FULL_SYS_FORMATS[variant], prompts.FULL_TASK
        else:
            fmt, task = prompts.CROSS_SYS_FORMAT, prompts.CROSS_TASK
            partner = sample.inspirations[np.random.randint(len(sample.inspirations))]
        history = ""
        if sample.inspirations:
            history = (
                "Here are the performance metrics of a set of previously "
                "implemented programs:\n\n# Prior programs\n\n"
            )
            for program in sample.inspirations:
                history += (
                    f"```python\n{program.solution.code.strip()}\n```\n\n"
                    f"Performance metrics:\n{self._fitness_text(program)}\n\n"
                )
        body = (
            f"{history}# Current program\n\nHere is the current program we "
            "are trying to improve:\n\n"
            f"```python\n{parent.solution.code.strip()}\n```\n\n"
            "Here are the performance metrics of the program:\n\n"
            f"{self._fitness_text(parent)}\n\n{task}"
        )
        if partner is not None:
            body += (
                "\n\n# Crossover Inspiration Programs\n"
                f"```python\n{partner.solution.code.strip()}\n```\n\n"
                f"Performance metrics: {self._fitness_text(partner)}"
            )
        return f"{system}\n\n{fmt}\n\n{body}", partner, variant

    # endregion

    # region Meta recommendations.
    def _after_evaluation(self, program: Program) -> None:
        interval = self.params["meta_interval"]
        if interval is None:
            return
        self._since_meta.append(program)
        if len(self._since_meta) >= interval and self._evaluations < self.budget:
            self._update_meta(self._since_meta)
            self._since_meta = []

    def _program_text(self, program: Program) -> str:
        status = (
            "The program is correct and passes all validation tests."
            if program.correct
            else "The program is incorrect and does not pass all validation tests."
        )
        return (
            f"```python\n{program.solution.code.strip()}\n```\n\n"
            f"Performance metrics:\n{self._fitness_text(program)}\n\n{status}"
        )

    def _update_meta(self, programs: list[Program]) -> None:
        """Shinka's three meta steps: program summaries, global insights,
        recommendations."""
        ask = lambda text: self.llm.query([{"role": "user", "content": text}])
        summaries = "\n\n".join(
            ask(prompts.META_STEP1.format(program=self._program_text(p)))
            for p in programs
        )
        best = self.database.best()
        best_text = self._program_text(best) if best else "None yet."
        self.insights = ask(
            prompts.META_STEP2.format(
                summaries=summaries, previous=self.insights or "None.", best=best_text
            )
        )
        previous = "\n".join(
            f"{i}. {r}" for i, r in enumerate(self.recommendations, start=1)
        )
        reply = ask(
            prompts.META_STEP3.format(
                insights=self.insights,
                previous=previous or "None.",
                best=best_text,
                n=self.params["meta_max_recommendations"],
            )
        )
        recommendations = _numbered_items(reply)
        if recommendations:
            self.recommendations = recommendations[
                : self.params["meta_max_recommendations"]
            ]

    # endregion

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.name if self.name is not None else "ShinkaEvolve",
            "budget": self.budget,
            **self.params,
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "name": "ShinkaEvolve",
            "source": "https://github.com/XAI-liacs/BLADE/blob/main/iohblade/methods/shinka_evolve",
            "config": dict(self.params),
        }


def _numbered_items(text: str) -> list[str]:
    """The items of a numbered list (an item continues on following lines
    until the next number)."""
    items: list[str] = []
    for line in text.splitlines():
        match = re.match(r"\s*\d+[.)]\s+(.*)", line)
        if match:
            items.append(match.group(1).strip())
        elif items and line.strip():
            items[-1] += " " + line.strip()
    return items


def _patch_type_probs(probs: dict[str, float] | None) -> dict[str, float]:
    if probs is None:
        return {"diff": 0.6, "full": 0.3, "cross": 0.1}
    if not isinstance(probs, dict) or not probs:
        raise ValueError(
            "patch_type_probs must be a mapping of diff, full and cross to "
            f"probabilities (got {probs!r})."
        )
    unknown = set(probs) - set(PATCH_TYPES)
    if unknown:
        raise ValueError(
            f"Unknown patch types {sorted(unknown)}; use {', '.join(PATCH_TYPES)}."
        )
    for key, value in probs.items():
        _check_number(f"patch_type_probs.{key}", value, 0.0)
    if sum(probs.values()) <= 0:
        raise ValueError("patch_type_probs must have a positive probability.")
    return {key: float(value) for key, value in probs.items()}


def _check_int(name: str, value: Any, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value!r}).")


def _check_number(name, value, low, high=float("inf")) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not low <= value <= high
    ):
        bounds = f">= {low}" if high == float("inf") else f"in [{low}, {high}]"
        raise ValueError(f"{name} must be a number {bounds} (got {value!r}).")
