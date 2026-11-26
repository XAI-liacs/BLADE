from __future__ import annotations

import builtins
import contextlib
import logging
import os
import re
import tempfile
import textwrap
import threading
import traceback
from types import SimpleNamespace
from typing import Any

from ..fitness import from_minimised, to_minimised
from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import class_info, first_class_name
from .prompt_utils import declared_interface_matches, example_interface

try:
    import reevo as _reevo_module
    from reevo import ReEvo as ReEvoAlgorithm
    from reevo.utils.llm_client.base import BaseClient
except Exception:  # pragma: no cover - optional dependency
    _reevo_module = None
    ReEvoAlgorithm = None
    BaseClient = object  # type: ignore


# The ReEvo library writes its reflection files
# (``problem_iter*_short_term_reflections.txt`` etc.) with ``open()`` on bare
# relative file names, i.e. into the current working directory, whatever
# ``root_dir`` is. Changing the working directory would affect every thread
# of the process (parallel runs, loggers with relative paths), so instead the
# library module gets its own ``open`` that puts relative paths into the
# output directory of the run executing in the calling thread.
_output_dir = threading.local()


def _reevo_open(file, *args, **kwargs):
    directory = getattr(_output_dir, "path", None)
    if directory and isinstance(file, (str, os.PathLike)) and not os.path.isabs(file):
        file = os.path.join(directory, file)
    return builtins.open(file, *args, **kwargs)


def _install_open_shim(module) -> None:
    if module is not None and getattr(module, "open", None) is not _reevo_open:
        module.open = _reevo_open


@contextlib.contextmanager
def _files_in(directory: str):
    """Make ReEvo write its relative files into ``directory`` (this thread)."""
    _install_open_shim(_reevo_module)
    previous = getattr(_output_dir, "path", None)
    _output_dir.path = directory
    try:
        yield
    finally:
        _output_dir.path = previous


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


def _interface(problem: Problem) -> tuple[list[str], str, list[str]]:
    """``(init_inputs, func_name, func_inputs)`` to state in ReEvo's prompts.

    The interface the problem declares, unless it does not match the class
    in the problem's example code: ``Problem`` defaults it to the BBOB
    interface (``__init__(self, budget, dim)``, ``__call__(self, func)``),
    which contradicts the task of most other problems, so the example's
    interface is used then (see
    :func:`~iohblade.methods.prompt_utils.declared_interface_matches`).
    """
    if not declared_interface_matches(problem):
        example = example_interface(problem)
        if example["func_name"]:
            return (
                example["init_inputs"],
                example["func_name"],
                example["func_inputs"],
            )
        return example["init_inputs"], "__call__", []
    return (
        list(getattr(problem, "init_inputs", []) or []),
        getattr(problem, "func_name", "__call__"),
        list(getattr(problem, "func_inputs", []) or []),
    )


def _func_signature(problem: Problem) -> str:
    _, func_name, func_inputs = _interface(problem)
    if func_inputs:
        return f"{func_name}(self, {', '.join(func_inputs)})"
    return f"{func_name}(self)"


def _func_desc(problem: Problem, class_name: str) -> str:
    init, func_name, call = _interface(problem)
    init_inputs = ", ".join(init)
    call_inputs = ", ".join(call)
    parts = [
        f"Implement a single Python class called `{class_name}`.",
    ]
    if init_inputs:
        parts.append(
            f"The constructor signature should be `__init__(self, {init_inputs})`."
        )
    if call_inputs:
        parts.append(
            f"The main inference signature should be `{func_name}(self, {call_inputs})`."
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
    """Wrapper for the ReEvo baseline.

    ReEvo writes reflection files (``problem_iter*_*.txt``). They go to
    ``output_path`` if given; otherwise to a ``reevo`` folder in the run's
    log directory when the problem has a logger, else to a temporary
    directory that is removed afterwards. (Never to the working directory.)
    """

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
                # No code in the response: invalid (ReEvo minimises, so inf).
                # ReEvo still reads the code of an all-invalid population.
                individual["code"] = ""
                individual["exec_success"] = False
                individual["obj"] = float("inf")
                population[response_id] = reevo.mark_invalid_individual(
                    individual, "No code found in the response."
                )
                continue
            individual["code"] = _normalize_code_snippet(raw_code)
            solution = Solution(
                code=individual["code"],
                name=first_class_name(individual["code"]) or "AlgorithmName",
                description=class_info(individual["code"])[1]
                or "No description provided.",
            )
            solution = problem(solution)

            objective = to_minimised(solution.fitness, minimisation)
            if solution.error != "" or objective == float("inf"):
                # An error or a failed evaluation (non-finite fitness): invalid.
                individual["exec_success"] = False
                individual["obj"] = float("inf")
                population[response_id] = reevo.mark_invalid_individual(
                    individual,
                    solution.error or solution.feedback or "Invalid fitness.",
                )
                continue
            # ReEvo always minimises; to_minimised negates maximised fitness.
            individual["obj"] = objective

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

        with contextlib.ExitStack() as stack:
            output_dir = self._output_dir(problem, stack)
            stack.enter_context(_files_in(output_dir))
            reevo = ReEvoAlgorithm(cfg, root_dir=output_dir, generator_llm=client)
            # Override evaluation to use BLADE problems
            reevo.evaluate_population = lambda pop: self._eval_population(
                reevo, pop, problem
            )
            reevo.init_population()
            code, _ = reevo.evolve()
        code = _normalize_code_snippet(code)
        name = first_class_name(code) or "AlgorithmName"
        sol = Solution(code=code, name=name)
        sol.set_scores(
            from_minimised(
                reevo.best_obj_overall, getattr(problem, "minimisation", False)
            ),
            "",
            "",
        )
        return sol

    def _output_dir(self, problem: Problem, stack: contextlib.ExitStack) -> str:
        """The directory for ReEvo's files (see the class docstring)."""
        explicit = self.kwargs.get("output_path")
        if explicit:
            path = os.path.abspath(explicit)
        elif getattr(problem, "logger", None) is not None and getattr(
            problem, "logger_dir", ""
        ):
            path = os.path.abspath(os.path.join(problem.logger_dir, "reevo"))
        else:
            return stack.enter_context(tempfile.TemporaryDirectory(prefix="reevo-"))
        os.makedirs(path, exist_ok=True)
        return path

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
