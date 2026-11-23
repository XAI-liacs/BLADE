import ast
import json
import logging
import math
import os
import re
import textwrap
import threading
import warnings
from typing import Any, Optional, Tuple

from ..fitness import from_minimised, to_minimised
from ..llm import LLM
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import class_info, first_class_name

try:
    from eoh import eoh as eoh_main
    from eoh.methods.eoh import eoh_evolution
    from eoh.utils.getParas import Paras
except Exception:  # pragma: no cover - optional dependency
    eoh_main = None
    Paras = None
    eoh_evolution = None


class _BladePrompts:
    """Adapter exposing the prompt interface expected by EoH."""

    def __init__(self, problem: Problem):
        self.problem = problem

    def get_task(self):
        return self.problem.task_prompt

    def get_func_name(self):
        return self.problem.func_name

    def get_func_inputs(self):
        return self.problem.func_inputs

    def get_init_inputs(self):
        return self.problem.init_inputs

    def get_func_outputs(self):
        return self.problem.func_outputs

    def get_inout_inf(self):
        return (
            f"Implement a Python class called `AlgorithmName` with"
            f" an __init__(self, {', '.join(self.get_init_inputs())}) and a function {self.get_func_name()}(self, {', '.join(self.get_func_inputs())})"
            f" returning {', '.join(self.get_func_outputs())}."
        )

    def get_other_inf(self):
        return (
            self.problem.task_prompt
            + self.problem.example_prompt
            + self.problem.format_prompt
        )


# Evaluations EoH makes, read from the installed ``eoh`` package
# (``eoh/methods/eoh/eoh.py`` and ``eoh_interface_EC.py``):
#
# * initialisation: ``population_generation`` calls
#   ``get_algorithm([], "i1")`` ``n_create = 2`` times, and each call makes
#   ``pop_size`` offspring, so ``2 * pop_size`` evaluations;
# * each generation: every operator (default ``e1, e2, m1, m2``, all with
#   weight 1) calls ``get_algorithm`` once, so ``n_operators * pop_size``
#   evaluations.
#
# Every offspring is evaluated exactly once: a duplicate is regenerated (more
# LLM calls) before its single evaluation, and a failed evaluation still
# counts. Only an exception before the evaluation (for example an LLM error)
# skips it, so the real total can be lower, never higher.
_EOH_INIT_BATCHES = 2


def _eoh_schedule(budget: int, pop_size: int, n_operators: int) -> tuple[int, int]:
    """Population size and number of generations that fit in ``budget``.

    EoH makes ``2 * pop_size`` evaluations to initialise and
    ``n_operators * pop_size`` per generation, so the number of generations is
    ``(budget - 2 * pop_size) // (n_operators * pop_size)``: the most whole
    generations within the budget. The remainder (less than one generation)
    is not used. When the budget does not cover the initialisation, the
    population size is reduced to ``budget // 2`` (at least 1) and no
    generation is run; with a budget of 1 the second initial evaluation is
    stopped by the adapter's budget cap.

    Args:
        budget: Number of evaluations the method may make (at least 1).
        pop_size: Requested EoH population size (at least 1).
        n_operators: Number of EoH operators applied per generation.

    Returns:
        ``(pop_size, n_generations)`` to give to EoH.

    Raises:
        ValueError: If ``budget`` or ``pop_size`` is smaller than 1.
    """
    if budget < 1:
        raise ValueError(f"EoH needs a budget of at least 1, got {budget}.")
    if pop_size < 1:
        raise ValueError(f"EoH needs pop_size of at least 1, got {pop_size}.")
    init_evals = _EOH_INIT_BATCHES * pop_size
    if budget < init_evals:
        return max(1, budget // _EOH_INIT_BATCHES), 0
    n_generations = (budget - init_evals) // (max(1, n_operators) * pop_size)
    return pop_size, n_generations


class _BladeProblemAdapter:
    """Wraps a BLADE problem for use with EoH.

    Args:
        problem: The BLADE problem.
        budget: Maximum number of calls to ``problem``; once used, further
            evaluations return the failure objective (``inf``, EoH minimises)
            without calling the problem. ``None`` means no limit.
    """

    def __init__(self, problem: Problem, budget: int | None = None):
        self.problem = problem
        self.prompts = _BladePrompts(problem)
        self.budget = budget
        self.evaluations = 0
        self._lock = threading.Lock()

        # Get optimisation direction/ set default as maximisaiton.
        minimisation = getattr(problem, "minimisation", False)
        self.minimisation = minimisation

    def evaluate(self, code_string):
        with self._lock:
            if self.budget is not None and self.evaluations >= self.budget:
                # Safety net: the budget is used, do not call the problem.
                return math.inf
            self.evaluations += 1
        solution = Solution(
            code=code_string,
            name=first_class_name(code_string) or "AlgorithmName",
            description=class_info(code_string)[1] or "No description provided.",
        )
        solution = self.problem(solution)
        return to_minimised(solution.fitness, self.minimisation)


class _BladeInterfaceLLM:
    """Provide EoH with the BLADE LLM interface."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def get_response(self, prompt_content):
        return self.llm.query([{"role": "user", "content": prompt_content}])


class EoH(Method):
    def __init__(self, llm: LLM, budget: int, name="EoH", **kwargs):
        """
        Initializes the EoH algorithm within the benchmarking framework.

        Args:
            problem (Problem): The problem instance to optimize.
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            name (str): The name of the method.
            kwargs: Additional arguments for configuring EoH.
        """
        super().__init__(llm, budget, name)
        self.kwargs = kwargs

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via EoH.

        Returns:
            Solution: The best solution found.
        """
        if eoh_main is None:
            raise ImportError(
                "EoH package is not installed, , please install it using `poetry install --with methods`."
            )

        # Patch the EoH LLM interface to use our LLM
        if eoh_evolution is not None:
            eoh_evolution.InterfaceLLM = lambda *args, **kwargs: _BladeInterfaceLLM(
                self.llm
            )

        requested_pop_size = self.kwargs.get("pop_size", 4)
        paras = Paras()
        paras.set_paras(
            method="eoh",
            problem=_BladeProblemAdapter(problem, budget=self.budget),
            llm_api_endpoint="unused",
            llm_api_key="unused",
            llm_model=self.llm.model,
            ec_pop_size=requested_pop_size,
            exp_output_path=self.kwargs.get("output_path", "./"),
            exp_n_proc=1,
            exp_debug_mode=False,
            eva_timeout=self.kwargs.get("timeout", 600),
        )
        # Fit the population size and generation count to the budget (see
        # _eoh_schedule); EoH's operator list is known after set_paras.
        pop_size, n_generations = _eoh_schedule(
            self.budget, requested_pop_size, len(paras.ec_operators)
        )
        if pop_size != requested_pop_size:
            warnings.warn(
                f"EoH budget {self.budget} does not cover the initial "
                f"population ({_EOH_INIT_BATCHES} * pop_size = "
                f"{_EOH_INIT_BATCHES * requested_pop_size} evaluations); "
                f"using pop_size={pop_size}.",
                stacklevel=2,
            )
        paras.ec_pop_size = pop_size
        paras.ec_n_pop = n_generations

        evol = eoh_main.EVOL(paras, prob=paras.problem)
        evol.run()

        if n_generations > 0:
            result_file = os.path.join(
                paras.exp_output_path,
                "results",
                "pops_best",
                f"population_generation_{n_generations}.json",
            )
        else:
            # No generation ran: the initial population (sorted best first
            # by EoH's population management) holds the result.
            result_file = os.path.join(
                paras.exp_output_path,
                "results",
                "pops",
                "population_generation_0.json",
            )
        if not os.path.exists(result_file):
            raise FileNotFoundError(result_file)

        with open(result_file) as f:
            best = json.load(f)

        if isinstance(best, list):
            if not best:
                raise RuntimeError(
                    f"EoH produced no evaluated individual ({result_file})."
                )
            best = best[0]

        code = best.get("code", "")
        desc = best.get("algorithm", "")
        name_match = re.search(r"class\s+(\w+)", code)
        name = name_match.group(1) if name_match else "OptimizationAlgorithm"
        solution = Solution(code=code, name=name, description=desc)
        solution.set_scores(
            from_minimised(
                best.get("objective"), getattr(problem, "minimisation", False)
            )
        )
        return solution

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "EoH",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }

    def get_config(self) -> dict[str, Any]:
        args = self.kwargs.copy()
        args["budget"] = self.budget
        return {
            "name": "EoH",
            "source": "https://github.com/nikivanstein/EoH/commit/f0d21b2f2f62c452bac545fd7f68d3af50845321",
            "config": args,
        }
