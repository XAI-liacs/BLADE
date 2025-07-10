import ast
import json
import logging
import os
import re
import textwrap
from typing import Optional, Tuple

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


class _BladeProblemAdapter:
    """Wraps a BLADE problem for use with EoH."""

    def __init__(self, problem: Problem):
        self.problem = problem
        self.prompts = _BladePrompts(problem)

    def evaluate(self, code_string):
        solution = Solution(
            code=code_string,
            name=first_class_name(code_string) or "AlgorithmName",
            description=class_info(code_string)[1] or "No description provided.",
        )
        solution = self.problem(solution)
        return (
            -solution.fitness
        )  # EoH minimizes the fitness, so we return negative value.


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

        paras = Paras()
        paras.set_paras(
            method="eoh",
            problem=_BladeProblemAdapter(problem),
            llm_api_endpoint="unused",
            llm_api_key="unused",
            llm_model=self.llm.model,
            ec_pop_size=self.kwargs.get("pop_size", 4),
            ec_n_pop=max(1, self.budget // self.kwargs.get("pop_size", 4)),
            exp_output_path=self.kwargs.get("output_path", "./"),
            exp_n_proc=1,
            exp_debug_mode=False,
        )

        evol = eoh_main.EVOL(paras, prob=paras.problem)
        evol.run()

        result_file = os.path.join(
            paras.exp_output_path,
            "results",
            "pops_best",
            f"population_generation_{paras.ec_n_pop}.json",
        )
        if not os.path.exists(result_file):
            raise FileNotFoundError(result_file)

        with open(result_file) as f:
            best = json.load(f)

        if isinstance(best, list):
            best = best[0]

        code = best.get("code", "")
        desc = best.get("algorithm", "")
        name_match = re.search(r"class\s+(\w+)", code)
        name = name_match.group(1) if name_match else "OptimizationAlgorithm"
        solution = Solution(code=code, name=name, description=desc)
        solution.set_scores(-best.get("objective", 0))
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
