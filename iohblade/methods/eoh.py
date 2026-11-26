import ast
import contextlib
import json
import logging
import math
import os
import re
import tempfile
import textwrap
import threading
import warnings
from typing import Any, Optional, Tuple

from ..fitness import from_minimised, to_minimised
from ..llm import LLM, _extract_python_block
from ..method import Method
from ..problem import Problem
from ..solution import Solution
from ..utils import class_info, first_class_name
from .prompt_utils import declared_interface_matches, example_interface

try:
    from eoh import eoh as eoh_main
    from eoh.methods.eoh import eoh_evolution, eoh_interface_EC
    from eoh.utils.getParas import Paras
except Exception:  # pragma: no cover - optional dependency
    eoh_main = None
    Paras = None
    eoh_evolution = None
    eoh_interface_EC = None


class _BladePrompts:
    """Adapter exposing the prompt interface expected by EoH.

    EoH states the interface of the code to write (``func_name``,
    ``init_inputs``, ``func_inputs`` and ``func_outputs``) in every prompt.
    ``Problem`` defaults these to the BBOB interface (``__init__(self,
    budget, dim)``, ``__call__(self, func)`` returning ``f_opt, x_opt``),
    which contradicts the task of most other problems. The declared
    interface is used only when it matches the class in the problem's
    example code (or there is no example class); otherwise the interface of
    the example class is stated instead, and no output names (the task
    describes what to return).
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.declared_interface = declared_interface_matches(problem)
        self.example = example_interface(problem)

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

    def get_example_interface(self) -> str:
        """The instruction that replaces EoH's interface sentence when the
        declared interface does not match the example (see the class
        docstring)."""
        ex = self.example
        init = ", ".join(["self"] + ex["init_inputs"])
        text = (
            "implement it in Python as a class with the same interface as "
            f"the example class `{ex['class']}` in the task: an "
            f"`__init__({init})`"
        )
        if ex["func_name"]:
            call = ", ".join(["self"] + ex["func_inputs"])
            text += f" and a method `{ex['func_name']}({call})`"
        return text + " that returns what the task asks for. "

    def get_other_inf(self):
        return (
            self.problem.task_prompt
            + self.problem.example_prompt
            + self.problem.format_prompt
        )


# Per-run state of the EoH runs in this thread. Runs execute in parallel
# threads of one process (Experiment n_jobs > 1) and EoH looks its LLM
# interface and Evolution class up as module globals, so the globals are
# patched once (_install_eoh_patches) and read the LLM and output folder of
# the run in the current thread from here.
_EOH_RUN = threading.local()

_DESC_BRACES = re.compile(r"\{(.*?)\}", re.DOTALL)
_DESC_LINE = re.compile(r"#\s*Description\s*:\s*(.*)", re.IGNORECASE)
_CODE_START = re.compile(r"^(?:import|from|class|def|@)\b", re.MULTILINE)


def _extract_eoh_response(response: str) -> tuple[str, str]:
    """The code and the algorithm description in an LLM response to EoH.

    EoH's own parser keeps the text from the first ``import`` (or ``def``)
    up to the *last* ``return`` and appends ``return`` + the problem's
    output names (``f_opt, x_opt`` by default), which breaks every class
    that returns something else, drops a class line that is not preceded
    by an import, and takes the description from the first ``{`` to the
    last ``}`` of the response (dict literals included). Here the code is
    the fenced python block (or, unfenced, everything from the first
    ``import``/``from``/``class``/``def`` line), unchanged, and the
    description the first ``{...}`` outside the code, else a
    ``# Description:`` line, else the first line of text before the code.

    Returns:
        ``(code, description)``; ``code`` is ``""`` if there is none.
    """
    code = _extract_python_block(response)
    if code is not None:
        outside = response.replace(code, "\n")
    else:
        start = _CODE_START.search(response)
        code = response[start.start() :] if start else ""
        outside = response[: start.start()] if start else response
    code = code.strip("\n")
    match = _DESC_BRACES.search(outside) or _DESC_LINE.search(outside)
    if match:
        description = match.group(1)
    else:
        lines = [
            ln.strip("# ").strip()
            for ln in outside.splitlines()
            if ln.strip() and not ln.strip().startswith("```")
        ]
        description = lines[0] if lines else ""
    return code, " ".join(description.split())


if eoh_evolution is not None:

    class _BladeEvolution(eoh_evolution.Evolution):
        """EoH's ``Evolution`` with BLADE's prompt interface and code
        extraction (see :class:`_BladePrompts` and
        :func:`_extract_eoh_response`)."""

        _MAX_RETRIES = 3

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            prompts = args[6] if len(args) > 6 else kwargs["prompts"]
            self._replacement = None
            if isinstance(prompts, _BladePrompts) and not prompts.declared_interface:
                # The sentence EoH builds in every prompt (eoh_evolution.py).
                original = (
                    "implement it in Python as a function named "
                    + self.prompt_func_name
                    + ". This function should accept "
                    + str(len(self.prompt_func_inputs))
                    + " input(s): "
                    + self.joined_inputs
                    + ". The function should return "
                    + str(len(self.prompt_func_outputs))
                    + " output(s): "
                    + self.joined_outputs
                    + ". "
                    + self.prompt_inout_inf
                    + " "
                )
                self._replacement = (original, prompts.get_example_interface())

        def _fix_prompt(self, prompt: str) -> str:
            if self._replacement is None:
                return prompt
            original, replacement = self._replacement
            prompt = prompt.replace(original, replacement)
            # m3 states only the class sentence.
            return prompt.replace(self.prompt_inout_inf, replacement.strip())

        def get_prompt_i1(self):
            return self._fix_prompt(super().get_prompt_i1())

        def get_prompt_e1(self, indivs):
            return self._fix_prompt(super().get_prompt_e1(indivs))

        def get_prompt_e2(self, indivs):
            return self._fix_prompt(super().get_prompt_e2(indivs))

        def get_prompt_m1(self, indiv1):
            return self._fix_prompt(super().get_prompt_m1(indiv1))

        def get_prompt_m2(self, indiv1):
            return self._fix_prompt(super().get_prompt_m2(indiv1))

        def get_prompt_m3(self, indiv1):
            return self._fix_prompt(super().get_prompt_m3(indiv1))

        def _get_alg(self, prompt_content):
            """Query the LLM (EoH retries up to 4 times when there is no
            code) and return ``[code, description]`` as written by the LLM."""
            for _ in range(1 + self._MAX_RETRIES):
                response = self.interface_llm.get_response(prompt_content)
                code, description = _extract_eoh_response(response)
                if code:
                    return [code, description]
            raise ValueError("EoH: no code found in the LLM response.")

    class _JsonlinesToRunFolder:
        """Stand-in for the ``jsonlines`` module in ``eoh_interface_EC``:
        EoH appends every offspring to ``./EoHrun.jsonl``, shared by all
        runs in the working directory; relative paths go to the run's
        output folder instead."""

        def __init__(self, jsonlines_module):
            self._jsonlines = jsonlines_module

        def open(self, path, *args, **kwargs):
            folder = getattr(_EOH_RUN, "output_dir", None)
            if folder and not os.path.isabs(path):
                path = os.path.join(folder, os.path.basename(path))
            return self._jsonlines.open(path, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._jsonlines, name)


def _run_llm(*args, **kwargs):
    """EoH's ``InterfaceLLM`` replacement: the LLM of this thread's run
    (None outside a run; querying it then fails)."""
    return _BladeInterfaceLLM(getattr(_EOH_RUN, "llm", None))


_EOH_PATCH_LOCK = threading.Lock()


def _install_eoh_patches() -> None:
    """Route EoH's LLM calls, prompts, code extraction and run log through
    BLADE (idempotent; the patched globals read per-thread run state)."""
    with _EOH_PATCH_LOCK:
        eoh_evolution.InterfaceLLM = _run_llm
        eoh_interface_EC.Evolution = _BladeEvolution
        if not isinstance(eoh_interface_EC.jsonlines, _JsonlinesToRunFolder):
            eoh_interface_EC.jsonlines = _JsonlinesToRunFolder(
                eoh_interface_EC.jsonlines
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

        _install_eoh_patches()

        with contextlib.ExitStack() as stack:
            # A folder of its own per run: EoH writes its populations to
            # <output>/results and the result is read back from there.
            output_dir = self._output_dir(problem, stack)
            _EOH_RUN.llm = self.llm
            _EOH_RUN.output_dir = output_dir
            stack.callback(_EOH_RUN.__dict__.clear)
            requested_pop_size = self.kwargs.get("pop_size", 4)
            paras = Paras()
            paras.set_paras(
                method="eoh",
                problem=_BladeProblemAdapter(problem, budget=self.budget),
                llm_api_endpoint="unused",
                llm_api_key="unused",
                llm_model=self.llm.model,
                ec_pop_size=requested_pop_size,
                exp_output_path=output_dir,
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

    def _output_dir(self, problem: Problem, stack: contextlib.ExitStack) -> str:
        """The directory EoH writes its ``results`` folder to.

        ``output_path`` if given; otherwise an ``eoh`` folder in the run's log
        directory when the problem has a logger, else a temporary directory
        that is removed afterwards. Never the working directory, which
        parallel runs share.
        """
        explicit = self.kwargs.get("output_path")
        if explicit:
            path = os.path.abspath(explicit)
        elif getattr(problem, "logger", None) is not None and getattr(
            problem, "logger_dir", ""
        ):
            path = os.path.abspath(os.path.join(problem.logger_dir, "eoh"))
        else:
            return stack.enter_context(tempfile.TemporaryDirectory(prefix="eoh-"))
        # EoH creates these with a check-then-create; create them up front
        # so that runs sharing an explicit output_path do not race.
        for sub in ("history", "pops", "pops_best"):
            os.makedirs(os.path.join(path, "results", sub), exist_ok=True)
        return path

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
