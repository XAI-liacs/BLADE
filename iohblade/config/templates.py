"""Starter projects for ``iohblade init``.

Each template is a small folder with an ``experiment.yml`` (and stub code)
that passes ``iohblade validate`` and ``iohblade check`` as generated; edit
the TODOs to make it your own.
"""

from __future__ import annotations

from pathlib import Path

_HEADER = """\
# yaml-language-server: $schema=./experiment.schema.json
# Created by `iohblade init --template {template}`.
#   uv run iohblade validate experiment.yml   # check the configuration
#   uv run iohblade check experiment.yml      # smoke test, no LLM calls
#   uv run iohblade run experiment.yml        # run (or resume)
version: 1

"""

_LLMS = """\
llms:
  # TODO: choose your LLM. API keys are read from the environment (or a
  # .env file next to this file), never written here.
  local:
    provider: ollama
    model: "qwen2.5-coder:14b"
  # gemini:
  #   provider: gemini
  #   model: gemini-2.0-flash
  #   api_key: ${env:GEMINI_API_KEY}
"""

_LLAMEA = """\
  - type: llamea
    name: "LLaMEA-{llm}"
    llm: local
    params:
      n_parents: 2
      n_offspring: 2
      operators:
        - name: refine
          prompt: "Refine the strategy of the selected solution to improve it."
          weight: 0.7
          number_of_parents: 1
        - name: crossover
          prompt: "Combine the ideas of the two selected solutions into a better one."
          weight: 0.3
          number_of_parents: 2
"""

_CIRCLE_REFERENCE_PY = """\
import math
import random


class CirclePackingSolver:
    \"\"\"Greedy reference: place the largest radii first at random free spots.\"\"\"

    def __init__(self, container, candidate_radii, tolerance):
        self.cx, self.cy, self.R = container
        self.radii = sorted(candidate_radii, reverse=True)
        self.tol = tolerance

    def __call__(self):
        rng = random.Random(0)
        circles = []
        for r in self.radii:
            for _ in range(200):
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(0, max(self.R - r - 2 * self.tol, 0))
                x = self.cx + dist * math.cos(angle)
                y = self.cy + dist * math.sin(angle)
                if all(
                    math.hypot(x - x2, y - y2) > r + r2 + 2 * self.tol
                    for x2, y2, r2 in circles
                ):
                    circles.append([x, y, r])
                    break
        return circles
"""

BUILTIN = {
    "experiment.yml": _HEADER.format(template="builtin")
    + """\
experiment:
  output_dir: results
  runs: 3
  budget: 20

"""
    + _LLMS
    + """
methods:
"""
    + _LLAMEA
    + """\
  - type: random_search
    name: "RS-{llm}"
    llm: local

problems:
  # `uv run iohblade list problems` shows all built-in problems.
  - type: circle_packing
    smoke_test_solution: reference_solution.py   # used by `iohblade check`
""",
    "reference_solution.py": _CIRCLE_REFERENCE_PY,
}

_PROBLEM_PY = '''\
"""TODO: describe your benchmark.

The LLM writes a class ``Optimizer(budget, dim)`` with ``__call__(func)``
that minimises ``func``; the fitness is the negated best value found
(BLADE maximises fitness by default, set ``self.minimisation = True`` to
minimise instead).
"""

from typing import Any

import numpy as np

from iohblade.problem import Problem
from iohblade.solution import Solution
from iohblade.utils import first_class_name

# Shown to the LLM as an example (the same code as reference_solution.py).
EXAMPLE = """__REFERENCE__"""


class MyProblem(Problem):
    def __init__(self, dim: int = 5, budget: int = 200, name="MyProblem"):
        super().__init__(name=name, eval_timeout=60)
        self.dim = dim
        self.budget = budget
        self.minimisation = False
        # TODO: describe the task to the LLM.
        self.task_prompt = (
            f"Write a Python class that minimises a black-box function of {dim} "
            f"variables in [-5, 5] using at most {budget} function evaluations.\\n"
        )
        self.example_prompt = "An example:\\n```python\\n" + EXAMPLE + "```\\n"
        self.format_prompt = "Give the full class in one ```python code block.\\n"

    def evaluate(self, solution: Solution) -> Solution:
        # Packages needed only here can be imported inside this method and
        # listed under 'dependencies' in experiment.yml.
        namespace: dict[str, Any] = {}
        exec(solution.code, namespace)
        algorithm = namespace[first_class_name(solution.code)](
            budget=self.budget, dim=self.dim
        )
        shift = np.random.default_rng(0).uniform(-3, 3, self.dim)

        def func(x):  # TODO: your objective
            return float(np.sum((np.asarray(x) - shift) ** 2))

        best_f, _ = algorithm(func)
        solution.set_scores(-float(best_f), f"Best value found: {best_f:.4g}.")
        return solution

    def test(self, solution: Solution) -> Solution:
        return self.evaluate(solution)

    def to_dict(self) -> dict[str, Any]:
        return {"dim": self.dim, "budget": self.budget}

    def get_config(self) -> dict[str, Any]:
        return {
            "tags": [],
            "name": self.name,
            "prompt": self.get_prompt(),
            "minimisation": self.minimisation,
            "config": self.to_dict(),
        }
'''

_REFERENCE_PY = """\
import numpy as np


class RandomSearch:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        best_f, best_x = float("inf"), None
        for _ in range(self.budget):
            x = np.random.uniform(-5, 5, self.dim)
            f = func(x)
            if f < best_f:
                best_f, best_x = f, x
        return best_f, best_x
"""

_PROBLEM_PY = _PROBLEM_PY.replace("__REFERENCE__", _REFERENCE_PY)

CUSTOM_PROBLEM = {
    "experiment.yml": _HEADER.format(template="custom-problem")
    + """\
experiment:
  output_dir: results
  runs: 3
  budget: 20

"""
    + _LLMS
    + """
methods:
"""
    + _LLAMEA
    + """
problems:
  - type: custom
    source: my_problem.py              # or a package folder
    class: MyProblem
    params: {dim: 5, budget: 200}
    smoke_test_solution: reference_solution.py   # used by `iohblade check`
    dependencies: []                   # extra pip packages for evaluate()
""",
    "my_problem.py": _PROBLEM_PY,
    "reference_solution.py": _REFERENCE_PY,
}

_METHOD_PY = '''\
"""TODO: describe your search method.

A (1+1) strategy: ask the LLM for a solution, then repeatedly ask it to
improve the best one so far, keeping the better of parent and child.
"""

from typing import Any

from iohblade.method import Method
from iohblade.problem import Problem


def better(a, b, minimise):
    """Whether fitness a is strictly better than fitness b."""
    return a < b if minimise else a > b


class MyMethod(Method):
    def __init__(self, llm, budget, name="MyMethod", improve_prompt=None):
        super().__init__(llm, budget, name)
        self.improve_prompt = improve_prompt or (
            "Improve the following solution. Feedback from its evaluation: "
        )

    def __call__(self, problem: Problem):
        minimise = getattr(problem, "minimisation", False)
        prompt = [{"role": "user", "content": problem.get_prompt()}]
        best = problem(self.llm.sample_solution(prompt))
        for _ in range(self.budget - 1):
            message = (
                f"{problem.get_prompt()}\\n{self.improve_prompt}{best.feedback}\\n"
                f"```python\\n{best.code}\\n```"
            )
            child = problem(
                self.llm.sample_solution([{"role": "user", "content": message}])
            )
            if better(child.fitness, best.fitness, minimise):
                best = child
        return best

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "budget": self.budget}

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "my_method.py",
            "config": {"improve_prompt": self.improve_prompt},
        }
'''

CUSTOM_METHOD = {
    "experiment.yml": _HEADER.format(template="custom-method")
    + """\
experiment:
  output_dir: results
  runs: 3
  budget: 20

"""
    + _LLMS
    + """
methods:
  - type: custom
    source: my_method.py
    class: MyMethod
    name: "MyMethod-{llm}"
    llm: local
  - type: random_search            # a baseline to compare with
    name: "RS-{llm}"
    llm: local

problems:
  - type: circle_packing
    smoke_test_solution: reference_solution.py   # used by `iohblade check`
""",
    "my_method.py": _METHOD_PY,
    "reference_solution.py": _CIRCLE_REFERENCE_PY,
}

TEMPLATES: dict[str, dict[str, str]] = {
    "builtin": BUILTIN,
    "custom-problem": CUSTOM_PROBLEM,
    "custom-method": CUSTOM_METHOD,
}


def write_template(name: str, directory: str | Path, force: bool = False) -> list[Path]:
    """Write template ``name`` into ``directory``.

    Args:
        name: One of :data:`TEMPLATES`.
        directory: Target folder (created if needed).
        force: Overwrite existing files.

    Returns:
        list[Path]: The files written.

    Raises:
        FileExistsError: If a file exists and ``force`` is False.
    """
    import json

    from .schema import json_schema

    directory = Path(directory)
    files = dict(TEMPLATES[name])
    files["experiment.schema.json"] = json.dumps(json_schema(), indent=2) + "\n"
    existing = [directory / f for f in files if (directory / f).exists()]
    if existing and not force:
        raise FileExistsError(
            "Not overwriting existing files: "
            + ", ".join(str(p) for p in existing)
            + " (use --force)."
        )
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, content in files.items():
        path = directory / filename
        path.write_text(content)
        written.append(path)
    return written
